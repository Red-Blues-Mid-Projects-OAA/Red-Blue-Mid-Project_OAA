"""
SVM 하이퍼파라미터 최적화 (누수 방지 + Deep Scaling 버전).

목표 제약:
- Accuracy >= 0.52
- IC >= 0.05
- Train-Test Gap(여기서는 Train-Validation Gap) <= 0.25

주의:
- 이 스크립트는 튜닝 시 test 데이터를 사용하지 않습니다.
- 튜닝 평가는 stride별 train/validation만 사용합니다.
- 최종 test 평가는 svm_pipeline.py에서만 수행합니다.
"""

import json
import os
import sys

import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.svm import SVC

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLASSIFICATION_DIR = os.path.dirname(_MODELS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

from split_dataset import get_stride_splits, split_dataset
from model_config import (
    SVM_PARAMS_ARTIFACT_PATH,
    save_json_artifact_only,
)

optuna.logging.set_verbosity(optuna.logging.INFO)

N_TRIALS = 400

TARGET_ACC = 0.52
TARGET_IC = 0.05
TARGET_GAP_MAX = 0.25
TARGET_BALANCE_CENTER = 0.50
TARGET_BALANCE_TOL = 0.15
FIXED_THRESHOLD = 0.50

SEED_PARAMS = {
    "kernel": "sigmoid",
    "C": 0.6,
    "gamma": 0.03,
}


def safe_ic(pred_proba, actual_excess_return):
    """짧은/상수열에서 NaN 안전 처리 포함 Spearman IC 계산."""
    x = np.asarray(pred_proba)
    y = np.asarray(actual_excess_return)
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan
    ic, p_value = spearmanr(x, y)
    return float(ic), float(p_value)


def evaluate_params(params, stride_splits, feature_cols, threshold=FIXED_THRESHOLD):
    """stride train/validation 기준으로 누수 없이 성능을 계산합니다."""
    train_accs = []
    val_accs = []
    val_precs = []
    val_ics = []
    val_ps = []
    val_loglosses = []
    val_pos_rates = []

    for ss in stride_splits:
        x_train = ss.train[feature_cols]
        y_train = ss.train["Target_Class"].astype(int)
        x_val = ss.val[feature_cols]
        y_val = ss.val["Target_Class"].astype(int)

        # split_dataset()에서 이미 스케일된 입력을 받아 추가 스케일링을 하지 않습니다.
        model = SVC(**params)
        model.fit(x_train, y_train)

        train_proba = model.predict_proba(x_train)[:, 1]
        val_proba = model.predict_proba(x_val)[:, 1]
        y_train_pred = (train_proba >= threshold).astype(int)
        y_val_pred = (val_proba >= threshold).astype(int)

        val_excess = ss.val["Target_AAPL_3M"] - ss.val["Target_SP500_3M"]
        ic, ic_p = safe_ic(val_proba, val_excess)

        train_accs.append(accuracy_score(y_train, y_train_pred))
        val_accs.append(accuracy_score(y_val, y_val_pred))
        val_precs.append(precision_score(y_val, y_val_pred, zero_division=0))
        val_ics.append(ic)
        val_ps.append(ic_p)
        val_loglosses.append(log_loss(y_val, val_proba, labels=[0, 1]))
        val_pos_rates.append(float(y_val_pred.mean()))

    avg_train_acc = float(np.mean(train_accs))
    avg_val_acc = float(np.mean(val_accs))
    gap = avg_train_acc - avg_val_acc

    valid_ics = [v for v in val_ics if not np.isnan(v)]
    valid_ps = [v for v in val_ps if not np.isnan(v)]

    return {
        "accuracy": avg_val_acc,
        "precision": float(np.mean(val_precs)),
        "avg_train_acc": avg_train_acc,
        "gap": gap,
        "ic": float(np.mean(valid_ics)) if valid_ics else np.nan,
        "ic_p": float(np.mean(valid_ps)) if valid_ps else np.nan,
        "val_logloss": float(np.mean(val_loglosses)),
        "val_pos_rate": float(np.mean(val_pos_rates)),
        "threshold": float(threshold),
    }


def trial_to_params(trial):
    """Optuna trial 값을 SVC 파라미터 딕셔너리로 변환합니다."""
    # IC 안정성을 위해 과도한 굴곡을 만들기 쉬운 poly는 제외
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"])

    params = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "kernel": kernel,
        "probability": True,
        "random_state": 42,
        "class_weight": None,
    }

    if kernel in ["rbf", "sigmoid"]:
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)

    return params


def build_objective(stride_splits, feature_cols):
    """목표 정렬형 강건 objective 함수를 생성합니다."""

    def objective(trial):
        params = trial_to_params(trial)
        metrics = evaluate_params(params, stride_splits, feature_cols, threshold=FIXED_THRESHOLD)

        acc = metrics["accuracy"]
        ic = metrics["ic"]
        gap = metrics["gap"]
        val_logloss = metrics["val_logloss"]
        pos_rate = metrics["val_pos_rate"]

        # 목표 정렬형 점수(낮을수록 좋음)
        p_ic = max(0.0, TARGET_IC - (ic if not np.isnan(ic) else 0.0))
        p_gap = max(0.0, gap - TARGET_GAP_MAX)
        p_balance = max(0.0, abs(pos_rate - TARGET_BALANCE_CENTER) - TARGET_BALANCE_TOL)

        score = (
            (1.0 - acc)
            + 0.25 * val_logloss
            + 2.2 * p_ic
            + 1.3 * p_gap
            + 1.0 * p_balance
        )

        constraints = {
            "acc_ok": acc >= TARGET_ACC,
            "ic_ok": (not np.isnan(ic)) and ic >= TARGET_IC,
            "gap_ok": gap <= TARGET_GAP_MAX,
            "balance_ok": abs(pos_rate - TARGET_BALANCE_CENTER) <= TARGET_BALANCE_TOL,
        }

        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("params_expanded", params)
        trial.set_user_attr("constraints", constraints)
        return score

    return objective


def choose_best_trial(study):
    """제약 충족 우선순위로 best trial을 선택합니다."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    fully_feasible = [
        t for t in completed
        if all(
            t.user_attrs.get("constraints", {}).get(k, False)
            for k in ("acc_ok", "ic_ok", "gap_ok", "balance_ok")
        )
    ]
    if fully_feasible:
        return min(fully_feasible, key=lambda t: t.value), "all_constraints"

    return min(completed, key=lambda t: t.value), "fallback_objective_best"


def compute_validation_threshold(params, stride_splits, feature_cols):
    """정책상 threshold=0.5를 고정하고 해당 validation 정확도만 계산합니다."""
    all_val_proba = []
    all_val_y = []

    for ss in stride_splits:
        x_train = ss.train[feature_cols]
        y_train = ss.train["Target_Class"].astype(int)
        x_val = ss.val[feature_cols]
        y_val = ss.val["Target_Class"].astype(int)

        # split_dataset()에서 이미 스케일된 입력을 받아 추가 스케일링을 하지 않습니다.
        model = SVC(**params)
        model.fit(x_train, y_train)
        all_val_proba.append(model.predict_proba(x_val)[:, 1])
        all_val_y.append(y_val.to_numpy())

    val_proba = np.concatenate(all_val_proba)
    val_y = np.concatenate(all_val_y)

    fixed_acc = accuracy_score(val_y, (val_proba >= FIXED_THRESHOLD).astype(int))
    return float(FIXED_THRESHOLD), float(fixed_acc)


def optimize():
    """제약조건 기반 튜닝 실행 후 best_svm_params.json에 저장합니다."""
    print("=" * 70)
    print("SVM 제약조건 기반 최적화 (누수 방지, validation 기준)")
    print("=" * 70)
    print(
        f"목표: acc>={TARGET_ACC:.2f}, ic>={TARGET_IC:.2f}, "
        f"gap<={TARGET_GAP_MAX:.2f}, threshold={FIXED_THRESHOLD:.2f} (고정)"
    )

    split = split_dataset()
    stride_splits = get_stride_splits(split)

    study = optuna.create_study(direction="minimize", study_name="svm_constraints_tuning")
    study.enqueue_trial(SEED_PARAMS)
    study.optimize(
        build_objective(stride_splits, split.feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best, best_source = choose_best_trial(study)

    if best_source == "fallback_objective_best":
        print("\n[WARN] 모든 제약을 동시 충족한 trial이 없어 objective 최적 trial 사용")

    best_params = dict(best.user_attrs["params_expanded"])
    best_metrics = dict(best.user_attrs["metrics"])
    best_constraints = dict(best.user_attrs["constraints"])
    best_threshold, best_threshold_acc = compute_validation_threshold(
        best_params, stride_splits, split.feature_cols
    )

    print("\n" + "=" * 70)
    print("최적 Trial")
    print("=" * 70)
    print(f"Trial: #{best.number}")
    print(f"Score: {best.value:.6f}")
    print("파라미터:")
    for k, v in best_params.items():
        print(f"  {k:15s}: {v}")

    gap_bp = best_metrics["gap"] * 10000.0
    print("지표 (validation 기준):")
    print(f"  Accuracy      : {best_metrics['accuracy']:.4f}")
    print(f"  Precision     : {best_metrics['precision']:.4f}")
    print(f"  IC            : {best_metrics['ic']:.4f} (p={best_metrics['ic_p']:.4f})")
    print(f"  Avg Train Acc : {best_metrics['avg_train_acc']:.4f}")
    print(f"  Gap           : {best_metrics['gap']:.4f} ({gap_bp:.0f}bp)")
    print(f"  Best Threshold: {best_threshold:.2f} (val_acc={best_threshold_acc:.4f})")
    print("제약 충족 여부:")
    print(f"  Accuracy >= {TARGET_ACC:.2f} : {best_constraints['acc_ok']}")
    print(f"  IC >= {TARGET_IC:.2f}        : {best_constraints['ic_ok']}")
    print(f"  Gap <= {TARGET_GAP_MAX:.2f}      : {best_constraints['gap_ok']}")
    print(f"  |PosRate-0.5| <= {TARGET_BALANCE_TOL:.2f} : {best_constraints['balance_ok']}")

    save_data = {
        "best_params": best_params,
        "best_score": best.value,
        "best_trial_number": best.number,
        "best_source": best_source,
        "n_trials": N_TRIALS,
        "stride": 5,
        "n_models": 5,
        "objective": "deep_target_aligned_v3_fixed_threshold_0.5_no_class_weight",
        "targets": {
            "accuracy_min": TARGET_ACC,
            "ic_min": TARGET_IC,
            "gap_max": TARGET_GAP_MAX,
            "balance_center": TARGET_BALANCE_CENTER,
            "balance_tolerance": TARGET_BALANCE_TOL,
        },
        "selected_metrics": best_metrics,
        "decision_threshold": best_threshold,
        "decision_threshold_metric": "fixed_policy_0.5_validation_accuracy",
        "decision_threshold_score": best_threshold_acc,
        "fit_mode": "train_only",
        "constraints_satisfied": best_constraints,
        "leakage_policy": "test_not_used_in_tuning",
    }

    save_json_artifact_only(save_data, SVM_PARAMS_ARTIFACT_PATH)
    print(f"\n저장 완료: {SVM_PARAMS_ARTIFACT_PATH}")
    print("=" * 70)
    return save_data


if __name__ == "__main__":
    optimize()
