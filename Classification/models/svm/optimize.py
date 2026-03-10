"""
이 파일은 최적화 관련 작업을 담당합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.svm import SVC

from Classification.Preprocessing.split_dataset import get_stride_splits, split_dataset
from Classification.model_config import (
    get_model_params_path,
    save_json_artifact_only,
)

optuna.logging.set_verbosity(optuna.logging.INFO)

# Optuna가 시도할 하이퍼파라미터 탐색 횟수입니다.
N_TRIALS = 100
# 현재 파일이 기대하는 목적 함수 버전 문자열입니다.
BASE_OBJECTIVE_VERSION = "target_aligned_v3_svm_no_class_weight"
# 저장된 파라미터와 비교할 교차검증 분할 정책 이름입니다.
CV_MODE = "single_holdout_2024Q2Q3"
# 단계형 게이트 전략 전용 튜닝 분기를 구분하는 프로필 이름입니다.
TSM_GATE_PROFILE = "tsm_gatehard_v1"
# 단계형 게이트 전략에서 허용하는 스테이지 집합입니다.
TSM_STAGES = {"stage1", "stage2"}

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

def _get_feature_hash(feature_cols):
    """피처 hash 정보를 조회해 반환합니다."""
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _get_current_data_end_date(split):
    """current 데이터 end date 정보를 조회해 반환합니다."""
    candidates = []
    for df in [split.train, split.val, split.test, split.final_train]:
        if len(df) > 0:
            candidates.append(df.index.max())
    if not candidates:
        return None
    return max(candidates).strftime("%Y-%m-%d")

def _get_objective_version(profile: str, strategy_stage: str) -> str:
    """objective version 정보를 조회해 반환합니다."""
    if profile == TSM_GATE_PROFILE:
        return f"{TSM_GATE_PROFILE}_{strategy_stage}"
    return BASE_OBJECTIVE_VERSION

def _apply_direction(proba: np.ndarray, direction_mode: str) -> np.ndarray:
    """지표 방향성을 반영해 점수가 클수록 좋도록 맞춥니다."""
    if direction_mode == "normal":
        return proba
    if direction_mode == "inverted":
        return 1.0 - proba
    raise ValueError(f"지원하지 않는 direction_mode 입니다: {direction_mode}")

def _compute_recency_weights(index, recency_weight_lambda: float) -> np.ndarray | None:
    """최근 데이터에 더 큰 비중을 주는 학습 가중치를 계산합니다."""
    if recency_weight_lambda <= 0.0:
        return None
    rank = np.arange(len(index), dtype=float)
    weights = 1.0 + recency_weight_lambda * rank
    weights = np.clip(weights, 1e-8, None)
    return weights / np.mean(weights)

def _normalize_class_weight_dict(class_weight):
    """class 비중 dict 값을 서로 비교하기 쉽게 정규화합니다."""
    if not isinstance(class_weight, dict):
        return class_weight
    out = {}
    for k, v in class_weight.items():
        try:
            key = int(k)
        except Exception:
            key = k
        out[key] = float(v)
    return out

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

def evaluate_params(
    params,
    stride_splits,
    feature_cols,
    target_col,
    benchmark_target_col,
    threshold=FIXED_THRESHOLD,
    direction_mode="normal",
    recency_weight_lambda=0.0,
):
    """stride train/validation 기준으로 누수 없이 성능을 계산합니다."""
    train_accs = []
    val_accs = []
    val_precs = []
    val_ics = []
    val_ps = []
    val_loglosses = []
    val_pos_rates = []

    all_val_dates = []
    all_val_proba = []
    all_val_alpha = []
    all_val_pred = []

    for ss in stride_splits:
        x_train = ss.train[feature_cols]
        y_train = ss.train["Target_Class"].astype(int)
        x_val = ss.val[feature_cols]
        y_val = ss.val["Target_Class"].astype(int)

        model = SVC(**params)
        fit_kwargs = {}
        sw = _compute_recency_weights(ss.train.index, recency_weight_lambda)
        if sw is not None:
            fit_kwargs["sample_weight"] = sw

        model.fit(x_train, y_train, **fit_kwargs)

        train_proba_raw = model.predict_proba(x_train)[:, 1]
        val_proba_raw = model.predict_proba(x_val)[:, 1]

        train_proba = _apply_direction(train_proba_raw, direction_mode)
        val_proba = _apply_direction(val_proba_raw, direction_mode)

        y_train_pred = (train_proba >= threshold).astype(int)
        y_val_pred = (val_proba >= threshold).astype(int)

        val_excess = ss.val[target_col] - ss.val[benchmark_target_col]
        ic, ic_p = safe_ic(val_proba, val_excess)

        train_accs.append(accuracy_score(y_train, y_train_pred))
        val_accs.append(accuracy_score(y_val, y_val_pred))
        val_precs.append(precision_score(y_val, y_val_pred, zero_division=0))
        val_ics.append(ic)
        val_ps.append(ic_p)
        val_loglosses.append(log_loss(y_val, val_proba, labels=[0, 1]))
        val_pos_rates.append(float(y_val_pred.mean()))

        all_val_dates.extend(ss.val.index.to_pydatetime())
        all_val_proba.extend(val_proba.tolist())
        all_val_alpha.extend(val_excess.astype(float).tolist())
        all_val_pred.extend(y_val_pred.tolist())

    avg_train_acc = float(np.mean(train_accs))
    avg_val_acc = float(np.mean(val_accs))
    gap_signed = avg_train_acc - avg_val_acc
    gap_abs = abs(gap_signed)
    val_proba_std = float(np.std(np.asarray(all_val_proba, dtype=float))) if all_val_proba else 0.0
    val_proba_unique = (
        int(np.unique(np.round(np.asarray(all_val_proba, dtype=float), 6)).size)
        if all_val_proba
        else 0
    )
    val_pred_unique = int(np.unique(np.asarray(all_val_pred, dtype=int)).size) if all_val_pred else 0

    valid_ics = [v for v in val_ics if not np.isnan(v)]
    valid_ps = [v for v in val_ps if not np.isnan(v)]

    ic_early = np.nan
    ic_late = np.nan
    if all_val_dates:
        order = np.argsort(np.array(all_val_dates, dtype="datetime64[ns]"))
        ordered_proba = np.asarray(all_val_proba, dtype=float)[order]
        ordered_alpha = np.asarray(all_val_alpha, dtype=float)[order]
        mid = len(ordered_proba) // 2
        if mid > 0:
            ic_early, _ = safe_ic(ordered_proba[:mid], ordered_alpha[:mid])
            ic_late, _ = safe_ic(ordered_proba[mid:], ordered_alpha[mid:])

    return {
        "accuracy": avg_val_acc,
        "precision": float(np.mean(val_precs)),
        "avg_train_acc": avg_train_acc,
        "gap": gap_abs,  # backward-compat alias
        "gap_signed": gap_signed,
        "gap_abs": gap_abs,
        "ic": float(np.mean(valid_ics)) if valid_ics else np.nan,
        "ic_p": float(np.mean(valid_ps)) if valid_ps else np.nan,
        "val_logloss": float(np.mean(val_loglosses)),
        "val_pos_rate": float(np.mean(val_pos_rates)),
        "val_proba_std": val_proba_std,
        "val_proba_unique": val_proba_unique,
        "val_pred_unique": val_pred_unique,
        "threshold": float(threshold),
        "ic_early": float(0.0 if np.isnan(ic_early) else ic_early),
        "ic_late": float(0.0 if np.isnan(ic_late) else ic_late),
    }

def _suggest_class_weight(trial: optuna.Trial, profile: str):
    """클래스 불균형 완화를 위한 권장 가중치를 계산합니다."""
    if profile != TSM_GATE_PROFILE:
        return None, "none"

    mode = trial.suggest_categorical("class_weight_mode_key", ["none", "balanced", "custom"])
    if mode == "none":
        return None, "none"
    if mode == "balanced":
        return "balanced", "balanced"

    w = float(trial.suggest_float("class_weight_1", 1.0, 5.0))
    return {0: 1.0, 1: w}, f"custom_1:{w:.4f}"

def trial_to_params(trial, profile):
    """Optuna trial 값을 SVC 파라미터 딕셔너리로 변환합니다."""
    class_weight, class_weight_mode = _suggest_class_weight(trial, profile)

    if profile == "balanced":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"])
        c_range = (1e-3, 100.0)
        gamma_range = (1e-4, 1.0)
    elif profile == "regularized":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"])
        c_range = (1e-3, 30.0)
        gamma_range = (1e-4, 0.5)
    elif profile == TSM_GATE_PROFILE:
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid", "poly"])
        c_range = (1e-4, 300.0)
        gamma_range = (1e-5, 5.0)
    else:
        raise ValueError(f"지원하지 않는 profile 입니다: {profile}")

    params = {
        "C": trial.suggest_float("C", c_range[0], c_range[1], log=True),
        "kernel": kernel,
        "probability": True,
        "random_state": 42,
        "class_weight": _normalize_class_weight_dict(class_weight),
    }

    if kernel in ["rbf", "sigmoid", "poly"]:
        params["gamma"] = trial.suggest_float("gamma", gamma_range[0], gamma_range[1], log=True)

    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 4)

    return params, class_weight_mode

def _compute_score(metrics: dict, profile: str, strategy_stage: str) -> float:
    """최적화에 사용할 종합 점수를 계산합니다."""
    acc = metrics["accuracy"]
    ic = metrics["ic"]
    gap_abs = metrics["gap_abs"]
    val_logloss = metrics["val_logloss"]
    pos_rate = metrics["val_pos_rate"]
    ic_early = metrics["ic_early"]
    ic_late = metrics["ic_late"]
    val_proba_std = float(metrics.get("val_proba_std", 0.0))
    val_proba_unique = int(metrics.get("val_proba_unique", 0))

    p_ic = max(0.0, TARGET_IC - (ic if not np.isnan(ic) else 0.0))
    p_gap = max(0.0, gap_abs - TARGET_GAP_MAX)
    p_balance = max(0.0, abs(pos_rate - TARGET_BALANCE_CENTER) - TARGET_BALANCE_TOL)
    flat_penalty = 4.0 * max(0.0, 0.03 - val_proba_std)
    unique_penalty = 0.25 * max(0.0, 8 - float(val_proba_unique))

    if profile in {"balanced", "regularized"}:
        return (
            (1.0 - acc)
            + 0.25 * val_logloss
            + 2.2 * p_ic
            + 1.3 * p_gap
            + 1.0 * p_balance
            + flat_penalty
            + unique_penalty
        )

    # tsm_gatehard_v1
    p_acc = max(0.0, TARGET_ACC - acc)
    score = (
        (1.0 - acc)
        + 0.20 * val_logloss
        + 3.8 * p_acc
        + 3.8 * p_ic
        + 2.6 * p_gap
        + 1.6 * p_balance
        + flat_penalty
        + unique_penalty
    )

    if strategy_stage == "stage2":
        score += 2.0 * abs(ic_early - ic_late)
        if ic_early * ic_late < 0:
            score += 1.6

    return float(score)

def build_objective(stride_splits, feature_cols, target_col, benchmark_target_col, profile, strategy_stage):
    """목표 정렬형 강건 objective 함수를 생성합니다."""

    def objective(trial):
        """하이퍼파라미터 탐색에서 사용할 목적 함수를 계산합니다."""
        params, class_weight_mode = trial_to_params(trial, profile=profile)

        recency_weight_lambda = 0.0
        if profile == TSM_GATE_PROFILE and strategy_stage == "stage2":
            recency_weight_lambda = float(
                trial.suggest_float("recency_weight_lambda", 0.0, 0.02)
            )

        direction_candidates = ["normal"]
        if profile == TSM_GATE_PROFILE and strategy_stage == "stage2":
            direction_candidates = ["normal", "inverted"]

        best_candidate = None
        for direction_mode in direction_candidates:
            metrics = evaluate_params(
                params,
                stride_splits,
                feature_cols,
                target_col=target_col,
                benchmark_target_col=benchmark_target_col,
                threshold=FIXED_THRESHOLD,
                direction_mode=direction_mode,
                recency_weight_lambda=recency_weight_lambda,
            )
            is_degenerate = (
                float(metrics.get("val_proba_std", 0.0)) < 1e-6
                or int(metrics.get("val_pred_unique", 0)) < 2
            )
            if is_degenerate:
                score = 9e9
            else:
                score = _compute_score(metrics, profile=profile, strategy_stage=strategy_stage)

            constraints = {
                "acc_ok": metrics["accuracy"] >= TARGET_ACC,
                "ic_ok": (not np.isnan(metrics["ic"])) and metrics["ic"] >= TARGET_IC,
                "gap_ok": metrics["gap_abs"] <= TARGET_GAP_MAX,
                "balance_ok": abs(metrics["val_pos_rate"] - TARGET_BALANCE_CENTER) <= TARGET_BALANCE_TOL,
            }

            candidate = {
                "score": float(score),
                "metrics": metrics,
                "constraints": constraints,
                "params_expanded": params,
                "direction_mode": direction_mode,
                "recency_weight_lambda": float(recency_weight_lambda),
                "class_weight_mode": class_weight_mode,
                "degenerate_rejected": bool(is_degenerate),
            }
            if best_candidate is None or candidate["score"] < best_candidate["score"]:
                best_candidate = candidate

        trial.set_user_attr("metrics", best_candidate["metrics"])
        trial.set_user_attr("params_expanded", best_candidate["params_expanded"])
        trial.set_user_attr("constraints", best_candidate["constraints"])
        trial.set_user_attr("direction_mode", best_candidate["direction_mode"])
        trial.set_user_attr("recency_weight_lambda", best_candidate["recency_weight_lambda"])
        trial.set_user_attr("class_weight_mode", best_candidate["class_weight_mode"])
        trial.set_user_attr("degenerate_rejected", best_candidate["degenerate_rejected"])
        trial.set_user_attr("score", best_candidate["score"])
        return best_candidate["score"]

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

def compute_validation_threshold(params, stride_splits, feature_cols, direction_mode="normal"):
    """정책상 threshold=0.5를 고정하고 해당 validation 정확도만 계산합니다."""
    all_val_proba = []
    all_val_y = []

    for ss in stride_splits:
        x_train = ss.train[feature_cols]
        y_train = ss.train["Target_Class"].astype(int)
        x_val = ss.val[feature_cols]
        y_val = ss.val["Target_Class"].astype(int)

        model = SVC(**params)
        model.fit(x_train, y_train)
        val_raw = model.predict_proba(x_val)[:, 1]
        all_val_proba.append(_apply_direction(val_raw, direction_mode))
        all_val_y.append(y_val.to_numpy())

    val_proba = np.concatenate(all_val_proba)
    val_y = np.concatenate(all_val_y)

    fixed_acc = accuracy_score(val_y, (val_proba >= FIXED_THRESHOLD).astype(int))
    return float(FIXED_THRESHOLD), float(fixed_acc)

def optimize(
    profile="balanced",
    n_trials=N_TRIALS,
    ticker="AAPL",
    benchmark="SP500",
    auto_update=True,
    persist_total_features_on_update=True,
    feature_source_mode="db_first",
    strategy_stage="stage2",
):
    """제약조건 기반 튜닝 실행 후 best_svm_params.json에 저장합니다."""
    if profile not in {"balanced", "regularized", TSM_GATE_PROFILE}:
        raise ValueError(
            "profile은 'balanced', 'regularized', 'tsm_gatehard_v1'만 허용됩니다: "
            f"{profile}"
        )

    resolved_stage = "default"
    if profile == TSM_GATE_PROFILE:
        if strategy_stage not in TSM_STAGES:
            raise ValueError(f"strategy_stage는 {sorted(TSM_STAGES)} 중 하나여야 합니다: {strategy_stage}")
        resolved_stage = strategy_stage

    objective_version = _get_objective_version(profile, resolved_stage)

    print("=" * 70)
    print("SVM 제약조건 기반 최적화 (누수 방지, validation 기준)")
    print("=" * 70)
    print(
        f"목표: acc>={TARGET_ACC:.2f}, ic>={TARGET_IC:.2f}, "
        f"gap<={TARGET_GAP_MAX:.2f}, threshold={FIXED_THRESHOLD:.2f} (고정)"
    )

    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()

    split = split_dataset(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=auto_update,
        persist_total_features_on_update=persist_total_features_on_update,
        feature_source_mode=feature_source_mode,
    )
    stride_splits = get_stride_splits(split)

    study = optuna.create_study(
        direction="minimize",
        study_name=f"svm_constraints_tuning_{ticker}_{profile}_{resolved_stage}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.enqueue_trial(SEED_PARAMS)
    study.optimize(
        build_objective(
            stride_splits,
            split.feature_cols,
            target_col=split.target_col,
            benchmark_target_col=split.benchmark_target_col,
            profile=profile,
            strategy_stage=resolved_stage,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best, best_source = choose_best_trial(study)

    if best_source == "fallback_objective_best":
        print("\n[WARN] 모든 제약을 동시 충족한 trial이 없어 objective 최적 trial 사용")

    best_params = dict(best.user_attrs["params_expanded"])
    best_metrics = dict(best.user_attrs["metrics"])
    best_constraints = dict(best.user_attrs["constraints"])
    best_direction_mode = str(best.user_attrs.get("direction_mode", "normal"))
    best_recency_weight_lambda = float(best.user_attrs.get("recency_weight_lambda", 0.0))
    best_class_weight_mode = str(best.user_attrs.get("class_weight_mode", "none"))
    best_threshold, best_threshold_acc = compute_validation_threshold(
        best_params,
        stride_splits,
        split.feature_cols,
        direction_mode=best_direction_mode,
    )

    print("\n" + "=" * 70)
    print("최적 Trial")
    print("=" * 70)
    print(f"Trial: #{best.number}")
    print(f"Score: {best.value:.6f}")
    print(f"Direction Mode: {best_direction_mode}")
    print(f"Recency λ: {best_recency_weight_lambda:.6f}")
    print(f"Class Weight Mode: {best_class_weight_mode}")
    print("파라미터:")
    for k, v in best_params.items():
        print(f"  {k:15s}: {v}")

    gap_bp_signed = best_metrics["gap_signed"] * 10000.0
    gap_bp_abs = best_metrics["gap_abs"] * 10000.0
    print("지표 (validation 기준):")
    print(f"  Accuracy      : {best_metrics['accuracy']:.4f}")
    print(f"  Precision     : {best_metrics['precision']:.4f}")
    print(f"  IC            : {best_metrics['ic']:.4f} (p={best_metrics['ic_p']:.4f})")
    print(f"  Avg Train Acc : {best_metrics['avg_train_acc']:.4f}")
    print(
        f"  Gap           : signed={best_metrics['gap_signed']:.4f} ({gap_bp_signed:.0f}bp), "
        f"abs={best_metrics['gap_abs']:.4f} ({gap_bp_abs:.0f}bp)"
    )
    print(f"  Val Proba Std        : {best_metrics['val_proba_std']:.6f}")
    print(f"  Val Proba Unique(6d) : {best_metrics['val_proba_unique']}")
    print(f"  Val Pred Unique      : {best_metrics['val_pred_unique']}")
    print(f"  Best Threshold: {best_threshold:.2f} (val_acc={best_threshold_acc:.4f})")
    print("제약 충족 여부:")
    print(f"  Accuracy >= {TARGET_ACC:.2f} : {best_constraints['acc_ok']}")
    print(f"  IC >= {TARGET_IC:.2f}        : {best_constraints['ic_ok']}")
    print(f"  |Gap| <= {TARGET_GAP_MAX:.2f}      : {best_constraints['gap_ok']}")
    print(f"  |PosRate-0.5| <= {TARGET_BALANCE_TOL:.2f} : {best_constraints['balance_ok']}")

    degenerate_rejected_count = int(
        sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and bool(t.user_attrs.get("degenerate_rejected", False))
        )
    )

    save_data = {
        "best_params": best_params,
        "best_score": float(best.value),
        "best_trial_number": int(best.number),
        "best_source": best_source,
        "n_trials": int(n_trials),
        "stride": 5,
        "n_models": 5,
        "optimization_method": "Single Holdout (Target-Aligned Objective)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_end_date": _get_current_data_end_date(split),
        "feature_count": len(split.feature_cols),
        "feature_hash": _get_feature_hash(split.feature_cols),
        "split_definition": {
            "train": ["2021-01-01", "2023-12-31"],
            "embargo": ["2024-01-01", "2024-03-31"],
            "validation": ["2024-04-01", "2024-09-30"],
            "golden_gap": ["2024-10-01", "2024-12-31"],
            "test": ["2025-01-01", None],
        },
        "profile": profile,
        "objective_version": objective_version,
        "cv_mode": CV_MODE,
        "objective": "deep_target_aligned_v3_fixed_threshold_0.5",
        "targets": {
            "accuracy_min": TARGET_ACC,
            "ic_min": TARGET_IC,
            "gap_max": TARGET_GAP_MAX,
            "balance_center": TARGET_BALANCE_CENTER,
            "balance_tolerance": TARGET_BALANCE_TOL,
        },
        "selected_metrics": {
            **best_metrics,
            "train_val_gap": float(best_metrics["gap_abs"]),
            "train_val_gap_signed": float(best_metrics["gap_signed"]),
            "train_val_gap_abs": float(best_metrics["gap_abs"]),
            "val_proba_std": float(best_metrics["val_proba_std"]),
            "val_proba_unique": int(best_metrics["val_proba_unique"]),
            "val_pred_unique": int(best_metrics["val_pred_unique"]),
        },
        "decision_threshold": best_threshold,
        "decision_threshold_metric": "fixed_policy_0.5_validation_accuracy",
        "decision_threshold_score": best_threshold_acc,
        "fit_mode": "train_only",
        "constraints_satisfied": best_constraints,
        "leakage_policy": "test_not_used_in_tuning",
        "strategy_stage": resolved_stage,
        "direction_mode": best_direction_mode,
        "recency_weight_lambda": best_recency_weight_lambda,
        "class_weight_mode": best_class_weight_mode,
        "gate_objective_version": objective_version,
        "train_val_gap_signed": float(best_metrics["gap_signed"]),
        "train_val_gap_abs": float(best_metrics["gap_abs"]),
        "val_proba_std": float(best_metrics["val_proba_std"]),
        "val_proba_unique": int(best_metrics["val_proba_unique"]),
        "val_pred_unique": int(best_metrics["val_pred_unique"]),
        "degenerate_rejected_count": degenerate_rejected_count,
    }

    params_path = get_model_params_path("svm", ticker)
    save_json_artifact_only(save_data, params_path)
    print(f"\n저장 완료: {params_path}")
    print("=" * 70)
    return save_data

if __name__ == "__main__":
    optimize()
