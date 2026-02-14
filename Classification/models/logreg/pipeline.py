"""
Logistic Regression 분류 파이프라인 (Stride 앙상블).
"""

from __future__ import annotations

import hashlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLASSIFICATION_DIR = os.path.dirname(_MODELS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

from model_config import (
    LOGREG_PARAMS_ARTIFACT_PATH,
    LOGREG_RESULT_ARTIFACT_PATH,
    ensure_artifact_dirs,
    load_json_artifact_only,
)
from model_gate import evaluate_gate, print_gate_result
from split_dataset import N_MODELS, get_stride_splits, split_dataset

EXPECTED_OBJECTIVE_VERSION = "target_aligned_v1_logreg_elasticnet_stride"
EXPECTED_CV_MODE = "single_holdout_2024Q2Q3"


def _safe_spearman(x, y):
    ic, p_value = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0, 1.0
    if np.isnan(p_value):
        return float(ic), 1.0
    return float(ic), float(p_value)


def _get_feature_hash(feature_cols):
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_current_data_end_date(split):
    candidates = []
    for df in [split.train, split.val, split.test, split.final_train]:
        if len(df) > 0:
            candidates.append(df.index.max())
    if not candidates:
        return None
    return max(candidates).strftime("%Y-%m-%d")


def _is_param_file_stale(param_data, feature_cols, current_data_end_date, optimize_profile):
    required_meta = ["feature_hash", "data_end_date", "profile", "objective_version", "cv_mode"]
    for key in required_meta:
        if key not in param_data:
            return True, f"메타데이터 누락: {key}"

    current_feature_hash = _get_feature_hash(feature_cols)
    if param_data["feature_hash"] != current_feature_hash:
        return True, "feature_hash 불일치"

    file_data_end = param_data["data_end_date"]
    if current_data_end_date is not None and file_data_end < current_data_end_date:
        return True, f"data_end_date 구버전 ({file_data_end} < {current_data_end_date})"

    if param_data["profile"] != optimize_profile:
        return True, f"profile 불일치 ({param_data['profile']} != {optimize_profile})"

    if param_data["objective_version"] != EXPECTED_OBJECTIVE_VERSION:
        return True, (
            f"objective_version 불일치 "
            f"({param_data['objective_version']} != {EXPECTED_OBJECTIVE_VERSION})"
        )

    if param_data["cv_mode"] != EXPECTED_CV_MODE:
        return True, f"cv_mode 불일치 ({param_data['cv_mode']} != {EXPECTED_CV_MODE})"

    return False, "최신 파라미터 사용 가능"


def _load_params_or_default():
    default_params = {
        "solver": "saga",
        "penalty": "elasticnet",
        "C": 1.0,
        "l1_ratio": 0.5,
        "tol": 1e-4,
        "max_iter": 5000,
        "random_state": 42,
    }
    default_cw1 = 1.0

    data, used_path = load_json_artifact_only(LOGREG_PARAMS_ARTIFACT_PATH)
    if data is None:
        return default_params, default_cw1, None

    best_params = data.get("best_params", {})
    common_params = data.get("common_params", {})
    merged = {**default_params, **common_params, **best_params}

    cw_1 = merged.pop("class_weight_1", best_params.get("class_weight_1", default_cw1))
    if "eval_metric" in merged:
        del merged["eval_metric"]
    if "scale_pos_weight" in merged:
        del merged["scale_pos_weight"]
    if "early_stopping_rounds" in merged:
        del merged["early_stopping_rounds"]

    if "l1_ratio" in merged:
        merged["penalty"] = "elasticnet"
        merged["solver"] = "saga"

    print(f"  logreg 파라미터 로드 경로: {used_path}")
    return merged, float(cw_1), data


def _ensure_best_params(split, auto_optimize=False, optimize_profile="balanced"):
    param_data, _ = load_json_artifact_only(LOGREG_PARAMS_ARTIFACT_PATH)
    current_data_end_date = _get_current_data_end_date(split)
    feature_cols = split.feature_cols

    needs_optimize = False
    reason = ""

    if param_data is None:
        needs_optimize = True
        reason = "파라미터 파일 미존재"
    else:
        stale, reason = _is_param_file_stale(
            param_data,
            feature_cols,
            current_data_end_date,
            optimize_profile=optimize_profile,
        )
        needs_optimize = stale

    if needs_optimize and auto_optimize:
        print("\n  ⚠️ Logistic 자동 재튜닝을 실행합니다.")
        print(f"    사유: {reason}")
        from optimize_logreg import optimize

        optimize(profile=optimize_profile, n_trials=100)


def run_pipeline(auto_optimize=False, optimize_profile="balanced", return_metrics=False):
    print("\n" + "=" * 70)
    print("Logistic Regression 실전 파이프라인 시작")
    print("=" * 70)

    split = split_dataset()
    stride_splits = get_stride_splits(split)

    _ensure_best_params(
        split,
        auto_optimize=auto_optimize,
        optimize_profile=optimize_profile,
    )
    params, cw_1, _ = _load_params_or_default()

    print("\n  적용할 파라미터:")
    for key, val in params.items():
        print(f"    {key:20s}: {val}")
    print(f"    {'class_weight_1':20s}: {cw_1}")

    feature_cols = split.feature_cols
    x_test = split.test[feature_cols]
    y_test = split.test["Target_Class"].astype(int)

    models = []
    train_accs = []
    all_test_probas = []
    all_train_probas_full = []
    all_coefs = []

    x_final_train_full = split.final_train[feature_cols]
    y_final_train_full = split.final_train["Target_Class"].astype(int)

    print("\n" + "=" * 70)
    print(f"6. 스트라이드 앙상블 Final Refit ({N_MODELS}개 모델)")
    print("=" * 70)

    for ss in stride_splits:
        x_refit = ss.final_train[feature_cols]
        y_refit = ss.final_train["Target_Class"].astype(int)

        model_params = params.copy()
        model_params["class_weight"] = {0: 1.0, 1: cw_1}

        model = LogisticRegression(**model_params)
        model.fit(x_refit, y_refit)
        models.append(model)

        all_coefs.append(np.abs(model.coef_[0]))

        y_refit_pred = model.predict(x_refit)
        t_acc = accuracy_score(y_refit, y_refit_pred)
        train_accs.append(t_acc)

        proba = model.predict_proba(x_test)[:, 1]
        all_test_probas.append(proba)

        train_full_proba = model.predict_proba(x_final_train_full)[:, 1]
        all_train_probas_full.append(train_full_proba)

        print(
            f"  모델 {ss.offset}: Refit {len(x_refit):>4d}건 | "
            f"Train Acc {t_acc * 100:.1f}% | CW_1 {cw_1:.3f}"
        )

    ensemble_proba = np.mean(all_test_probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    ensemble_train_proba = np.mean(all_train_probas_full, axis=0)
    ensemble_train_pred = (ensemble_train_proba >= 0.5).astype(int)
    avg_train_acc = accuracy_score(y_final_train_full, ensemble_train_pred)

    print(f"\n  앙상블 Train Accuracy (평균): {avg_train_acc * 100:.1f}%")

    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)

    print("\n" + "=" * 70)
    print("7. 앙상블 모델 평가")
    print("=" * 70)
    print(f"    Accuracy : {acc * 100:.2f}%")
    print(f"    Precision: {prec * 100:.2f}%")
    print(classification_report(y_test, ensemble_pred, target_names=["Lose(0)", "Win(1)"]))

    actual_excess = split.test["Target_AAPL_3M"] - split.test["Target_SP500_3M"]
    ic, p_value = _safe_spearman(ensemble_proba, actual_excess)
    print(f"    IC       : {ic:.4f} (p={p_value:.4f})")

    avg_coefs = np.mean(all_coefs, axis=0)
    feat_imp = pd.Series(avg_coefs, index=feature_cols).sort_values(ascending=False)

    print("\n  [Feature Importance Top 5]")
    for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        print(f"    {i}. {feat:25s} {imp:.4f}")

    gap = avg_train_acc - acc
    print(f"\n  Train-Test Gap: {gap * 100:.1f}%p")

    mid_idx = len(split.test) // 2
    first_half = split.test.iloc[:mid_idx]
    second_half = split.test.iloc[mid_idx:]
    proba_first = ensemble_proba[:mid_idx]
    proba_second = ensemble_proba[mid_idx:]
    excess_first = first_half["Target_AAPL_3M"] - first_half["Target_SP500_3M"]
    excess_second = second_half["Target_AAPL_3M"] - second_half["Target_SP500_3M"]
    ic_first, _ = _safe_spearman(proba_first, excess_first)
    ic_second, _ = _safe_spearman(proba_second, excess_second)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    feat_imp.plot(kind="barh", ax=axes[0, 0], color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Feature Importance (Stride Ensemble Avg)", fontsize=13)
    axes[0, 0].set_xlabel("Importance")
    axes[0, 0].invert_yaxis()

    axes[0, 1].hist(
        ensemble_proba[y_test == 1], bins=30, alpha=0.6,
        label="Win (AAPL > SP500)", color="green", edgecolor="black"
    )
    axes[0, 1].hist(
        ensemble_proba[y_test == 0], bins=30, alpha=0.6,
        label="Lose (AAPL <= SP500)", color="red", edgecolor="black"
    )
    axes[0, 1].axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    axes[0, 1].set_title("Ensemble Probability Distribution", fontsize=13)
    axes[0, 1].set_xlabel("P(AAPL beats SP500)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    gap_labels = ["Avg Train", "Test"]
    gap_values = [avg_train_acc * 100, acc * 100]
    gap_colors = ["#4CAF50", "#FF5722"]
    bars = axes[1, 0].bar(gap_labels, gap_values, color=gap_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, gap_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[1, 0].set_title(f"Ensemble Train-Test Gap ({gap * 100:.1f}%p)", fontsize=13)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[1, 0].legend()

    ic_labels = ["1st Half", "2nd Half", "Full"]
    ic_values = [ic_first, ic_second, ic]
    ic_colors = ["green" if v > 0 else "red" for v in ic_values]
    bars = axes[1, 1].bar(ic_labels, ic_values, color=ic_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, ic_values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.005 if val >= 0 else -0.02),
                        f"{val:+.3f}", ha="center", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("IC Stability (Stride Ensemble)", fontsize=13)
    axes[1, 1].set_ylabel("Information Coefficient")
    axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    axes[1, 1].axhline(y=0.05, color="blue", linestyle="--", alpha=0.5, label="IC=0.05 (Strong)")
    axes[1, 1].legend()

    plt.tight_layout()
    ensure_artifact_dirs()
    plt.savefig(LOGREG_RESULT_ARTIFACT_PATH, dpi=150)
    print(f"\n  차트 저장 완료: {LOGREG_RESULT_ARTIFACT_PATH}")
    plt.close()

    gate = evaluate_gate({"accuracy": acc, "ic": ic, "gap": gap})
    print_gate_result("LogisticRegression", gate)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "ic": float(ic),
        "ic_p_value": float(p_value),
        "gap": float(gap),
        "overall_pass": bool(gate["pass_all"]),
    }

    if return_metrics:
        return metrics
    return models, ensemble_proba, ic


if __name__ == "__main__":
    run_pipeline(auto_optimize=False, optimize_profile="balanced")
