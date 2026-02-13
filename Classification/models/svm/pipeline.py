"""
SVM 분류 파이프라인 (Stride 앙상블).

차트 형식은 xgboost_classifier_result.png와 동일하게 2x2 구성으로 맞춥니다.
1) Feature Importance (Permutation)
2) Ensemble Probability Distribution
3) Ensemble Train-Test Gap
4) IC Stability
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.svm import SVC

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLASSIFICATION_DIR = os.path.dirname(_MODELS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

from split_dataset import N_MODELS, get_stride_splits, split_dataset
from model_config import (
    SVM_PARAMS_ARTIFACT_PATH,
    SVM_RESULT_ARTIFACT_PATH,
    ensure_artifact_dirs,
    load_json_artifact_only,
)
from model_gate import evaluate_gate, print_gate_result

PERM_SEED = 42


def load_svm_params():
    """best_svm_params.json을 로드하고, 없으면 기본값을 반환합니다."""
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
    }

    default_meta = {
        "decision_threshold": 0.5,
        "fit_mode": "final_train",
    }

    data, used_path = load_json_artifact_only(SVM_PARAMS_ARTIFACT_PATH)
    if data is None:
        return default_params, default_meta

    try:
        params = data.get("best_params", data)
        if not isinstance(params, dict):
            return default_params, default_meta

        merged = {**default_params, **params}
        merged["probability"] = True
        merged.setdefault("random_state", 42)
        meta = {
            "decision_threshold": float(data.get("decision_threshold", 0.5)),
            "fit_mode": data.get("fit_mode", "final_train"),
        }
        print(f"  SVM 파라미터 로드 경로: {used_path}")
        return merged, meta
    except Exception as exc:
        print(f"[WARN] 파라미터 로드 실패: {exc}")
        return default_params, default_meta


def safe_spearmanr(x, y):
    """상수열/짧은 시계열에서 NaN 안전 처리를 포함한 Spearman 계산."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan
    return spearmanr(x, y)


def ensemble_predict_proba(models, x_df):
    """(model, scaler, offset) 리스트에 대해 앙상블 평균 확률을 반환합니다."""
    probas = []
    for model, scaler, _ in models:
        if scaler is None:
            probas.append(model.predict_proba(x_df)[:, 1])
        else:
            x_scaled = scaler.transform(x_df)
            probas.append(model.predict_proba(x_scaled)[:, 1])
    return np.mean(probas, axis=0)


def compute_permutation_importance(models, x_test, y_test, feature_cols, threshold):
    """
    앙상블 기준 permutation importance를 계산합니다.
    기준: Accuracy drop (baseline_acc - permuted_acc)
    """
    rng = np.random.default_rng(PERM_SEED)
    baseline_proba = ensemble_predict_proba(models, x_test)
    baseline_pred = (baseline_proba >= threshold).astype(int)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    importances = []
    for col in feature_cols:
        x_perm = x_test.copy()
        shuffled = x_perm[col].to_numpy(copy=True)
        rng.shuffle(shuffled)
        x_perm[col] = shuffled

        perm_proba = ensemble_predict_proba(models, x_perm)
        perm_pred = (perm_proba >= threshold).astype(int)
        perm_acc = accuracy_score(y_test, perm_pred)
        # 시각화 안정성을 위해 음수 importance는 0으로 절단합니다.
        importance = max(0.0, baseline_acc - perm_acc)
        importances.append(importance)

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    return feat_imp


def run_pipeline(return_metrics=False):
    """SVM stride 앙상블 학습/평가 및 결과 차트 저장."""
    print("\n" + "=" * 70)
    print("1. SVM Stride 앙상블 파이프라인 시작")
    print("=" * 70)

    split = split_dataset()
    stride_splits = get_stride_splits(split)
    feature_cols = split.feature_cols
    x_test = split.test[feature_cols]
    y_test = split.test["Target_Class"].astype(int)

    svm_params, meta = load_svm_params()
    threshold = meta.get("decision_threshold", 0.5)
    fit_mode = meta.get("fit_mode", "final_train")
    print(f"\n사용 SVM 파라미터: {svm_params}")
    print(f"추론 임계값: {threshold:.2f} | 학습 모드: {fit_mode}")

    models = []
    train_accs = []
    all_test_probas = []

    print("\n" + "=" * 70)
    print(f"2. Stride split별 Final Refit ({N_MODELS}개 모델)")
    print("=" * 70)

    for ss in stride_splits:
        if fit_mode == "train_only":
            x_refit = ss.train[feature_cols]
            y_refit = ss.train["Target_Class"].astype(int)
        else:
            x_refit = ss.final_train[feature_cols]
            y_refit = ss.final_train["Target_Class"].astype(int)

        # split_dataset()에서 이미 스케일된 입력을 받아 추가 스케일링을 하지 않습니다.
        model = SVC(**svm_params)
        model.fit(x_refit, y_refit)
        models.append((model, None, ss.offset))

        y_refit_pred = model.predict(x_refit)
        train_acc = accuracy_score(y_refit, y_refit_pred)
        train_accs.append(train_acc)
        all_test_probas.append(model.predict_proba(x_test)[:, 1])

        print(
            f"  모델 {ss.offset}: Refit {len(x_refit):>4d}건 | "
            f"Train Acc {train_acc * 100:.1f}%"
        )

    ensemble_proba = np.mean(all_test_probas, axis=0)
    ensemble_pred = (ensemble_proba >= threshold).astype(int)
    avg_train_acc = float(np.mean(train_accs))

    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)
    gap = avg_train_acc - acc

    excess_return = split.test["Target_AAPL_3M"] - split.test["Target_SP500_3M"]
    ic, p_value = safe_spearmanr(ensemble_proba, excess_return)

    mid_idx = len(split.test) // 2
    first_half = split.test.iloc[:mid_idx]
    second_half = split.test.iloc[mid_idx:]
    proba_first = ensemble_proba[:mid_idx]
    proba_second = ensemble_proba[mid_idx:]
    excess_first = first_half["Target_AAPL_3M"] - first_half["Target_SP500_3M"]
    excess_second = second_half["Target_AAPL_3M"] - second_half["Target_SP500_3M"]
    ic_first, p_first = safe_spearmanr(proba_first, excess_first)
    ic_second, p_second = safe_spearmanr(proba_second, excess_second)

    feat_imp = compute_permutation_importance(models, x_test, y_test, feature_cols, threshold)
    top1_share = float(feat_imp.iloc[0]) if len(feat_imp) > 0 else np.nan
    top3_share = float(feat_imp.iloc[:3].sum()) if len(feat_imp) >= 3 else np.nan
    hhi = float((feat_imp ** 2).sum()) if len(feat_imp) > 0 else np.nan

    print("\n" + "=" * 70)
    print("3. 성능 평가")
    print("=" * 70)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"IC       : {ic:+.4f} (p={p_value:.4f})")
    print(f"Gap      : {gap * 100:.2f}%p ({gap * 10000:.0f}bp)")
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=["Lose(0)", "Win(1)"]))

    print("\n[Permutation Feature Importance Top 5]")
    for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        bar = "#" * int(imp * 200)
        print(f"  {i}. {feat:25s} {imp:.4f} {bar}")

    print("\n" + "=" * 70)
    print("4. 결과 차트 저장 (xgboost 형식과 동일 2x2 레이아웃)")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) Feature Importance (xgboost 차트와 동일 위치/형식)
    feat_imp.plot(kind="barh", ax=axes[0, 0], color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Feature Importance (Stride Ensemble Avg)", fontsize=13)
    axes[0, 0].set_xlabel("Importance")
    axes[0, 0].invert_yaxis()

    # (2) Predicted Probability Distribution
    axes[0, 1].hist(
        ensemble_proba[y_test == 1], bins=30, alpha=0.6,
        label="Win (AAPL > SP500)", color="green", edgecolor="black"
    )
    axes[0, 1].hist(
        ensemble_proba[y_test == 0], bins=30, alpha=0.6,
        label="Lose (AAPL <= SP500)", color="red", edgecolor="black"
    )
    axes[0, 1].axvline(
        x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})"
    )
    axes[0, 1].set_title("Ensemble Probability Distribution", fontsize=13)
    axes[0, 1].set_xlabel("P(AAPL beats SP500)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # (3) Train vs Test Accuracy Gap
    gap_labels = ["Avg Train", "Test"]
    gap_values = [avg_train_acc * 100, acc * 100]
    gap_colors = ["#4CAF50", "#FF5722"]
    bars = axes[1, 0].bar(gap_labels, gap_values, color=gap_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, gap_values):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
    axes[1, 0].set_title(f"Ensemble Train-Test Gap ({gap * 100:.1f}%p)", fontsize=13)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[1, 0].legend()

    # (4) IC Stability
    ic_labels = ["1st Half", "2nd Half", "Full"]
    ic_values = [0.0 if np.isnan(ic_first) else ic_first, 0.0 if np.isnan(ic_second) else ic_second, 0.0 if np.isnan(ic) else ic]
    ic_colors = ["green" if v > 0 else "red" for v in ic_values]
    bars = axes[1, 1].bar(ic_labels, ic_values, color=ic_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, ic_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.005 if val >= 0 else -0.02),
            f"{val:+.3f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    axes[1, 1].set_title("IC Stability (Stride Ensemble)", fontsize=13)
    axes[1, 1].set_ylabel("Information Coefficient")
    axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    axes[1, 1].axhline(y=0.05, color="blue", linestyle="--", alpha=0.5, label="IC=0.05 (Strong)")
    axes[1, 1].legend()

    plt.tight_layout()
    ensure_artifact_dirs()
    plt.savefig(SVM_RESULT_ARTIFACT_PATH, dpi=150)
    plt.close()
    print(f"차트 저장 완료: {SVM_RESULT_ARTIFACT_PATH}")

    print("\n" + "=" * 70)
    print(f"최종 평가 요약 (Stride {N_MODELS}개 모델 SVM 앙상블)")
    print("=" * 70)
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  IC (전체)      : {ic:+.4f} (p={p_value:.4f})")
    print(f"  IC (전반기)    : {ic_first:+.4f} (p={p_first:.4f})")
    print(f"  IC (후반기)    : {ic_second:+.4f} (p={p_second:.4f})")
    print(f"  Train-Test Gap : {gap * 100:.1f}%p ({gap * 10000:.0f}bp)")
    if len(feat_imp) > 0:
        print(f"  Top Feature    : {feat_imp.index[0]} ({feat_imp.iloc[0]:.4f})")
    print(f"  Top3 합계      : {top3_share:.4f}" if not np.isnan(top3_share) else "  Top3 합계      : N/A")
    print(f"  HHI 집중도     : {hhi:.4f}" if not np.isnan(hhi) else "  HHI 집중도     : N/A")
    print("=" * 70)

    gate = evaluate_gate({"accuracy": acc, "ic": ic, "gap": gap})
    print_gate_result("SVM", gate)

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
    run_pipeline()
