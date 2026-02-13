"""
XGBoost 실전 파이프라인 (실전 엔진)

스트라이드 앙상블: 5개 모델(offset 0~4)을 각각 학습시켜 예측치를 평균냅니다.
  - 각 모델은 STRIDE=5 간격의 독립적인 시퀀스로 학습 → 과적합 해소
  - 5개 모델의 예측 평균 → 견고한 확률값 (데이터 100% 활용)

파이프라인:
  1. split_dataset() → get_stride_splits()
  2. best_params.json 로드
  3. 5개 모델 Final Refit (각 StrideSplit.final_train)
  4. Test 세트 앙상블 예측 (5개 모델 평균)
  5. 과적합 진단 3종 세트
  6. 시각화 및 결과 저장

★ 자주 실행하는 모듈
"""

import sys
import os
import json

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from scipy.stats import spearmanr

from split_dataset import split_dataset, get_stride_splits, N_MODELS

# best_params.json 경로
BEST_PARAMS_PATH = os.path.join(_THIS_DIR, "best_params.json")


def load_best_params():
    """best_params.json을 로드합니다."""
    if not os.path.exists(BEST_PARAMS_PATH):
        print(f"  ⚠️ {BEST_PARAMS_PATH} 파일이 없습니다.")
        print(f"  먼저 optimize_hyperparams.py를 실행하세요.")
        return None

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  best_params.json 로드 완료 (Trial #{data['best_trial_number']}, "
          f"LogLoss={data['best_logloss']:.6f})")
    return data


def run_pipeline():
    """
    5-모델 스트라이드 앙상블 파이프라인을 실행합니다.
    """
    # ══════════════════════════════════════════════════════════════
    #  1. 데이터 로드 및 분할
    # ══════════════════════════════════════════════════════════════
    split = split_dataset()
    stride_splits = get_stride_splits(split)

    # ══════════════════════════════════════════════════════════════
    #  2. 최적 파라미터 로드
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. 최적 파라미터 로드")
    print("=" * 70)

    param_data = load_best_params()
    if param_data is None:
        return None

    best_params = param_data["best_params"]

    print(f"\n  적용할 파라미터:")
    for key, val in best_params.items():
        print(f"    {key:20s}: {val}")

    # ══════════════════════════════════════════════════════════════
    #  3. 5-모델 앙상블 Final Refit
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"6. 스트라이드 앙상블 Final Refit ({N_MODELS}개 모델)")
    print("=" * 70)

    feature_cols = split.feature_cols
    X_test = split.test[feature_cols]
    y_test = split.test["Target_Class"].astype(int)

    models = []
    train_accs = []
    all_test_probas = []

    for ss in stride_splits:
        X_refit = ss.final_train[feature_cols]
        y_refit = ss.final_train["Target_Class"].astype(int)

        # 모델 생성 (각 스트라이드의 scale_pos_weight 사용)
        model_params = {**best_params}
        model_params["scale_pos_weight"] = ss.scale_pos_weight
        model_params["random_state"] = 42
        model_params["eval_metric"] = "logloss"
        model_params.pop("early_stopping_rounds", None)

        model = XGBClassifier(**model_params)
        model.fit(X_refit, y_refit)
        models.append(model)

        # 개별 모델 Train Accuracy
        y_refit_pred = model.predict(X_refit)
        t_acc = accuracy_score(y_refit, y_refit_pred)
        train_accs.append(t_acc)

        # 개별 모델 Test 확률 예측
        proba = model.predict_proba(X_test)[:, 1]
        all_test_probas.append(proba)

        print(f"  모델 {ss.offset}: Refit {len(X_refit):>4d}건 | "
              f"Train Acc {t_acc*100:.1f}% | SPW {ss.scale_pos_weight:.3f}")

    # ── 앙상블 평균 확률 ──
    ensemble_proba = np.mean(all_test_probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    avg_train_acc = np.mean(train_accs)

    print(f"\n  앙상블 Train Accuracy (평균): {avg_train_acc*100:.1f}%")

    # ══════════════════════════════════════════════════════════════
    #  4. 퀀트 관점 평가
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("7. 앙상블 모델 평가 (퀀트 관점)")
    print("=" * 70)

    # ── ① 기본 ML 평가 지표 ──
    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)

    print("\n  [기본 ML 지표]")
    print(f"    Accuracy (정확도)  : {acc * 100:.2f}%")
    print(f"    Precision (정밀도) : {prec * 100:.2f}%")

    print("\n  [상세 Classification Report]")
    print(classification_report(y_test, ensemble_pred, target_names=["Lose(0)", "Win(1)"]))

    # ── ② Information Coefficient (IC) ──
    actual_excess_return = split.test["Target_AAPL_3M"] - split.test["Target_SP500_3M"]
    ic, p_value = spearmanr(ensemble_proba, actual_excess_return)

    print("  [★ 퀀트 핵심 지표: Information Coefficient (IC)]")
    print(f"    Test Set IC : {ic:.4f}")
    print(f"    P-value     : {p_value:.4f}")

    if ic > 0.10:
        print("    → 🔥 매우 강한 예측력 (IC > 0.10).")
    elif ic > 0.05:
        print("    → 💡 매우 훌륭한 예측력 (IC > 0.05).")
    elif ic > 0.02:
        print("    → 💡 실전 퀀트 수준 (IC 0.02~0.05).")
    elif ic > 0:
        print("    → 양의 예측력 있음.")
    else:
        print("    → ⚠️ 예측력 부족 (IC ≤ 0).")

    # ── ③ Feature Importance (앙상블 평균) ──
    avg_importance = np.mean(
        [m.feature_importances_ for m in models], axis=0
    )
    feat_imp = pd.Series(avg_importance, index=feature_cols).sort_values(ascending=False)

    print("\n  [Feature Importance Top 5 (앙상블 평균)]")
    for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        bar = "█" * int(imp * 50)
        print(f"    {i}. {feat:25s} {imp:.4f}  {bar}")

    # ══════════════════════════════════════════════════════════════
    #  5. 과적합(Overfitting) 진단 3종 세트
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("8. 과적합(Overfitting) 진단")
    print("=" * 70)

    # ── 진단 ①: Train-Test Accuracy Gap ──
    gap = avg_train_acc - acc

    print("\n  [진단 ①] Train-Test Accuracy Gap (앙상블)")
    print(f"    Avg Train Accuracy : {avg_train_acc * 100:.2f}%")
    print(f"    Test Accuracy      : {acc * 100:.2f}%")
    print(f"    Gap                : {gap * 100:.2f}%p")

    if gap <= 0.10:
        print("    → ✅ 건강 (Gap ≤ 10%p). 스트라이드 앙상블의 과적합 억제 효과가 나타났습니다.")
    elif gap <= 0.20:
        print("    → ⚠️ 주의 (Gap 10~20%p). 약간의 과적합 가능성이 있습니다.")
    else:
        print("    → 🚨 과적합 (Gap > 20%p). 추가적인 규제 강화가 필요합니다.")

    # ── 진단 ②: IC 안정성 (테스트 기간 전반/후반 분할) ──
    mid_idx = len(split.test) // 2
    test_first_half = split.test.iloc[:mid_idx]
    test_second_half = split.test.iloc[mid_idx:]

    proba_first = ensemble_proba[:mid_idx]
    proba_second = ensemble_proba[mid_idx:]

    excess_first = test_first_half["Target_AAPL_3M"] - test_first_half["Target_SP500_3M"]
    excess_second = test_second_half["Target_AAPL_3M"] - test_second_half["Target_SP500_3M"]

    ic_first, p_first = spearmanr(proba_first, excess_first)
    ic_second, p_second = spearmanr(proba_second, excess_second)

    mid_date = split.test.index[mid_idx].strftime("%Y-%m-%d")

    print(f"\n  [진단 ②] IC 안정성 (테스트 기간 분할: 기준일 {mid_date})")
    print(f"    전반기 IC : {ic_first:+.4f} (p={p_first:.4f}) | {len(test_first_half)}건")
    print(f"    후반기 IC : {ic_second:+.4f} (p={p_second:.4f}) | {len(test_second_half)}건")
    print(f"    전체   IC : {ic:+.4f}")

    if ic_first > 0 and ic_second > 0:
        print("    → ✅ 안정 — 두 기간 모두 양(+)의 IC를 유지합니다.")
    elif ic_first * ic_second > 0:
        print("    → ⚠️ 부호 일관 — 방향은 같지만 크기 차이를 모니터링하세요.")
    else:
        print("    → 🚨 불안정 — 전반/후반 IC 부호가 다릅니다.")

    # ── 진단 ③: Feature Importance 평탄화 (분산도 확인) ──
    top1_share = feat_imp.iloc[0]
    top3_share = feat_imp.iloc[:3].sum()
    hhi = (feat_imp ** 2).sum()

    print(f"\n  [진단 ③] Feature Importance 분산도 (앙상블 평균)")
    print(f"    Top 1 비중 : {top1_share * 100:.1f}% ({feat_imp.index[0]})")
    print(f"    Top 3 비중 : {top3_share * 100:.1f}%")
    print(f"    HHI 집중도 : {hhi:.4f} (낮을수록 분산, 균일 분배={1/len(feature_cols):.4f})")

    if top1_share > 0.50:
        print(f"    → 🚨 위험 — '{feat_imp.index[0]}'에 과도 의존.")
    elif top1_share > 0.25:
        print(f"    → ⚠️ 주의 — 상위 피처 집중도가 높습니다.")
    else:
        print("    → ✅ 양호 — 피처가 골고루 활용되고 있어 건강한 모델입니다.")

    # ══════════════════════════════════════════════════════════════
    #  6. 시각화
    # ══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) Feature Importance (앙상블 평균)
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
        label="Lose (AAPL ≤ SP500)", color="red", edgecolor="black"
    )
    axes[0, 1].axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    axes[0, 1].set_title("Ensemble Probability Distribution", fontsize=13)
    axes[0, 1].set_xlabel("P(AAPL beats SP500)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()

    # (3) Train vs Test Accuracy Gap (앙상블)
    gap_labels = ["Avg Train", "Test"]
    gap_values = [avg_train_acc * 100, acc * 100]
    gap_colors = ["#4CAF50", "#FF5722"]
    bars = axes[1, 0].bar(gap_labels, gap_values, color=gap_colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, gap_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axes[1, 0].set_title(f"Ensemble Train-Test Gap ({gap*100:.1f}%p)", fontsize=13)
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylim(0, 100)
    axes[1, 0].axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    axes[1, 0].legend()

    # (4) IC Stability
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
    save_path = os.path.join(_THIS_DIR, "xgboost_classifier_result.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  차트 저장 완료: {save_path}")
    plt.close()

    # ══════════════════════════════════════════════════════════════
    #  최종 요약
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"★ 최종 평가 요약 (스트라이드 {N_MODELS}-모델 앙상블)")
    print(f"{'=' * 70}")
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  IC (전체)      : {ic:+.4f} (p={p_value:.4f})")
    print(f"  IC (전반기)    : {ic_first:+.4f}")
    print(f"  IC (후반기)    : {ic_second:+.4f}")
    print(f"  Train-Test Gap : {gap * 100:.1f}%p")
    print(f"  Top Feature    : {feat_imp.index[0]} ({feat_imp.iloc[0]*100:.1f}%)")
    print(f"  HHI 집중도     : {hhi:.4f}")
    print(f"{'=' * 70}")

    return models, ensemble_proba, ic


if __name__ == "__main__":
    run_pipeline()
