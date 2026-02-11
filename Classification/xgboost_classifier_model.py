"""
XGBoost Classification 및 퀀트 모델 평가 모듈

파이프라인:
  1. generate_target() → 피처 + 타겟이 포함된 Master DataFrame 로드
  2. Train (2021.01~2024.09) / Test (2025.01~현재) 분할
  3. XGBClassifier 학습 (규제 적용, 과적합 방지)
  4. 퀀트 평가: Accuracy, Precision, IC (Information Coefficient), Feature Importance

★ DB 적재 없음 / Classification 폴더 외 파일 수정 없음
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from scipy.stats import spearmanr

from generate_target import generate_target


def train_and_evaluate_xgboost():
    """
    XGBoost 분류 모델을 학습·평가하고 퀀트 관점의 성과 지표를 출력합니다.
    """
    # ══════════════════════════════════════════════════════════════
    #  1. 데이터 로드 및 분할
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("1. 데이터 로드 및 분할")
    print("=" * 70)

    df = generate_target()

    # 피처(X) 컬럼 추출 (Target 컬럼 제외)
    exclude_cols = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 타겟 결측치(최근 63일, Inference용) 제거
    df_valid = df.dropna(subset=["Target_Class"])

    # Train / Test 분할
    # ★ Lookback Window 축소: 최근 국면(2021~)만 학습하여 Regime Shift 대응
    train_df = df_valid.loc["2021-01-01":"2024-09-30"]
    test_df = df_valid.loc["2025-01-01":]

    X_train = train_df[feature_cols]
    y_train = train_df["Target_Class"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["Target_Class"].astype(int)

    print(f"\n  학습 데이터(Train) : {len(X_train)}건 | 피처 수: {len(feature_cols)}개")
    print(f"  테스트 데이터(Test): {len(X_test)}건")
    print(f"  피처 목록: {feature_cols}")

    # ══════════════════════════════════════════════════════════════
    #  2. XGBoost 모델 학습
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. XGBoost 모델 학습 (Classifier)")
    print("=" * 70)

    # ★ Class 불균형(Imbalance) 보정
    # scale_pos_weight = (Class 0 수) / (Class 1 수)
    ratio = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)
    print(f"  Class 불균형 보정: scale_pos_weight = {ratio:.4f}")

    # 퀀트 모델 특화 규제(Regularization) 파라미터 적용
    # 얕은 트리(depth=3) + 무작위 샘플링 → 과적합 방지
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)
    print("  모델 학습 완료!")

    # ══════════════════════════════════════════════════════════════
    #  3. 예측 및 퀀트 관점 평가
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. 모델 예측 및 퀀트 관점의 평가")
    print("=" * 70)

    # 확률 예측: predict_proba[:, 1] = Class 1(시장을 이길) 확률
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = model.predict(X_test)

    # ── ① 기본 ML 평가 지표 ──
    acc = accuracy_score(y_test, y_pred_class)
    prec = precision_score(y_test, y_pred_class, zero_division=0)

    print("\n  [기본 ML 지표]")
    print(f"    Accuracy (정확도)  : {acc * 100:.2f}%")
    print(f"    Precision (정밀도) : {prec * 100:.2f}%")
    print(f"    (모델이 '이긴다'고 했을 때 실제로 이긴 비율)")

    print("\n  [상세 Classification Report]")
    print(classification_report(y_test, y_pred_class, target_names=["Lose(0)", "Win(1)"]))

    # ── ② Information Coefficient (IC) ──
    # 예측 확률(P)의 순위와 실제 초과수익률(Alpha)의 순위 간 Spearman 상관
    actual_excess_return = (
        test_df["Target_AAPL_3M"] - test_df["Target_SP500_3M"]
    )
    ic, p_value = spearmanr(y_pred_proba, actual_excess_return)

    print("  [★ 퀀트 핵심 지표: Information Coefficient (IC)]")
    print(f"    Test Set IC : {ic:.4f}")
    print(f"    P-value     : {p_value:.4f}")

    if ic > 0.10:
        print("    → 🔥 매우 강한 예측력 (IC > 0.10). 데이터 누수 여부도 재확인 권장.")
    elif ic > 0.05:
        print("    → 💡 매우 훌륭한 예측력 (IC > 0.05). Scale Factor에 사용 가능.")
    elif ic > 0.02:
        print("    → 💡 실전 퀀트 수준의 예측력 (IC 0.02~0.05). 포트폴리오 최적화 가능.")
    elif ic > 0:
        print("    → 양의 예측력 있음. 추가 피처 튜닝으로 개선 여지 있음.")
    else:
        print("    → ⚠️ 예측력 부족 (IC ≤ 0). 피처 재검토 필요.")

    # ── ③ Feature Importance ──
    feat_imp = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print("\n  [Feature Importance Top 5]")
    for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        bar = "█" * int(imp * 50)
        print(f"    {i}. {feat:25s} {imp:.4f}  {bar}")

    # 과도한 단일 피처 의존도 경고
    if feat_imp.iloc[0] > 0.50:
        print(f"\n    ⚠️ 경고: '{feat_imp.index[0]}'에 대한 의존도가 {feat_imp.iloc[0]*100:.1f}%로 과도합니다.")
        print("       시장 전환기에 모델이 무너질 위험이 있습니다.")
    else:
        print("\n    ✓ 피처 분산도 양호 — 시장 국면을 균형 있게 파악하는 건강한 모델입니다.")

    # ── ④ 시각화: Feature Importance 차트 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Feature Importance 바 차트
    feat_imp.plot(kind="barh", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Feature Importance (XGBoost Classifier)", fontsize=13)
    axes[0].set_xlabel("Importance")
    axes[0].invert_yaxis()

    # 예측 확률 분포 (Test Set)
    axes[1].hist(
        y_pred_proba[y_test == 1], bins=30, alpha=0.6,
        label="Win (AAPL > SP500)", color="green", edgecolor="black"
    )
    axes[1].hist(
        y_pred_proba[y_test == 0], bins=30, alpha=0.6,
        label="Lose (AAPL ≤ SP500)", color="red", edgecolor="black"
    )
    axes[1].axvline(x=0.5, color="black", linestyle="--", label="Threshold (0.5)")
    axes[1].set_title("Predicted Probability Distribution (Test Set)", fontsize=13)
    axes[1].set_xlabel("P(AAPL beats SP500)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(_THIS_DIR, "xgboost_classifier_result.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n  차트 저장 완료: {save_path}")
    plt.close()

    # ── 요약 ──
    print(f"\n{'=' * 70}")
    print("★ 최종 평가 요약")
    print(f"{'=' * 70}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  IC        : {ic:.4f} (p={p_value:.4f})")
    print(f"  Top Feature: {feat_imp.index[0]} ({feat_imp.iloc[0]*100:.1f}%)")
    print(f"{'=' * 70}")

    return model, y_pred_proba, ic


if __name__ == "__main__":
    train_and_evaluate_xgboost()
