"""
Logistic Regression 실전 파이프라인 (Renewed)

스트라이드 앙상블: 5개 모델(offset 0~4)을 각각 학습시켜 예측치를 평균냅니다.
파라미터는 logic_params.json에서 로드하며, 공통 파라미터와 최적화된 파라미터를 모두 적용합니다.

파이프라인:
  1. split_dataset() → get_stride_splits()
  2. Logic_params.json 로드 (best_params + common_params)
  3. 5개 모델 Final Refit (각 StrideSplit.final_train)
     * 주의: Logistic Regression은 스케일링 필수 (StandardScaler)
  4. Test 세트 앙상블 예측
  5. 과적합 진단 3종 세트
  6. 시각화 및 결과 저장 (logic_regression_result.png)
"""

import sys
import os
import json

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report
from scipy.stats import spearmanr

from split_dataset import split_dataset, get_stride_splits, N_MODELS

LOGIC_PARAMS_PATH = os.path.join(_THIS_DIR, "logic_params.json")


def load_params():
    """logic_params.json을 로드하여 통합된 파라미터 딕셔너리를 반환합니다."""
    if not os.path.exists(LOGIC_PARAMS_PATH):
        print(f"  ⚠️ {LOGIC_PARAMS_PATH} 파일이 없습니다.")
        print(f"  먼저 optimize_logreg.py를 실행하세요.")
        return None, None

    with open(LOGIC_PARAMS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 최적화된 파라미터 (C, l1_ratio)
    best_params = data.get("best_params", {})
    
    # 2. 공통 파라미터 (solver, max_iter 등)
    common_params = data.get("common_params", {})
    
    
    # 3. 통합 파라미터 생성 (class_weight_1 포함)
    final_params = {**common_params, **best_params}
    
    # 4. class_weight_1 추출 (없으면 기본값 1.0)
    cw_1 = final_params.get("class_weight_1", 1.0)
    if "class_weight_1" in final_params:
        del final_params["class_weight_1"] # LogisticRegression 생성자에는 전달하지 않음
    
    print(f"  logic_params.json 로드 완료 (Acc={data.get('best_accuracy', 0):.4f})")
    
    # 로드된 파라미터 검증/정제 (구버전 sklearn 호환성 등)
    if "penalty" in final_params:
        del final_params["penalty"] # l1_ratio 사용 시 penalty 지정 불필요 (saga solver)
        
    return final_params, cw_1


def run_pipeline():
    print("\n" + "=" * 70)
    print("Logistic Regression 실전 파이프라인 시작")
    print("=" * 70)

    # 1. 데이터 로드
    split = split_dataset()
    stride_splits = get_stride_splits(split)

    # 2. 파라미터 로드
    params, cw_1 = load_params()
    if params is None:
        return

    print(f"\n  적용할 파라미터:")
    for key, val in params.items():
        print(f"    {key:20s}: {val}")
    print(f"    {'class_weight_1':20s}: {cw_1}")

    # 3. 5-모델 앙상블 Refit
    print("\n" + "=" * 70)
    print(f"6. 스트라이드 앙상블 Final Refit ({N_MODELS}개 모델)")
    print("=" * 70)

    feature_cols = split.feature_cols
    X_test_raw = split.test[feature_cols]
    y_test = split.test["Target_Class"].astype(int)

    models = []
    scalers = []
    train_accs = []
    all_test_probas = []
    all_coefs = []

    for ss in stride_splits:
        X_refit_raw = ss.final_train[feature_cols]
        y_refit = ss.final_train["Target_Class"].astype(int)

        # 스케일링 (제거됨: split_dataset.py에서 이미 스케일링된 데이터를 제공함)
        # scaler = StandardScaler()
        # X_refit = scaler.fit_transform(X_refit_raw)
        # X_test = scaler.transform(X_test_raw)
        
        X_refit = X_refit_raw
        X_test = X_test_raw # Loop 밖에서 정의된 X_test_raw 사용 (이미 Scaled)

        # 모델 파라미터 설정
        model_params = params.copy()
        
        # class_weight: {0: 1, 1: cw_1} 
        # (Optuna에서 튜닝된 값을 고정적으로 사용. SPW 동적 계산 대체)
        model_params["class_weight"] = {0: 1, 1: cw_1}
        
        # eval_metric 등 sklearn에 없는 파라미터 제거
        if "eval_metric" in model_params:
            del model_params["eval_metric"]

        model = LogisticRegression(**model_params)
        model.fit(X_refit, y_refit)
        models.append(model)
        
        # Coef 저장
        all_coefs.append(np.abs(model.coef_[0]))

        # Train Accuracy
        y_refit_pred = model.predict(X_refit)
        t_acc = accuracy_score(y_refit, y_refit_pred)
        train_accs.append(t_acc)

        # Test Probability
        proba = model.predict_proba(X_test)[:, 1]
        all_test_probas.append(proba)

        print(f"  모델 {ss.offset}: Refit {len(X_refit):>4d}건 | "
              f"Train Acc {t_acc*100:.1f}% | CW_1 {cw_1:.3f}")

    # 앙상블
    ensemble_proba = np.mean(all_test_probas, axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    avg_train_acc = np.mean(train_accs)
    
    print(f"\n  앙상블 Train Accuracy (평균): {avg_train_acc*100:.1f}%")

    # 4. 평가
    print("\n" + "=" * 70)
    print("7. 앙상블 모델 평가")
    print("=" * 70)

    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)

    print(f"    Accuracy : {acc * 100:.2f}%")
    print(f"    Precision: {prec * 100:.2f}%")
    
    # IC 계산
    actual_excess = split.test["Target_AAPL_3M"] - split.test["Target_SP500_3M"]
    
    # 예외 처리: 표준편차가 0이면 모든 예측값이 동일함 -> IC 계산 불가 (0 처리)
    if np.std(ensemble_proba) == 0:
        ic, p_value = 0.0, 1.0
        print("    ! 경고: 모든 예측 확률이 동일하여 IC를 계산할 수 없습니다. (Constant Prediction)")
    else:
        ic, p_value = spearmanr(ensemble_proba, actual_excess)
    
    print(f"    IC       : {ic:.4f} (p={p_value:.4f})")
    
    # Feature Importance
    avg_coefs = np.mean(all_coefs, axis=0)
    feat_imp = pd.Series(avg_coefs, index=feature_cols).sort_values(ascending=False)
    
    print("\n  [Feature Importance Top 5]")
    for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        print(f"    {i}. {feat:25s} {imp:.4f}")

    # 5. 과적합 진단 및 시각화 (간소화)
    gap = avg_train_acc - acc
    print(f"\n  Train-Test Gap: {gap*100:.1f}%p")
    
    # 시각화 저장
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    feat_imp.head(20).plot(kind="barh", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Feature Importance")
    axes[0, 0].invert_yaxis()
    
    axes[0, 1].hist(ensemble_proba[y_test==1], bins=20, alpha=0.5, label="Win", color="g")
    axes[0, 1].hist(ensemble_proba[y_test==0], bins=20, alpha=0.5, label="Lose", color="r")
    axes[0, 1].legend()
    axes[0, 1].set_title("Probability Distribution")
    
    axes[1, 0].bar(["Train", "Test"], [avg_train_acc, acc], color=["blue", "orange"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title(f"Accuracy Gap ({gap*100:.1f}%p)")
    
    save_path = os.path.join(_THIS_DIR, "logic_regression_result.png")
    plt.savefig(save_path)
    print(f"\n  차트 저장 완료: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_pipeline()
