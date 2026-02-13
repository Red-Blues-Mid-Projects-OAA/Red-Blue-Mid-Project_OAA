"""
Logistic Regression Hyperparameter 최적화 모듈 (Custom Objective)

Optuna를 사용하여 Logistic Regression의 최적 파라미터를 찾습니다.
목표: 
  1. Test Accuracy > 0.52 (최대화)
  2. Test IC < 0.25 (과적합 방지 제약)

결과는 logic_params.json에 저장됩니다. (공통 파라미터 포함)
"""

import sys
import os
import json

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

import optuna
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr

from split_dataset import split_dataset, get_stride_splits

# Optuna 로그 레벨 (INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

# 결과 저장 경로
LOGIC_PARAMS_PATH = os.path.join(_THIS_DIR, "logic_params.json")
N_TRIALS = 100


def create_objective(stride_split, feature_cols, target_3m_diff):
    """
    Optuna objective 함수를 생성합니다.
    """
    X_val = stride_split.val[feature_cols]
    y_val = stride_split.val["Target_Class"].astype(int)
    
    # 3개월 수익률 차이 (IC 계산용) - Validation 구간
    # split_dataset의 val_df 인덱스와 매칭되는 데이터
    val_excess_return = target_3m_diff.loc[X_val.index]

    # Pre-computation: Validation (Already Scaled by split_dataset.py)
    X_train = stride_split.train[feature_cols]
    y_train = stride_split.train["Target_Class"].astype(int)
    spw = stride_split.scale_pos_weight

    # StandardScaler 제거: split_dataset.py에서 이미 스케일링된 데이터를 제공함
    # X_train_scaled -> X_train
    # X_val_scaled -> X_val

    def objective(trial):
        # 3. 하이퍼파라미터 공간 정의
        # C: 0.1 ~ 100.0 (규제 완화 -> 학습 능력 증대)
        C = trial.suggest_float("C", 0.1, 100.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0) # 0=L2, 1=L1
        
        # 1. tol: 1e-4 ~ 1e-3
        tol = trial.suggest_float("tol", 1e-4, 1e-3, log=True)
        
        # 2. class_weight: {0: 1, 1: 0.5 ~ 2.0}
        cw_1 = trial.suggest_float("class_weight_1", 0.5, 2.0)
        
        params = {
            "solver": "saga",
            "C": C,
            "l1_ratio": l1_ratio,
            "tol": tol,
            "class_weight": {0: 1, 1: cw_1},
            "random_state": 42,
            "max_iter": 5000,
        }

        # 2. 모델 학습
        model = LogisticRegression(**params)
        try:
            model.fit(X_train, y_train)
        except Exception:
            return 0.0  # 학습 실패 시 0점

        # 3. 예측 및 평가
        # Validation
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Train (Gap 계산용)
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        gap = train_acc - val_acc

        # IC 계산 (Validation)
        try:
            if np.std(y_val_proba) == 0:
                ic = 0.0
            else:
                ic, _ = spearmanr(y_val_proba, val_excess_return)
        except:
            ic = 0.0
        
        # 4. Custom Objective & Penalty
        # Base Score: Validation Accuracy
        score = val_acc
        
        # Penalty 1: IC < 0.01 (최소한의 예측력 보장)
        if ic < 0.01:
            score -= 0.2  # 강력한 페널티
            
        # Penalty 2: Gap >= 0.25 (과적합 방지)
        if gap >= 0.25:
            score -= 0.2

        # (Optional) IC > 0.25 과적합 의심 (기존 유지)
        if ic >= 0.25:
            score -= 0.1

        return score

    return objective


def optimize():
    print("=" * 70)
    print("Logistic Regression Optuna Optimization (Acc > 0.52 & IC < 0.25)")
    print("=" * 70)

    # 데이터 로드
    split = split_dataset()
    stride_splits = get_stride_splits(split)
    
    # IC 계산을 위한 타겟 데이터 준비
    df_all = split.final_train.copy() # train + val
    if split.test is not None:
         df_all = pd.concat([df_all, split.test])
         
    # 전체 데이터에서 3개월 수익률 차이 미리 계산 (인덱스 매칭용)
    # 실제로는 split_dataset 내에 포함되어 있거나 새로 로드해야 함
    # 여기서는 split.val 인덱스로 접근 가능하다고 가정
    # (Target_AAPL_3M, Target_SP500_3M 컬럼이 split 된 df에 남아있어야 함)
    # split_dataset.py를 보면 feature_cols만 뽑아서 저장하지 않고 df 전체를 저장하므로 OK
    target_3m_diff = split.val["Target_AAPL_3M"] - split.val["Target_SP500_3M"]

    tuning_stride = stride_splits[0]
    
    print(f"  Target: Maximize Validation Accuracy")
    print(f"  Constraint: IC < 0.25")
    print(f"  Trials: {N_TRIALS}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        create_objective(tuning_stride, split.feature_cols, target_3m_diff),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    # ── 결과 출력 ──
    best = study.best_trial
    print("\n" + "=" * 70)
    print(f"★ Best Trial (#{best.number})")
    print(f"  Value (Acc): {best.value:.4f}")
    print(f"  Params: {best.params}")

    # 공통 파라미터 포함하여 저장할 데이터 구성
    # XGBoost의 scale_pos_weight도 참고용으로 보존? -> Logistic은 spw 값을 class_weight로 변환해 씀
    save_data = {
        "best_params": {
            "C": best.params["C"],
            "l1_ratio": best.params["l1_ratio"],
            "tol": best.params["tol"],
            "class_weight_1": best.params["class_weight_1"]
        }, 
        "common_params": {
            "solver": "saga",
            "random_state": 42,
            "max_iter": 5000,
            "eval_metric": "logloss"
        },
        "best_accuracy": best.value,
        "n_trials": N_TRIALS,
    }

    with open(LOGIC_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  저장 완료: {LOGIC_PARAMS_PATH}")

if __name__ == "__main__":
    optimize()
