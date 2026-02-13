"""
Hyperparameter 최적화 모듈 (Walk-Forward CV + Sample Weighting)

Data Squeezing 전략 적용:
  1. Walk-Forward CV (2-Fold):
     - Fold 1: Train(2021~2022) | Val(2023)
     - Fold 2: Train(2021~2023) | Val(2024)
     → 검증 기회 2배 확대, 과거/최신 시장 모두 대응
  2. Sample Weighting:
     - 확실한 Alpha(승리)를 가진 샘플에 가중치를 부여하여 학습

결과는 best_params.json에 저장됩니다.
"""

import sys
import os
import json
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from generate_target import generate_target
from split_dataset import calculate_sample_weight

# Optuna 로그 레벨 (INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

BEST_PARAMS_PATH = os.path.join(_THIS_DIR, "best_params.json")
N_TRIALS = 100

def create_objective(full_df, feature_cols):
    """
    Walk-Forward CV를 적용한 Objective 함수
    """
    # Fold 정의 (날짜 기준)
    folds = [
        # Fold 1: 과거 검증 (2023년 시장)
        {
            "train": ("2021-01-01", "2022-12-31"),
            "val":   ("2023-04-01", "2023-12-31") 
        },
        # Fold 2: 최신 검증 (2024년 시장)
        {
            "train": ("2021-01-01", "2023-12-31"),
            "val":   ("2024-04-01", "2024-09-30")
        }
    ]

    def objective(trial):
        # 파라미터 탐색 공간 (Balanced Strategy)
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 3), # 깊이 1~3 (복잡도 증가)
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.20, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 30, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 30),
            "gamma": trial.suggest_float("gamma", 0.1, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "random_state": 42,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        }

        fold_scores = []
        
        for i, fold in enumerate(folds):
            # 1. 데이터 슬라이싱
            train_fold = full_df.loc[fold["train"][0]:fold["train"][1]].copy()
            val_fold = full_df.loc[fold["val"][0]:fold["val"][1]].copy()

            # 2. Scaling (Fit on Train, Transform Val) - Leakage 방지
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_fold[feature_cols])
            X_val = scaler.transform(val_fold[feature_cols])

            y_train = train_fold["Target_Class"].astype(int)
            y_val = val_fold["Target_Class"].astype(int)

            # 3. Sample Weighting (확실한 놈만 팬다)
            #    Val 데이터에는 가중치 적용 X (평가는 공정하게)
            w_train = calculate_sample_weight(train_fold)

            # 4. scale_pos_weight 계산 (Fold별 자동 산출)
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            spw = float(n_neg) / max(n_pos, 1)
            
            # 5. 모델 학습
            model = XGBClassifier(**params, scale_pos_weight=spw)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 6. 평가 (LogLoss + Gap Penalty)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            val_loss = log_loss(y_val, y_pred_proba)

            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            gap = train_acc - val_acc
            # Gap Penalty: 20% 초과 시 강력 제재
            gap_penalty = max(0, gap - 0.20) * 3.0

            total_score = val_loss + gap_penalty
            fold_scores.append(total_score)

        # Fold 평균 점수 반환
        return np.mean(fold_scores)

    return objective


def optimize():
    print("=" * 70)
    print("Hyperparameter 최적화 (Walk-Forward CV + Sample Weighting)")
    print("=" * 70)

    # 1. 전체 데이터 로드
    full_df = generate_target()
    
    # Feature 컬럼 발라내기
    exclude = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class", "Alpha_Diff"]
    feature_cols = [c for c in full_df.columns if c not in exclude]
    
    # 타겟 있는 데이터만 사용
    full_df = full_df.dropna(subset=["Target_Class"])
    
    print(f"\n  전체 데이터: {len(full_df)}건 ({full_df.index.min().date()} ~ {full_df.index.max().date()})")
    print(f"  CV 전략: 2-Fold Walk-Forward")
    print(f"    - Fold 1: Train 21~22 / Val 23")
    print(f"    - Fold 2: Train 21~23 / Val 24 (Current)")
    print(f"  가중치 전략: Sample Weighting 적용")
    
    study = optuna.create_study(direction="minimize", study_name="xgb_wf_cv")
    study.optimize(
        create_objective(full_df, feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    best = study.best_trial
    print("\n" + "=" * 70)
    print("★ 최적화 결과")
    print("=" * 70)
    print(f"  Best LogLoss (CV Avg) : {best.value:.6f}")
    print(f"  Best Parameters:")
    for key, val in best.params.items():
        print(f"    {key:20s}: {val}")

    # 저장
    save_data = {
        "best_params": best.params,
        "best_logloss": best.value,
        "best_trial_number": best.number,
        "n_trials": N_TRIALS,
        "optimization_method": "Walk-Forward CV 2-Fold",
        # SPW는 Fold마다 다르므로 저장하지 않거나, 최신 Fold(전체) 기준으로 하나 계산해둘 수 있음
        # 여기서는 run_model_pipeline.py가 재계산하도록 유도 (pipeline은 split_dataset을 쓰므로 거기서 계산됨)
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  저장 완료: {BEST_PARAMS_PATH}")

if __name__ == "__main__":
    optimize()
