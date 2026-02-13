"""
Hyperparameter 최적화 모듈 (튜닝 랩)

Optuna를 사용하여 XGBoost 분류기의 최적 파라미터를 찾습니다.
스트라이드 샘플링(STRIDE=5, offset=0)을 적용하여 과적합을 방지합니다.
결과는 best_params.json에 저장되며, run_model_pipeline.py에서 로드하여 사용합니다.

핵심 설계:
  - 스트라이드 샘플링된 Train/Val 사용 (offset=0 대표)
  - Early Stopping (early_stopping_rounds=20) 적용
  - learning_rate: 로그 스케일 탐색
  - 최적화 기준: Validation LogLoss 최소화
  - scale_pos_weight: Train 클래스 비율로 자동 산출

★ 가끔 실행 (파라미터 재탐색 필요 시)
"""

import sys
import os
import json

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

from split_dataset import split_dataset, get_stride_splits

# Optuna 로그 레벨 조정 (INFO만 표시)
optuna.logging.set_verbosity(optuna.logging.INFO)

# 결과 저장 경로
BEST_PARAMS_PATH = os.path.join(_THIS_DIR, "best_params.json")
N_TRIALS = 100


def create_objective(stride_split, feature_cols):
    """
    Optuna objective 함수를 생성합니다 (클로저 패턴).
    스트라이드 샘플링된 Train/Val을 사용합니다.
    """
    X_train = stride_split.train[feature_cols]
    y_train = stride_split.train["Target_Class"].astype(int)
    X_val = stride_split.val[feature_cols]
    y_val = stride_split.val["Target_Class"].astype(int)
    spw = stride_split.scale_pos_weight

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 1), # 깊이 1 고정 (Stump)
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15, log=True), # 학습률 상향 (트리 수 감소 보상)
            "n_estimators": trial.suggest_int("n_estimators", 40, 70),  # 트리 수 엄격 제한
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),   # 규제 완화 (정확도 복구)
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),      # 샘플링 비율 상향 (정보량 증대)
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "scale_pos_weight": spw,
            "random_state": 42,
            "eval_metric": "logloss",
            "early_stopping_rounds": 20,
        }

        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation LogLoss 계산
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_logloss = log_loss(y_val, y_val_proba)

        return val_logloss

    return objective


def optimize():
    """
    Optuna 최적화를 실행하고 결과를 best_params.json에 저장합니다.
    """
    print("=" * 70)
    print("Hyperparameter 최적화 (Optuna + 스트라이드 샘플링)")
    print("=" * 70)

    # 데이터 로드 및 스트라이드 분할
    split = split_dataset()
    stride_splits = get_stride_splits(split)

    # 튜닝은 offset=0의 스트라이드로 대표 수행
    tuning_stride = stride_splits[0]
    print(f"\n  튜닝 대표 모델: offset=0 (Train {len(tuning_stride.train)}건, "
          f"Val {len(tuning_stride.val)}건)")
    print(f"  최적화 기준: Validation LogLoss 최소화")
    print(f"  Trial 횟수 : {N_TRIALS}")
    print(f"  Early Stopping: 20 rounds")
    print(f"\n  탐색 시작...\n")

    # Optuna Study 생성 및 실행
    study = optuna.create_study(
        direction="minimize",
        study_name="xgb_stride_tuning",
    )
    study.optimize(
        create_objective(tuning_stride, split.feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── 결과 출력 ──
    best = study.best_trial
    print("\n" + "=" * 70)
    print("★ 최적화 결과")
    print("=" * 70)
    print(f"  Best Trial    : #{best.number}")
    print(f"  Best LogLoss  : {best.value:.6f}")
    print(f"\n  Best Parameters:")
    for key, val in best.params.items():
        print(f"    {key:20s}: {val}")

    # ── max_depth 경고 ──
    if best.params.get("max_depth", 0) >= 5:
        print(f"\n  ⚠️ 주의: max_depth={best.params['max_depth']}로 높은 편입니다.")
        print(f"     Train Accuracy가 지나치게 높지 않은지 확인하세요.")

    # ── best_params.json 저장 ──
    save_data = {
        "best_params": best.params,
        "best_logloss": best.value,
        "best_trial_number": best.number,
        "n_trials": N_TRIALS,
        "scale_pos_weight": tuning_stride.scale_pos_weight,
        "stride": 5,
        "n_models": 5,
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n  저장 완료: {BEST_PARAMS_PATH}")
    print("=" * 70)

    return save_data


if __name__ == "__main__":
    optimize()
