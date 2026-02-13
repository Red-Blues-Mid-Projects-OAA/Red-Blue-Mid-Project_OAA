"""
Hyperparameter 최적화 모듈 (단일 Holdout 검증)

고정 분할 정책:
  - Train      : 2021-01-01 ~ 2023-12-31
  - Embargo    : 2024-01-01 ~ 2024-03-31 (튜닝 학습 제외)
  - Validation : 2024-04-01 ~ 2024-09-30
  - Golden Gap : 2024-10-01 ~ 2024-12-31 (튜닝/평가 제외)
  - Test       : 2025-01-01 ~ 현재 (튜닝 미사용)

결과는 xgb_best_params.json에 저장됩니다.
"""

import sys
import os
import json
import hashlib
from datetime import datetime

import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLASSIFICATION_DIR = os.path.dirname(_MODELS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

from generate_target import generate_target
from model_config import (
    XGB_PARAMS_ARTIFACT_PATH,
    save_json_artifact_only,
)

# Optuna 로그 레벨 (INFO)
optuna.logging.set_verbosity(optuna.logging.INFO)

N_TRIALS = 100
OBJECTIVE_VERSION = "target_aligned_v4_stride_consistent"
CV_MODE = "single_holdout_2024Q2Q3"
STRIDE = 5
N_STRIDE_MODELS = 5

# 튜닝에 사용하는 단일 분할 정의
TUNE_SPLIT = {
    "train": ("2021-01-01", "2023-12-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}


def _get_feature_hash(feature_cols):
    """피처 목록 기반 해시를 생성합니다(순서 민감)."""
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_spearman(x, y):
    """상수 벡터 등으로 IC가 NaN이 되는 경우 0으로 보정합니다."""
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0
    return float(ic)


def _get_search_space(trial, profile):
    """
    프로파일별 하이퍼파라미터 탐색 공간을 정의합니다.
    - balanced   : 경량 확장
    - regularized: 규제 강화 프로파일
    """
    if profile == "balanced":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.18, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 60, 140),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 70.0),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "eval_metric": "logloss",
            "early_stopping_rounds": 25,
        }

    if profile == "regularized":
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 2),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 180),
            "min_child_weight": trial.suggest_int("min_child_weight", 4, 24),
            "gamma": trial.suggest_float("gamma", 1.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 2.0, 12.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 20.0, 140.0),
            "subsample": trial.suggest_float("subsample", 0.80, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "eval_metric": "logloss",
            "early_stopping_rounds": 25,
        }

    raise ValueError(f"지원하지 않는 profile 입니다: {profile}")


def create_objective(full_df, feature_cols, profile):
    """
    단일 Holdout 분할에서 'stride 5-모델 앙상블' 기준 점수를 최소화하는 Objective를 생성합니다.

    핵심:
    - 튜닝 단계에서도 실전과 동일하게 STRIDE=5, 5개 모델을 학습/평균합니다.
    - scaler는 Train 구간에만 fit하여 누수를 방지합니다.
    score = (1 - val_acc) + 0.25*val_logloss + 1.8*max(0, 0.05 - val_ic) + 1.2*max(0, gap - 0.25)
            + 2.2*max(0, 0.03 - std(val_proba))
            + 1.5*max(0, 0.08 - min(pos_rate, 1-pos_rate))
    """
    train_start, train_end = TUNE_SPLIT["train"]
    val_start, val_end = TUNE_SPLIT["validation"]

    train_fold = full_df.loc[train_start:train_end].copy()
    val_fold = full_df.loc[val_start:val_end].copy()

    if train_fold.empty or val_fold.empty:
        raise ValueError(
            "튜닝 분할 데이터가 비어 있습니다. "
            f"Train[{train_start}~{train_end}] / Val[{val_start}~{val_end}]"
        )

    def objective(trial):
        params = _get_search_space(trial, profile)

        # Train 기준 스케일링으로 누수를 방지합니다.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_fold[feature_cols].values)
        X_val = scaler.transform(val_fold[feature_cols].values)

        y_train = train_fold["Target_Class"].astype(int).reset_index(drop=True)
        y_val = val_fold["Target_Class"].astype(int).reset_index(drop=True)
        alpha_val = val_fold["Alpha_Diff"].astype(float).values

        # 튜닝도 실전과 같은 stride 5-모델 앙상블로 평가합니다.
        val_proba_list = []
        train_proba_list = []

        for offset in range(N_STRIDE_MODELS):
            stride_idx = np.arange(offset, len(y_train), STRIDE)
            X_train_stride = X_train[stride_idx]
            y_train_stride = y_train.iloc[stride_idx]

            # 클래스 불균형 보정 비율(각 stride별)
            n_neg = int((y_train_stride == 0).sum())
            n_pos = max(int((y_train_stride == 1).sum()), 1)
            spw = float(n_neg) / n_pos

            # 정책 반영: 튜닝 단계에서만 random_state=42 사용
            model = XGBClassifier(**params, scale_pos_weight=spw, random_state=42)
            model.fit(
                X_train_stride,
                y_train_stride,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_proba_list.append(model.predict_proba(X_val)[:, 1])
            train_proba_list.append(model.predict_proba(X_train)[:, 1])

        val_proba = np.mean(val_proba_list, axis=0)
        train_proba = np.mean(train_proba_list, axis=0)
        train_pred = (train_proba >= 0.5).astype(int)
        val_pred = (val_proba >= 0.5).astype(int)

        # 목표 정렬형 점수 계산 요소
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        val_logloss = log_loss(y_val, val_proba, labels=[0, 1])
        val_ic = _safe_spearman(val_proba, alpha_val)
        gap = train_acc - val_acc
        val_proba_std = float(np.std(val_proba))
        pos_rate = float(val_pred.mean())
        balance = min(pos_rate, 1.0 - pos_rate)

        # 상수 확률/단일 클래스 예측으로 붕괴되는 해를 억제합니다.
        noncollapse_penalty = 2.2 * max(0.0, 0.03 - val_proba_std)
        class_balance_penalty = 1.5 * max(0.0, 0.08 - balance)

        # score = (1 - val_acc) + 0.25*val_logloss + 1.8*max(0, 0.05 - val_ic) + 1.2*max(0, gap - 0.25)
        #         + 2.2*max(0, 0.03 - std(val_proba))
        #         + 1.5*max(0, 0.08 - min(pos_rate, 1-pos_rate))
        score = (
            (1.0 - val_acc)
            + 0.25 * val_logloss
            + 1.8 * max(0.0, 0.05 - val_ic)
            + 1.2 * max(0.0, gap - 0.25)
            + noncollapse_penalty
            + class_balance_penalty
        )

        # 추후 디버깅/리포팅을 위한 메타 기록
        trial.set_user_attr("train_acc", float(train_acc))
        trial.set_user_attr("val_acc", float(val_acc))
        trial.set_user_attr("val_logloss", float(val_logloss))
        trial.set_user_attr("val_ic", float(val_ic))
        trial.set_user_attr("gap", float(gap))
        trial.set_user_attr("val_proba_std", val_proba_std)
        trial.set_user_attr("pos_rate", pos_rate)
        trial.set_user_attr("noncollapse_penalty", float(noncollapse_penalty))
        trial.set_user_attr("class_balance_penalty", float(class_balance_penalty))
        trial.set_user_attr("score", float(score))

        return score

    return objective


def optimize(profile="balanced", n_trials=N_TRIALS):
    """
    단일 Holdout 기준으로 XGBoost 하이퍼파라미터를 최적화합니다.

    Args:
        profile (str): 'balanced' 또는 'regularized'
        n_trials (int): Optuna trial 수
    """
    if profile not in {"balanced", "regularized"}:
        raise ValueError(f"profile은 'balanced' 또는 'regularized'만 허용됩니다: {profile}")

    print("=" * 70)
    print("Hyperparameter 최적화 (Single Holdout)")
    print("=" * 70)

    # 1. 전체 데이터 로드
    full_df = generate_target()

    # Feature 컬럼 발라내기
    exclude = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class", "Alpha_Diff"]
    feature_cols = [c for c in full_df.columns if c not in exclude]

    # 타겟 있는 데이터만 사용
    full_df = full_df.dropna(subset=["Target_Class"])

    print(f"\n  전체 데이터: {len(full_df)}건 ({full_df.index.min().date()} ~ {full_df.index.max().date()})")
    print(f"  튜닝 프로파일: {profile}")
    print(f"  CV 모드      : {CV_MODE}")
    print(
        f"  Train 구간   : {TUNE_SPLIT['train'][0]} ~ {TUNE_SPLIT['train'][1]}\n"
        f"  Validation   : {TUNE_SPLIT['validation'][0]} ~ {TUNE_SPLIT['validation'][1]}"
    )
    print(f"  Trial 수     : {n_trials}")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=f"xgb_{profile}_{CV_MODE}",
        sampler=sampler,
    )
    study.optimize(
        create_objective(full_df, feature_cols, profile=profile),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_logloss = float(best.user_attrs.get("val_logloss", np.nan))
    best_val_acc = float(best.user_attrs.get("val_acc", np.nan))
    best_val_ic = float(best.user_attrs.get("val_ic", np.nan))
    best_gap = float(best.user_attrs.get("gap", np.nan))
    best_proba_std = float(best.user_attrs.get("val_proba_std", np.nan))
    best_pos_rate = float(best.user_attrs.get("pos_rate", np.nan))

    print("\n" + "=" * 70)
    print("★ 최적화 결과")
    print("=" * 70)
    print(f"  Best Objective Score : {best.value:.6f}")
    print(f"  Validation Accuracy  : {best_val_acc * 100:.2f}%")
    print(f"  Validation IC        : {best_val_ic:+.4f}")
    print(f"  Train-Val Gap        : {best_gap * 100:.2f}%p")
    print(f"  Validation LogLoss   : {best_logloss:.6f}")
    print(f"  Val Proba Std        : {best_proba_std:.4f}")
    print(f"  Val Positive Rate    : {best_pos_rate * 100:.2f}%")
    print("  Best Parameters:")
    for key, val in best.params.items():
        print(f"    {key:20s}: {val}")

    # 저장 (기존 스키마 호환 키 + 확장 메타데이터)
    save_data = {
        "best_params": best.params,
        "best_logloss": best_logloss,
        "best_objective_score": float(best.value),
        "best_trial_number": best.number,
        "n_trials": int(n_trials),
        "optimization_method": "Single Holdout (Target-Aligned Objective)",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_end_date": full_df.index.max().strftime("%Y-%m-%d"),
        "feature_count": len(feature_cols),
        "feature_hash": _get_feature_hash(feature_cols),
        "split_definition": {
            "train": [TUNE_SPLIT["train"][0], TUNE_SPLIT["train"][1]],
            "embargo": [TUNE_SPLIT["embargo"][0], TUNE_SPLIT["embargo"][1]],
            "validation": [TUNE_SPLIT["validation"][0], TUNE_SPLIT["validation"][1]],
            "golden_gap": [TUNE_SPLIT["golden_gap"][0], TUNE_SPLIT["golden_gap"][1]],
            "test": [TUNE_SPLIT["test"][0], TUNE_SPLIT["test"][1]],
        },
        "profile": profile,
        "objective_version": OBJECTIVE_VERSION,
        "cv_mode": CV_MODE,
    }

    save_json_artifact_only(save_data, XGB_PARAMS_ARTIFACT_PATH)
    print(f"\n  저장 완료: {XGB_PARAMS_ARTIFACT_PATH}")
    return save_data


if __name__ == "__main__":
    optimize()
