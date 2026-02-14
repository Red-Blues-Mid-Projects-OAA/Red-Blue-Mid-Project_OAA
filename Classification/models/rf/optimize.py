"""
RandomForest 하이퍼파라미터 최적화 모듈 (단일 Holdout + stride 앙상블).
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime

import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.dirname(_THIS_DIR)
_CLASSIFICATION_DIR = os.path.dirname(_MODELS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

from generate_target import generate_target
from model_config import RF_PARAMS_ARTIFACT_PATH, save_json_artifact_only

optuna.logging.set_verbosity(optuna.logging.INFO)

N_TRIALS = 100
OBJECTIVE_VERSION = "target_aligned_v3_rf_no_class_weight"
CV_MODE = "single_holdout_2024Q2Q3"
STRIDE = 5
N_STRIDE_MODELS = 5

TUNE_SPLIT = {
    "train": ("2021-01-01", "2023-12-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}


def _get_feature_hash(feature_cols: list[str]) -> str:
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0
    return float(ic)


def _get_search_space(trial: optuna.Trial, profile: str) -> dict:
    if profile == "balanced":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 520),
            # RF는 test gap 안정화를 위해 shallow tree(깊이 2)로 고정합니다.
            "max_depth": trial.suggest_int("max_depth", 2, 2),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 120),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 60),
            "max_features": trial.suggest_float("max_features", 0.35, 0.75),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0005, 0.03),
            "random_state": 42,
            "n_jobs": -1,
        }

    if profile == "regularized":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 80, 320),
            "max_depth": trial.suggest_int("max_depth", 2, 2),
            "min_samples_split": trial.suggest_int("min_samples_split", 40, 140),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 80),
            "max_features": trial.suggest_float("max_features", 0.25, 0.55),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.004, 0.04),
            "random_state": 42,
            "n_jobs": -1,
        }

    raise ValueError(f"지원하지 않는 profile 입니다: {profile}")


def create_objective(full_df, feature_cols: list[str], profile: str):
    train_start, train_end = TUNE_SPLIT["train"]
    val_start, val_end = TUNE_SPLIT["validation"]

    train_fold = full_df.loc[train_start:train_end].copy()
    val_fold = full_df.loc[val_start:val_end].copy()

    if train_fold.empty or val_fold.empty:
        raise ValueError(
            "튜닝 분할 데이터가 비어 있습니다. "
            f"Train[{train_start}~{train_end}] / Val[{val_start}~{val_end}]"
        )

    x_train = train_fold[feature_cols].values
    y_train = train_fold["Target_Class"].astype(int).to_numpy()
    x_val = val_fold[feature_cols].values
    y_val = val_fold["Target_Class"].astype(int).to_numpy()
    alpha_val = val_fold["Alpha_Diff"].astype(float).to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = _get_search_space(trial, profile)

        val_proba_list = []
        train_proba_list = []

        for offset in range(N_STRIDE_MODELS):
            stride_idx = np.arange(offset, len(y_train), STRIDE)
            if len(stride_idx) == 0:
                continue

            x_train_stride = x_train[stride_idx]
            y_train_stride = y_train[stride_idx]

            model = RandomForestClassifier(**params)
            model.fit(x_train_stride, y_train_stride)

            val_proba_list.append(model.predict_proba(x_val)[:, 1])
            train_proba_list.append(model.predict_proba(x_train)[:, 1])

        if not val_proba_list:
            return 9e9

        val_proba = np.mean(val_proba_list, axis=0)
        train_proba = np.mean(train_proba_list, axis=0)

        val_pred = (val_proba >= 0.5).astype(int)
        train_pred = (train_proba >= 0.5).astype(int)

        val_acc = accuracy_score(y_val, val_pred)
        train_acc = accuracy_score(y_train, train_pred)
        val_logloss = log_loss(y_val, val_proba, labels=[0, 1])
        val_ic = _safe_spearman(val_proba, alpha_val)
        gap = train_acc - val_acc

        val_proba_std = float(np.std(val_proba))
        pos_rate = float(val_pred.mean())
        balance = min(pos_rate, 1.0 - pos_rate)

        noncollapse_penalty = 2.2 * max(0.0, 0.02 - val_proba_std)
        class_balance_penalty = 1.2 * max(0.0, 0.08 - balance)

        score = (
            (1.0 - val_acc)
            + 0.25 * val_logloss
            + 2.2 * max(0.0, 0.05 - val_ic)
            + 2.0 * max(0.0, gap - 0.20)
            + 0.8 * max(0.0, train_acc - 0.72)
            + noncollapse_penalty
            + class_balance_penalty
        )

        trial.set_user_attr("train_acc", float(train_acc))
        trial.set_user_attr("val_acc", float(val_acc))
        trial.set_user_attr("val_logloss", float(val_logloss))
        trial.set_user_attr("val_ic", float(val_ic))
        trial.set_user_attr("gap", float(gap))
        trial.set_user_attr("val_proba_std", val_proba_std)
        trial.set_user_attr("pos_rate", pos_rate)
        trial.set_user_attr("score", float(score))

        return score

    return objective


def optimize(profile: str = "balanced", n_trials: int = N_TRIALS):
    if profile not in {"balanced", "regularized"}:
        raise ValueError(f"profile은 'balanced' 또는 'regularized'만 허용됩니다: {profile}")

    print("=" * 70)
    print("RandomForest Hyperparameter Optimization (Single Holdout)")
    print("=" * 70)

    full_df = generate_target()
    exclude = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class", "Alpha_Diff"]
    feature_cols = [c for c in full_df.columns if c not in exclude]
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
        study_name=f"rf_{profile}_{CV_MODE}",
        sampler=sampler,
    )
    study.optimize(
        create_objective(full_df, feature_cols, profile=profile),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_val_acc = float(best.user_attrs.get("val_acc", np.nan))
    best_val_ic = float(best.user_attrs.get("val_ic", np.nan))
    best_gap = float(best.user_attrs.get("gap", np.nan))
    best_val_logloss = float(best.user_attrs.get("val_logloss", np.nan))

    print("\n" + "=" * 70)
    print("★ 최적화 결과")
    print("=" * 70)
    print(f"  Best Objective Score : {best.value:.6f}")
    print(f"  Validation Accuracy  : {best_val_acc * 100:.2f}%")
    print(f"  Validation IC        : {best_val_ic:+.4f}")
    print(f"  Train-Val Gap        : {best_gap * 100:.2f}%p")
    print(f"  Validation LogLoss   : {best_val_logloss:.6f}")
    print("  Best Parameters:")
    for key, val in best.params.items():
        print(f"    {key:20s}: {val}")

    save_data = {
        "best_params": {
            "n_estimators": int(best.params["n_estimators"]),
            "max_depth": int(best.params["max_depth"]),
            "min_samples_split": int(best.params["min_samples_split"]),
            "min_samples_leaf": int(best.params["min_samples_leaf"]),
            "max_features": float(best.params["max_features"]),
            "ccp_alpha": float(best.params["ccp_alpha"]),
        },
        "common_params": {
            "random_state": 42,
            "n_jobs": -1,
        },
        "best_logloss": best_val_logloss,
        "best_objective_score": float(best.value),
        "best_trial_number": int(best.number),
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
        "selected_metrics": {
            "validation_accuracy": best_val_acc,
            "validation_ic": best_val_ic,
            "validation_logloss": best_val_logloss,
            "train_val_gap": best_gap,
        },
    }

    save_json_artifact_only(save_data, RF_PARAMS_ARTIFACT_PATH)
    print(f"\n  저장 완료: {RF_PARAMS_ARTIFACT_PATH}")
    return save_data


if __name__ == "__main__":
    optimize()
