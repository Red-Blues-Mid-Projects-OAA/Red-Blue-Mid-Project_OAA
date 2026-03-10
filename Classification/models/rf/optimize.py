"""
이 파일은 최적화 관련 작업을 담당합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

import numpy as np
import optuna
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from Classification.Preprocessing.generate_target import generate_target
from Classification.model_config import get_model_params_path, save_json_artifact_only

optuna.logging.set_verbosity(optuna.logging.INFO)

# Optuna가 시도할 하이퍼파라미터 탐색 횟수입니다.
N_TRIALS = 100
# 현재 파일이 기대하는 목적 함수 버전 문자열입니다.
BASE_OBJECTIVE_VERSION = "target_aligned_v3_rf_no_class_weight"
# 저장된 파라미터와 비교할 교차검증 분할 정책 이름입니다.
CV_MODE = "single_holdout_2024Q2Q3"
# 스트라이드 분할에서 다음 모델로 이동할 날짜 간격입니다.
STRIDE = 5
# 한 번의 스트라이드 실험에서 학습할 모델 수입니다.
N_STRIDE_MODELS = 5
# 단계형 게이트 전략 전용 튜닝 분기를 구분하는 프로필 이름입니다.
TSM_GATE_PROFILE = "tsm_gatehard_v1"
# 단계형 게이트 전략에서 허용하는 스테이지 집합입니다.
TSM_STAGES = {"stage1", "stage2"}
# 튜닝 전용 날짜 분할 규칙으로, 학습/검증/테스트 구간 경계를 명시합니다.
TUNE_SPLIT = {
    "train": ("2021-01-01", "2023-12-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}

def _get_feature_hash(feature_cols: list[str]) -> str:
    """피처 hash 정보를 조회해 반환합니다."""
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """예외 상황을 감안해 스피어만 상관계수를 안전하게 계산합니다."""
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0
    return float(ic)

def _apply_direction(proba: np.ndarray, direction_mode: str) -> np.ndarray:
    """지표 방향성을 반영해 점수가 클수록 좋도록 맞춥니다."""
    if direction_mode == "normal":
        return proba
    if direction_mode == "inverted":
        return 1.0 - proba
    raise ValueError(f"지원하지 않는 direction_mode 입니다: {direction_mode}")

def _compute_recency_weights(index, recency_weight_lambda: float) -> np.ndarray | None:
    """최근 데이터에 더 큰 비중을 주는 학습 가중치를 계산합니다."""
    if recency_weight_lambda <= 0.0:
        return None
    rank = np.arange(len(index), dtype=float)
    weights = 1.0 + recency_weight_lambda * rank
    weights = np.clip(weights, 1e-8, None)
    return weights / np.mean(weights)

def _split_half_ic(proba: np.ndarray, alpha_diff: np.ndarray) -> tuple[float, float]:
    """half ic를 기준에 따라 나눕니다."""
    mid = len(proba) // 2
    if mid == 0:
        ic = _safe_spearman(proba, alpha_diff)
        return ic, ic
    first_ic = _safe_spearman(proba[:mid], alpha_diff[:mid])
    second_ic = _safe_spearman(proba[mid:], alpha_diff[mid:])
    return first_ic, second_ic

def _get_objective_version(profile: str, strategy_stage: str) -> str:
    """objective version 정보를 조회해 반환합니다."""
    if profile == TSM_GATE_PROFILE:
        return f"{TSM_GATE_PROFILE}_{strategy_stage}"
    return BASE_OBJECTIVE_VERSION

def _suggest_class_weight(trial: optuna.Trial, profile: str):
    """클래스 불균형 완화를 위한 권장 가중치를 계산합니다."""
    if profile != TSM_GATE_PROFILE:
        return None, "none"

    mode = trial.suggest_categorical(
        "class_weight_mode_key",
        ["none", "balanced", "balanced_subsample", "custom"],
    )
    if mode == "none":
        return None, "none"
    if mode == "balanced":
        return "balanced", "balanced"
    if mode == "balanced_subsample":
        return "balanced_subsample", "balanced_subsample"

    w = float(trial.suggest_float("class_weight_1", 1.0, 5.0))
    return {0: 1.0, 1: w}, f"custom_1:{w:.4f}"

def _get_search_space(trial: optuna.Trial, profile: str) -> tuple[dict, str]:
    """search space 정보를 조회해 반환합니다."""
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
            "class_weight": None,
        }, "none"

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
            "class_weight": None,
        }, "none"

    if profile == TSM_GATE_PROFILE:
        class_weight, class_weight_mode = _suggest_class_weight(trial, profile)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 120, 700),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 160),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "max_features": trial.suggest_float("max_features", 0.20, 1.0),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.06),
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": class_weight,
        }, class_weight_mode

    raise ValueError(f"지원하지 않는 profile 입니다: {profile}")

def _compute_score(
    *,
    profile: str,
    strategy_stage: str,
    val_acc: float,
    val_logloss: float,
    val_ic: float,
    gap_abs: float,
    val_proba_std: float,
    val_proba_unique: int,
    pos_rate: float,
    ic_first: float,
    ic_second: float,
) -> float:
    """최적화에 사용할 종합 점수를 계산합니다."""
    balance = min(pos_rate, 1.0 - pos_rate)
    flat_penalty = 4.0 * max(0.0, 0.03 - val_proba_std)
    unique_penalty = 0.25 * max(0.0, 8 - float(val_proba_unique))

    if profile in {"balanced", "regularized"}:
        class_balance_penalty = 1.2 * max(0.0, 0.08 - balance)
        return (
            (1.0 - val_acc)
            + 0.25 * val_logloss
            + 2.2 * max(0.0, 0.05 - val_ic)
            + 2.0 * max(0.0, gap_abs - 0.20)
            + 0.8 * max(0.0, val_acc - 0.72)
            + flat_penalty
            + unique_penalty
            + class_balance_penalty
        )

    # tsm_gatehard_v1
    p_acc = max(0.0, 0.52 - val_acc)
    p_ic = max(0.0, 0.05 - val_ic)
    p_gap = max(0.0, gap_abs - 0.25)
    class_balance_penalty = 1.6 * max(0.0, 0.08 - balance)

    score = (
        (1.0 - val_acc)
        + 0.20 * val_logloss
        + 3.6 * p_acc
        + 3.6 * p_ic
        + 2.8 * p_gap
        + flat_penalty
        + unique_penalty
        + class_balance_penalty
    )

    if strategy_stage == "stage2":
        stability_penalty = 2.0 * abs(ic_first - ic_second)
        sign_penalty = 1.6 if ic_first * ic_second < 0 else 0.0
        score += stability_penalty + sign_penalty

    return float(score)

def create_objective(full_df, feature_cols: list[str], profile: str, strategy_stage: str):
    """objective를 준비하거나 생성합니다."""
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
        """하이퍼파라미터 탐색에서 사용할 목적 함수를 계산합니다."""
        params, class_weight_mode = _get_search_space(trial, profile)

        recency_weight_lambda = 0.0
        if profile == TSM_GATE_PROFILE and strategy_stage == "stage2":
            recency_weight_lambda = float(
                trial.suggest_float("recency_weight_lambda", 0.0, 0.02)
            )

        train_weight_full = _compute_recency_weights(train_fold.index, recency_weight_lambda)

        val_proba_list = []
        train_proba_list = []

        for offset in range(N_STRIDE_MODELS):
            stride_idx = np.arange(offset, len(y_train), STRIDE)
            if len(stride_idx) == 0:
                continue

            x_train_stride = x_train[stride_idx]
            y_train_stride = y_train[stride_idx]

            sw_stride = None
            if train_weight_full is not None:
                sw_stride = train_weight_full[stride_idx]

            model = RandomForestClassifier(**params)
            fit_kwargs = {}
            if sw_stride is not None:
                fit_kwargs["sample_weight"] = sw_stride
            model.fit(x_train_stride, y_train_stride, **fit_kwargs)

            val_proba_list.append(model.predict_proba(x_val)[:, 1])
            train_proba_list.append(model.predict_proba(x_train)[:, 1])

        if not val_proba_list:
            return 9e9

        val_proba_raw = np.mean(val_proba_list, axis=0)
        train_proba_raw = np.mean(train_proba_list, axis=0)

        direction_candidates = ["normal"]
        if profile == TSM_GATE_PROFILE and strategy_stage == "stage2":
            direction_candidates = ["normal", "inverted"]

        best_candidate = None
        non_degenerate_found = False
        for direction_mode in direction_candidates:
            val_proba = _apply_direction(val_proba_raw, direction_mode)
            train_proba = _apply_direction(train_proba_raw, direction_mode)

            val_pred = (val_proba >= 0.5).astype(int)
            train_pred = (train_proba >= 0.5).astype(int)

            val_acc = accuracy_score(y_val, val_pred)
            train_acc = accuracy_score(y_train, train_pred)
            val_logloss = log_loss(y_val, val_proba, labels=[0, 1])
            val_ic = _safe_spearman(val_proba, alpha_val)
            gap_signed = train_acc - val_acc
            gap_abs = abs(gap_signed)

            val_proba_std = float(np.std(val_proba))
            val_proba_unique = int(np.unique(np.round(val_proba, 6)).size)
            val_pred_unique = int(np.unique(val_pred).size)
            pos_rate = float(val_pred.mean())
            ic_first, ic_second = _split_half_ic(val_proba, alpha_val)

            is_degenerate = (val_proba_std < 1e-6) or (val_pred_unique < 2)
            if is_degenerate:
                score = 9e9
            else:
                non_degenerate_found = True

                score = _compute_score(
                    profile=profile,
                    strategy_stage=strategy_stage,
                    val_acc=val_acc,
                    val_logloss=val_logloss,
                    val_ic=val_ic,
                    gap_abs=gap_abs,
                    val_proba_std=val_proba_std,
                    val_proba_unique=val_proba_unique,
                    pos_rate=pos_rate,
                    ic_first=ic_first,
                    ic_second=ic_second,
                )

            candidate = {
                "score": float(score),
                "direction_mode": direction_mode,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "val_logloss": float(val_logloss),
                "val_ic": float(val_ic),
                "gap": float(gap_abs),
                "gap_signed": float(gap_signed),
                "gap_abs": float(gap_abs),
                "val_proba_std": val_proba_std,
                "val_proba_unique": int(val_proba_unique),
                "val_pred_unique": int(val_pred_unique),
                "pos_rate": pos_rate,
                "ic_first": float(ic_first),
                "ic_second": float(ic_second),
                "is_degenerate": bool(is_degenerate),
            }
            if best_candidate is None or candidate["score"] < best_candidate["score"]:
                best_candidate = candidate

        if not non_degenerate_found:
            trial.set_user_attr("degenerate_rejected", True)
            trial.set_user_attr("score", float(9e9))
            return float(9e9)

        trial.set_user_attr("train_acc", best_candidate["train_acc"])
        trial.set_user_attr("val_acc", best_candidate["val_acc"])
        trial.set_user_attr("val_logloss", best_candidate["val_logloss"])
        trial.set_user_attr("val_ic", best_candidate["val_ic"])
        trial.set_user_attr("gap", best_candidate["gap"])
        trial.set_user_attr("train_val_gap_signed", best_candidate["gap_signed"])
        trial.set_user_attr("train_val_gap_abs", best_candidate["gap_abs"])
        trial.set_user_attr("val_proba_std", best_candidate["val_proba_std"])
        trial.set_user_attr("val_proba_unique", best_candidate["val_proba_unique"])
        trial.set_user_attr("val_pred_unique", best_candidate["val_pred_unique"])
        trial.set_user_attr("pos_rate", best_candidate["pos_rate"])
        trial.set_user_attr("ic_first", best_candidate["ic_first"])
        trial.set_user_attr("ic_second", best_candidate["ic_second"])
        trial.set_user_attr("direction_mode", best_candidate["direction_mode"])
        trial.set_user_attr("recency_weight_lambda", float(recency_weight_lambda))
        trial.set_user_attr("class_weight_mode", class_weight_mode)
        trial.set_user_attr("degenerate_rejected", bool(best_candidate["is_degenerate"]))
        trial.set_user_attr("score", best_candidate["score"])

        return best_candidate["score"]

    return objective

def optimize(
    profile: str = "balanced",
    n_trials: int = N_TRIALS,
    ticker: str = "AAPL",
    benchmark: str = "SP500",
    auto_update: bool = True,
    persist_total_features_on_update: bool = True,
    feature_source_mode: str = "db_first",
    strategy_stage: str = "stage2",
):
    """핵심 결과가 더 좋아지도록 최적화합니다."""
    if profile not in {"balanced", "regularized", TSM_GATE_PROFILE}:
        raise ValueError(
            "profile은 'balanced', 'regularized', 'tsm_gatehard_v1'만 허용됩니다: "
            f"{profile}"
        )

    resolved_stage = "default"
    if profile == TSM_GATE_PROFILE:
        if strategy_stage not in TSM_STAGES:
            raise ValueError(f"strategy_stage는 {sorted(TSM_STAGES)} 중 하나여야 합니다: {strategy_stage}")
        resolved_stage = strategy_stage

    objective_version = _get_objective_version(profile, resolved_stage)

    print("=" * 70)
    print("RandomForest Hyperparameter Optimization (Single Holdout)")
    print("=" * 70)

    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    target_col = f"Target_{ticker}_3M"
    benchmark_target_col = f"Target_{benchmark}_3M"
    params_path = get_model_params_path("rf", ticker)

    full_df = generate_target(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=auto_update,
        persist_total_features_on_update=persist_total_features_on_update,
        feature_source_mode=feature_source_mode,
    )
    exclude = [target_col, benchmark_target_col, "Target_Class", "Alpha_Diff", "TICKER"]
    feature_cols = [c for c in full_df.columns if c not in exclude]
    full_df = full_df.dropna(subset=["Target_Class"])

    print(f"\n  전체 데이터: {len(full_df)}건 ({full_df.index.min().date()} ~ {full_df.index.max().date()})")
    print(f"  튜닝 프로파일: {profile}")
    print(f"  전략 스테이지: {resolved_stage}")
    print(f"  CV 모드      : {CV_MODE}")
    print(
        f"  Train 구간   : {TUNE_SPLIT['train'][0]} ~ {TUNE_SPLIT['train'][1]}\n"
        f"  Validation   : {TUNE_SPLIT['validation'][0]} ~ {TUNE_SPLIT['validation'][1]}"
    )
    print(f"  Trial 수     : {n_trials}")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=f"rf_{profile}_{resolved_stage}_{CV_MODE}",
        sampler=sampler,
    )
    study.optimize(
        create_objective(full_df, feature_cols, profile=profile, strategy_stage=resolved_stage),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    best_val_acc = float(best.user_attrs.get("val_acc", np.nan))
    best_val_ic = float(best.user_attrs.get("val_ic", np.nan))
    best_gap_signed = float(best.user_attrs.get("train_val_gap_signed", np.nan))
    if np.isnan(best_gap_signed):
        best_gap_signed = float(best.user_attrs.get("gap", np.nan))
    best_gap_abs = float(best.user_attrs.get("train_val_gap_abs", np.nan))
    if np.isnan(best_gap_abs):
        best_gap_abs = abs(best_gap_signed) if not np.isnan(best_gap_signed) else np.nan
    best_val_logloss = float(best.user_attrs.get("val_logloss", np.nan))
    best_val_proba_std = float(best.user_attrs.get("val_proba_std", np.nan))
    best_val_proba_unique = int(best.user_attrs.get("val_proba_unique", 0))
    best_val_pred_unique = int(best.user_attrs.get("val_pred_unique", 0))
    best_direction_mode = str(best.user_attrs.get("direction_mode", "normal"))
    best_recency_weight_lambda = float(best.user_attrs.get("recency_weight_lambda", 0.0))
    best_class_weight_mode = str(best.user_attrs.get("class_weight_mode", "none"))

    print("\n" + "=" * 70)
    print("★ 최적화 결과")
    print("=" * 70)
    print(f"  Best Objective Score : {best.value:.6f}")
    print(f"  Validation Accuracy  : {best_val_acc * 100:.2f}%")
    print(f"  Validation IC        : {best_val_ic:+.4f}")
    print(
        f"  Train-Val Gap        : signed={best_gap_signed * 100:.2f}%p, "
        f"abs={best_gap_abs * 100:.2f}%p"
    )
    print(f"  Validation LogLoss   : {best_val_logloss:.6f}")
    print(f"  Val Proba Std        : {best_val_proba_std:.4f}")
    print(f"  Val Proba Unique(6d) : {best_val_proba_unique}")
    print(f"  Val Pred Unique      : {best_val_pred_unique}")
    print(f"  Direction Mode       : {best_direction_mode}")
    print(f"  Recency λ            : {best_recency_weight_lambda:.6f}")
    print(f"  Class Weight Mode    : {best_class_weight_mode}")
    print("  Best Parameters:")
    for key, val in best.params.items():
        print(f"    {key:20s}: {val}")

    class_weight_value = None
    if "class_weight_mode_key" in best.params:
        cw_mode = best.params["class_weight_mode_key"]
        if cw_mode == "balanced":
            class_weight_value = "balanced"
        elif cw_mode == "balanced_subsample":
            class_weight_value = "balanced_subsample"
        elif cw_mode == "custom":
            class_weight_value = {0: 1.0, 1: float(best.params.get("class_weight_1", 1.0))}

    degenerate_rejected_count = int(
        sum(
            1
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and bool(t.user_attrs.get("degenerate_rejected", False))
        )
    )

    save_data = {
        "best_params": {
            "n_estimators": int(best.params["n_estimators"]),
            "max_depth": int(best.params["max_depth"]),
            "min_samples_split": int(best.params["min_samples_split"]),
            "min_samples_leaf": int(best.params["min_samples_leaf"]),
            "max_features": float(best.params["max_features"]),
            "ccp_alpha": float(best.params["ccp_alpha"]),
            "class_weight": class_weight_value,
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
        "objective_version": objective_version,
        "cv_mode": CV_MODE,
        "strategy_stage": resolved_stage,
        "direction_mode": best_direction_mode,
        "recency_weight_lambda": best_recency_weight_lambda,
        "class_weight_mode": best_class_weight_mode,
        "gate_objective_version": objective_version,
        "train_val_gap_signed": best_gap_signed,
        "train_val_gap_abs": best_gap_abs,
        "val_proba_std": best_val_proba_std,
        "val_proba_unique": best_val_proba_unique,
        "val_pred_unique": best_val_pred_unique,
        "degenerate_rejected_count": degenerate_rejected_count,
        "selected_metrics": {
            "validation_accuracy": best_val_acc,
            "validation_ic": best_val_ic,
            "validation_logloss": best_val_logloss,
            "train_val_gap": best_gap_abs,
            "train_val_gap_signed": best_gap_signed,
            "train_val_gap_abs": best_gap_abs,
            "val_proba_std": best_val_proba_std,
            "val_proba_unique": best_val_proba_unique,
            "val_pred_unique": best_val_pred_unique,
        },
    }

    save_json_artifact_only(save_data, params_path)
    print(f"\n  저장 완료: {params_path}")
    return save_data

if __name__ == "__main__":
    optimize()
