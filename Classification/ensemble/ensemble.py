"""
4모델 Equal-weight 앙상블 모듈 (티커 파라미터화).

Execution:
  python3 -m Classification.ensemble.ensemble
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

from common import pd
from Classification.model_config import get_ensemble_result_path
from Classification.models.logreg.pipeline import run_pipeline as run_logreg_pipeline
from Classification.models.rf.pipeline import run_pipeline as run_rf_pipeline
from Classification.models.svm.pipeline import run_pipeline as run_svm_pipeline
from Classification.models.xgb.pipeline import run_pipeline as run_xgb_pipeline
from Classification.Preprocessing.generate_target import generate_target
from Classification.Preprocessing.split_dataset import SPLIT_CONFIG, split_dataset


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ic, p_value = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0, 1.0
    if np.isnan(p_value):
        return float(ic), 1.0
    return float(ic), float(p_value)


def _validate_proba_vector(name: str, proba: Any, expected_len: int) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} proba must be 1D. got shape={arr.shape}")
    if len(arr) != expected_len:
        raise ValueError(f"{name} proba length mismatch. expected={expected_len}, got={len(arr)}")
    return arr


def _predict_mean_proba(models: list[Any], x_df: pd.DataFrame) -> np.ndarray:
    probas = []
    for item in models:
        if isinstance(item, tuple):
            model = item[0]
            scaler = item[1] if len(item) > 1 else None
            x_input = x_df if scaler is None else scaler.transform(x_df)
            proba = model.predict_proba(x_input)[:, 1]
        else:
            proba = item.predict_proba(x_df)[:, 1]
        probas.append(np.asarray(proba, dtype=float))
    return np.mean(np.vstack(probas), axis=0)


def _distribution_summary(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _get_scaled_future_features(split, ticker: str, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train-fit scaler 기준으로 test/future를 재구성해 미래 확률 입력을 생성합니다.
    """
    df_all = generate_target(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    feature_cols = split.feature_cols
    df_valid = df_all.dropna(subset=["Target_Class"]).copy()

    train_start, train_end = SPLIT_CONFIG["train"]
    raw_train = df_valid.loc[train_start:train_end, feature_cols].copy()
    scaler = StandardScaler()
    scaler.fit(raw_train)

    raw_test = df_valid.loc[SPLIT_CONFIG["test"][0]:, feature_cols].reindex(split.test.index)
    rebuilt_scaled_test = scaler.transform(raw_test)
    expected_scaled_test = split.test[feature_cols].to_numpy()
    if not np.allclose(rebuilt_scaled_test, expected_scaled_test, atol=1e-8, rtol=1e-6):
        raise RuntimeError("Scaled feature mismatch between rebuilt scaler output and split.test.")

    df_future = df_all[df_all["Target_Class"].isna()].copy()
    if df_future.empty:
        return pd.DataFrame(columns=feature_cols), df_future

    x_future_scaled = pd.DataFrame(
        scaler.transform(df_future[feature_cols]),
        index=df_future.index,
        columns=feature_cols,
    )
    return x_future_scaled, df_future


def run_equal_weight_ensemble(
    ticker: str = "AAPL",
    benchmark: str = "SP500",
    auto_update: bool = False,
    persist_total_features_on_update: bool = False,
    save_json: bool = True,
) -> dict[str, Any]:
    """
    4모델 동일가중치(0.25) 앙상블을 수행하고 지표를 저장합니다.
    """
    if auto_update or persist_total_features_on_update:
        print("[INFO] Ensemble 모듈은 no-update 정책을 강제합니다. 전달된 update 플래그는 무시됩니다.")

    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()

    split = split_dataset(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    expected_len = len(split.test)

    xgb_models, p_xgb_raw, _ = run_xgb_pipeline(
        auto_optimize=False,
        return_metrics=False,
        ticker=ticker,
        benchmark=benchmark,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    svm_models, p_svm_raw, _ = run_svm_pipeline(
        return_metrics=False,
        ticker=ticker,
        benchmark=benchmark,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    rf_models, p_rf_raw, _ = run_rf_pipeline(
        auto_optimize=False,
        return_metrics=False,
        ticker=ticker,
        benchmark=benchmark,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    logreg_models, p_logreg_raw, _ = run_logreg_pipeline(
        auto_optimize=False,
        return_metrics=False,
        ticker=ticker,
        benchmark=benchmark,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )

    p_xgb = _validate_proba_vector("xgb", p_xgb_raw, expected_len)
    p_svm = _validate_proba_vector("svm", p_svm_raw, expected_len)
    p_rf = _validate_proba_vector("rf", p_rf_raw, expected_len)
    p_logreg = _validate_proba_vector("logreg", p_logreg_raw, expected_len)

    p_ens_test = (p_xgb + p_svm + p_rf + p_logreg) / 4.0
    y_test = split.test["Target_Class"].astype(int).to_numpy()
    y_pred_test = (p_ens_test >= 0.5).astype(int)
    alpha_diff_test = (split.test[split.target_col] - split.test[split.benchmark_target_col]).to_numpy()

    ic_full, ic_pvalue = _safe_spearman(p_ens_test, alpha_diff_test)
    mid = len(p_ens_test) // 2
    ic_first, _ = _safe_spearman(p_ens_test[:mid], alpha_diff_test[:mid])
    ic_second, _ = _safe_spearman(p_ens_test[mid:], alpha_diff_test[mid:])

    accuracy_ref = float(accuracy_score(y_test, y_pred_test))
    precision_ref = float(precision_score(y_test, y_pred_test, zero_division=0))

    df_test_probs = pd.DataFrame(
        {
            "p_xgb": p_xgb,
            "p_svm": p_svm,
            "p_rf": p_rf,
            "p_logreg": p_logreg,
            "p_ens": p_ens_test,
        },
        index=split.test.index,
    )

    x_future_scaled, df_future_raw = _get_scaled_future_features(split, ticker=ticker, benchmark=benchmark)
    if x_future_scaled.empty:
        df_future_probs = pd.DataFrame(columns=["p_xgb", "p_svm", "p_rf", "p_logreg", "p_ens"])
    else:
        p_xgb_future = _predict_mean_proba(xgb_models, x_future_scaled)
        p_svm_future = _predict_mean_proba(svm_models, x_future_scaled)
        p_rf_future = _predict_mean_proba(rf_models, x_future_scaled)
        p_logreg_future = _predict_mean_proba(logreg_models, x_future_scaled)
        p_ens_future = (p_xgb_future + p_svm_future + p_rf_future + p_logreg_future) / 4.0
        df_future_probs = pd.DataFrame(
            {
                "p_xgb": p_xgb_future,
                "p_svm": p_svm_future,
                "p_rf": p_rf_future,
                "p_logreg": p_logreg_future,
                "p_ens": p_ens_future,
            },
            index=x_future_scaled.index,
        )

    if df_future_probs.empty:
        latest_future_trade_date = None
        latest_future_p_ens = None
    else:
        latest_future_trade_date = df_future_probs.index.max().strftime("%Y-%m-%d")
        latest_future_p_ens = float(df_future_probs.iloc[-1]["p_ens"])

    payload = {
        "ticker": ticker,
        "benchmark": benchmark,
        "weights": {"xgb": 0.25, "svm": 0.25, "rf": 0.25, "logreg": 0.25},
        "n_test_samples": int(len(df_test_probs)),
        "test_start": split.test.index.min().strftime("%Y-%m-%d"),
        "test_end": split.test.index.max().strftime("%Y-%m-%d"),
        "ic_full": float(ic_full),
        "ic_pvalue": float(ic_pvalue),
        "ic_first_half": float(ic_first),
        "ic_second_half": float(ic_second),
        "accuracy_ref": float(accuracy_ref),
        "precision_ref": float(precision_ref),
        "p_ens_test_stats": _distribution_summary(df_test_probs["p_ens"].to_numpy()),
        "future_window": {
            "n_samples": int(len(df_future_probs)),
            "start": df_future_raw.index.min().strftime("%Y-%m-%d") if not df_future_raw.empty else None,
            "end": df_future_raw.index.max().strftime("%Y-%m-%d") if not df_future_raw.empty else None,
        },
        "p_ens_future_stats": (
            _distribution_summary(df_future_probs["p_ens"].to_numpy())
            if not df_future_probs.empty
            else {"min": None, "mean": None, "max": None}
        ),
        "latest_future_prediction": {
            "trade_date": latest_future_trade_date,
            "p_ens": float(latest_future_p_ens) if latest_future_p_ens is not None else None,
        },
        "p_ens_column_location": {
            "test": "df_test_probs['p_ens'] (runtime)",
            "future": "df_future_probs['p_ens'] (runtime)",
        },
    }

    result_path = get_ensemble_result_path(ticker)
    if save_json:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEqual-weight ensemble complete")
    print(f"  ticker                : {ticker}")
    print(f"  Test samples          : {len(df_test_probs)}")
    print(f"  IC / p-value          : {ic_full:+.4f} / {ic_pvalue:.4f}")
    print(f"  IC first half         : {ic_first:+.4f}")
    print(f"  IC second half        : {ic_second:+.4f}")
    test_stats = payload["p_ens_test_stats"]
    print(
        "  p_ens (test) min/mean/max: "
        f"{test_stats['min']:.6f} / {test_stats['mean']:.6f} / {test_stats['max']:.6f}"
    )
    if not df_future_probs.empty:
        future_stats = payload["p_ens_future_stats"]
        print(f"  Future samples        : {len(df_future_probs)}")
        print(
            "  p_ens (future) min/mean/max: "
            f"{future_stats['min']:.6f} / {future_stats['mean']:.6f} / {future_stats['max']:.6f}"
        )
        print(f"  latest future p_ens   : {latest_future_trade_date} -> {latest_future_p_ens:.6f}")
    else:
        print("  Future samples        : 0 (no unrealized target rows)")
    print(f"  p_ens column location : {payload['p_ens_column_location']}")
    print(f"  JSON saved            : {result_path}")

    return payload


if __name__ == "__main__":
    run_equal_weight_ensemble()
