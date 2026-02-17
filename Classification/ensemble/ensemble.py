"""
Equal-weight ensemble for XGB/SVM/RF/LogReg.

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
from Classification.model_config import ENSEMBLE_RESULT_PATH, ensure_artifact_dirs
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
        raise ValueError(
            f"{name} proba length mismatch. expected={expected_len}, got={len(arr)}"
        )
    return arr


def _predict_mean_proba(models: list[Any], x_df: pd.DataFrame) -> np.ndarray:
    probas = []
    for item in models:
        if isinstance(item, tuple):
            model = item[0]
            scaler = item[1] if len(item) > 1 else None
            if scaler is None:
                x_input = x_df
            else:
                x_input = scaler.transform(x_df)
            proba = model.predict_proba(x_input)[:, 1]
        else:
            proba = item.predict_proba(x_df)[:, 1]
        probas.append(np.asarray(proba, dtype=float))
    return np.mean(np.vstack(probas), axis=0)


def _to_json_number(v: Any) -> float | None:
    if v is None:
        return None
    x = float(v)
    if np.isnan(x) or np.isinf(x):
        return None
    return x


def _distribution_summary(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _get_scaled_future_features(split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build scaled feature matrices for test/future using a scaler fitted on raw train period.
    """
    df_all = generate_target(
        auto_update=False,
        persist_total_features_on_update=False,
    )
    feature_cols = split.feature_cols
    df_valid = df_all.dropna(subset=["Target_Class"]).copy()

    train_start, train_end = SPLIT_CONFIG["train"]
    raw_train = df_valid.loc[train_start:train_end, feature_cols].copy()
    scaler = StandardScaler()
    scaler.fit(raw_train)

    # Consistency check: rebuilt scaled test should match split.test features.
    raw_test = df_valid.loc[SPLIT_CONFIG["test"][0]:, feature_cols].reindex(split.test.index)
    rebuilt_scaled_test = scaler.transform(raw_test)
    expected_scaled_test = split.test[feature_cols].to_numpy()
    if not np.allclose(rebuilt_scaled_test, expected_scaled_test, atol=1e-8, rtol=1e-6):
        raise RuntimeError(
            "Scaled feature mismatch between rebuilt scaler output and split.test."
        )

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
    auto_update: bool = False,
    persist_total_features_on_update: bool = False,
    save_json: bool = True,
) -> dict[str, Any]:
    """
    Run equal-weight ensemble on test set and future (unrealized target) window.
    """
    if auto_update or persist_total_features_on_update:
        print(
            "[INFO] Ensemble module enforces no-update policy. "
            "auto_update/persist flags are ignored."
        )

    split = split_dataset(
        auto_update=False,
        persist_total_features_on_update=False,
    )
    expected_len = len(split.test)

    xgb_models, p_xgb_raw, _ = run_xgb_pipeline(
        auto_optimize=False,
        return_metrics=False,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    svm_models, p_svm_raw, _ = run_svm_pipeline(
        return_metrics=False,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    rf_models, p_rf_raw, _ = run_rf_pipeline(
        auto_optimize=False,
        return_metrics=False,
        save_plot=False,
        compute_importance=False,
        split_override=split,
    )
    logreg_models, p_logreg_raw, _ = run_logreg_pipeline(
        auto_optimize=False,
        return_metrics=False,
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
    alpha_diff_test = (
        split.test["Target_AAPL_3M"] - split.test["Target_SP500_3M"]
    ).to_numpy()

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

    x_future_scaled, df_future_raw = _get_scaled_future_features(split)
    if x_future_scaled.empty:
        df_future_probs = pd.DataFrame(
            columns=["p_xgb", "p_svm", "p_rf", "p_logreg", "p_ens"]
        )
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
        "weights": {"xgb": 0.25, "svm": 0.25, "rf": 0.25, "logreg": 0.25},
        "n_test_samples": int(len(df_test_probs)),
        "test_start": split.test.index.min().strftime("%Y-%m-%d"),
        "test_end": split.test.index.max().strftime("%Y-%m-%d"),
        "ic_full": _to_json_number(ic_full),
        "ic_pvalue": _to_json_number(ic_pvalue),
        "ic_first_half": _to_json_number(ic_first),
        "ic_second_half": _to_json_number(ic_second),
        "accuracy_ref": _to_json_number(accuracy_ref),
        "precision_ref": _to_json_number(precision_ref),
        "p_ens_test_stats": _distribution_summary(df_test_probs["p_ens"].to_numpy()),
        "future_window": {
            "n_samples": int(len(df_future_probs)),
            "start": (
                df_future_raw.index.min().strftime("%Y-%m-%d")
                if not df_future_raw.empty
                else None
            ),
            "end": (
                df_future_raw.index.max().strftime("%Y-%m-%d")
                if not df_future_raw.empty
                else None
            ),
        },
        "p_ens_future_stats": (
            _distribution_summary(df_future_probs["p_ens"].to_numpy())
            if not df_future_probs.empty
            else {"min": None, "mean": None, "max": None}
        ),
        "latest_future_prediction": {
            "trade_date": latest_future_trade_date,
            "p_ens": _to_json_number(latest_future_p_ens),
        },
        "p_ens_column_location": {
            "test": "df_test_probs['p_ens'] (runtime)",
            "future": "df_future_probs['p_ens'] (runtime)",
        },
    }

    if save_json:
        ensure_artifact_dirs()
        ENSEMBLE_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ENSEMBLE_RESULT_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print("\nEqual-weight ensemble complete")
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
        print(
            "  latest future p_ens   : "
            f"{latest_future_trade_date} -> {latest_future_p_ens:.6f}"
        )
    else:
        print("  Future samples        : 0 (no unrealized target rows)")
    print(f"  p_ens column location : {payload['p_ens_column_location']}")
    print(f"  JSON saved            : {ENSEMBLE_RESULT_PATH}")

    return payload


if __name__ == "__main__":
    run_equal_weight_ensemble()
