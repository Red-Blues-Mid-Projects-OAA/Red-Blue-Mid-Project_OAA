"""
모델 공통 permutation importance 유틸.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ic, _ = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0
    return float(ic)


def compute_permutation_importance_ic(
    x_test: pd.DataFrame,
    y_test,
    alpha_diff,
    predict_proba_fn: Callable[[pd.DataFrame], np.ndarray],
    threshold: float,
    n_repeats: int,
    seed: int,
) -> tuple[float, float, pd.DataFrame]:
    """
    Permutation 기반 ΔIC 중요도를 계산합니다.

    ΔIC = IC_baseline - IC_permuted(feature)
    """
    if n_repeats <= 0:
        raise ValueError(f"n_repeats must be positive: {n_repeats}")

    y_arr = np.asarray(y_test).astype(int)
    alpha_arr = np.asarray(alpha_diff).astype(float)

    baseline_proba = np.asarray(predict_proba_fn(x_test), dtype=float)
    if baseline_proba.ndim != 1 or baseline_proba.shape[0] != len(x_test):
        raise ValueError(
            "predict_proba_fn must return a 1D array with the same length as x_test"
        )

    baseline_pred = (baseline_proba >= threshold).astype(int)
    baseline_acc = float(accuracy_score(y_arr, baseline_pred))
    baseline_ic = _safe_spearman(baseline_proba, alpha_arr)

    norm_denom = max(abs(baseline_ic), 1e-8)
    rng = np.random.default_rng(seed)

    records = []
    for col in x_test.columns:
        ic_drops = np.zeros(n_repeats, dtype=float)
        acc_drops = np.zeros(n_repeats, dtype=float)
        base_values = x_test[col].to_numpy(copy=True)

        for i in range(n_repeats):
            shuffled = base_values.copy()
            rng.shuffle(shuffled)

            x_perm = x_test.copy()
            x_perm[col] = shuffled

            perm_proba = np.asarray(predict_proba_fn(x_perm), dtype=float)
            perm_pred = (perm_proba >= threshold).astype(int)
            perm_acc = float(accuracy_score(y_arr, perm_pred))
            perm_ic = _safe_spearman(perm_proba, alpha_arr)

            ic_drops[i] = baseline_ic - perm_ic
            acc_drops[i] = baseline_acc - perm_acc

        ic_drop_mean = float(ic_drops.mean())
        ic_drop_std = float(ic_drops.std(ddof=0))
        acc_drop_mean = float(acc_drops.mean())
        acc_drop_std = float(acc_drops.std(ddof=0))

        records.append(
            {
                "feature": col,
                "ic_drop_mean": ic_drop_mean,
                "ic_drop_std": ic_drop_std,
                "ic_drop_norm_mean": float(ic_drop_mean / norm_denom),
                "ic_drop_norm_std": float(ic_drop_std / norm_denom),
                "acc_drop_mean": acc_drop_mean,
                "acc_drop_std": acc_drop_std,
            }
        )

    importance_df = pd.DataFrame(records).sort_values(
        by="ic_drop_mean", ascending=False, ignore_index=True
    )
    return float(baseline_ic), float(baseline_acc), importance_df
