"""
이 파일은 모델이 어떤 피처를 중요하게 보는지 정리해 해석하기 쉽게 만듭니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    스피어만 상관계수를 NaN 안전하게 계산합니다.

    상수열/결측 상황으로 상관계수가 NaN이 되면 0.0으로 보정해
    후속 중요도 계산이 중단되지 않도록 처리합니다.
    """
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
    Permutation 기반 중요도를 계산합니다.

    Args:
        x_test (pd.DataFrame): 테스트 피처 데이터프레임입니다.
        y_test: 테스트 정답 클래스(0/1) 시퀀스입니다.
        alpha_diff: 알파 차이(연속 타깃) 시퀀스입니다.
        predict_proba_fn (Callable): x_test 형태 입력을 받아
            양성 클래스 확률(1차원 배열)을 반환하는 함수입니다.
        threshold (float): 확률을 클래스(0/1)로 이진화할 임계값입니다.
        n_repeats (int): 피처별 셔플 반복 횟수입니다.
        seed (int): 난수 시드입니다.

    Returns:
        tuple[float, float, pd.DataFrame]:
            - baseline_ic: 원본 예측의 IC
            - baseline_acc: 원본 예측의 Accuracy
            - importance_df: 피처별 ΔIC/ΔAccuracy 통계 테이블

    Raises:
        ValueError: n_repeats가 0 이하이거나,
            predict_proba_fn 출력 형태가 입력 길이와 맞지 않을 때 발생합니다.

    Notes:
        ΔIC = IC_baseline - IC_permuted(feature)
        값이 클수록 해당 피처를 섞었을 때 신호가 더 크게 훼손되어
        모델 의존도가 높은 피처로 해석합니다.
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
