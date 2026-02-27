"""
3개월 기대 초과수익률(alpha) 매핑 모듈 (티커 파라미터화).

Execution:
  python3 -m Classification.mapping.mapping
"""

from __future__ import annotations

import math
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

from common import pd
from Classification.model_config import (
    get_ensemble_result_path,
    get_mapping_result_path,
    load_json_artifact_only,
    save_json_artifact_only,
)
from DB import StockDBManager

LOOKBACK_DAYS = 252
HORIZON_DAYS = 60
PHI = 1.0


def _as_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"필수 키 누락 또는 None: {key}")
    out = float(value)
    if np.isnan(out) or np.isinf(out):
        raise ValueError(f"유효하지 않은 수치값: {key}={value}")
    return out


def _coerce_series(series_like, name: str) -> pd.Series:
    if series_like is None:
        raise ValueError(f"{name} is None")
    if isinstance(series_like, pd.DataFrame):
        if "LOG_RETURN" in series_like.columns:
            s = series_like["LOG_RETURN"].copy()
        elif series_like.shape[1] == 1:
            s = series_like.iloc[:, 0].copy()
        else:
            raise ValueError(f"{name} DataFrame은 단일 컬럼이어야 합니다.")
    elif isinstance(series_like, pd.Series):
        s = series_like.copy()
    else:
        raise TypeError(f"{name}는 pd.Series 또는 단일컬럼 DataFrame이어야 합니다.")

    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s, errors="coerce").sort_index()


def _build_active_returns(
    ticker: str,
    benchmark: str,
    ticker_logret_series: pd.Series,
    sp500_logret_series: pd.Series,
) -> pd.Series:
    ticker_lr = _coerce_series(ticker_logret_series, "ticker_logret_series").rename(ticker)
    benchmark_lr = _coerce_series(sp500_logret_series, "sp500_logret_series").rename(benchmark)

    merged = pd.concat([ticker_lr, benchmark_lr], axis=1, join="inner").dropna().sort_index()
    if merged.empty:
        raise RuntimeError(f"{ticker}/{benchmark} 로그수익률 교집합이 비어 있습니다.")

    active = merged[ticker] - merged[benchmark]
    if active.empty:
        raise RuntimeError("active return 시계열이 비어 있습니다.")
    return active


def _compute_mapping_result(
    *,
    ticker: str,
    benchmark: str,
    p_latest: float,
    ic_full: float,
    latest_trade_date: str | None,
    active_ret: pd.Series,
) -> dict[str, Any]:
    warnings: list[str] = []

    # Step 4: Shrinkage Volatility (Tracking Error)
    # 1) OLS Sigma (장기/2024-01-01 이후)
    ols_series = active_ret.loc["2024-01-01":]
    if len(ols_series) < 2:
        sigma_ols = 0.0
        warnings.append("OLS Sigma 계산 표본 부족 (2024-01-01 이후).")
    else:
        sigma_ols = float(np.std(ols_series.to_numpy(), ddof=1))

    # 2) EWMA Sigma (단기)
    if len(active_ret) < 2:
        sigma_ewma = 0.0
        warnings.append("EWMA Sigma 계산 표본 부족.")
    else:
        sigma_ewma_series = active_ret.ewm(alpha=0.06, adjust=False).std()
        sigma_ewma = float(sigma_ewma_series.iloc[-1])

    # 3) Shrinkage 결합 (70% OLS + 30% EWMA)
    sigma_active = (0.7 * sigma_ols) + (0.3 * sigma_ewma)

    if sigma_active == 0:
        te_annual = 0.0
        warnings.append("sigma_active=0 처리 (입력 데이터 확인 필요).")
    else:
        te_annual = sigma_active * math.sqrt(252.0)

    te_3m = te_annual * math.sqrt(HORIZON_DAYS / 252.0)
    signal_raw = (p_latest - 0.5) / 0.5
    signal_clipped = float(np.clip(signal_raw, -0.8, 0.8))
    e_alpha_3m_log = signal_clipped * ic_full * te_3m * PHI
    e_alpha_3m_simple = math.exp(e_alpha_3m_log) - 1.0

    if ic_full <= 0:
        warnings.append("예측력 부재/역방향 가능성: IC_full <= 0")
    if abs(signal_raw) > 0.95:
        warnings.append("확률 과확신 가능성: |signal_raw| > 0.95")
    if e_alpha_3m_simple > 0.15 or e_alpha_3m_simple < -0.15:
        warnings.append("매핑 과대/과소 가능성: |E_alpha_3M_simple| > 0.15")
    if te_annual == 0:
        warnings.append("TE_annual == 0: 변동성 입력 점검 필요")

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "p_latest": float(p_latest),
        "signal_raw": float(signal_raw),
        "signal_clipped": float(signal_clipped),
        "IC_full": float(ic_full),
        "sigma_ols": float(sigma_ols),
        "sigma_ewma": float(sigma_ewma),
        "sigma_final": float(sigma_active),
        "TE_annual": float(te_annual),
        "TE_3M": float(te_3m),
        "phi": float(PHI),
        "E_alpha_3M_log": float(e_alpha_3m_log),
        "E_alpha_3M_simple": float(e_alpha_3m_simple),
        "warnings": warnings,
        "lookback_days_requested": int(LOOKBACK_DAYS),
        "lookback_days_used": int(len(active_ret)),
        "horizon_days": int(HORIZON_DAYS),
        "latest_trade_date": latest_trade_date,
    }


def _print_mapping_result(result: dict[str, Any], result_path: Path) -> None:
    warnings = result.get("warnings", [])
    print("\n[Mapping Result]")
    print(f"  ticker             : {result['ticker']}")
    print(f"  latest_trade_date  : {result['latest_trade_date']}")
    print(f"  p_latest           : {result['p_latest']:.6f}")
    print(f"  signal(raw/clipped): {result['signal_raw']:.6f} / {result['signal_clipped']:.6f}")
    print(f"  IC_full            : {result['IC_full']:.6f}")
    print(f"  Sigma (OLS/EWMA)   : {result['sigma_ols']:.6f} / {result['sigma_ewma']:.6f}")
    print(f"  Sigma Final        : {result['sigma_final']:.6f}")
    print(f"  TE_annual          : {result['TE_annual']:.6f}")
    print(f"  TE_3M              : {result['TE_3M']:.6f}")
    print(f"  E_alpha_3M_log     : {result['E_alpha_3M_log']:.6f}")
    print(f"  E_alpha_3M_simple  : {result['E_alpha_3M_simple']:.6f}")
    print(f"  warnings           : {len(warnings)}")
    if warnings:
        for idx, msg in enumerate(warnings, 1):
            print(f"    {idx}. {msg}")
    print(f"  saved              : {result_path}")
    print(
        "\n본 E[alpha_3M]는 확률신호와 IC를 결합한 기대 초과수익 추정치이며, "
        "포트폴리오 최적화 입력(기대효용, VaR 시뮬레이션)으로 사용한다."
    )


def run_mapping_from_inputs(
    ticker: str,
    benchmark: str,
    p_latest: float,
    ic_full: float,
    latest_trade_date: str | None,
    ticker_logret_series: pd.Series,
    sp500_logret_series: pd.Series,
    save_json: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")

    p_latest_f = float(p_latest)
    ic_full_f = float(ic_full)
    if np.isnan(p_latest_f) or np.isinf(p_latest_f):
        raise ValueError(f"유효하지 않은 p_latest: {p_latest}")
    if np.isnan(ic_full_f) or np.isinf(ic_full_f):
        raise ValueError(f"유효하지 않은 ic_full: {ic_full}")

    active_ret = _build_active_returns(
        ticker=ticker,
        benchmark=benchmark,
        ticker_logret_series=ticker_logret_series,
        sp500_logret_series=sp500_logret_series,
    )

    result = _compute_mapping_result(
        ticker=ticker,
        benchmark=benchmark,
        p_latest=p_latest_f,
        ic_full=ic_full_f,
        latest_trade_date=latest_trade_date,
        active_ret=active_ret,
    )

    result_path = get_mapping_result_path(ticker)
    if save_json:
        save_json_artifact_only(result, result_path)

    if verbose:
        _print_mapping_result(result, result_path)

    return result


def _load_inputs_from_ensemble(ticker: str) -> tuple[float, float, str | None]:
    ensemble_path = get_ensemble_result_path(ticker)
    data, used_path = load_json_artifact_only(ensemble_path)
    if data is None:
        raise FileNotFoundError(f"ensemble 결과 파일을 찾을 수 없습니다: {ensemble_path}")

    p_block = data.get("latest_future_prediction", {})
    p_latest = _as_float(p_block, "p_ens")
    latest_trade_date = p_block.get("trade_date")
    ic_full = _as_float(data, "ic_full")

    print(f"[INPUT] ensemble source: {used_path}")
    print(f"[INPUT] p_latest={p_latest:.6f}, IC_full={ic_full:.6f}, date={latest_trade_date}")
    return p_latest, ic_full, latest_trade_date


def _load_logret_series_from_db(ticker: str, benchmark: str) -> tuple[pd.Series, pd.Series]:
    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")

    db = StockDBManager()
    db.connect()
    try:
        ticker_lr = db.fetch_log_returns_by_ticker(ticker)
        benchmark_lr = db.fetch_sp500_log_returns()
    finally:
        db.close()

    if ticker_lr is None or len(ticker_lr) == 0:
        raise RuntimeError(f"LOG_RETURNS에 {ticker} 로그수익률 시계열이 없습니다.")
    if benchmark_lr is None or len(benchmark_lr) == 0:
        raise RuntimeError("SP500_DATA(LOG_RETURN) 데이터가 비어 있습니다.")

    return ticker_lr, benchmark_lr


def run_mapping(ticker: str = "AAPL", benchmark: str = "SP500") -> dict[str, Any]:
    """
    하위호환 엔트리포인트.
    ensemble.json + DB 로그수익률을 직접 로드해 매핑을 수행합니다.
    """
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()

    p_latest, ic_full, latest_trade_date = _load_inputs_from_ensemble(ticker)
    ticker_lr, benchmark_lr = _load_logret_series_from_db(ticker, benchmark)

    return run_mapping_from_inputs(
        ticker=ticker,
        benchmark=benchmark,
        p_latest=p_latest,
        ic_full=ic_full,
        latest_trade_date=latest_trade_date,
        ticker_logret_series=ticker_lr,
        sp500_logret_series=benchmark_lr,
        save_json=True,
        verbose=True,
    )


if __name__ == "__main__":
    for ticker in ["AAPL", "TSLA"]:
        try:
            run_mapping(ticker)
        except Exception as e:
            print(f"Error for {ticker}: {e}")
