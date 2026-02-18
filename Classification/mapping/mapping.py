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
from Classification.model_config import get_ensemble_result_path, get_mapping_result_path, load_json_artifact_only, save_json_artifact_only
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


def _load_active_returns(ticker: str, benchmark: str) -> pd.Series:
    db = StockDBManager()
    db.connect()
    try:
        log_ret = db.fetch_log_returns()
        sp500 = db.fetch_sp500_data()
    finally:
        db.close()

    if log_ret.empty:
        raise RuntimeError("LOG_RETURNS 데이터가 비어 있습니다.")
    if ticker not in log_ret.columns:
        raise RuntimeError(f"LOG_RETURNS에 {ticker} 컬럼이 없습니다.")
    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")
    if sp500.empty or "LOG_RETURN" not in sp500.columns:
        raise RuntimeError("SP500_DATA(LOG_RETURN) 데이터가 비어 있습니다.")

    ticker_lr = pd.to_numeric(log_ret[ticker], errors="coerce")
    bench_lr = pd.to_numeric(sp500["LOG_RETURN"], errors="coerce")
    ticker_lr.index = pd.to_datetime(ticker_lr.index)
    bench_lr.index = pd.to_datetime(bench_lr.index)

    merged = pd.concat(
        [ticker_lr.rename(ticker), bench_lr.rename(benchmark)],
        axis=1,
        join="inner",
    ).dropna().sort_index()
    if merged.empty:
        raise RuntimeError(f"{ticker}/{benchmark} 로그수익률 교집합이 비어 있습니다.")

    active = merged[ticker] - merged[benchmark]
    if active.empty:
        raise RuntimeError("active return 시계열이 비어 있습니다.")
    return active


def run_mapping(ticker: str = "AAPL", benchmark: str = "SP500") -> dict[str, Any]:
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    p_latest, ic_full, latest_trade_date = _load_inputs_from_ensemble(ticker)
    active_ret = _load_active_returns(ticker, benchmark)

    warnings: list[str] = []
    lookback_series = active_ret.tail(LOOKBACK_DAYS)
    lookback_used = int(len(lookback_series))
    if lookback_used < LOOKBACK_DAYS:
        warnings.append(f"최근 {LOOKBACK_DAYS}거래일 데이터 부족: 가용 {lookback_used}거래일로 TE 계산.")

    if lookback_used < 2:
        te_annual = 0.0
        warnings.append("TE 계산 표본 부족으로 TE_annual=0 처리.")
    else:
        sigma_active = float(np.std(lookback_series.to_numpy(), ddof=1))
        if np.isnan(sigma_active) or np.isinf(sigma_active):
            te_annual = 0.0
            warnings.append("active return 표준편차 비정상값으로 TE_annual=0 처리.")
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

    result = {
        "ticker": ticker,
        "benchmark": benchmark,
        "p_latest": float(p_latest),
        "signal_raw": float(signal_raw),
        "signal_clipped": float(signal_clipped),
        "IC_full": float(ic_full),
        "TE_annual": float(te_annual),
        "TE_3M": float(te_3m),
        "phi": float(PHI),
        "E_alpha_3M_log": float(e_alpha_3m_log),
        "E_alpha_3M_simple": float(e_alpha_3m_simple),
        "warnings": warnings,
        "lookback_days_requested": int(LOOKBACK_DAYS),
        "lookback_days_used": int(lookback_used),
        "horizon_days": int(HORIZON_DAYS),
        "latest_trade_date": latest_trade_date,
    }

    result_path = get_mapping_result_path(ticker)
    save_json_artifact_only(result, result_path)

    print("\n[Mapping Result]")
    print(f"  ticker             : {ticker}")
    print(f"  latest_trade_date  : {result['latest_trade_date']}")
    print(f"  p_latest           : {result['p_latest']:.6f}")
    print(f"  signal(raw/clipped): {result['signal_raw']:.6f} / {result['signal_clipped']:.6f}")
    print(f"  IC_full            : {result['IC_full']:.6f}")
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
    return result


if __name__ == "__main__":
    run_mapping()
