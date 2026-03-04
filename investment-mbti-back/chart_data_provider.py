"""
차트 데이터 전용 모듈 (DB 기반, yfinance 미사용).

DB의 LOG_RETURNS / SP500_DATA 테이블을 활용하여
포트폴리오 누적수익률 차트, S&P 500 벤치마크, Monte Carlo placeholder 데이터를 생성합니다.

모든 수익률은 단순수익률(Simple Return)로 변환하여 반환합니다.
"""

import sys
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 프로젝트 루트 경로 설정
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DB import StockDBManager


# ──────────────────────────────────────────────────────────────────
# 글로벌 캐시 (FastAPI startup 시 1회 로드)
# ──────────────────────────────────────────────────────────────────
_chart_cache = {
    "log_ret_df": pd.DataFrame(),
    "sp500_df": pd.DataFrame(),
}


def load_chart_cache():
    """
    서버 시작 시 1회 호출되어 차트용 데이터를 메모리에 캐싱합니다.
    """
    print("[Cache] 차트/수익률 데이터 캐싱 시작...")
    db = StockDBManager()
    try:
        db.connect()
        _chart_cache["log_ret_df"] = db.fetch_log_returns()
        _chart_cache["sp500_df"] = db.fetch_sp500_data()

        # 날짜 인덱스 정규화 (전처리 미리 수행)
        if not _chart_cache["log_ret_df"].empty:
            _chart_cache["log_ret_df"].index = pd.to_datetime(_chart_cache["log_ret_df"].index)
        if not _chart_cache["sp500_df"].empty:
            _chart_cache["sp500_df"].index = pd.to_datetime(_chart_cache["sp500_df"].index)

        print(f"  [SYNC][CHART] LOG_RETURNS: {len(_chart_cache['log_ret_df'])}일분")
        print(f"  [SYNC][CHART] SP500_DATA: {len(_chart_cache['sp500_df'])}일분")
        print("[Cache] 차트용 데이터 캐싱 완료!")
    except Exception as e:
        print(f"  [SYNC][CHART][ERROR] {e}")
    finally:
        db.close()



# ──────────────────────────────────────────────────────────────────
# 유틸리티: 로그수익률 → 단순수익률 변환
# ──────────────────────────────────────────────────────────────────
def _log_to_simple(log_return):
    """로그수익률을 단순수익률로 변환: simple = e^r - 1"""
    return math.exp(log_return) - 1.0


def _log_to_simple_array(log_returns):
    """numpy 배열의 로그수익률을 단순수익률로 일괄 변환"""
    return np.exp(log_returns) - 1.0


# ──────────────────────────────────────────────────────────────────
# 과거 기간별 누적수익률 (1M / 3M / 6M / 12M)
# ──────────────────────────────────────────────────────────────────
def get_historical_returns(tickers: list[str]) -> dict:
    """
    캐시된 LOG_RETURNS 데이터에서 과거 로그수익률을 로드하여 계산합니다.
    (DB 접속 없이 메모리 데이터 사용)
    """
    log_ret_df = _chart_cache.get("log_ret_df")
    
    if log_ret_df is None or log_ret_df.empty:
        return {t: {"1M": 0, "3M": 0, "6M": 0, "12M": 0} for t in tickers}

    # 기간별 거래일 수 (근사값)
    periods = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

    result = {}
    for t in tickers:
        t_upper = t.strip().upper()
        if t_upper not in log_ret_df.columns:
            result[t_upper] = {"1M": 0, "3M": 0, "6M": 0, "12M": 0}
            continue

        col = log_ret_df[t_upper].dropna()
        returns = {}
        for label, days in periods.items():
            if len(col) >= days:
                cum_log = col.iloc[-days:].sum()
                returns[label] = round(_log_to_simple(cum_log) * 100, 2)
            else:
                cum_log = col.sum()
                returns[label] = round(_log_to_simple(cum_log) * 100, 2)

        result[t_upper] = returns

    return result


# ──────────────────────────────────────────────────────────────────
# 포트폴리오 vs S&P 500 누적수익률 시계열 (과거 1년)
# ──────────────────────────────────────────────────────────────────
def get_cumulative_return_chart(tickers: list[str], weights: list[float]) -> dict:
    """
    캐시된 데이터를 활용하여 과거 1년 포트폴리오 누적수익률 시계열을 생성합니다.
    """
    log_ret_df = _chart_cache.get("log_ret_df")
    sp500_df = _chart_cache.get("sp500_df")

    warnings = []

    if log_ret_df is None or log_ret_df.empty or sp500_df is None or sp500_df.empty:
        return {"dates": [], "portfolio": [], "sp500": [], "warnings": ["캐시 데이터를 로드할 수 없습니다."]}

    # 최근 252거래일(약 1년) 범위 설정 (캐시는 이미 datetime index)

    # 최근 252거래일(약 1년) 범위 설정
    latest_date = log_ret_df.index.max()
    lookback = 252
    if len(log_ret_df) > lookback:
        log_ret_df = log_ret_df.iloc[-lookback:]

    # 포트폴리오 가중 일별 로그수익률 계산
    valid_tickers = []
    valid_weights = []
    for t, w in zip(tickers, weights):
        t_upper = t.strip().upper()
        if t_upper in log_ret_df.columns:
            valid_tickers.append(t_upper)
            valid_weights.append(w)
        else:
            warnings.append(f"{t_upper}: 과거 데이터 없음, 평균값으로 대체됨")

    if not valid_tickers:
        return {"dates": [], "portfolio": [], "sp500": [], "warnings": warnings}

    # 비중 재정규화 (누락 종목 존재 시)
    w_arr = np.array(valid_weights)
    w_arr = w_arr / w_arr.sum()

    # 포트폴리오 일별 로그수익률
    port_log_ret = log_ret_df[valid_tickers].fillna(0).values @ w_arr

    # 누적 로그수익률 → 단순수익률 변환
    cum_log_port = np.cumsum(port_log_ret)
    cum_simple_port = _log_to_simple_array(cum_log_port) * 100  # 퍼센트 단위

    # S&P 500 동일 기간 추출
    sp500_aligned = sp500_df.reindex(log_ret_df.index).fillna(0)
    cum_log_sp = np.cumsum(sp500_aligned["LOG_RETURN"].values)
    cum_simple_sp = _log_to_simple_array(cum_log_sp) * 100

    # 날짜를 문자열로 변환
    dates = [d.strftime("%Y-%m-%d") for d in log_ret_df.index]

    return {
        "dates": dates,
        "portfolio": [round(float(v), 2) for v in cum_simple_port],
        "sp500": [round(float(v), 2) for v in cum_simple_sp],
        "warnings": warnings,
    }


# ──────────────────────────────────────────────────────────────────
# Monte Carlo 시뮬레이션 Placeholder (향후 실제 로직으로 대체)
# ──────────────────────────────────────────────────────────────────
def get_forecast_placeholder(expected_return_log: float, volatility_60d: float, n_paths: int = 50) -> dict:
    """
    향후 3개월(약 63거래일) 예측 선 + Monte Carlo 시뮬레이션 더미 데이터를 생성합니다.
    현재는 placeholder이며, 추후 실제 시뮬레이션 로직으로 대체될 예정입니다.

    Args:
        expected_return_log: 3개월 예측 로그수익률 (소수)
        volatility_60d: 60일 기준 포트폴리오 표준편차
        n_paths: 시뮬레이션 경로 수

    Returns:
        dict: {
            "forecast_days": [0, 1, ..., 60],
            "expected_line": [단순수익률 누적 리스트],
            "mc_paths": [[path1], [path2], ...],
            "final_distribution": {"mean": float, "std": float, "percentile_5": float, "percentile_95": float}
        }
    """
    horizon = 60  # 약 3개월 거래일
    dt = 1.0 / 252  # 일별 시간 스텝

    # 일별 드리프트 및 변동성 (간단한 GBM 가정)
    daily_drift = expected_return_log / horizon
    daily_vol = volatility_60d * np.sqrt(dt / (60 / 252))  # 60일 스케일 → 1일 스케일

    # 기대 경로 (결정론적)
    expected_cum_log = np.array([daily_drift * i for i in range(horizon + 1)])
    expected_line = (_log_to_simple_array(expected_cum_log) * 100).tolist()

    # Monte Carlo 시뮬레이션 경로 (placeholder)
    np.random.seed(42)
    mc_paths = []
    final_values = []
    for _ in range(n_paths):
        shocks = np.random.normal(daily_drift, daily_vol, horizon)
        path_log = np.concatenate([[0], np.cumsum(shocks)])
        path_simple = (_log_to_simple_array(path_log) * 100).tolist()
        mc_paths.append([round(v, 2) for v in path_simple])
        final_values.append(path_simple[-1])

    final_values = np.array(final_values)

    return {
        "forecast_days": list(range(horizon + 1)),
        "expected_line": [round(v, 2) for v in expected_line],
        "mc_paths": mc_paths,
        "final_distribution": {
            "mean": round(float(np.mean(final_values)), 2),
            "std": round(float(np.std(final_values)), 2),
            "percentile_5": round(float(np.percentile(final_values, 5)), 2),
            "percentile_95": round(float(np.percentile(final_values, 95)), 2),
        },
    }


# ──────────────────────────────────────────────────────────────────
# 실데이터 기반 Monte Carlo 시뮬레이션
# ──────────────────────────────────────────────────────────────────
def get_real_forecast(
    tickers: list[str],
    weights: list[float],
    expected_return_3m_log: float,
    n_paths: int = 300,
) -> dict:
    """
    실제 포트폴리오 과거 데이터 기반 Monte Carlo 시뮬레이션.

    1. 캐시된 일별 로그수익률에서 포트폴리오 가중 일별 수익률을 계산
    2. 실제 daily drift 와 volatility를 추출
    3. GBM: dS = μ·dt + σ·dW 으로 60일(3개월) 시뮬레이션
    4. 기대 경로: adjusted expected return 기반 결정론적 라인
    5. VaR 5%: 시뮬레이션 최종값의 5th percentile

    Args:
        tickers: 포트폴리오 티커 리스트
        weights: 각 종목 비중 (합=1)
        expected_return_3m_log: 3개월 조정 기대 로그수익률 (소수)
        n_paths: 시뮬레이션 경로 수

    Returns:
        dict: forecast_days, expected_line, mc_paths, final_distribution
    """
    log_ret_df = _chart_cache.get("log_ret_df")
    horizon = 60  # 약 3개월 거래일

    # ── 포트폴리오 일별 로그수익률 계산 ──
    valid_tickers = []
    valid_weights = []
    if log_ret_df is not None and not log_ret_df.empty:
        for t, w in zip(tickers, weights):
            t_upper = t.strip().upper()
            if t_upper in log_ret_df.columns:
                valid_tickers.append(t_upper)
                valid_weights.append(w)

    if valid_tickers:
        w_arr = np.array(valid_weights)
        w_arr = w_arr / w_arr.sum()
        port_daily_log = log_ret_df[valid_tickers].fillna(0).values @ w_arr

        # 최근 252일 데이터로 drift/vol 추정
        recent = port_daily_log[-252:] if len(port_daily_log) > 252 else port_daily_log
        daily_vol = float(np.std(recent))
    else:
        daily_vol = 0.01  # fallback

    # daily drift (기대 경로는 MC 시뮬레이션 후 일별 평균으로 계산)
    daily_drift_expected = expected_return_3m_log / horizon

    # ── Monte Carlo GBM 시뮬레이션 ──
    rng = np.random.default_rng(seed=42)
    all_paths_matrix = np.zeros((n_paths, horizon + 1))  # (300, 61) matrix
    mc_paths = []
    final_values = []
    for path_idx in range(n_paths):
        # GBM: log(S_t/S_0) = (μ - σ²/2)·t + σ·W_t
        shocks = rng.normal(
            daily_drift_expected - 0.5 * daily_vol ** 2,
            daily_vol,
            horizon,
        )
        path_log = np.concatenate([[0], np.cumsum(shocks)])
        path_simple = _log_to_simple_array(path_log) * 100
        all_paths_matrix[path_idx] = path_simple
        mc_paths.append([round(v, 2) for v in path_simple.tolist()])
        final_values.append(float(path_simple[-1]))

    final_values = np.array(final_values)

    # ── 기대 경로: Brownian Bridge ──
    # 시작점(0%)과 끝점(예측 3개월 수익률)을 고정하고
    # 중간 과정은 자연스럽게 변동하는 조건부 브라운 경로 생성
    target_simple = float((_log_to_simple_array(np.array([expected_return_3m_log]))[0]) * 100)
    bridge_rng = np.random.default_rng(seed=123)
    W = np.concatenate([[0], np.cumsum(bridge_rng.normal(0, daily_vol, horizon))])
    W_T = W[-1]
    T = horizon
    bridge_log = np.array([
        W[t] - (t / T) * W_T + (t / T) * expected_return_3m_log
        for t in range(T + 1)
    ])
    bridge_simple = _log_to_simple_array(bridge_log) * 100
    expected_line = [round(float(v), 2) for v in bridge_simple]

    # ── VaR 5% (MC 기반) ──
    var_5_mc = float(np.percentile(final_values, 5))

    # ── 보수적 VaR 5% (drift=0, 순수 위험 기반) ──
    # Parametric VaR: VaR_5% = -1.645 × σ_portfolio × √T
    portfolio_vol_3m = daily_vol * np.sqrt(horizon)
    var_5_conservative_log = -1.645 * portfolio_vol_3m
    var_5_conservative = float((_log_to_simple_array(np.array([var_5_conservative_log]))[0]) * 100)

    # ── 히스토그램 빈 데이터 (프론트 vertical AreaChart 용) ──
    counts, bin_edges = np.histogram(final_values, bins=30)
    distribution_bins = []
    for i in range(len(counts)):
        bin_center = float((bin_edges[i] + bin_edges[i + 1]) / 2)
        distribution_bins.append({
            "returnBin": round(bin_center, 2),
            "frequency": int(counts[i]),
        })

    return {
        "forecast_days": list(range(horizon + 1)),
        "expected_line": expected_line,
        "mc_paths": mc_paths,
        "final_distribution": {
            "mean": round(float(np.mean(final_values)), 2),
            "std": round(float(np.std(final_values)), 2),
            "percentile_5": round(var_5_mc, 2),
            "percentile_95": round(float(np.percentile(final_values, 95)), 2),
        },
        "distribution_bins": distribution_bins,
        "var_5_value": round(var_5_mc, 2),
        "var_5_conservative": round(var_5_conservative, 2),
    }

