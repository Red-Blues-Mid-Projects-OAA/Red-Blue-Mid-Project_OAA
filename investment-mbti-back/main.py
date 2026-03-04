"""
Investment MBTI Integration API — 실제 데이터 기반 FastAPI 서버.

설문 응답 + 손실 한도 → λ 산출 → 종목 추천 → MVO 최적화 → 결과 반환
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 프로젝트 루트 경로 (investment-mbti-back의 상위 폴더)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import math
from risk_profile import calculate_risk_profile
from real_data_provider import (
    load_cache,
    get_recommended_stocks,
    get_risk_level_portfolio_summary,
    get_mvo_inputs,
    get_all_300_stocks,
    calculate_portfolio_scores,
    TICKER_NAME_MAP,
    normalize_risk_score,
    _cache,
)
from portfolio_optimizer import optimize_portfolio
from chart_data_provider import (
    get_historical_returns,
    get_cumulative_return_chart,
    get_forecast_placeholder,
    get_real_forecast,
    load_chart_cache,
    get_last_updated_date,
)


# ──────────────────────────────────────────────────────────────────
# FastAPI 앱 생성 (서버 시작 시 데이터 캐싱)
# ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 DB/스냅샷 캐시 로드."""
    print("=" * 60)
    print("Investment MBTI API 서버 시작 — 데이터 캐싱 중...")
    print("=" * 60)
    load_cache()
    load_chart_cache()
    print("✅ 서버 준비 완료!")
    yield
    print("서버 종료.")


app = FastAPI(title="Investment MBTI Integration API", lifespan=lifespan)

# 프론트엔드 연동을 위한 CORS 미들웨어 적용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# 요청/응답 모델
# ──────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    """설문 분석 요청."""

    answers: List[str]  # ['A', 'B', 'B', ...] 형식 (12개)
    loss_limit_value: int  # 원 단위 금액 (예: 9500000)


class OptimizeRequest(BaseModel):
    """포트폴리오 최적화 요청 (사용자 종목 선택 후)."""

    selected_tickers: List[str]  # 사용자가 선택한 종목 리스트
    lambda_final: float  # λ 값


class OptimizeFinalRequest(BaseModel):
    """최종 포트폴리오 최적화 요청 (종목 선택 후)."""

    selected_tickers: List[str]
    lambda_final: float
    final_level: int


class PortfolioScoresRequest(BaseModel):
    """포트폴리오 스코어 산출 요청."""

    selected_tickers: List[str]  # 선택된 종목 티커 리스트


# ──────────────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
def analyze_portfolio(req: AnalyzeRequest):
    """
    1단계: 설문 분석 → λ 산출 → 종목 추천 → MVO 최적화 → 차트 데이터 생성.
    """
    # 1. MBTI 기반 리스크 프로필 산출
    risk_result = calculate_risk_profile(req.answers, req.loss_limit_value)
    lambda_final = risk_result["lambda_final"]
    persona = risk_result["persona"]
    desc = risk_result["persona_desc"]

    # 2. final_level 추출 (종목 추천 풀 선택 및 리스크 카테고리 매핑 등에서 사용)
    final_level = risk_result.get("final_level", 2)

    # 3. 리스크 타입별 종목 추천/비중은 DB snapshot을 직접 사용
    recommended_stocks = get_recommended_stocks(final_level, top_n=10)
    portfolio_summary = get_risk_level_portfolio_summary(final_level)
    if not recommended_stocks or portfolio_summary is None:
        raise HTTPException(
            status_code=503,
            detail="RISK_LEVEL_PORTFOLIO_SNAPSHOT 준비가 완료되지 않았습니다. snapshot 동기화를 먼저 실행하세요.",
        )

    # "전체 종목 보기" 모달을 위해 DB 조건에 맞는 전체 종목 풀도 가져옵니다.
    all_matching_stocks = get_recommended_stocks(final_level, top_n=500)

    # 4. 종목 목록 확보 + MVO 입력(차트/보조 계산용)
    tickers = [
        str(stock.get("ticker", "")).strip().upper()
        for stock in recommended_stocks
        if str(stock.get("ticker", "")).strip()
    ]
    if not tickers:
        raise HTTPException(status_code=503, detail="추천 종목 데이터가 비어 있습니다.")

    mu, cov_sub, valid_tickers = get_mvo_inputs(tickers)
    if len(valid_tickers) == 0:
        raise HTTPException(status_code=503, detail="추천 종목의 MVO 입력 데이터를 구성할 수 없습니다.")

    weights = np.array(optimize_portfolio(mu, cov_sub, lambda_final), dtype=float)
    ticker_weight_map_mvo = {
        ticker: round(float(weight) * 100.0, 2) for ticker, weight in zip(valid_tickers, weights)
    }

    # 5. snapshot 비중을 우선 사용하고, 누락 시 MVO 비중으로 채움
    for stock in recommended_stocks:
        ticker = str(stock.get("ticker", "")).strip().upper()
        try:
            existing_weight = float(stock.get("weight", 0.0))
        except (TypeError, ValueError):
            existing_weight = 0.0
        stock["weight"] = existing_weight if existing_weight > 0 else ticker_weight_map_mvo.get(ticker, 0.0)
        stock["risk_score"] = int(stock.get("risk_score", 0) or 0)
        if "expectedReturn3M_simple" not in stock:
            try:
                expected_log_pct = float(stock.get("expectedReturn3M", 0.0) or 0.0)
            except (TypeError, ValueError):
                expected_log_pct = 0.0
            stock["expectedReturn3M_simple"] = round((math.exp(expected_log_pct / 100.0) - 1.0) * 100.0, 2)

    # 6. 포트폴리오 메트릭은 snapshot 값을 우선 사용 (로그수익률 -> 단순수익률 명시 변환)
    summary_return_log_pct = float(portfolio_summary.get("expected_portfolio_return_3m", 0.0) or 0.0)
    portfolio_return_log = summary_return_log_pct / 100.0
    portfolio_return_simple = float(
        portfolio_summary.get(
            "expected_portfolio_return_3m_simple",
            round((math.exp(portfolio_return_log) - 1.0) * 100.0, 2),
        )
    )

    summary_sigma = portfolio_summary.get("portfolio_std_60d")
    if summary_sigma is None:
        portfolio_var = float(weights.T @ cov_sub @ weights)
        portfolio_volatility = round(float(np.sqrt(max(portfolio_var, 0.0))), 4)
    else:
        portfolio_volatility = round(float(summary_sigma), 4)

    # 7. 과거 기간별 수익률 (1M/3M/6M/12M, 단순수익률 %)
    historical_returns = get_historical_returns(tickers)
    for stock in recommended_stocks:
        ticker = str(stock.get("ticker", "")).strip().upper()
        stock["historical_returns"] = historical_returns.get(
            ticker, {"1M": 0, "3M": 0, "6M": 0, "12M": 0}
        )

    # 8. 누적수익률 차트 데이터 (포트폴리오 vs S&P 500)
    chart_weight_pct = np.array(
        [float(stock.get("weight", 0.0) or 0.0) for stock in recommended_stocks], dtype=float
    )
    if chart_weight_pct.sum() <= 0:
        chart_weight_pct = np.array([ticker_weight_map_mvo.get(ticker, 0.0) for ticker in tickers], dtype=float)
    if chart_weight_pct.sum() <= 0:
        chart_weight_pct = np.ones(len(tickers), dtype=float)
    chart_weight_list = (chart_weight_pct / chart_weight_pct.sum()).tolist()
    chart_data = get_cumulative_return_chart(tickers, chart_weight_list)

    # 9. Monte Carlo 예측 placeholder
    forecast_data = get_forecast_placeholder(portfolio_return_log, portfolio_volatility)

    # 10. 리스크 카테고리 매핑
    risk_categories = {
        4: "지수를 간신히 이기는 유형",
        3: "지수를 가볍게 이기는 유형",
        2: "지수를 거뜬히 이기는 유형",
        1: "지수를 무참히 이기는 유형",
    }

    # 11. 프론트엔드 호환 응답 구성
    features = [
        f"투자 MBTI 성향: {risk_result['mbti']}",
        f"위험 회피 계수(Lambda): {lambda_final:.2f}",
        f"손실 한도 선택: -{risk_result['loss_ratio_percent']}%",
    ]

    return {
        "status": "success",
        "data": {
            "persona": persona,
            "description": desc,
            "features": features,
            "mbti": risk_result["mbti"],
            "recommended_stocks": recommended_stocks,
            "all_matching_stocks": all_matching_stocks,
            "portfolio_analysis": {
                "expected_return_simple": round(portfolio_return_simple, 2),
                "expected_return_log": round(portfolio_return_log * 100.0, 2),
                "volatility_60d": portfolio_volatility,
                "risk_category": risk_categories.get(final_level, "균형 잡힌 성장형"),
                "var_5": forecast_data["final_distribution"]["percentile_5"],
            },
            "chart_data": chart_data,
            "forecast_data": forecast_data,
            "lambda_final": lambda_final,
            "final_level": final_level,
            "raw_answers": req.answers,
        },
    }


@app.post("/api/optimize")
def optimize_selected(req: OptimizeRequest):
    """
    2단계: 사용자가 체크박스로 종목을 변경한 후 재최적화 요청.
    """
    tickers = [t.strip().upper() for t in req.selected_tickers]
    mu, cov_sub, valid_tickers = get_mvo_inputs(tickers)

    if len(valid_tickers) == 0:
        raise HTTPException(status_code=400, detail="유효한 종목이 없습니다.")

    weights = optimize_portfolio(mu, cov_sub, req.lambda_final)

    # 종목별 비중 응답
    result_stocks = []
    for i, ticker in enumerate(valid_tickers):
        result_stocks.append(
            {
                "ticker": ticker,
                "weight": round(float(weights[i]) * 100, 2),
                "expectedReturn3M": round(float(mu[i]) * 100, 2),
            }
        )

    portfolio_return = float(np.dot(weights, mu)) * 100

    return {
        "status": "success",
        "data": {
            "optimized_stocks": result_stocks,
            "portfolio_expected_return": f"+{portfolio_return:.2f}%" if portfolio_return >= 0 else f"{portfolio_return:.2f}%",
            "weight_sum": round(float(np.sum(weights)) * 100, 2),
        },
    }


@app.get("/api/all-stocks")
def get_all_stocks():
    """전체 300개 종목의 return_score, risk_score, market_cap_rank를 반환합니다."""
    try:
        stocks = get_all_300_stocks()
        return {"status": "success", "data": stocks}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/last-updated")
def last_updated():
    """DB에 적재된 개별 종목 데이터의 가장 최근 날짜를 반환합니다."""
    date_str = get_last_updated_date()
    if not date_str:
        raise HTTPException(status_code=503, detail="날짜 정보를 가져올 수 없습니다.")
    return {"status": "success", "date": date_str}


@app.post("/api/portfolio-scores")
def get_portfolio_scores(req: PortfolioScoresRequest):
    """선택 종목의 동일 비중 포트폴리오 Return/Risk Score를 산출합니다."""
    try:
        scores = calculate_portfolio_scores(req.selected_tickers)
        return {"status": "success", "data": scores}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/optimize-final")
def optimize_final(req: OptimizeFinalRequest):
    """
    종목 선택 완료 후 최종 포트폴리오 최적화 + 차트 + Monte Carlo.
    DashboardResult가 필요로 하는 전체 데이터를 반환합니다.
    """
    tickers = [t.strip().upper() for t in req.selected_tickers]
    if not tickers:
        raise HTTPException(status_code=400, detail="종목이 선택되지 않았습니다.")

    # 1. MVO 입력 데이터
    mu, cov_sub, valid_tickers = get_mvo_inputs(tickers)
    if len(valid_tickers) == 0:
        raise HTTPException(status_code=400, detail="유효한 종목이 없습니다.")

    # 2. 포트폴리오 최적화
    weights = np.array(optimize_portfolio(mu, cov_sub, req.lambda_final), dtype=float)

    # 3. 최적화 결과 종목별 데이터 구성
    optimized_stocks = []
    for i, ticker in enumerate(valid_tickers):
        w_pct = round(float(weights[i]) * 100, 2)
        if w_pct < 0.1:  # 비중 0에 가까운 종목은 제외
            continue
        expected_log = float(mu[i])  # 이미 decimal (예: 0.152 = 15.2%)
        expected_simple = round((math.exp(expected_log) - 1.0) * 100.0, 2)
        optimized_stocks.append({
            "ticker": ticker,
            "name": TICKER_NAME_MAP.get(ticker, ticker),
            "weight": w_pct,
            "expectedReturn3M": expected_simple,
            "expectedReturn3M_simple": expected_simple,
        })

    # 3.5. risk_score 및 risk_rank 추가
    variance_map = _cache.get("variance_map") or {}
    all_sigmas = [v ** 0.5 for v in variance_map.values()] if variance_map else []
    sigma_min = min(all_sigmas) if all_sigmas else 0
    sigma_max = max(all_sigmas) if all_sigmas else 1
    
    all_stocks_data = get_all_300_stocks()
    rank_map = {s.get("ticker"): s.get("risk_rank", 300) for s in all_stocks_data}

    # 과거 기간별 수익률 (chart_cache 기반 직접 계산)
    active_tickers_all = [s["ticker"] for s in optimized_stocks]
    historical_returns = get_historical_returns(active_tickers_all)

    for stock in optimized_stocks:
        t = stock["ticker"]
        ewma_std = variance_map.get(t, 0.0) ** 0.5 if t in variance_map else 0.0
        stock["risk_score"] = normalize_risk_score(ewma_std, sigma_min, sigma_max)
        stock["risk_rank"] = rank_map.get(t, 300)
        stock["historical_returns"] = historical_returns.get(t, {"1M": 0, "3M": 0, "6M": 0, "12M": 0})

    # 4. 포트폴리오 메트릭
    portfolio_return_log = float(np.dot(weights, mu))  # 가중 로그수익률 (decimal, 예: 0.14)
    portfolio_return_simple = round((math.exp(portfolio_return_log) - 1.0) * 100.0, 2)
    portfolio_var = float(weights.T @ cov_sub @ weights)
    portfolio_volatility = round(float(np.sqrt(max(portfolio_var, 0.0))), 4)
    # 단순 가중 변동성 (분산효과 0일 때)
    naive_var = float(np.dot(weights**2, np.diag(cov_sub)))
    portfolio_volatility_naive = round(float(np.sqrt(max(naive_var, 0.0))), 4)

    # 5. 누적수익률 차트 (포트폴리오 vs S&P 500)
    active_tickers = [s["ticker"] for s in optimized_stocks]
    active_weights_pct = [s["weight"] for s in optimized_stocks]
    w_sum = sum(active_weights_pct)
    chart_weights = [w / w_sum for w in active_weights_pct] if w_sum > 0 else []
    chart_data = get_cumulative_return_chart(active_tickers, chart_weights)

    # 7. 실데이터 Monte Carlo 시뮬레이션
    forecast_data = get_real_forecast(
        tickers=active_tickers,
        weights=chart_weights,
        expected_return_3m_log=portfolio_return_log,  # 이미 decimal
        n_paths=300,
    )

    # 8. 과거 3개월 가중 수익률
    hist_3m_values = []
    for s in optimized_stocks:
        h = s.get("historical_returns", {})
        hist_3m_values.append(float(h.get("3M", 0)))
    hist_weights_norm = np.array(active_weights_pct)
    if hist_weights_norm.sum() > 0:
        hist_weights_norm = hist_weights_norm / hist_weights_norm.sum()
    past_3m_return = round(float(np.dot(hist_weights_norm, hist_3m_values)), 2)

    # 9. VaR 5% (보수적: drift=0, 순수 위험 기반 — 카드 표시용)
    var_5_conservative = forecast_data.get("var_5_conservative", forecast_data["final_distribution"]["percentile_5"])

    # 10. 포트폴리오 스코어 (PortfolioSelection과 동일 로직)
    portfolio_scores = calculate_portfolio_scores(active_tickers)

    return {
        "status": "success",
        "data": {
            "optimized_stocks": optimized_stocks,
            "portfolio_expected_return_3m_simple": portfolio_return_simple,
            "portfolio_volatility": portfolio_volatility,
            "portfolio_volatility_naive": portfolio_volatility_naive,
            "past_3m_return": past_3m_return,
            "var_5": round(var_5_conservative, 2),
            "chart_data": chart_data,
            "forecast_data": forecast_data,
            "weight_sum": round(float(np.sum(weights)) * 100, 2),
            "portfolio_scores": portfolio_scores,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
