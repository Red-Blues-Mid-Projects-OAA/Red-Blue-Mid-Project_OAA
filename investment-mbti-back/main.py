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
)
from portfolio_optimizer import optimize_portfolio
from chart_data_provider import (
    get_historical_returns,
    get_cumulative_return_chart,
    get_forecast_placeholder,
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
        4: "극도로 보수적인 안전형",
        3: "신중한 배당/가치형",
        2: "균형 잡힌 성장형",
        1: "공격적 테마/변동성",
    }

    # 11. 프론트엔드 호환 응답 구성
    features = [
        f"투자 MBTI 성향: {risk_result['mbti']} ({risk_result['mbti_nickname']})",
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
            "mbti_nickname": risk_result["mbti_nickname"],
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
