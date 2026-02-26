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
from typing import List, Optional

import math
from risk_profile import calculate_risk_profile
from real_data_provider import load_cache, get_recommended_stocks, get_mvo_inputs
from portfolio_optimizer import optimize_portfolio
from chart_data_provider import get_historical_returns, get_cumulative_return_chart, get_forecast_placeholder


# ──────────────────────────────────────────────────────────────────
# FastAPI 앱 생성 (서버 시작 시 데이터 캐싱)
# ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 CSV + EWMA 공분산 캐시 로드."""
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
    answers: List[str]        # ['A', 'B', 'B', ...] 형식 (12개)
    loss_limit_value: int     # 원 단위 금액 (예: 9500000)


class OptimizeRequest(BaseModel):
    """포트폴리오 최적화 요청 (사용자 종목 선택 후)."""
    selected_tickers: List[str]  # 사용자가 선택한 종목 리스트
    lambda_final: float          # λ 값


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

    # 3. 리스크 타입별 종목 추천 (유틸리티 스코어 기반)
    recommended_stocks = get_recommended_stocks(final_level, top_n=10)

    # 4. 추천 종목으로 초기 MVO 최적화 수행
    tickers = [s["ticker"] for s in recommended_stocks]
    mu, cov_sub, valid_tickers = get_mvo_inputs(tickers)
    weights = optimize_portfolio(mu, cov_sub, lambda_final)

    # 5. 비중을 종목 객체에 할당 (퍼센트 단위)
    ticker_weight_map = {t: round(float(w) * 100, 2) for t, w in zip(valid_tickers, weights)}
    for stock in recommended_stocks:
        stock["weight"] = ticker_weight_map.get(stock["ticker"], 0.0)

    # 6. 포트폴리오 메트릭 계산
    portfolio_return_log = float(np.dot(weights, mu))                    # 로그수익률
    portfolio_return_simple = round((math.exp(portfolio_return_log) - 1) * 100, 2)  # 단순수익률(%)
    portfolio_var = float(weights.T @ cov_sub @ weights)
    portfolio_volatility = round(float(np.sqrt(portfolio_var)), 4)       # 60일 기준 표준편차

    # 7. 과거 기간별 수익률 (1M/3M/6M/12M, 단순수익률 %)
    historical_returns = get_historical_returns(valid_tickers)
    for stock in recommended_stocks:
        stock["historical_returns"] = historical_returns.get(stock["ticker"], {"1M": 0, "3M": 0, "6M": 0, "12M": 0})

    # 8. 누적수익률 차트 데이터 (포트폴리오 vs S&P 500)
    weight_list = [float(weights[valid_tickers.index(t)]) if t in valid_tickers else 0 for t in tickers]
    chart_data = get_cumulative_return_chart(tickers, weight_list)

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
            "portfolio_analysis": {
                "expected_return_simple": portfolio_return_simple,       # 단순수익률 (%)
                "expected_return_log": round(portfolio_return_log * 100, 2),  # 로그수익률 (%, 참조용)
                "volatility_60d": portfolio_volatility,                 # 60일 표준편차
                "risk_category": risk_categories.get(final_level, "균형 잡힌 성장형"),
                "var_5": forecast_data["final_distribution"]["percentile_5"],  # VaR 5% (Monte Carlo placeholder)
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
    for i, t in enumerate(valid_tickers):
        result_stocks.append({
            "ticker": t,
            "weight": round(float(weights[i]) * 100, 2),
            "expectedReturn3M": round(float(mu[i]) * 100, 2),
        })

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
