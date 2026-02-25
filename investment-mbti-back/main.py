from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from risk_profile import calculate_risk_profile
from mock_data_provider import get_mock_data, build_mvo_inputs
from portfolio_optimizer import optimize_portfolio

app = FastAPI(title="Investment MBTI Integration API")

# 프론트엔드 연동을 위한 CORS 미들웨어 적용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    answers: List[str]      # ['A', 'B', 'B', ...] 형식
    loss_limit_value: int   # 9500000 와 같은 원 단위 금액 형태

@app.post("/api/analyze")
def analyze_portfolio(req: AnalyzeRequest):
    # 1. 기생성 모듈 사용
    risk_result = calculate_risk_profile(req.answers, req.loss_limit_value)
    lambda_final = risk_result['lambda_final']
    persona = risk_result['persona']
    desc = risk_result['persona_desc']

    # 2. 결과(페르소나)에 따른 4단계 Mock Data 종목 및 메타데이터 선정
    mock = get_mock_data()
    if "거북이" in persona:
        recommended_stocks = mock["turtle_stocks"]
        expected_return = "+2.5%"
        risk_category = "극도로 보수적인 안전형"
    elif "강아지" in persona:
        recommended_stocks = mock["dog_stocks"]
        expected_return = "+4.8%"
        risk_category = "신중한 배당/가치형"
    elif "사자" in persona:
        recommended_stocks = mock["lion_stocks"]
        expected_return = "+9.5%"
        risk_category = "균형 잡힌 성장형"
    else: # 독수리
        recommended_stocks = mock["eagle_stocks"]
        expected_return = "+14.2%"
        risk_category = "공격적 테마/변동성"

    # 3. 평균-분산 최적화 (MVO) 엔진 수행하여 비중 산출
    # 10개 종목에 대한 mu, cov 도출
    mu, cov = build_mvo_inputs(recommended_stocks)
    
    # MVO 최적 비중 산출 (scipy.optimize)
    weights = optimize_portfolio(mu, cov, lambda_final)
    
    # 4. 프론트엔드 연동을 위해 개별 주식 객체에 할당 비중(weight) 추가 (퍼센트 단위 반환)
    for i, stock in enumerate(recommended_stocks):
        stock['weight'] = round(float(weights[i]) * 100, 2)
    
    # 5. 프론트엔드에서 요구하는 JSON 응답 형태로 반환
    features = [
        f"투자 MBTI 성향: {risk_result['mbti']} ({risk_result['mbti_nickname']})",
        f"위험 회피 계수(Lambda): {lambda_final:.2f}",
        f"손실 한도 선택: -{risk_result['loss_ratio_percent']}%"
    ]

    return {
        "status": "success",
        "data": {
            "persona": persona,
            "description": desc,
            "features": features,
            "recommended_stocks": recommended_stocks,
            "portfolio_analysis": {
                "expected_portfolio_return": expected_return,
                "risk_category": risk_category
            },
            "raw_answers": req.answers
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
