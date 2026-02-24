from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from risk_profile import calculate_risk_profile
from mock_data_provider import get_mock_data

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

    # 2. 결과에 따른 Mock Data 종목 선정 (lambda가 8.0 이상이면 보수적, 이하면 공격적)
    is_conservative = lambda_final > 8.0
    mock = get_mock_data()
    recommended_stocks = mock["conservative_stocks"] if is_conservative else mock["aggressive_stocks"]
    
    # 3. 프론트엔드에서 요구하는 JSON 응답 형태로 반환
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
                "expected_portfolio_return": "+4.5%" if is_conservative else "+14.2%",
                "risk_category": "보수적 안정형" if is_conservative else "공격적 변동성"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
