def get_mock_data():
    """
    4가지 MBTI(거북이, 강아지, 사자, 독수리) 유형별 맞춤형 포트폴리오
    추후 DB/모델 데이터로 대체됩니다.
    """
    turtle_stocks = [
        { "rank": 1, "ticker": "SHV", "name": "단기국채 ETF", "volatility": "Very Low", "color": "#10b981", "expectedReturn3M": 1.2 },
        { "rank": 2, "ticker": "TLT", "name": "장기국채 ETF", "volatility": "Very Low", "color": "#10b981", "expectedReturn3M": 2.1 },
        { "rank": 3, "ticker": "KO", "name": "코카콜라", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 2.5 },
        { "rank": 4, "ticker": "PEP", "name": "펩시코", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 2.8 },
        { "rank": 5, "ticker": "JNJ", "name": "존슨앤드존슨", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.1 },
        { "rank": 6, "ticker": "PG", "name": "프록터앤갬블", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.0 },
        { "rank": 7, "ticker": "WMT", "name": "월마트", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.5 },
        { "rank": 8, "ticker": "COST", "name": "코스트코", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 4.1 },
        { "rank": 9, "ticker": "MCD", "name": "맥도날드", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.8 },
        { "rank": 10, "ticker": "NEE", "name": "넥스트에라 에너지", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.2 }
    ]

    dog_stocks = [
        { "rank": 1, "ticker": "BRK.B", "name": "버크셔 해서웨이", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.5 },
        { "rank": 2, "ticker": "V", "name": "비자", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.2 },
        { "rank": 3, "ticker": "MA", "name": "마스터카드", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.5 },
        { "rank": 4, "ticker": "UNH", "name": "유나이티드헬스", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.8 },
        { "rank": 5, "ticker": "MRK", "name": "머크", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.2 },
        { "rank": 6, "ticker": "ABBV", "name": "애브비", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.9 },
        { "rank": 7, "ticker": "HD", "name": "홈디포", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.1 },
        { "rank": 8, "ticker": "LMT", "name": "록히드마틴", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.6 },
        { "rank": 9, "ticker": "XOM", "name": "엑슨모빌", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.8 },
        { "rank": 10, "ticker": "CVX", "name": "쉐브론", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.2 }
    ]

    lion_stocks = [
        { "rank": 1, "ticker": "AAPL", "name": "애플", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 7.5 },
        { "rank": 2, "ticker": "MSFT", "name": "마이크로소프트", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 8.2 },
        { "rank": 3, "ticker": "GOOGL", "name": "알파벳", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 8.0 },
        { "rank": 4, "ticker": "AMZN", "name": "아마존", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 9.5 },
        { "rank": 5, "ticker": "META", "name": "메타", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 10.2 },
        { "rank": 6, "ticker": "AVGO", "name": "브로드컴", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 11.0 },
        { "rank": 7, "ticker": "ASML", "name": "ASML", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 9.8 },
        { "rank": 8, "ticker": "LLY", "name": "일라이릴리", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 11.5 },
        { "rank": 9, "ticker": "NFLX", "name": "넷플릭스", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 10.8 },
        { "rank": 10, "ticker": "CRM", "name": "세일즈포스", "volatility": "Medium", "color": "#8b5cf6", "expectedReturn3M": 8.5 }
    ]

    eagle_stocks = [
        { "rank": 1, "ticker": "MSTR", "name": "마이크로스트레티지", "volatility": "High", "color": "#ef4444", "expectedReturn3M": 18.5 },
        { "rank": 2, "ticker": "COIN", "name": "코인베이스", "volatility": "High", "color": "#ef4444", "expectedReturn3M": 15.0 },
        { "rank": 3, "ticker": "SMCI", "name": "슈퍼마이크로", "volatility": "High", "color": "#ef4444", "expectedReturn3M": 22.0 },
        { "rank": 4, "ticker": "TSLA", "name": "테슬라", "volatility": "High", "color": "#ef4444", "expectedReturn3M": 12.5 },
        { "rank": 5, "ticker": "PLTR", "name": "팔란티어", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 14.0 },
        { "rank": 6, "ticker": "NVDA", "name": "엔비디아", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 11.5 },
        { "rank": 7, "ticker": "ARM", "name": "ARM 홀딩스", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 13.0 },
        { "rank": 8, "ticker": "HOOD", "name": "로빈후드", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 16.5 },
        { "rank": 9, "ticker": "ROKU", "name": "로쿠", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 9.5 },
        { "rank": 10, "ticker": "AMD", "name": "AMD", "volatility": "High", "color": "#f59e0b", "expectedReturn3M": 10.0 }
    ]

    return {
        "turtle_stocks": turtle_stocks,
        "dog_stocks": dog_stocks,
        "lion_stocks": lion_stocks,
        "eagle_stocks": eagle_stocks
    }

def build_mvo_inputs(stocks):
    import numpy as np
    
    # mu: 백분율 수익률(ex: 18.5)을 소수점(0.185)으로 변환
    mu = [s['expectedReturn3M'] / 100.0 for s in stocks]
    n = len(stocks)
    
    # 공분산 행렬(Covariance Matrix) 생성
    # 수익률이 높을수록 변동성이 기하급수적으로 커지도록 가정하여 대각선 분산(Variance) 계산
    cov = np.zeros((n, n))
    base_cov = 0.0005
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i][j] = (mu[i] * 1.5)**2 + 0.001
            else:
                cov[i][j] = base_cov * (mu[i] + mu[j]) # 임의의 양의 상관관계
                
    return mu, cov.tolist()
