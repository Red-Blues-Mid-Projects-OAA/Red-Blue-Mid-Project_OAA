def get_mock_data():
    """
    임시로 사용할 포트폴리오(가상) 주식 정보
    추후 실제 DB/모델 데이터로 대체됩니다.
    """
    conservative_stocks = [
        { "rank": 1, "ticker": "JNJ", "name": "존슨앤드존슨", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 4.1 },
        { "rank": 2, "ticker": "PG", "name": "프로스터 앤 갬블", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 3.8 },
        { "rank": 3, "ticker": "KO", "name": "코카콜라", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 4.5 },
        { "rank": 4, "ticker": "PEP", "name": "펩시코", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 4.2 },
        { "rank": 5, "ticker": "WM", "name": "웨이스트 매니지먼트", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 5.1 },
        { "rank": 6, "ticker": "COST", "name": "코스트코", "volatility": "Low", "color": "#10b981", "expectedReturn3M": 6.0 },
        { "rank": 7, "ticker": "V", "name": "비자", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 7.2 },
        { "rank": 8, "ticker": "MRK", "name": "머크", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 4.9 },
        { "rank": 9, "ticker": "UNH", "name": "유나이티드헬스", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.5 },
        { "rank": 10, "ticker": "BRK.B", "name": "버크셔 해서웨이", "volatility": "Low", "color": "#3b82f6", "expectedReturn3M": 5.8 }
    ]

    aggressive_stocks = [
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
        "expected_returns": {},
        "covariance_matrix": {},
        "conservative_stocks": conservative_stocks,
        "aggressive_stocks": aggressive_stocks
    }
