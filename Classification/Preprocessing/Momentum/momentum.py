"""
Momentum feature engineering module.
"""

import numpy as np
import pandas as pd


def calculate_features(ticker, db=None):
    """
    주가 데이터를 DB에서 로드하여 기본 모멘텀 피처를 계산합니다.
    
    Args:
        ticker (str): 종목 티커 (예: 'AAPL')
        db (StockDBManager): DB 연결 객체

    Returns:
        pd.DataFrame: 기술적 지표가 포함된 DataFrame
    """
    print(f"\n  [Feature Engineering] {ticker} 기술적 지표 계산 중...")

    # 1. 데이터 로드
    if db is None:
        raise ValueError("DB Connection is required")
        
    df = db.fetch_ticker_data(ticker)
    if df.empty:
        print(f"  ⚠️ {ticker} 데이터가 없습니다.")
        return pd.DataFrame()

    # 2. 기본 전처리
    df = df.sort_index()
    
    # 3. 기술적 지표 계산
    # (1) Log Returns (20, 120일) - Log_Ret_60은 제거 정책 적용
    for period in [20, 120]:
        df[f'Log_Ret_{period}'] = np.log(df['Close'] / df['Close'].shift(period))

    # (2) MA Envelope (20일 이동평균 대비 이격도)
    ma_20 = df['Close'].rolling(window=20).mean()
    df['MA_Envelope'] = (df['Close'] - ma_20) / ma_20

    # (3) 52주 최고가 대비 현재가 위치
    df["High_52W"] = df["High"].rolling(window=252, min_periods=252).max()
    df["High_52W_Proximity"] = (df["Close"] / df["High_52W"]) - 1.0

    # (4) RSI (14일) - 선택 사항이지만 유용하므로 추가
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 필요한 컬럼만 선택하여 반환
    features = [
        "Log_Ret_20",
        "Log_Ret_120",
        "MA_Envelope",
        "High_52W_Proximity",
        "RSI_14",
    ]
    
    # NaN 제거 (지표 계산으로 인한 앞부분 결측치)
    df_features = df[features].dropna()
    
    print(f"  지표 계산 완료: {len(df_features)}건")
    return df_features
