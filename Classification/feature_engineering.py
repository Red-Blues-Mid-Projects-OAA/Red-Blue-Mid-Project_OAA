"""
Feature Engineering Utils

데이터 전처리와 파생 변수 생성을 위한 유틸리티 함수들을 모아둔 모듈입니다.
"""

import numpy as np
import pandas as pd

def apply_rolling_z_score(df, window=60, exclude_cols=None):
    """
    데이터프레임의 수치형 컬럼에 대해 Rolling Z-Score를 적용합니다.
    공식: (현재값 - 이동평균) / 이동표준편차

    Args:
        df (pd.DataFrame): 기술적 지표가 포함된 데이터프레임
        window (int): 기준 기간 (기본 60일 = 1분기)
        exclude_cols (list): 변환하지 않을 컬럼 리스트 (Target, Date 등)

    Returns:
        pd.DataFrame: Z-Score 변환된 데이터프레임 (앞부분 NaN 제거됨)
    """
    if exclude_cols is None:
        exclude_cols = ['Date', 'Target', 'Target_Class', 'Target_AAPL_3M', 'Target_SP500_3M', 'Alpha_Diff']

    # 변환할 컬럼 식별 (수치형이면서 제외 목록에 없는 것)
    cols_to_transform = [
        c for c in df.columns 
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    print(f"\n[Feature Engineering] Rolling Z-Score 적용 중 (Window={window})...")
    print(f"  대상 컬럼 ({len(cols_to_transform)}개): {cols_to_transform}")

    # 원본 데이터 보존을 위해 복사
    df_transformed = df.copy()

    for col in cols_to_transform:
        # 1. 롤링 평균과 표준편차 계산
        rolling_mean = df[col].rolling(window=window).mean()
        rolling_std = df[col].rolling(window=window).std()
        
        # 2. 표준편차가 0인 경우(값이 안 변함) 대비하여 아주 작은 값(epsilon) 더함
        rolling_std = rolling_std.replace(0, 1e-8)
        
        # 3. Z-Score 변환
        z_score = (df[col] - rolling_mean) / rolling_std
        
        # 4. 무한대(inf)나 너무 큰 값 클리핑 (이상치 제어, -5 ~ +5 사이로)
        df_transformed[col] = z_score.clip(-5, 5)

    # 앞부분 NaN 제거 (윈도우 크기만큼 데이터 손실 발생)
    original_len = len(df)
    df_transformed = df_transformed.dropna()
    dropped_len = original_len - len(df_transformed)
    
    print(f"  완료: {len(df_transformed)}건 생성 (초기 {dropped_len}건 제외됨)")
    
    return df_transformed


def calculate_features(ticker, db=None):
    """
    주가 데이터를 DB에서 로드하여 기본 기술적 지표(모멘텀, 트렌드 등)를 계산합니다.
    (기존 feature_engineering.py의 핵심 기능 복원)
    
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

    # (3) High-Low Proximity (삭제: DB에 Low Price 없음)
    # hl_range = df['High'] - df['Low']
    # df['High_Low_Proximity'] = (df['Close'] - df['Low']) / hl_range.replace(0, np.nan)

    # (4) RSI (14일) - 선택 사항이지만 유용하므로 추가
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 필요한 컬럼만 선택하여 반환
    features = [
        'Log_Ret_20', 'Log_Ret_120',
        'MA_Envelope', 
        # 'High_Low_Proximity', # 삭제
        'RSI_14' # 추가
    ]
    
    # NaN 제거 (지표 계산으로 인한 앞부분 결측치)
    df_features = df[features].dropna()
    
    print(f"  지표 계산 완료: {len(df_features)}건")
    return df_features
