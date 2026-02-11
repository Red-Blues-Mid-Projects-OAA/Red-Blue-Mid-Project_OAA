"""
AAPL 거래량 분석 모듈 (Self-contained)

생성되는 DataFrame:
  df_volume_features : Volume_Ratio, OBV_ROC_20 (정상성 확보)

★ DB 적재 없음 / 다른 모듈 수정 없음
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import sys
import os

# 상위 디렉토리의 모듈을 import하기 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_aapl_volume_analysis():
    """
    AAPL 거래량 분석 (Volume Ratio + OBV 변화율)
    - Volume_Ratio = 당일 거래량 / 직전 20일 평균 거래량
    - OBV_ROC_20   = OBV의 20일간 변화율 (정상성 확보)
    """
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"[{ticker}] {start_date} ~ {end_date} 거래량 분석 데이터 계산 중...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print("데이터를 가져오지 못했습니다.")
        return None

    # 멀티 컬럼 처리
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
        volume = data["Volume"].iloc[:, 0]
    else:
        close = data["Close"]
        volume = data["Volume"]

    # Volume Ratio 계산
    # 직전 20일 평균 거래량 (shift(1)로 당일 제외 → 누수 차단)
    avg_20 = volume.rolling(window=20).mean().shift(1)
    volume_ratio = volume / avg_20

    # OBV 계산
    price_change = np.sign(close.diff())
    obv = (price_change * volume).cumsum()

    # OBV 변화율 (20일) — 비정상성(Non-stationarity) 해결
    # 원시 OBV는 누적합이라 시간에 따라 무한히 커져 ML 모델에 부적합
    # 대신 20일간 OBV 변화율을 사용하여 정상성(Stationarity) 확보
    obv_roc_20 = obv.pct_change(20)

    result = pd.DataFrame({
        "Close": close.values,
        "Volume": volume.values,
        "Avg_Volume_20d": avg_20.values,
        "Volume_Ratio": volume_ratio.values,
        "OBV": obv.values,
        "OBV_ROC_20": obv_roc_20.values,
    }, index=close.index)
    result.index.name = "Date"

    # NaN 제거 (rolling window + OBV ROC로 인한 초기 데이터 불완전)
    result = result.dropna()

    print(f"\n총 {len(result)}개 거래일")

    # ML용 피처 DataFrame (Volume_Ratio + OBV_ROC_20만 추출)
    df_volume_features = result[["Volume_Ratio", "OBV_ROC_20"]].copy()

    print(f"\n{'=' * 60}")
    print("[결과] AAPL 거래량 피처 (df_volume_features)")
    print(f"{'=' * 60}")
    print(f"  기간: {df_volume_features.index[0].date()} ~ {df_volume_features.index[-1].date()}")
    print(f"  건수: {len(df_volume_features)}")
    print(df_volume_features.head(10))
    print("  ...")
    print(df_volume_features.tail(5))

    # 거래량 기본 통계
    print(f"\n평균 거래량: {result['Volume'].mean():,.0f}")
    print(f"최대 거래량: {result['Volume'].max():,.0f} ({result['Volume'].idxmax().strftime('%Y-%m-%d')})")
    print(f"최소 거래량: {result['Volume'].min():,.0f} ({result['Volume'].idxmin().strftime('%Y-%m-%d')})")

    return df_volume_features


if __name__ == "__main__":
    calculate_aapl_volume_analysis()
