"""
시장 데이터 수집 및 변환 모듈 (Self-contained)

생성되는 DataFrame:
  1. df_dxy_log_returns  : 달러 인덱스(FRED: DTWEXBGS) 일별 로그 수익률
  2. df_vix              : VIX 종가 + VIX 로그 수익률 (yfinance: ^VIX)
  3. df_sp500_momentum   : S&P 500 1개월(21d)/3개월(63d) Rolling 누적 수익률

★ DB 적재 없음 / 다른 모듈 수정 없음
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime
import sys
import os

# 상위 디렉토리의 모듈을 import하기 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_db_manager import StockDBManager

START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")


def fetch_dollar_index_log_returns():
    """
    FRED에서 달러 인덱스(DTWEXBGS)를 가져와 일별 로그 수익률로 변환
    """
    print("=" * 60)
    print("[1] 달러 인덱스 로그 수익률 (FRED: DTWEXBGS)")
    print("=" * 60)

    dxy = web.DataReader("DTWEXBGS", "fred", START_DATE, END_DATE)
    dxy.columns = ["DXY_Close"]
    dxy = dxy.dropna()

    # 로그 수익률 계산
    dxy["DXY_Log_Return"] = np.log(dxy["DXY_Close"] / dxy["DXY_Close"].shift(1))
    df_dxy = dxy[["DXY_Log_Return"]].dropna()

    print(f"  기간: {df_dxy.index[0].date()} ~ {df_dxy.index[-1].date()}")
    print(f"  건수: {len(df_dxy)}")
    print(f"\n  ★ DataFrame 이름: df_dxy_log_returns")
    print(df_dxy.head())
    print("  ...")
    print(df_dxy.tail())

    return df_dxy


def fetch_vix_data():
    """
    yfinance에서 VIX 종가 데이터를 가져와 DataFrame으로 반환
    """
    print("\n" + "=" * 60)
    print("[2] VIX 지수 (yfinance: ^VIX)")
    print("=" * 60)

    vix_raw = yf.download("^VIX", start=START_DATE, auto_adjust=True, progress=False)
    vix_close = vix_raw["Close"]

    # MultiIndex 처리 (yfinance가 단일 종목도 MultiIndex로 반환하는 경우)
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    vix_close.index = vix_close.index.tz_localize(None)
    vix_close = vix_close.dropna()

    df_vix = pd.DataFrame({"VIX_Close": vix_close})

    # VIX 로그 수익률 추가 — "어제보다 VIX가 얼마나 튀었는가(Shock)" 지표
    # VIX 수준(Level)보다 급등 여부가 하락장 방어에 더 유효
    df_vix['VIX_Log_Return'] = np.log(df_vix['VIX_Close'] / df_vix['VIX_Close'].shift(1))
    df_vix = df_vix.dropna()

    print(f"  기간: {df_vix.index[0].date()} ~ {df_vix.index[-1].date()}")
    print(f"  건수: {len(df_vix)}")
    print(f"\n  ★ DataFrame 이름: df_vix")
    print(df_vix.head())
    print("  ...")
    print(df_vix.tail())

    return df_vix


def calculate_sp500_momentum():
    """
    DB에서 S&P 500 일별 로그수익률을 조회하여
    1개월(21거래일) / 3개월(63거래일) Rolling 누적 수익률 계산
    """
    print("\n" + "=" * 60)
    print("[3] S&P 500 1개월 / 3개월 수익률 (DB: SP500_DATA)")
    print("=" * 60)

    db = StockDBManager()
    db.connect()
    try:
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA ORDER BY TRADE_DATE"
        db.cursor.execute(query)
        rows = db.cursor.fetchall()
    finally:
        db.close()

    df = pd.DataFrame(rows, columns=["TRADE_DATE", "LOG_RETURN"])
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df = df.set_index("TRADE_DATE").sort_index()
    df["LOG_RETURN"] = df["LOG_RETURN"].astype(float)

    # Rolling 누적 로그 수익률 (일별 로그 수익률의 합 = 기간 누적 수익률)
    df["SP500_1M_Return"] = df["LOG_RETURN"].rolling(window=21).sum()
    df["SP500_3M_Return"] = df["LOG_RETURN"].rolling(window=63).sum()

    df_sp500 = df[["SP500_1M_Return", "SP500_3M_Return"]].dropna()

    print(f"  기간: {df_sp500.index[0].date()} ~ {df_sp500.index[-1].date()}")
    print(f"  건수: {len(df_sp500)}")
    print(f"\n  ★ DataFrame 이름: df_sp500_momentum")
    print(df_sp500.head())
    print("  ...")
    print(df_sp500.tail())

    return df_sp500


def main():
    df_dxy = fetch_dollar_index_log_returns()
    df_vix = fetch_vix_data()
    df_sp500_mom = calculate_sp500_momentum()

    print("\n" + "=" * 60)
    print("생성된 DataFrame 목록")
    print("=" * 60)
    print(f"  1. df_dxy_log_returns  : 달러 인덱스 일별 로그 수익률  ({len(df_dxy)}건)")
    print(f"  2. df_vix              : VIX 종가                    ({len(df_vix)}건)")
    print(f"  3. df_sp500_momentum   : S&P 500 1M/3M 누적 수익률   ({len(df_sp500_mom)}건)")


if __name__ == "__main__":
    main()
