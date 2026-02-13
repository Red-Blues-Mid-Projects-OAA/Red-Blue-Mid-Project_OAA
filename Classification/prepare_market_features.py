"""
시장 피처 생성 모듈 (DB 기반)

생성되는 DataFrame:
  1. df_dxy_log_returns  : 달러 인덱스 일별 로그 수익률 (DB: MARKET_FEATURES)
  2. df_vix              : VIX 종가 + VIX 로그 수익률 (DB: MARKET_FEATURES)
  3. df_sp500_momentum   : S&P 500 1개월(20d)/3개월(60d) Rolling 누적 수익률 (DB: SP500_DATA)

★ 외부 API 의존성 없음 — 모든 데이터를 Oracle DB에서 조회
"""

import sys, os
# 상위 디렉토리의 모듈 및 common 패키지 import를 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import pd, np
from stock_db_manager import StockDBManager


def fetch_dollar_index_log_returns(db):
    """
    DB에서 달러 인덱스(DXY) 로그 수익률을 조회하여 DataFrame으로 반환
    """
    print("=" * 60)
    print("[1] 달러 인덱스 로그 수익률 (DB: MARKET_FEATURES)")
    print("=" * 60)

    df = db.fetch_market_features("DXY")

    if df.empty:
        print("DXY 데이터를 가져오지 못했습니다. fetch_market_data.py를 먼저 실행하세요.")
        return pd.DataFrame()

    df_dxy = pd.DataFrame({"DXY_Log_Return": df["Log_Return"].astype(float)})

    print(f"  기간: {df_dxy.index[0].date()} ~ {df_dxy.index[-1].date()}")
    print(f"  건수: {len(df_dxy)}")
    print(f"\n  ★ DataFrame 이름: df_dxy_log_returns")
    print(df_dxy.head())
    print("  ...")
    print(df_dxy.tail())

    return df_dxy


def fetch_vix_data(db):
    """
    DB에서 VIX 종가 및 로그 수익률을 조회하여 DataFrame으로 반환
    """
    print("\n" + "=" * 60)
    print("[2] VIX 지수 (DB: MARKET_FEATURES)")
    print("=" * 60)

    df = db.fetch_market_features("VIX")

    if df.empty:
        print("VIX 데이터를 가져오지 못했습니다. fetch_market_data.py를 먼저 실행하세요.")
        return pd.DataFrame()

    df_vix = pd.DataFrame({
        "VIX_Close": df["Close"].astype(float),
        "VIX_Log_Return": df["Log_Return"].astype(float)
    })

    print(f"  기간: {df_vix.index[0].date()} ~ {df_vix.index[-1].date()}")
    print(f"  건수: {len(df_vix)}")
    print(f"\n  ★ DataFrame 이름: df_vix")
    print(df_vix.head())
    print("  ...")
    print(df_vix.tail())

    return df_vix


def calculate_sp500_momentum(db):
    """
    DB에서 S&P 500 일별 로그수익률을 조회하여
    1개월(20거래일) / 3개월(60거래일) Rolling 누적 수익률 계산
    """
    print("\n" + "=" * 60)
    print("[3] S&P 500 1개월 / 3개월 수익률 (DB: SP500_DATA)")
    print("=" * 60)

    df = db.fetch_sp500_data()

    if df.empty:
        print("S&P 500 데이터를 가져오지 못했습니다. update_sp500_data.py를 먼저 실행하세요.")
        return pd.DataFrame()

    # Rolling 누적 로그 수익률 (일별 로그 수익률의 합 = 기간 누적 수익률)
    df["SP500_1M_Return"] = df["LOG_RETURN"].rolling(window=20).sum()
    df["SP500_3M_Return"] = df["LOG_RETURN"].rolling(window=60).sum()

    df_sp500 = df[["SP500_1M_Return", "SP500_3M_Return"]].dropna()

    print(f"  기간: {df_sp500.index[0].date()} ~ {df_sp500.index[-1].date()}")
    print(f"  건수: {len(df_sp500)}")
    print(f"\n  ★ DataFrame 이름: df_sp500_momentum")
    print(df_sp500.head())
    print("  ...")
    print(df_sp500.tail())

    return df_sp500


def get_market_features(db):
    """
    모든 시장 피처를 생성하여 반환하는 통합 진입점
    """
    df_dxy = fetch_dollar_index_log_returns(db)
    df_vix = fetch_vix_data(db)
    df_sp500_mom = calculate_sp500_momentum(db)
    
    return df_dxy, df_vix, df_sp500_mom


if __name__ == "__main__":
    db = StockDBManager()
    db.connect()
    try:
        df_dxy, df_vix, df_sp500_mom = get_market_features(db)

        print("\n" + "=" * 60)
        print("생성된 DataFrame 목록")
        print("=" * 60)
        print(f"  1. df_dxy_log_returns  : 달러 인덱스 일별 로그 수익률  ({len(df_dxy)}건)")
        print(f"  2. df_vix              : VIX 종가                    ({len(df_vix)}건)")
        print(f"  3. df_sp500_momentum   : S&P 500 1M/3M 누적 수익률   ({len(df_sp500_mom)}건)")
    finally:
        db.close()
