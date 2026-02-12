import sys, os
# 상위 디렉토리의 모듈 및 common 패키지 import를 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import pd, np
from stock_db_manager import StockDBManager


def calculate_features(ticker):
    """
    AAPL 종목의 가격/모멘텀 feature를 계산하여 단일 DataFrame으로 반환.
    - 중장기 수익률 (Log Returns): 20일, 60일, 120일 rolling mean
    - 이동평균 이격도 (MA Envelope): (Close - MA60) / MA60
    - 52주 고점 대비 위치 (High-Low Proximity): (Close - 52W High) / 52W High
    """
    print(f"[{ticker}] Oracle DB에서 데이터 로드 중...")
    db_manager = StockDBManager()
    db_manager.connect()
    df = db_manager.fetch_ticker_data(ticker)
    db_manager.close()

    if df.empty:
        print("DB에서 데이터를 가져오지 못했습니다. fetch_stock_data.py를 먼저 실행하세요.")
        return pd.DataFrame()

    # --- Feature Engineering ---

    # 1. 중장기 수익률 (Log Returns): 20일/60일/120일 rolling mean
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Ret_20'] = df['Log_Return'].rolling(window=20).mean()
    df['Log_Ret_60'] = df['Log_Return'].rolling(window=60).mean()
    df['Log_Ret_120'] = df['Log_Return'].rolling(window=120).mean()

    # 2. 이동평균 이격도 (MA Envelope): (Close - MA60) / MA60
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA_Envelope'] = (df['Close'] - df['MA60']) / df['MA60']

    # 3. 52주 고점 대비 위치 (High-Low Proximity): (Close - 52W High) / 52W High
    df['High_52W'] = df['High'].rolling(window=252).max()
    df['High_Low_Proximity'] = (df['Close'] / df['High_52W']) - 1

    # Feature 변수들
    feature_cols = ['Log_Ret_20', 'Log_Ret_60', 'Log_Ret_120', 'MA_Envelope', 'High_Low_Proximity']

    # NaN 제거 (rolling window로 인해 초기 데이터 결측치)
    df = df.dropna(subset=feature_cols).copy()

    return df[feature_cols]


def main():
    ticker = 'AAPL'

    df = calculate_features(ticker)

    if df.empty:
        print("Feature 계산 실패")
        return

    print("\n" + "=" * 60)
    print("Combined Feature DataFrame")
    print(f"Shape: {df.shape}")
    print(f"Period: {df.index.min().date()} ~ {df.index.max().date()}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("=" * 60)


if __name__ == "__main__":
    main()
