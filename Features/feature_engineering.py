
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler


def calculate_features(ticker, start_date, end_date):
    """
    AAPL 종목의 가격/모멘텀 feature를 계산하여 단일 DataFrame으로 반환.
    - 중장기 수익률 (Log Returns): 20일, 60일, 120일 rolling mean
    - 이동평균 이격도 (MA Envelope): (Close - MA60) / MA60
    - 52주 고점 대비 위치 (High-Low Proximity): (Close / 52W High) - 1
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if df.empty:
        print("No data found.")
        return pd.DataFrame()

    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        if 'Ticker' in df.columns.names:
            df.columns = df.columns.droplevel('Ticker')
        else:
            try:
                df = df.xs(ticker, axis=1, level=1)
            except Exception:
                pass

    # --- Feature Engineering ---

    # 1. 중장기 수익률 (Log Returns): 20일/60일/120일 rolling mean
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Log_Ret_20'] = df['Log_Return'].rolling(window=20).mean()
    df['Log_Ret_60'] = df['Log_Return'].rolling(window=60).mean()
    df['Log_Ret_120'] = df['Log_Return'].rolling(window=120).mean()

    # 2. 이동평균 이격도 (MA Envelope): (Close - MA60) / MA60
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA_Envelope'] = (df['Close'] - df['MA60']) / df['MA60']

    # 3. 52주 고점 대비 위치 (High-Low Proximity): (Close / 52W High) - 1
    df['High_52W'] = df['High'].rolling(window=252).max()
    df['High_Low_Proximity'] = (df['Close'] / df['High_52W']) - 1

    # Feature columns
    feature_cols = ['Log_Ret_20', 'Log_Ret_60', 'Log_Ret_120', 'MA_Envelope', 'High_Low_Proximity']

    # NaN 제거 (rolling window로 인해 초기 데이터 불완전)
    df = df.dropna(subset=feature_cols).copy()

    # % 형태 normalization (* 100)
    df[feature_cols] = df[feature_cols] * 100

    return df[feature_cols]


def main():
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2026-02-11'

    df = calculate_features(ticker, start_date, end_date)

    if df.empty:
        return

    # StandardScaler: Train 구간(2016~2024)으로 fit, 전체 데이터에 transform
    train_mask = (df.index >= '2016-01-01') & (df.index <= '2024-12-31')
    df_train = df.loc[train_mask]

    scaler = StandardScaler()
    print("Fitting StandardScaler on Train data (2016-2024)...")
    scaler.fit(df_train)

    df_scaled = pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns
    )

    # Output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n" + "=" * 60)
    print("Combined Feature DataFrame (Standardized)")
    print(f"Shape: {df_scaled.shape}")
    print(f"Period: {df_scaled.index.min().date()} ~ {df_scaled.index.max().date()}")
    print("\nFirst 5 rows:")
    print(df_scaled.head())
    print("\nLast 5 rows:")
    print(df_scaled.tail())
    print("=" * 60)


if __name__ == "__main__":
    main()
