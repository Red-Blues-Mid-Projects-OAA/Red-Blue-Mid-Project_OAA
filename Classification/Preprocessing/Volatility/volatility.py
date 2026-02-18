"""
EWMA 기반 리스크 피처 생성 모듈.

생성되는 DataFrame:
  1. df_ticker_daily_vol : {ticker}_EWMA_Vol
  2. df_ticker_avg_vol   : {ticker}_Vol_20d_Avg, {ticker}_Vol_60d_Avg
  3. df_ticker_ewma_corr : {ticker}_{benchmark}_EWMA_Corr
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

from common import np, pd
from DB import StockDBManager

LAMBDA = 0.94
ALPHA = 1 - LAMBDA


def calculate_risk_features(ticker="AAPL", benchmark="SP500", db=None):
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()

    print("=" * 60)
    print(f"{ticker} EWMA 리스크 피처 생성 (benchmark={benchmark}, λ=0.94)")
    print("=" * 60)
    print("\n[데이터 로드]")

    should_close = False
    if db is None:
        db = StockDBManager()
        db.connect()
        should_close = True

    try:
        lr_all = db.fetch_log_returns()
        sp500 = db.fetch_sp500_data()
    finally:
        if should_close:
            db.close()

    if lr_all.empty or ticker not in lr_all.columns:
        print(f"{ticker} 로그수익률 데이터를 가져오지 못했습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")
    if sp500.empty:
        print("SP500 로그수익률 데이터를 가져오지 못했습니다.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ticker_lr = pd.to_numeric(lr_all[ticker], errors="coerce").dropna()
    bench_lr = pd.to_numeric(sp500["LOG_RETURN"], errors="coerce").dropna()

    ticker_lr.index = pd.to_datetime(ticker_lr.index)
    bench_lr.index = pd.to_datetime(bench_lr.index)

    combined = pd.DataFrame({ticker: ticker_lr, benchmark: bench_lr}).dropna()
    ewma_cov = combined.ewm(alpha=ALPHA).cov()

    ticker_var = ewma_cov.loc[(slice(None), ticker), ticker]
    ticker_var.index = ticker_var.index.droplevel(1)
    ticker_vol_raw = np.sqrt(ticker_var)
    ticker_vol_ann = ticker_vol_raw * np.sqrt(252)

    daily_col = f"{ticker}_EWMA_Vol"
    df_ticker_daily_vol = pd.DataFrame({daily_col: ticker_vol_ann}).dropna()

    avg20_col = f"{ticker}_Vol_20d_Avg"
    avg60_col = f"{ticker}_Vol_60d_Avg"
    df_ticker_avg_vol = pd.DataFrame(
        {
            avg20_col: ticker_vol_ann.rolling(window=20).mean(),
            avg60_col: ticker_vol_ann.rolling(window=60).mean(),
        }
    ).dropna()

    cross_cov = ewma_cov.loc[(slice(None), ticker), benchmark]
    cross_cov.index = cross_cov.index.droplevel(1)
    bench_var = ewma_cov.loc[(slice(None), benchmark), benchmark]
    bench_var.index = bench_var.index.droplevel(1)
    bench_vol_raw = np.sqrt(bench_var)
    ewma_corr = cross_cov / (ticker_vol_raw * bench_vol_raw)

    corr_col = f"{ticker}_{benchmark}_EWMA_Corr"
    df_ticker_ewma_corr = pd.DataFrame({corr_col: ewma_corr}).dropna()

    print(f"\n{'=' * 60}")
    print("생성된 DataFrame 목록")
    print(f"{'=' * 60}")
    print(f"  1. df_ticker_daily_vol : {daily_col:24s} ({len(df_ticker_daily_vol)}건)")
    print(f"  2. df_ticker_avg_vol   : {avg20_col}, {avg60_col} ({len(df_ticker_avg_vol)}건)")
    print(f"  3. df_ticker_ewma_corr : {corr_col:24s} ({len(df_ticker_ewma_corr)}건)")

    return df_ticker_daily_vol, df_ticker_avg_vol, df_ticker_ewma_corr


if __name__ == "__main__":
    calculate_risk_features()
