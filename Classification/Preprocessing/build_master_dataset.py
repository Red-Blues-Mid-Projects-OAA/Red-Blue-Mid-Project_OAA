# Master DataFrame 병합 모듈 (Panel Data 버전)
# 단일 MASTER_FEATURES 데이블에 300개의 종목 데이터를 통합하여(Trade_Date, Ticker) 저장합니다.

import sys
from pathlib import Path
import pandas as pd

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

from DB import (
    StockDBManager,
    TICKERS,
    update_market_data,
    update_sp500_data,
    update_stock_data,
)
from Classification.Preprocessing.Macro.macro import get_market_features
from Classification.Preprocessing.Momentum.momentum import calculate_features
from Classification.Preprocessing.Volatility.volatility import calculate_risk_features
from Classification.Preprocessing.Volume.volume import calculate_volume_analysis

BASE_FEATURE_COLUMNS = [
    "Log_Ret_20",
    "Log_Ret_120",
    "MA_Envelope",
    "High_52W_Proximity",
    "RSI_14",
    "Volume_Ratio",
    "OBV_ROC_20",
    "DXY_Log_Return",
    "VIX_Close",
    "VIX_Log_Return",
    "SP500_1M_Return",
    "SP500_3M_Return",
]

COMMON_DYNAMIC_COLS = [
    "EWMA_Vol",
    "Vol_20d_Avg",
    "Vol_60d_Avg",
    "SP500_EWMA_Corr",
]

def _safe_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("-", "_")

def build_all_master_datasets(benchmark="SP500", auto_update=True):
    """
    모든 Ticker에 대해 피처를 계산하고, MASTER_FEATURES 테이블에 패널 데이터 형태로 병합/저장합니다.
    """
    benchmark = str(benchmark).upper()
    
    db = StockDBManager()
    db.connect()
    
    try:
        if auto_update:
            print("======================================================================")
            print("0. 기초 데이터 일괄 업데이트 (STOCK_DATA, SP500, MARKET_FEATURES)")
            print("======================================================================")
            try:
                update_stock_data()
                update_sp500_data()
                update_market_data()
            except Exception as e:
                print(f"기초 데이터 업데이트 중 오류 발생: {e}")
                
        print("\n======================================================================")
        print("1. MASTER_FEATURES 테이블 준비 (TRUNCATE)")
        print("======================================================================")
        # We start fresh on each run so we don't hold stale or duplicated logic
        db.truncate_master_features()
        
        # Pre-fetch Market Features since they are common to all
        df_dxy, df_vix, df_sp500_mom = get_market_features(db=db)
        
        total_tickers = len(TICKERS)
        success_count = 0
        
        print(f"\n======================================================================")
        print(f"2. 종목별 피처 계산 및 병합 시작 (총 {total_tickers} 종목)")
        print(f"======================================================================")

        for i, ticker in enumerate(TICKERS, 1):
            target_ticker = _safe_symbol(ticker)
            print(f"[{i}/{total_tickers}] '{ticker}' 피처 계산 중...")
            
            try:
                df_momentum = calculate_features(ticker, db=db)
                df_vol = calculate_volume_analysis(ticker, db=db)
                df_ticker_daily_vol, df_ticker_avg_vol, df_ticker_ewma_corr = calculate_risk_features(
                    ticker=ticker,
                    benchmark=benchmark,
                    db=db,
                )
                
                master_index = df_momentum.index
                if hasattr(master_index, "tz") and master_index.tz is not None:
                    master_index = master_index.tz_localize(None)

                master_df = pd.DataFrame(index=master_index)
                dfs_to_join = [
                    df_momentum,
                    df_vol,
                    df_dxy,
                    df_vix,
                    df_sp500_mom,
                    df_ticker_daily_vol,
                    df_ticker_avg_vol,
                    df_ticker_ewma_corr,
                ]

                for frame in dfs_to_join:
                    if hasattr(frame.index, "tz") and frame.index.tz is not None:
                        frame = frame.copy()
                        frame.index = frame.index.tz_localize(None)
                    frame = frame[~frame.index.duplicated(keep="first")]
                    master_df = master_df.join(frame, how="left")

                # Rename columns from TICKER_EWMA_Vol to generic EWMA_Vol
                rename_map = {
                    f"{target_ticker}_EWMA_Vol": "EWMA_Vol",
                    f"{target_ticker}_Vol_20d_Avg": "Vol_20d_Avg",
                    f"{target_ticker}_Vol_60d_Avg": "Vol_60d_Avg",
                    f"{target_ticker}_{benchmark}_EWMA_Corr": "SP500_EWMA_Corr"
                }
                master_df = master_df.rename(columns=rename_map)

                required_cols = BASE_FEATURE_COLUMNS + COMMON_DYNAMIC_COLS
                
                # Check for required columns
                missing = [c for c in required_cols if c not in master_df.columns]
                if missing:
                    print(f"  ❌ 필수 컬럼 누락: {missing}, 병합 건너뜀.")
                    continue

                # Filter and Fill
                master_df = master_df.ffill()
                master_df = master_df[required_cols].dropna(subset=required_cols)
                
                if master_df.empty:
                    print(f"  ⚠️ 데이터가 비어있습니다. 건너뜀.")
                    continue
                    
                # Add Ticker column
                master_df["TICKER"] = ticker
                master_df.index.name = "TRADE_DATE"
                master_df.reset_index(inplace=True)
                
                # Insert into DB
                db.insert_master_features(master_df)
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ '{ticker}' 처리 실패: {e}")

        print("\n======================================================================")
        print(f"3. DB 재구조화 (Reorganization)")
        print("======================================================================")
        db.reorganize_master_features()
        print(f"✅ 처리가 완료되었습니다. {total_tickers} 중 {success_count} 종목 저장 완료.")
        
    finally:
        db.close()

if __name__ == "__main__":
    build_all_master_datasets(auto_update=False)
