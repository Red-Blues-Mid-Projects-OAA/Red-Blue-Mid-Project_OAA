# 마스터 데이터프레임 병합 모듈 (패널 데이터 버전)
# 단일 MASTER_FEATURES 테이블에 300개 종목 데이터를 통합하여 (TRADE_DATE, TICKER) 기준으로 저장합니다.

import sys
import time
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

INCREMENTAL_LOOKBACK_DAYS_DEFAULT = 756
MIN_INCREMENTAL_LOOKBACK_DAYS = 252

def _safe_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("-", "_")


def _timer_now() -> float:
    return time.perf_counter()

def build_all_master_datasets(
    benchmark="SP500",
    auto_update=True,
    mode="auto",
    incremental_lookback_days=INCREMENTAL_LOOKBACK_DAYS_DEFAULT,
    legacy_incremental=False,
    dry_run=False,
):
    """
    모든 Ticker에 대해 피처를 계산하고, MASTER_FEATURES 테이블에 패널 데이터 형태로 병합/저장합니다.

    mode:
      - auto: MASTER_FEATURES 테이블이 비어있으면 full, 데이터가 있으면 incremental
      - full: 테이블 초기화 후 전체 기간 재적재
      - incremental: 기존 데이터의 최신 날짜 이후만 계산하여 추가 적재

    incremental_lookback_days:
      - incremental 최적화 모드에서 사용하는 워밍업 조회 기간(일)

    legacy_incremental:
      - True면 기존 방식처럼 전체 기간을 조회하여 계산
      - False면 cutoff 기준 룩백 기간만 조회하여 계산

    dry_run:
      - True면 DB 쓰기(TRUNCATE/UPSERT/REORGANIZE) 없이 계산 시간만 측정
    """
    benchmark = str(benchmark).upper()
    run_started = _timer_now()
    
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

        # 모드 결정: auto일 때 기존 데이터 유무에 따라 full/incremental 자동 선택
        effective_mode = str(mode).lower().strip()
        if effective_mode not in {"auto", "full", "incremental"}:
            raise ValueError("mode는 auto/full/incremental 중 하나여야 합니다.")
        cutoff_date = None  # incremental 모드에서 이 날짜 이후 데이터만 계산

        if effective_mode == "auto":
            latest_date = db.get_latest_master_features_date()
            if latest_date is None:
                effective_mode = "full"
                print("\n[auto] MASTER_FEATURES 테이블이 비어있어 full 모드로 전체 적재를 수행합니다.")
            else:
                # STOCK_DATA의 최신 날짜와 비교
                db.cursor.execute("SELECT MAX(TRADE_DATE) FROM STOCK_DATA")
                stock_latest = db.cursor.fetchone()[0]
                if stock_latest and stock_latest > latest_date:
                    effective_mode = "incremental"
                    cutoff_date = latest_date
                    print(f"\n[auto] MASTER_FEATURES 최신: {latest_date.strftime('%Y-%m-%d')}, "
                          f"STOCK_DATA 최신: {stock_latest.strftime('%Y-%m-%d')}")
                    print(f"  -> incremental 모드로 {latest_date.strftime('%Y-%m-%d')} 이후 데이터만 추가합니다.")
                else:
                    print(f"\n[auto] MASTER_FEATURES가 이미 최신 상태입니다. (최신: {latest_date.strftime('%Y-%m-%d')})")
                    elapsed = _timer_now() - run_started
                    print(f"[소요시간] 총 소요 시간: {elapsed:.2f}초")
                    return {
                        "status": "skipped",
                        "mode": effective_mode,
                        "elapsed_seconds": round(elapsed, 3),
                    }

        elif effective_mode == "incremental":
            latest_date = db.get_latest_master_features_date()
            if latest_date is None:
                print("\n[incremental] 기존 데이터가 없어 full 모드로 전환합니다.")
                effective_mode = "full"
            else:
                cutoff_date = latest_date
                print(f"\n[incremental] {latest_date.strftime('%Y-%m-%d')} 이후 데이터만 추가합니다.")

        # 재실행 시 이미 최신이면 조기 종료 (incremental/auto 공통)
        if cutoff_date is not None:
            db.cursor.execute("SELECT MAX(TRADE_DATE) FROM STOCK_DATA")
            stock_latest = db.cursor.fetchone()[0]
            if stock_latest is None or stock_latest <= cutoff_date:
                elapsed = _timer_now() - run_started
                latest_str = cutoff_date.strftime("%Y-%m-%d")
                stock_latest_str = (
                    stock_latest.strftime("%Y-%m-%d") if stock_latest is not None else "None"
                )
                print(
                    f"\n[skip] 추가할 최신 데이터가 없습니다. "
                    f"(MASTER_FEATURES 최신: {latest_str}, STOCK_DATA 최신: {stock_latest_str})"
                )
                print(f"[소요시간] 총 소요 시간: {elapsed:.2f}초")
                return {
                    "status": "skipped",
                    "mode": effective_mode,
                    "cutoff_date": latest_str,
                    "stock_latest_date": stock_latest_str,
                    "elapsed_seconds": round(elapsed, 3),
                }

        # incremental 최적화: 필요한 워밍업 구간만 조회
        lookback_days = max(int(incremental_lookback_days), MIN_INCREMENTAL_LOOKBACK_DAYS)
        data_start_date = None
        if cutoff_date is not None:
            if legacy_incremental:
                print("[incremental] 기존 방식 모드: 전체 기간을 조회하여 기존 방식으로 계산합니다.")
            else:
                data_start_date = (pd.Timestamp(cutoff_date) - pd.Timedelta(days=lookback_days)).date()
                print(
                    f"[incremental] 최적화 모드: cutoff={cutoff_date.strftime('%Y-%m-%d')}, "
                    f"lookback={lookback_days}일, start={data_start_date}"
                )

        if effective_mode == "full":
            print("\n======================================================================")
            print("1. MASTER_FEATURES 테이블 준비 (TRUNCATE → 전체 재적재)")
            print("======================================================================")
            if dry_run:
                print("[드라이런] TRUNCATE는 수행하지 않습니다.")
            else:
                db.truncate_master_features()
        else:
            print("\n======================================================================")
            print(f"1. MASTER_FEATURES 증분 적재 (cutoff: {cutoff_date.strftime('%Y-%m-%d')})")
            print("======================================================================")

        # 공통 Market Features 및 로그수익률 사전 조회
        shared_stage_started = _timer_now()
        df_dxy, df_vix, df_sp500_mom = get_market_features(db=db, start_date=data_start_date)

        print("\n[DB 최적화] 전체 종목 로그수익률 및 S&P500 데이터를 한 번에 로드합니다 (N+1 문제 방지)...")
        lr_all = db.fetch_log_returns(start_date=data_start_date)
        sp500_data = db.fetch_sp500_data(start_date=data_start_date)
        shared_stage_elapsed = _timer_now() - shared_stage_started
        print(f"[소요시간] 공통 데이터 로드: {shared_stage_elapsed:.2f}초")
        
        total_tickers = len(TICKERS)
        success_count = 0
        processed_tickers = 0
        prepared_rows = 0
        
        print(f"\n======================================================================")
        print(f"2. 종목별 피처 계산 및 병합 시작 (총 {total_tickers} 종목, 모드: {effective_mode})")
        print(f"======================================================================")

        ticker_stage_started = _timer_now()
        for i, ticker in enumerate(TICKERS, 1):
            target_ticker = _safe_symbol(ticker)
            print(f"[{i}/{total_tickers}] '{ticker}' 피처 계산 중...")
            
            try:
                # DB 접근 최적화: 개별 종목 데이터 1번만 로드하여 공유
                df_ticker_data = db.fetch_ticker_data(ticker, start_date=data_start_date)
                if df_ticker_data.empty:
                     print(f"  ⚠️ {ticker} 데이터가 없습니다. 건너뜀.")
                     continue
                     
                df_momentum = calculate_features(ticker, db=db, df_data=df_ticker_data)
                df_vol = calculate_volume_analysis(ticker, db=db, df_data=df_ticker_data)
                df_ticker_daily_vol, df_ticker_avg_vol, df_ticker_ewma_corr = calculate_risk_features(
                    ticker=ticker,
                    benchmark=benchmark,
                    db=db,
                    lr_all=lr_all,
                    sp500=sp500_data
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

                # 종목별 변동성 컬럼명을 공통 컬럼명으로 변환
                rename_map = {
                    f"{target_ticker}_EWMA_Vol": "EWMA_Vol",
                    f"{target_ticker}_Vol_20d_Avg": "Vol_20d_Avg",
                    f"{target_ticker}_Vol_60d_Avg": "Vol_60d_Avg",
                    f"{target_ticker}_{benchmark}_EWMA_Corr": "SP500_EWMA_Corr"
                }
                master_df = master_df.rename(columns=rename_map)

                required_cols = BASE_FEATURE_COLUMNS + COMMON_DYNAMIC_COLS
                
                # 필수 컬럼 존재 여부 점검
                missing = [c for c in required_cols if c not in master_df.columns]
                if missing:
                    print(f"  ❌ 필수 컬럼 누락: {missing}, 병합 건너뜀.")
                    continue

                # 결측값 보간 후 최종 필수 컬럼만 유지
                master_df = master_df.ffill()
                master_df = master_df[required_cols].dropna(subset=required_cols)
                
                if master_df.empty:
                    print(f"  ⚠️ 데이터가 비어있습니다. 건너뜀.")
                    continue

                # incremental 모드: cutoff_date 이후 데이터만 남김
                if cutoff_date is not None:
                    cutoff_ts = pd.Timestamp(cutoff_date)
                    master_df = master_df[master_df.index > cutoff_ts]
                    if master_df.empty:
                        # 이 종목은 새로 추가할 데이터 없음
                        continue
                    
                # TICKER 컬럼 추가 및 인덱스 정리
                master_df["TICKER"] = ticker
                master_df.index.name = "TRADE_DATE"
                master_df.reset_index(inplace=True)

                rows_to_write = len(master_df)
                processed_tickers += 1
                prepared_rows += rows_to_write

                # DB 적재 (Upsert 방식이라 중복 키는 갱신 처리)
                if dry_run:
                    print(f"  [드라이런] {rows_to_write}건 적재 예정.")
                else:
                    db.insert_master_features(master_df)
                success_count += 1
                
                if cutoff_date is not None and not dry_run:
                    print(f"  ✅ {rows_to_write}건 추가 적재 완료.")
                
            except Exception as e:
                print(f"  ❌ '{ticker}' 처리 실패: {e}")

        ticker_stage_elapsed = _timer_now() - ticker_stage_started
        print(f"[소요시간] 종목별 계산 단계: {ticker_stage_elapsed:.2f}초")

        print("\n======================================================================")
        print(f"3. DB 재구조화 (Reorganization)")
        print("======================================================================")
        if dry_run:
            print("[드라이런] REORGANIZE는 수행하지 않습니다.")
        else:
            db.reorganize_master_features()

        total_elapsed = _timer_now() - run_started
        print(
            f"처리가 완료되었습니다. {total_tickers} 중 {success_count} 종목 저장 완료 "
            f"(모드: {effective_mode}, 총 {total_elapsed:.2f}초)"
        )
        return {
            "status": "success",
            "mode": effective_mode,
            "cutoff_date": cutoff_date.strftime("%Y-%m-%d") if cutoff_date is not None else None,
            "data_start_date": str(data_start_date) if data_start_date is not None else None,
            "legacy_incremental": bool(legacy_incremental),
            "dry_run": bool(dry_run),
            "tickers_total": int(total_tickers),
            "tickers_processed": int(processed_tickers),
            "tickers_succeeded": int(success_count),
            "rows_prepared": int(prepared_rows),
            "timing_seconds": {
                "shared_data": round(shared_stage_elapsed, 3),
                "per_ticker": round(ticker_stage_elapsed, 3),
                "total": round(total_elapsed, 3),
            },
        }
        
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MASTER_FEATURES 패널 데이터 빌드/업데이트")
    parser.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"],
                        help="auto: 자동 판단, full: 전체 재적재, incremental: 증분 적재")
    parser.add_argument("--auto-update", action="store_true", default=False,
                        help="기초 데이터(STOCK, SP500, MARKET) 자동 업데이트 여부")
    parser.add_argument(
        "--incremental-lookback-days",
        type=int,
        default=INCREMENTAL_LOOKBACK_DAYS_DEFAULT,
        help=f"증분 최적화 워밍업 조회 기간(일, 최소 {MIN_INCREMENTAL_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--legacy-incremental",
        action="store_true",
        default=False,
        help="증분 모드에서 기존 방식(전체 기간 조회) 사용",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="DB 쓰기 없이 계산만 수행(시간 측정용)",
    )
    args = parser.parse_args()
    
    build_all_master_datasets(
        auto_update=args.auto_update,
        mode=args.mode,
        incremental_lookback_days=args.incremental_lookback_days,
        legacy_incremental=args.legacy_incremental,
        dry_run=args.dry_run,
    )



