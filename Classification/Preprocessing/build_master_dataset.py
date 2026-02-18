"""
Master DataFrame 병합 모듈.

종목별 피처 테이블(`{TICKER}_TOTAL_FEATURES`)을 우선 소스로 사용하며,
없거나 무결성 실패 시 계산 후 DB에 적재합니다.
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

from common import pd
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


def _required_feature_columns(ticker: str, benchmark: str) -> list[str]:
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    return BASE_FEATURE_COLUMNS + [
        f"{ticker}_EWMA_Vol",
        f"{ticker}_Vol_20d_Avg",
        f"{ticker}_Vol_60d_Avg",
        f"{ticker}_{benchmark}_EWMA_Corr",
    ]


def _to_date(value):
    if value is None:
        return None
    return value.date() if hasattr(value, "date") else value


def _get_latest_snapshot(db):
    stock_latest_by_ticker = {}
    for ticker in TICKERS:
        stock_latest_by_ticker[ticker] = _to_date(db.get_latest_date(ticker))

    stock_dates = [d for d in stock_latest_by_ticker.values() if d is not None]
    stock_min_latest = min(stock_dates) if stock_dates else None

    return {
        "stock_min_latest": stock_min_latest,
        "stock_latest_by_ticker": stock_latest_by_ticker,
        "sp500_latest": _to_date(db.get_latest_sp500_date()),
        "vix_latest": _to_date(db.get_latest_market_date("VIX")),
        "dxy_latest": _to_date(db.get_latest_market_date("DXY")),
    }


def _print_latest_snapshot(snapshot, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"  STOCK_DATA 최소 최신일 : {snapshot['stock_min_latest']}")
    print(f"  SP500_DATA 최신일      : {snapshot['sp500_latest']}")
    print(f"  MARKET(VIX) 최신일     : {snapshot['vix_latest']}")
    print(f"  MARKET(DXY) 최신일     : {snapshot['dxy_latest']}")


def _run_latest_date_updates():
    print("\n" + "=" * 70)
    print("0. 최신일 기반 소스 데이터 업데이트 시도")
    print("=" * 70)

    stock_result = {"new_rows": 0}
    sp500_result = {"new_rows": 0}
    market_result = {"new_rows_total": 0}

    try:
        stock_result = update_stock_data()
    except Exception as e:
        print(f"  🚨 STOCK_DATA 업데이트 실행 실패: {e}")
    if not isinstance(stock_result, dict):
        stock_result = {"new_rows": 0}

    try:
        sp500_result = update_sp500_data()
    except Exception as e:
        print(f"  🚨 SP500_DATA 업데이트 실행 실패: {e}")
    if not isinstance(sp500_result, dict):
        sp500_result = {"new_rows": 0}

    try:
        market_result = update_market_data()
    except Exception as e:
        print(f"  🚨 MARKET_FEATURES 업데이트 실행 실패: {e}")
    if not isinstance(market_result, dict):
        market_result = {"new_rows_total": 0}

    stock_new = int(stock_result.get("new_rows", 0))
    sp500_new = int(sp500_result.get("new_rows", 0))
    market_new = int(market_result.get("new_rows_total", 0))
    total_new = stock_new + sp500_new + market_new

    print("\n  [업데이트 결과 요약]")
    print(f"    STOCK_DATA   신규: {stock_new}건")
    print(f"    SP500_DATA   신규: {sp500_new}건")
    print(f"    MARKET_FEAT  신규: {market_new}건")
    print(f"    TOTAL 신규 건수  : {total_new}건")

    return {
        "stock_new_rows": stock_new,
        "sp500_new_rows": sp500_new,
        "market_new_rows": market_new,
        "total_new_rows": total_new,
        "new_rows": total_new,
        "updated_any": total_new > 0,
    }


def _canonicalize_feature_columns(df: pd.DataFrame, required_feature_cols: list[str]) -> pd.DataFrame:
    """DB 조회 결과의 대문자 컬럼명을 프로젝트 표준 컬럼명으로 복원합니다."""
    canonical = {col.upper(): col for col in required_feature_cols}
    rename_map = {}
    for col in df.columns:
        upper = str(col).upper()
        if upper in canonical:
            rename_map[col] = canonical[upper]
    return df.rename(columns=rename_map)


def _try_load_features_from_db(db: StockDBManager, table_name: str, required_feature_cols: list[str]):
    df_db = db.fetch_features_table(table_name)
    if df_db.empty:
        return None

    df_db = _canonicalize_feature_columns(df_db, required_feature_cols)
    missing = [c for c in required_feature_cols if c not in df_db.columns]
    if missing:
        print(f"  [DB-FIRST] {table_name} 필수 컬럼 누락: {missing}")
        return None

    df_db = df_db.sort_index()[required_feature_cols]
    if df_db.isnull().any(axis=1).any():
        print(f"  [DB-FIRST] {table_name}에 NULL 행이 있어 재계산합니다.")
        return None

    print(f"  [DB-FIRST] {table_name}에서 피처 로드 완료: {len(df_db)}건")
    return df_db


def build_master_dataset(
    ticker="AAPL",
    benchmark="SP500",
    auto_update=True,
    persist_total_features_on_update=True,
    feature_source_mode="db_first",
    return_update_summary=False,
):
    """
    종목별 피처를 생성/조회하여 Master Feature DataFrame을 반환합니다.

    Args:
        ticker: 예측 대상 티커
        benchmark: 벤치마크 티커(현재 SP500 고정)
        auto_update: 최신일 기준 원천 데이터 업데이트 수행 여부
        persist_total_features_on_update: 피처 테이블 적재/동기화 수행 여부
        feature_source_mode: "db_first" 또는 "compute"
        return_update_summary: True면 (master_df, update_summary) 반환
    """
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")

    required_feature_cols = _required_feature_columns(ticker, benchmark)

    db_for_check = StockDBManager()
    db_for_check.connect()
    try:
        snapshot_before = _get_latest_snapshot(db_for_check)
    finally:
        db_for_check.close()

    _print_latest_snapshot(snapshot_before, "0-1. 업데이트 전 최신일 점검")

    update_summary = {
        "stock_new_rows": 0,
        "sp500_new_rows": 0,
        "market_new_rows": 0,
        "total_new_rows": 0,
        "new_rows": 0,
        "updated_any": False,
    }
    if auto_update:
        update_summary = _run_latest_date_updates()
    else:
        print("\n자동 업데이트를 비활성화하여 최신일 점검만 수행했습니다.")

    db_for_check = StockDBManager()
    db_for_check.connect()
    try:
        snapshot_after = _get_latest_snapshot(db_for_check)
    finally:
        db_for_check.close()

    _print_latest_snapshot(snapshot_after, "0-2. 업데이트 후 최신일 점검")

    table_name = StockDBManager().get_total_features_table_name(ticker)

    if feature_source_mode not in {"db_first", "compute"}:
        raise ValueError(f"지원하지 않는 feature_source_mode 입니다: {feature_source_mode}")

    db = StockDBManager()
    db.connect()
    try:
        if feature_source_mode == "db_first" and not update_summary["updated_any"]:
            loaded = _try_load_features_from_db(db, table_name, required_feature_cols)
            if loaded is not None:
                if return_update_summary:
                    return loaded, update_summary
                return loaded

        print("=" * 70)
        print("1. 개별 Feature 모듈 실행 및 데이터 수집")
        print("=" * 70)

        df_momentum = calculate_features(ticker, db=db)
        df_vol = calculate_volume_analysis(ticker=ticker, db=db)
        df_dxy, df_vix, df_sp500_mom = get_market_features(db=db)
        df_ticker_daily_vol, df_ticker_avg_vol, df_ticker_ewma_corr = calculate_risk_features(
            ticker=ticker,
            benchmark=benchmark,
            db=db,
        )

        print("\n" + "=" * 70)
        print(f"2. Master DataFrame 병합 (Master Index: {ticker} Trading Days)")
        print("=" * 70)

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

        master_df = master_df.ffill()
        master_df = master_df[required_feature_cols].dropna(subset=required_feature_cols)

        print(f"\n★ Master DataFrame 생성 완료! ({ticker})")
        print(f"  Shape  : {master_df.shape}")
        if not master_df.empty:
            print(f"  기간   : {master_df.index.min().date()} ~ {master_df.index.max().date()}")
            print(f"  컬럼({len(master_df.columns)}개): {list(master_df.columns)}")

        if persist_total_features_on_update and not master_df.empty:
            print("\n" + "=" * 70)
            print(f"3. DB 적재 및 Reorganization ({table_name})")
            print("=" * 70)
            try:
                print(f"  [1] {table_name} 테이블에 데이터 저장 중...")
                db.insert_features_table(master_df, table_name)

                print(f"  [2] {table_name} 무결성 동기화 진행...")
                db.sync_features_table_integrity(
                    table_name=table_name,
                    min_trade_date=master_df.index.min(),
                    max_trade_date=master_df.index.max(),
                    required_feature_cols=required_feature_cols,
                )

                print(f"  [3] {table_name} 테이블 재구조화(Reorganization) 진행...")
                db.reorganize_features_table(table_name)
            except Exception as e:
                print(f"  🚨 DB 적재/동기화 중 오류 발생: {e}")
        elif not persist_total_features_on_update:
            print("  피처 테이블 적재/동기화 비활성화: DB 저장 생략")

        if return_update_summary:
            return master_df, update_summary
        return master_df
    finally:
        db.close()


if __name__ == "__main__":
    df_master = build_master_dataset()
