"""
Master DataFrame 병합 모듈

4개 Feature 모듈의 결과를 하나의 통합 테이블로 조립합니다.

병합 3대 원칙:
  1. Master Index : AAPL 거래일 (NYSE 영업일) 기준
  2. Left Join    : AAPL 거래일에 나머지 데이터를 합침
  3. Forward Fill : 매크로 휴장일 등 빈칸은 직전 영업일 값으로 채움

"""
from common import pd
from DB import (
    StockDBManager,
    TICKERS,
    update_market_data,
    update_sp500_data,
    update_stock_data,
)
from Classification.Preprocessing.Momentum.momentum import calculate_features
from Classification.Preprocessing.Volume.volume import calculate_aapl_volume_analysis
from Classification.Preprocessing.Macro.macro import get_market_features
from Classification.Preprocessing.Volatility.volatility import calculate_risk_features


REQUIRED_FEATURE_COLUMNS = [
    "Log_Ret_20",
    "Log_Ret_120",
    "MA_Envelope",
    "High_Low_Proximity",
    "RSI_14",
    "Volume_Ratio",
    "OBV_ROC_20",
    "DXY_Log_Return",
    "VIX_Close",
    "VIX_Log_Return",
    "SP500_1M_Return",
    "SP500_3M_Return",
    "AAPL_EWMA_Vol",
    "AAPL_Vol_20d_Avg",
    "AAPL_Vol_60d_Avg",
    "AAPL_SP500_EWMA_Corr",
]


def _to_date(value):
    """DB 조회 결과를 date로 통일합니다."""
    if value is None:
        return None
    return value.date() if hasattr(value, "date") else value


def _get_latest_snapshot(db):
    """
    DB의 최신 적재 상태를 조회합니다.
    - 주가: 티커별 최신일 + 최소 최신일
    - S&P500, VIX, DXY 최신일
    """
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
    """최신일 점검 결과를 로그로 출력합니다."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"  STOCK_DATA 최소 최신일 : {snapshot['stock_min_latest']}")
    print(f"  SP500_DATA 최신일      : {snapshot['sp500_latest']}")
    print(f"  MARKET(VIX) 최신일     : {snapshot['vix_latest']}")
    print(f"  MARKET(DXY) 최신일     : {snapshot['dxy_latest']}")


def _run_latest_date_updates():
    """
    최신일 기반 업데이트를 실행합니다.
    stale 임계치 없이 업데이트를 시도하고, 신규 데이터 건수를 집계합니다.
    """
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


def build_master_dataset(
    auto_update=True,
    persist_total_features_on_update=True,
    return_update_summary=False,
):
    """
    4개 Feature 모듈을 호출하고, 결과를 Master Calendar 기반으로 병합하여
    하나의 통합 Feature DataFrame을 반환합니다.

    Args:
        auto_update (bool): 최신일 기반 업데이트 수행 여부
        persist_total_features_on_update (bool): 신규 소스 데이터가 있을 때만 TOTAL_FEATURES 적재 여부
        return_update_summary (bool): True면 (master_df, update_summary)를 반환
    
    """
    # ──────────────────────────────────────────────────────────────
    #  0. 최신일 점검 + 업데이트 시도
    # ──────────────────────────────────────────────────────────────
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

    # TOTAL_FEATURES 스키마에서 제거 정책 피처를 정리합니다.
    db_for_cleanup = StockDBManager()
    db_for_cleanup.connect()
    try:
        print("\n" + "=" * 70)
        print("0-3. TOTAL_FEATURES 레거시 컬럼 정리")
        print("=" * 70)
        db_for_cleanup.drop_total_features_column_if_exists("LOG_RET_60")
    finally:
        db_for_cleanup.close()

    print("=" * 70)
    print("1. 개별 Feature 모듈 실행 및 데이터 수집")
    print("=" * 70)

    # DB 연결 (전체 과정에서 공유)
    db = StockDBManager()
    db.connect()

    try:
        # ── [1] 가격 모멘텀 및 트렌드 지표 (Master Index 소스) ──
        df_momentum = calculate_features("AAPL", db=db)

        # ── [2] 거래량 지표 ──
        df_vol = calculate_aapl_volume_analysis(db=db)

        # ── [3] 시장 매크로 지표 ──
        df_dxy, df_vix, df_sp500_mom = get_market_features(db=db)

        # ── [4] 리스크 지표 ──
        df_aapl_daily_vol, df_aapl_avg_vol, df_aapl_ewma_corr = calculate_risk_features(db=db)

    finally:
        # 모든 데이터 로드 및 피처 생성 후 연결 해제
        db.close()

    # ──────────────────────────────────────────────────────────────
    #  병합 시작
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. Master DataFrame 병합 (Master Index: AAPL Trading Days)")
    print("=" * 70)

    # AAPL의 영업일을 Master Index로 설정
    master_index = df_momentum.index
    # timezone-naive 보장
    if hasattr(master_index, "tz") and master_index.tz is not None:
        master_index = master_index.tz_localize(None)

    master_df = pd.DataFrame(index=master_index)

    dfs_to_join = [
        df_momentum,
        df_vol,
        df_dxy,
        df_vix,
        df_sp500_mom,
        df_aapl_daily_vol,
        df_aapl_avg_vol,
        df_aapl_ewma_corr,
    ]

    # Left Join 수행
    for df in dfs_to_join:
        # timezone-naive 보장
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        # 중복 인덱스 제거 (안전장치)
        df = df[~df.index.duplicated(keep="first")]
        master_df = master_df.join(df, how="left")

    # 매크로 휴장일 빈칸 → 직전 영업일 값으로 채움 (Look-ahead Bias 방지)
    master_df = master_df.ffill()

    # ──────────────────────────────────────────────────────────────
    print(f"\n★ Master DataFrame 생성 완료!")
    print(f"  Shape  : {master_df.shape}")
    
    if not master_df.empty:
        print(f"  기간   : {master_df.index.min().date()} ~ {master_df.index.max().date()}")
        print(f"  컬럼({len(master_df.columns)}개): {list(master_df.columns)}")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(f"\n  First 3 rows:")
        print(master_df.head(3))
        print(f"\n  Last 3 rows:")
        print(master_df.tail(3))
        
        # ──────────────────────────────────────────────────────────────
        # 3. DB 적재 및 Reorganization (TOTAL_FEATURES)
        # ──────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("3. DB 적재 및 Reorganization (TOTAL_FEATURES)")
        print("=" * 70)

        if persist_total_features_on_update:
            db.connect()
            try:
                if update_summary["updated_any"]:
                    print("  [1] TOTAL_FEATURES 테이블에 데이터 저장 중...")
                    db.insert_total_features(master_df)
                else:
                    print("  신규 소스 데이터가 없어 TOTAL_FEATURES 업서트를 생략합니다.")

                print("  [2] TOTAL_FEATURES 무결성 동기화 진행...")
                db.sync_total_features_integrity(
                    min_trade_date=master_df.index.min(),
                    max_trade_date=master_df.index.max(),
                    required_feature_cols=REQUIRED_FEATURE_COLUMNS,
                )

                print("  [3] TOTAL_FEATURES 테이블 재구조화(Reorganization) 진행...")
                db.reorganize_total_features()
            except Exception as e:
                print(f"  🚨 DB 적재/동기화 중 오류 발생: {e}")
            finally:
                db.close()
        else:
            print("  TOTAL_FEATURES 적재/동기화가 비활성화되어 작업을 생략합니다.")
            
    else:
        print("  WARNING: Master DataFrame is empty!")

    if return_update_summary:
        return master_df, update_summary

    return master_df


if __name__ == "__main__":
    df_master = build_master_dataset()
