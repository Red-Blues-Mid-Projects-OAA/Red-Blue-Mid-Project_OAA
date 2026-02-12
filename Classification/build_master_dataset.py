"""
Master DataFrame 병합 모듈

4개 Feature 모듈의 결과를 하나의 통합 테이블로 조립합니다.

병합 3대 원칙:
  1. Master Index : AAPL 거래일 (NYSE 영업일) 기준
  2. Left Join    : AAPL 거래일에 나머지 데이터를 합침
  3. Forward Fill : 매크로 휴장일 등 빈칸은 직전 영업일 값으로 채움

★ DB 적재 없음 / Features 폴더 외 파일 수정 없음
"""

import sys, os

# 모듈 경로 설정 (Features/ 및 프로젝트 루트)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from common import pd, np
from stock_db_manager import StockDBManager
from feature_engineering import calculate_features
from calculate_aapl_volume_ratio import calculate_aapl_volume_analysis
from prepare_market_features import (
    fetch_dollar_index_log_returns,
    fetch_vix_data,
    calculate_sp500_momentum,
)
from risk_volatility_features import main as calc_risk_features


def build_master_dataset():
    """
    4개 Feature 모듈을 호출하고, 결과를 Master Calendar 기반으로 병합하여
    하나의 통합 Feature DataFrame을 반환합니다.
    """
    print("=" * 70)
    print("1. 개별 Feature 모듈 실행 및 데이터 수집")
    print("=" * 70)

    # ── [1] 기술적 지표 (Master Index 소스) ──
    df_tech = calculate_features("AAPL")

    # ── [2] 거래량 지표 ──
    df_vol = calculate_aapl_volume_analysis()

    # ── [3] 시장 매크로 지표 (DB 연결 공유) ──
    db = StockDBManager()
    db.connect()
    try:
        df_dxy = fetch_dollar_index_log_returns(db)
        df_vix = fetch_vix_data(db)
        df_sp500_mom = calculate_sp500_momentum(db)
    finally:
        db.close()

    # ── [4] 리스크 지표 (Tuple 3개) ──
    df_aapl_daily_vol, df_aapl_avg_vol, df_aapl_ewma_corr = calc_risk_features()

    # ──────────────────────────────────────────────────────────────
    #  병합 시작
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. Master DataFrame 병합 (Master Index: AAPL Trading Days)")
    print("=" * 70)

    # AAPL의 영업일을 Master Index로 설정
    master_index = df_tech.index
    # timezone-naive 보장
    if hasattr(master_index, "tz") and master_index.tz is not None:
        master_index = master_index.tz_localize(None)

    master_df = pd.DataFrame(index=master_index)

    dfs_to_join = [
        df_tech,
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

    # ★ 핵심: 매크로 휴장일 빈칸 → 직전 영업일 값으로 채움 (Look-ahead Bias 방지)
    master_df = master_df.ffill()

    # Burn-in 기간의 NaN 제거 (rolling window 초기값)
    burn_in_cols = ["Log_Ret_120", "Volume_Ratio", "AAPL_EWMA_Vol"]
    existing = [c for c in burn_in_cols if c in master_df.columns]
    if existing:
        master_df = master_df.dropna(subset=existing)

    # ──────────────────────────────────────────────────────────────
    print(f"\n★ Master DataFrame 생성 완료!")
    print(f"  Shape  : {master_df.shape}")
    print(f"  기간   : {master_df.index.min().date()} ~ {master_df.index.max().date()}")
    print(f"  컬럼({len(master_df.columns)}개): {list(master_df.columns)}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\n  First 3 rows:")
    print(master_df.head(3))
    print(f"\n  Last 3 rows:")
    print(master_df.tail(3))

    return master_df


if __name__ == "__main__":
    df_master = build_master_dataset()
