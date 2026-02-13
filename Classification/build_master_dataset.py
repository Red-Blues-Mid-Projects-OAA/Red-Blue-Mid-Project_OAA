"""
Master DataFrame 병합 모듈

4개 Feature 모듈의 결과를 하나의 통합 테이블로 조립합니다.

병합 3대 원칙:
  1. Master Index : AAPL 거래일 (NYSE 영업일) 기준
  2. Left Join    : AAPL 거래일에 나머지 데이터를 합침
  3. Forward Fill : 매크로 휴장일 등 빈칸은 직전 영업일 값으로 채움

"""

import sys, os

# 모듈 경로 설정 (Features/ 및 프로젝트 루트)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from common import pd, np
from sklearn.preprocessing import StandardScaler
from stock_db_manager import StockDBManager
from feature_engineering import calculate_features
from calculate_aapl_volume_ratio import calculate_aapl_volume_analysis
from prepare_market_features import get_market_features
from risk_volatility_features import calculate_risk_features


def build_master_dataset():
    """
    4개 Feature 모듈을 호출하고, 결과를 Master Calendar 기반으로 병합하여
    하나의 통합 Feature DataFrame을 반환합니다.
    ★ 모든 피처에 StandardScaling 적용
    """
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
    #  StandardScaling 적용 (전체 피처)
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2-1. StandardScaling 적용")
    print("=" * 70)

    feature_cols = list(master_df.columns)
    scaler = StandardScaler()
    master_df[feature_cols] = scaler.fit_transform(master_df[feature_cols])

    print(f"  스케일링 완료: {len(feature_cols)}개 피처")
    print(f"  스케일링 후 통계:")
    print(f"    Mean  범위: [{master_df[feature_cols].mean().min():.6f}, {master_df[feature_cols].mean().max():.6f}]")
    print(f"    Std   범위: [{master_df[feature_cols].std().min():.6f}, {master_df[feature_cols].std().max():.6f}]")

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
        
        # DB 재연결 (위에서 close() 했으므로)
        db.connect()
        try:
            # 1) 데이터 적재 (Upsert/Merge)
            print("  [1] TOTAL_FEATURES 테이블에 데이터 저장 중...")
            db.insert_total_features(master_df)
            
            # 2) 테이블 재구조화 (CTAS -> Rename)
            print("  [2] TOTAL_FEATURES 테이블 재구조화(Reorganization) 진행...")
            db.reorganize_total_features()
            
        except Exception as e:
            print(f"  🚨 DB 적재 중 오류 발생: {e}")
        finally:
            db.close()
            
    else:
        print("  WARNING: Master DataFrame is empty!")

    return master_df


if __name__ == "__main__":
    df_master = build_master_dataset()
