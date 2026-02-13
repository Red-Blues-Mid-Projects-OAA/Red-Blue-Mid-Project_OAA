"""
Target Variable 생성 모듈

Master DataFrame에 예측 타겟을 추가합니다:
  - Target_AAPL_3M  : AAPL 향후 60거래일 평균 일일 로그수익률
  - Target_SP500_3M : S&P500 향후 60거래일 평균 일일 로그수익률
  - Target_Class    : AAPL > SP500 이면 1, 아니면 0

데이터 소스 (DB, yfinance 미사용):
  - AAPL 로그 수익률  : LOG_RETURNS 테이블 (calculate_log_returns.py 결과)
  - S&P500 로그 수익률 : SP500_DATA 테이블 (update_sp500_data.py 결과)

★ DB 적재 없음
"""

import sys, os

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from common import pd, np

from build_master_dataset import build_master_dataset
from stock_db_manager import StockDBManager

FORWARD_DAYS = 60  # 3개월 ≈ 60거래일 (20d/60d/120d 규칙 통일)


def generate_target():
    """
    1. build_master_dataset()를 호출하여 통합 Feature DataFrame을 받고
    2. DB에서 AAPL/S&P500 로그수익률을 가져와 3개월 Forward Target을 생성
    3. 이진 분류 타겟 (AAPL이 시장을 이기는가?)을 추가하여 반환
    """
    # ──────────────────────────────────────────────────────────────
    #  1단계: Master Feature DataFrame 생성
    # ──────────────────────────────────────────────────────────────
    master_df = build_master_dataset()
    master_index = master_df.index

    # ──────────────────────────────────────────────────────────────
    #  2단계: DB에서 로그수익률 로드
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. Target Variable (정답지) 생성")
    print("=" * 70)

    print("  DB에서 로그 수익률 데이터 로드 중...")
    db = StockDBManager()
    db.connect()
    try:
        # AAPL 로그 수익률 (LOG_RETURNS 테이블)
        lr_all = db.fetch_log_returns()

        # S&P500 로그 수익률 (SP500_DATA 테이블)
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA ORDER BY TRADE_DATE"
        db.cursor.execute(query)
        sp500_rows = db.cursor.fetchall()
    finally:
        db.close()

    # AAPL 로그 수익률 → Master Index에 맞춤
    aapl_lr = lr_all["AAPL"].reindex(master_index).ffill()

    # S&P500 로그 수익률 → Master Index에 맞춤
    sp500_df = pd.DataFrame(sp500_rows, columns=["TRADE_DATE", "LOG_RETURN"])
    sp500_df = sp500_df.set_index("TRADE_DATE").sort_index()
    sp500_lr = sp500_df["LOG_RETURN"].astype(float).reindex(master_index).ffill()

    # ──────────────────────────────────────────────────────────────
    #  3단계: 향후 60거래일 평균 로그수익률 (Target)
    # ──────────────────────────────────────────────────────────────
    aapl_cumsum = aapl_lr.cumsum()
    sp500_cumsum = sp500_lr.cumsum()

    master_df["Target_AAPL_3M"] = (
        aapl_cumsum.shift(-FORWARD_DAYS) - aapl_cumsum
    ) / FORWARD_DAYS

    master_df["Target_SP500_3M"] = (
        sp500_cumsum.shift(-FORWARD_DAYS) - sp500_cumsum
    ) / FORWARD_DAYS

    # ★ 최종 이진 분류 타겟 (1: AAPL이 S&P500을 이김, 0: 짐)
    master_df["Target_Class"] = (
        master_df["Target_AAPL_3M"] > master_df["Target_SP500_3M"]
    ).astype(float)

    # Target이 NaN인 행(미래 미실현)의 Target_Class도 NaN으로 처리
    master_df.loc[master_df["Target_AAPL_3M"].isna(), "Target_Class"] = np.nan

    # ──────────────────────────────────────────────────────────────
    #  결과 출력
    # ──────────────────────────────────────────────────────────────
    target_realized = master_df["Target_Class"].dropna()
    target_nan = master_df["Target_Class"].isna().sum()

    print(f"\n  Target 컬럼 추가 완료 (Forward = {FORWARD_DAYS}거래일)")
    print(f"  타겟 실현 행 : {len(target_realized)}건")
    print(f"  타겟 미실현  : {target_nan}건 (최근 {FORWARD_DAYS}일, Inference 용도)")

    if len(target_realized) > 0:
        n_win = int(target_realized.sum())
        n_lose = len(target_realized) - n_win
        print(f"  Class 1 (AAPL > SP500) : {n_win}건 ({target_realized.mean()*100:.1f}%)")
        print(f"  Class 0 (AAPL ≤ SP500) : {n_lose}건 ({(1-target_realized.mean())*100:.1f}%)")

    print(f"\n  최근 5일 (Target이 NaN이면 정상 — Inference 용도):")
    print(master_df[["Target_AAPL_3M", "Target_SP500_3M", "Target_Class"]].tail())

    print(f"\n{'=' * 70}")
    print(f"★ 최종 Master DataFrame (피처 + 타겟)")
    print(f"{'=' * 70}")
    print(f"  Shape   : {master_df.shape}")
    print(f"  기간    : {master_df.index.min().date()} ~ {master_df.index.max().date()}")
    print(f"  전체 컬럼: {list(master_df.columns)}")

    return master_df


if __name__ == "__main__":
    df_final = generate_target()
