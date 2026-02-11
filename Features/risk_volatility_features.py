"""
AAPL EWMA 기반 리스크 피처 생성 모듈 (Self-contained)

calculate_ewma.py와 동일한 EWMA 방식(λ=0.94)을 사용하여:
  1. df_aapl_daily_vol    : AAPL 일별 EWMA 변동성 (√분산)
  2. df_aapl_avg_vol      : AAPL 20일/60일 평균 변동성
  3. df_aapl_ewma_corr    : AAPL–S&P500 일별 EWMA 상관계수

데이터 소스:
  - AAPL 로그 수익률 : LOG_RETURNS 테이블 (calculate_log_returns.py 결과)
  - S&P500 로그 수익률 : SP500_DATA 테이블 (update_sp500_data.py 결과)

★ DB 적재 없음 / 다른 모듈 수정 없음
"""

import numpy as np
import pandas as pd
import sys
import os

# 상위 디렉토리의 모듈을 import하기 위한 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stock_db_manager import StockDBManager

LAMBDA = 0.94
ALPHA = 1 - LAMBDA  # 0.06
TICKER = "AAPL"


def main():
    print("=" * 60)
    print("AAPL EWMA 리스크 피처 생성 (λ=0.94)")
    print("=" * 60)

    # ─── 데이터 로드 ───
    print("\n[데이터 로드]")

    db = StockDBManager()
    db.connect()
    try:
        # AAPL 로그 수익률 (LOG_RETURNS)
        lr_all = db.fetch_log_returns()

        # S&P 500 로그 수익률 (SP500_DATA)
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA ORDER BY TRADE_DATE"
        db.cursor.execute(query)
        rows = db.cursor.fetchall()
    finally:
        db.close()

    lr_all.index = pd.to_datetime(lr_all.index)
    aapl_lr = lr_all[TICKER].dropna()

    sp500 = pd.DataFrame(rows, columns=["TRADE_DATE", "LOG_RETURN"])
    sp500["TRADE_DATE"] = pd.to_datetime(sp500["TRADE_DATE"])
    sp500 = sp500.set_index("TRADE_DATE").sort_index()
    sp500_lr = sp500["LOG_RETURN"].astype(float).dropna()

    # 공통 거래일 정렬
    common = aapl_lr.index.intersection(sp500_lr.index).sort_values()
    aapl = aapl_lr.loc[common]
    spx = sp500_lr.loc[common]
    print(f"  AAPL: {len(aapl)}일 | S&P500: {len(spx)}일 | 공통: {len(common)}일")

    # ─── EWMA 공분산 행렬 계산 (일별, λ=0.94) ───
    combined = pd.DataFrame({"AAPL": aapl, "SP500": spx})
    ewma_cov = combined.ewm(alpha=ALPHA).cov()
    # ewma_cov: MultiIndex (date, ticker) × ticker

    # ═══════════════════════════════════════════════════════════
    # [1] AAPL 일별 EWMA 변동성
    # ═══════════════════════════════════════════════════════════
    # 공분산 행렬의 AAPL-AAPL 대각 원소 = AAPL 분산 → √ = 일일 변동성
    aapl_var = ewma_cov.loc[(slice(None), "AAPL"), "AAPL"]
    aapl_var.index = aapl_var.index.droplevel(1)
    aapl_vol_raw = np.sqrt(aapl_var)               # 일일 변동성 (상관계수 계산용)
    aapl_vol_ann = aapl_vol_raw * np.sqrt(252)      # 연율화 변동성 (출력용)

    df_aapl_daily_vol = pd.DataFrame({"AAPL_EWMA_Vol": aapl_vol_ann}).dropna()

    print(f"\n{'=' * 60}")
    print("[1] AAPL 일별 EWMA 변동성 (df_aapl_daily_vol)")
    print(f"{'=' * 60}")
    print(f"  기간: {df_aapl_daily_vol.index[0].date()} ~ {df_aapl_daily_vol.index[-1].date()}")
    print(f"  건수: {len(df_aapl_daily_vol)}")
    print(df_aapl_daily_vol.head(10))
    print("  ...")
    print(df_aapl_daily_vol.tail(5))

    # ═══════════════════════════════════════════════════════════
    # [2] 20일 / 60일 평균 변동성
    # ═══════════════════════════════════════════════════════════
    df_aapl_avg_vol = pd.DataFrame({
        "AAPL_Vol_20d_Avg": aapl_vol_ann.rolling(window=20).mean(),
        "AAPL_Vol_60d_Avg": aapl_vol_ann.rolling(window=60).mean()
    }).dropna()

    print(f"\n{'=' * 60}")
    print("[2] AAPL 20일/60일 평균 변동성 (df_aapl_avg_vol)")
    print(f"{'=' * 60}")
    print(f"  기간: {df_aapl_avg_vol.index[0].date()} ~ {df_aapl_avg_vol.index[-1].date()}")
    print(f"  건수: {len(df_aapl_avg_vol)}")
    print(df_aapl_avg_vol.head(10))
    print("  ...")
    print(df_aapl_avg_vol.tail(5))

    # ═══════════════════════════════════════════════════════════
    # [3] AAPL–S&P500 EWMA 상관계수
    # ═══════════════════════════════════════════════════════════
    # Corr = Cov(AAPL, SP500) / (σ_AAPL × σ_SP500)
    # ★ 상관계수 계산에는 반드시 일일(raw) 변동성 사용 (연율화 X)
    aapl_spx_cov = ewma_cov.loc[(slice(None), "AAPL"), "SP500"]
    aapl_spx_cov.index = aapl_spx_cov.index.droplevel(1)

    spx_var = ewma_cov.loc[(slice(None), "SP500"), "SP500"]
    spx_var.index = spx_var.index.droplevel(1)
    spx_vol_raw = np.sqrt(spx_var)

    ewma_corr = aapl_spx_cov / (aapl_vol_raw * spx_vol_raw)

    df_aapl_ewma_corr = pd.DataFrame({"AAPL_SP500_EWMA_Corr": ewma_corr}).dropna()

    print(f"\n{'=' * 60}")
    print("[3] AAPL–S&P500 EWMA 상관계수 (df_aapl_ewma_corr)")
    print(f"{'=' * 60}")
    print(f"  기간: {df_aapl_ewma_corr.index[0].date()} ~ {df_aapl_ewma_corr.index[-1].date()}")
    print(f"  건수: {len(df_aapl_ewma_corr)}")
    print(df_aapl_ewma_corr.head(10))
    print("  ...")
    print(df_aapl_ewma_corr.tail(5))

    # ─── 요약 ───
    print(f"\n{'=' * 60}")
    print("생성된 DataFrame 목록")
    print(f"{'=' * 60}")
    print(f"  1. df_aapl_daily_vol  : AAPL 일별 EWMA 변동성      ({len(df_aapl_daily_vol)}건)")
    print(f"  2. df_aapl_avg_vol    : AAPL 20d/60d 평균 변동성   ({len(df_aapl_avg_vol)}건)")
    print(f"  3. df_aapl_ewma_corr  : AAPL-S&P500 EWMA 상관계수 ({len(df_aapl_ewma_corr)}건)")

    return df_aapl_daily_vol, df_aapl_avg_vol, df_aapl_ewma_corr


if __name__ == "__main__":
    main()
