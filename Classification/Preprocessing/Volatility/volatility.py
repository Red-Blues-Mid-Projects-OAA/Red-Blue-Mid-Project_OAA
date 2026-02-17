"""
AAPL EWMA 기반 리스크 피처 생성 모듈 (Self-contained)

DB/calculate_ewma.py와 동일한 EWMA 방식(λ=0.94)을 사용하여:
  1. df_aapl_daily_vol    : AAPL 일별 EWMA 변동성 (√분산)
  2. df_aapl_avg_vol      : AAPL 20일/60일 평균 변동성
  3. df_aapl_ewma_corr    : AAPL–S&P500 일별 EWMA 상관계수

데이터 소스:
  - AAPL 로그 수익률 : LOG_RETURNS 테이블 (DB/calculate_log_returns.py 결과)
  - S&P500 로그 수익률 : SP500_DATA 테이블 (DB/update_sp500_data.py 결과)

★ DB 적재 없음 
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

from common import pd, np
from DB import StockDBManager

LAMBDA = 0.94
ALPHA = 1 - LAMBDA  # 0.06
TICKER = "AAPL"


def calculate_risk_features(db=None):
    print("=" * 60)
    print("AAPL EWMA 리스크 피처 생성 (λ=0.94)")
    print("=" * 60)

    # ─── 데이터 로드 ───
    print("\n[데이터 로드]")

    should_close = False
    if db is None:
        db = StockDBManager()
        db.connect()
        should_close = True
        
    try:
        # AAPL 로그 수익률 (LOG_RETURNS)
        lr_all = db.fetch_log_returns()

        # S&P 500 로그 수익률 (SP500_DATA)
        sp500 = db.fetch_sp500_data()
    finally:
        if should_close:
            db.close()

    aapl = lr_all[TICKER].dropna()

    if sp500.empty:
        print("S&P 500 데이터를 가져오지 못했습니다. DB/update_sp500_data.py를 먼저 실행하세요.")
        return

    spx = sp500["LOG_RETURN"].astype(float).dropna()

    # ─── EWMA 공분산 행렬 계산 (일별, λ=0.94) ───
    combined = pd.DataFrame({"AAPL": aapl, "SP500": spx})
    ewma_cov = combined.ewm(alpha=ALPHA).cov()
    # ewma_cov는 (날짜, 기준티커) 멀티 인덱스를 행으로 갖고,
    # 열은 비교 티커를 갖는 공분산 행렬 시계열 구조입니다.

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
    print(df_aapl_daily_vol.head())
    print("  ...")
    print(df_aapl_daily_vol.tail())

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
    print(df_aapl_avg_vol.head())
    print("  ...")
    print(df_aapl_avg_vol.tail())

    # ═══════════════════════════════════════════════════════════
    # [3] AAPL–S&P500 EWMA 상관계수
    # ═══════════════════════════════════════════════════════════
    # 상관계수(Correlation)는 공분산(Covariance)을
    # 각 자산 표준편차(σ_AAPL, σ_SP500)의 곱으로 정규화해 계산합니다.
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
    print(df_aapl_ewma_corr.head())
    print("  ...")
    print(df_aapl_ewma_corr.tail())

    # ─── 요약 ───
    print(f"\n{'=' * 60}")
    print("생성된 DataFrame 목록")
    print(f"{'=' * 60}")
    print(f"  1. df_aapl_daily_vol  : AAPL 일별 EWMA 변동성      ({len(df_aapl_daily_vol)}건)")
    print(f"  2. df_aapl_avg_vol    : AAPL 20d/60d 평균 변동성   ({len(df_aapl_avg_vol)}건)")
    print(f"  3. df_aapl_ewma_corr  : AAPL-S&P500 EWMA 상관계수 ({len(df_aapl_ewma_corr)}건)")

    return df_aapl_daily_vol, df_aapl_avg_vol, df_aapl_ewma_corr


if __name__ == "__main__":
    calculate_risk_features()
