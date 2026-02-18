"""
Target Variable 생성 모듈 (티커 파라미터화).
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

from Classification.Preprocessing.build_master_dataset import build_master_dataset
from DB import StockDBManager

FORWARD_DAYS = 60
ALPHA_MARGIN = 0.01


def generate_target(
    ticker="AAPL",
    benchmark="SP500",
    auto_update=True,
    persist_total_features_on_update=True,
    feature_source_mode="db_first",
):
    """
    종목별 3개월 Forward Target을 생성하여 master_df에 추가합니다.
    """
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    if benchmark != "SP500":
        raise ValueError(f"현재 benchmark는 SP500만 지원합니다: {benchmark}")

    target_col = f"Target_{ticker}_3M"
    benchmark_target_col = f"Target_{benchmark}_3M"

    master_df = build_master_dataset(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=auto_update,
        persist_total_features_on_update=persist_total_features_on_update,
        feature_source_mode=feature_source_mode,
    )
    master_index = master_df.index

    print("\n" + "=" * 70)
    print("3. Target Variable (정답지) 생성")
    print("=" * 70)
    print("  DB에서 로그 수익률 데이터 로드 중...")

    db = StockDBManager()
    db.connect()
    try:
        lr_all = db.fetch_log_returns()
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA ORDER BY TRADE_DATE"
        db.cursor.execute(query)
        sp500_rows = db.cursor.fetchall()
    finally:
        db.close()

    if lr_all.empty or ticker not in lr_all.columns:
        raise RuntimeError(f"LOG_RETURNS에 {ticker} 로그수익률 컬럼이 없습니다.")

    ticker_lr = pd.to_numeric(lr_all[ticker], errors="coerce").reindex(master_index).ffill()
    sp500_df = pd.DataFrame(sp500_rows, columns=["TRADE_DATE", "LOG_RETURN"])
    sp500_df = sp500_df.set_index("TRADE_DATE").sort_index()
    bench_lr = pd.to_numeric(sp500_df["LOG_RETURN"], errors="coerce").reindex(master_index).ffill()

    ticker_cumsum = ticker_lr.cumsum()
    bench_cumsum = bench_lr.cumsum()

    master_df[target_col] = ticker_cumsum.shift(-FORWARD_DAYS) - ticker_cumsum
    master_df[benchmark_target_col] = bench_cumsum.shift(-FORWARD_DAYS) - bench_cumsum
    master_df["Alpha_Diff"] = master_df[target_col] - master_df[benchmark_target_col]
    master_df["Target_Class"] = (master_df["Alpha_Diff"] > ALPHA_MARGIN).astype(float)
    master_df.loc[master_df[target_col].isna(), "Target_Class"] = np.nan

    target_realized = master_df["Target_Class"].dropna()
    target_nan = master_df["Target_Class"].isna().sum()

    print(f"\n  Target 컬럼 추가 완료 (Forward = {FORWARD_DAYS}거래일)")
    print(f"  Alpha Margin (ε) : {ALPHA_MARGIN} ({ALPHA_MARGIN*100:.1f}% 누적 초과수익률)")
    print(f"  타겟 실현 행 : {len(target_realized)}건")
    print(f"  타겟 미실현  : {target_nan}건 (최근 {FORWARD_DAYS}일, Inference 용도)")

    if len(target_realized) > 0:
        n_win = int(target_realized.sum())
        n_lose = len(target_realized) - n_win
        print(
            f"  Class 1 ({ticker} > {benchmark} + {ALPHA_MARGIN*100:.1f}%) : "
            f"{n_win}건 ({target_realized.mean()*100:.1f}%)"
        )
        print(f"  Class 0 (그 외)                              : {n_lose}건 ({(1-target_realized.mean())*100:.1f}%)")

        alpha_realized = master_df["Alpha_Diff"].dropna()
        print("\n  Alpha_Diff 분포:")
        print(f"    Mean   : {alpha_realized.mean()*100:.2f}%")
        print(f"    Median : {alpha_realized.median()*100:.2f}%")
        print(f"    Std    : {alpha_realized.std()*100:.2f}%")

    print("\n  최근 5일 (Target이 NaN이면 정상 — Inference 용도):")
    print(master_df[[target_col, benchmark_target_col, "Alpha_Diff", "Target_Class"]].tail())

    print(f"\n{'=' * 70}")
    print("★ 최종 Master DataFrame (피처 + 타겟)")
    print(f"{'=' * 70}")
    print(f"  Shape   : {master_df.shape}")
    print(f"  기간    : {master_df.index.min().date()} ~ {master_df.index.max().date()}")
    print(f"  전체 컬럼: {list(master_df.columns)}")

    return master_df


if __name__ == "__main__":
    df_final = generate_target()
