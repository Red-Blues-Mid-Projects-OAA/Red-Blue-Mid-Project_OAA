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

from DB import StockDBManager

FORWARD_DAYS = 60
ALPHA_MARGIN = 0.01


def _coerce_logret_series(series_like, name: str) -> pd.Series | None:
    if series_like is None:
        return None
    if isinstance(series_like, pd.Series):
        s = series_like.copy()
    elif isinstance(series_like, pd.DataFrame):
        if "LOG_RETURN" in series_like.columns:
            s = series_like["LOG_RETURN"].copy()
        elif series_like.shape[1] == 1:
            s = series_like.iloc[:, 0].copy()
        else:
            raise ValueError(f"{name} DataFrame은 단일 컬럼이어야 합니다.")
    else:
        raise TypeError(f"{name}는 pd.Series 또는 pd.DataFrame이어야 합니다.")
    s.index = pd.to_datetime(s.index)
    return pd.to_numeric(s.sort_index(), errors="coerce")


def generate_target(
    ticker="AAPL",
    benchmark="SP500",
    auto_update=True,
    persist_total_features_on_update=True,
    feature_source_mode="db_first",
    ticker_logret_series=None,
    sp500_logret_series=None,
    master_df_override=None,
    db=None,
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

    input_ticker_lr = _coerce_logret_series(ticker_logret_series, "ticker_logret_series")
    input_sp500_lr = _coerce_logret_series(sp500_logret_series, "sp500_logret_series")

    owns_db = False
    db_conn = db
    if db_conn is None:
        db_conn = StockDBManager()
        db_conn.connect()
        owns_db = True
    elif db_conn.connection is None or db_conn.cursor is None:
        db_conn.connect(ensure_tables=False, quiet=True)
        owns_db = True

    try:
        if master_df_override is not None:
            master_df = master_df_override.copy()
            if "TRADE_DATE" in master_df.columns:
                master_df = master_df.set_index("TRADE_DATE")
            master_df.index = pd.to_datetime(master_df.index)
        else:
            master_df = db_conn.fetch_master_features(ticker)
            if master_df.empty:
                raise ValueError(
                    f"MASTER_FEATURES 테이블에 '{ticker}' 데이터가 없습니다. 파이프라인 Phase 2를 재실행하세요."
                )
            master_df = master_df.set_index("TRADE_DATE")

        master_index = pd.to_datetime(master_df.index)

        print("\n" + "=" * 70)
        print("3. Target Variable (정답지) 생성")
        print("=" * 70)
        print("  DB에서 로그 수익률 데이터 로드 중...")

        ticker_lr_raw = input_ticker_lr
        if ticker_lr_raw is None:
            ticker_lr_raw = db_conn.fetch_log_returns_by_ticker(ticker)

        sp500_lr_raw = input_sp500_lr
        if sp500_lr_raw is None:
            sp500_lr_raw = db_conn.fetch_sp500_log_returns()
    finally:
        if owns_db:
            db_conn.close()

    if ticker_lr_raw is None or len(ticker_lr_raw) == 0:
        raise RuntimeError(f"LOG_RETURNS에 {ticker} 로그수익률 시계열이 없습니다.")
    if sp500_lr_raw is None or len(sp500_lr_raw) == 0:
        raise RuntimeError("SP500_DATA에 LOG_RETURN 시계열이 없습니다.")

    ticker_lr = pd.to_numeric(ticker_lr_raw, errors="coerce").reindex(master_index).ffill()
    bench_lr = pd.to_numeric(sp500_lr_raw, errors="coerce").reindex(master_index).ffill()

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
