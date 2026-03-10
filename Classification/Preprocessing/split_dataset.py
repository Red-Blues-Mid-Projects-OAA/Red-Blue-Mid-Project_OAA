"""
이 파일은 분할 데이터셋 관련 작업을 담당합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from collections import namedtuple
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
from sklearn.preprocessing import StandardScaler

from Classification.Preprocessing.generate_target import generate_target
from DB import StockDBManager

SPLIT_CONFIG = {
    "burn_in": ("2015-01-01", "2020-12-31"),
    "train": ("2021-01-01", "2023-12-31"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}

EXPECTED_SPLIT_POLICY = {
    "train": ("2021-01-01", "2023-12-31"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}

STRIDE = 5
N_MODELS = 5

DataSplit = namedtuple(
    "DataSplit",
    [
        "train",
        "val",
        "test",
        "final_train",
        "feature_cols",
        "ticker",
        "benchmark",
        "target_col",
        "benchmark_target_col",
        "feature_table_name",
    ],
)

StrideSplit = namedtuple(
    "StrideSplit",
    [
        "offset",
        "train",
        "val",
        "final_train",
    ],
)

def _assert_no_leakage(train_df, val_df, test_df, final_train_df):
    """데이터 분할에 미래 정보 누수가 없는지 검증합니다."""
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df.index.max() < val_df.index.min(), "Leakage detected: Train/Validation 경계 위반"
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df.index.max() < test_df.index.min(), "Leakage detected: Validation/Test 경계 위반"

    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert len(train_idx & val_idx) == 0, "Leakage detected: Train/Validation 교집합 존재"
    assert len(train_idx & test_idx) == 0, "Leakage detected: Train/Test 교집합 존재"
    assert len(val_idx & test_idx) == 0, "Leakage detected: Validation/Test 교집합 존재"

    emb_start, emb_end = SPLIT_CONFIG["embargo"]
    gap_start, gap_end = SPLIT_CONFIG["golden_gap"]

    assert len(train_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Train에 embargo 포함"
    assert len(val_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Validation에 embargo 포함"
    assert len(test_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Test에 embargo 포함"
    assert len(final_train_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Final Train에 embargo 포함"

    assert len(train_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Train에 golden gap 포함"
    assert len(val_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Validation에 golden gap 포함"
    assert len(test_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Test에 golden gap 포함"
    assert len(final_train_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Final Train에 golden gap 포함"

def _assert_split_policy_locked():
    """현재 데이터 분할 정책이 의도한 설정과 일치하는지 검증합니다."""
    for key, expected in EXPECTED_SPLIT_POLICY.items():
        actual = SPLIT_CONFIG.get(key)
        assert actual == expected, f"Split policy mismatch: {key}={actual}, expected={expected}"

def _print_split_policy():
    """현재 데이터 분할 정책을 로그로 출력합니다."""
    print("\n" + "=" * 70)
    print("4-0. 고정 분할 정책 확인")
    print("=" * 70)
    print(f"  Train      : {SPLIT_CONFIG['train'][0]} ~ {SPLIT_CONFIG['train'][1]}")
    print(f"  Embargo    : {SPLIT_CONFIG['embargo'][0]} ~ {SPLIT_CONFIG['embargo'][1]}")
    print(f"  Validation : {SPLIT_CONFIG['validation'][0]} ~ {SPLIT_CONFIG['validation'][1]}")
    print(f"  Golden Gap : {SPLIT_CONFIG['golden_gap'][0]} ~ {SPLIT_CONFIG['golden_gap'][1]}")
    print(f"  Test       : {SPLIT_CONFIG['test'][0]} ~ 현재")
    print("=" * 70)

def split_dataset(
    ticker="AAPL",
    benchmark="SP500",
    drop_features=None,
    auto_update=True,
    persist_total_features_on_update=True,
    feature_source_mode="db_first",
    cached_ticker_logret=None,
    cached_sp500_logret=None,
    master_df_override=None,
    db=None,
):
    """데이터셋를 기준에 따라 나눕니다."""
    ticker = str(ticker).upper()
    benchmark = str(benchmark).upper()
    target_col = f"Target_{ticker}_3M"
    benchmark_target_col = f"Target_{benchmark}_3M"
    feature_table_name = "MASTER_FEATURES"

    df = generate_target(
        ticker=ticker,
        benchmark=benchmark,
        auto_update=auto_update,
        persist_total_features_on_update=persist_total_features_on_update,
        feature_source_mode=feature_source_mode,
        ticker_logret_series=cached_ticker_logret,
        sp500_logret_series=cached_sp500_logret,
        master_df_override=master_df_override,
        db=db,
    )

    exclude_cols = [target_col, benchmark_target_col, "Target_Class", "Alpha_Diff", "TICKER"]
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    requested_drop = list(drop_features or [])
    unknown_drop = [c for c in requested_drop if c not in feature_cols]
    if unknown_drop:
        raise ValueError(f"drop_features에 존재하지 않는 컬럼이 포함되어 있습니다: {unknown_drop}")
    if requested_drop:
        drop_set = set(requested_drop)
        feature_cols = [c for c in feature_cols if c not in drop_set]
        print(f"\n  drop_features 적용: {requested_drop}")

    print(f"  최종 feature_cols 수: {len(feature_cols)}")
    df_valid = df.dropna(subset=["Target_Class"])

    s = SPLIT_CONFIG
    _assert_split_policy_locked()
    _print_split_policy()

    train_df = df_valid.loc[s["train"][0]:s["train"][1]].copy()
    val_df = df_valid.loc[s["validation"][0]:s["validation"][1]].copy()
    test_df = df_valid.loc[s["test"][0]:].copy()

    print("\n" + "=" * 70)
    print("4-1. StandardScaling 적용 (Data Leakage 방지)")
    print("=" * 70)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    final_train_df = pd.concat([train_df, val_df], axis=0).sort_index()
    print("  ✅ Scaling Completed (Train-Fit 단일 스케일러 적용)")
    _assert_no_leakage(train_df, val_df, test_df, final_train_df)
    print("  ✅ Leakage Check Passed (날짜 경계/교집합/격리구간)")

    split = DataSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        final_train=final_train_df,
        feature_cols=feature_cols,
        ticker=ticker,
        benchmark=benchmark,
        target_col=target_col,
        benchmark_target_col=benchmark_target_col,
        feature_table_name=feature_table_name,
    )
    print_split_summary(split)
    return split

def get_stride_splits(split):
    """stride splits 정보를 조회해 반환합니다."""
    stride_splits = []
    for offset in range(N_MODELS):
        train_sampled = split.train.iloc[offset::STRIDE]
        val_sampled = split.val.iloc[offset::STRIDE]
        refit_sampled = split.final_train.iloc[offset::STRIDE]
        stride_splits.append(
            StrideSplit(
                offset=offset,
                train=train_sampled,
                val=val_sampled,
                final_train=refit_sampled,
            )
        )

    print_stride_summary(stride_splits, split.feature_cols, split.target_col)
    return stride_splits

def print_split_summary(split):
    """데이터 분할 결과 요약을 로그로 출력합니다."""
    print("\n" + "=" * 70)
    print("4. 데이터셋 분할 (5단계)")
    print("=" * 70)
    datasets = [("Train", split.train), ("Validation", split.val), ("Test", split.test)]
    target_col = "Target_Class"
    for name, df in datasets:
        n = len(df)
        if n > 0:
            period = f"{df.index.min().date()} ~ {df.index.max().date()}"
            n_win = int(df[target_col].sum())
            n_lose = n - n_win
            ratio = df[target_col].mean() * 100
            print(
                f"  {name:12s}: {n:>5d}건 | {period} | "
                f"Win {n_win} ({ratio:.1f}%) / Lose {n_lose} ({100-ratio:.1f}%)"
            )
        else:
            print(f"  {name:12s}: 0건")
    print(f"\n  Final Train (Train ∪ Val) : {len(split.final_train)}건")
    print(f"  피처 수                    : {len(split.feature_cols)}개")
    print(f"  Ticker/Benchmark           : {split.ticker}/{split.benchmark}")
    print(f"  Feature Table              : {split.feature_table_name}")
    print("=" * 70)

def print_stride_summary(stride_splits, feature_cols, target_col):
    """스트라이드 설정 요약을 로그로 출력합니다."""
    print("\n" + "=" * 70)
    print(f"  스트라이드 앙상블 ({N_MODELS}개 모델, STRIDE={STRIDE})")
    print("=" * 70)
    for ss in stride_splits:
        n_tr = len(ss.train)
        n_val = len(ss.val)
        n_ref = len(ss.final_train)
        tr_ratio = ss.train[target_col].mean() * 100
        print(
            f"  모델 {ss.offset}: Train {n_tr:>4d}건 | Val {n_val:>3d}건 | "
            f"Refit {n_ref:>4d}건 | Win {tr_ratio:.1f}%"
        )
    print("=" * 70)

if __name__ == "__main__":
    split = split_dataset()
    stride_splits = get_stride_splits(split)
