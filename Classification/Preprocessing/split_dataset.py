"""
데이터셋 분할 모듈 (데이터 허브)

5단계 데이터 분할 + 스트라이드 샘플링 앙상블 지원:
  1. Burn-in  (2015.01 ~ 2020.12) : 피처 안정화 (학습 미사용)
  2. Train    (2021.01 ~ 2023.12) : 모델 학습
  3. Validation (2024.01 ~ 2024.09) : Hyperparameter Tuning
  4. Golden Gap (2024.10 ~ 2024.12) : Leakage 방지 격리 구간
  5. Test     (2025.01 ~ 현재)     : 최종 평가

스트라이드 앙상블:
  STRIDE=5 (1주), offset 0~4로 5개의 독립적인 학습 세트를 생성.
  각 모델은 타겟이 거의 겹치지 않는 시퀀스를 학습하여 과적합을 해소합니다.

★ 다른 모듈들이 이 모듈을 import 하여 데이터를 가져갑니다.
"""
from collections import namedtuple

from common import pd
from sklearn.preprocessing import StandardScaler
from Classification.Preprocessing.generate_target import generate_target

# ══════════════════════════════════════════════════════════════
#  구간 정의
# ══════════════════════════════════════════════════════════════
SPLIT_CONFIG = {
    "burn_in":    ("2015-01-01", "2020-12-31"),
    "train":      ("2021-01-01", "2023-12-31"),
    "embargo":    ("2024-01-01", "2024-03-31"), # ★ 신규: 학습에서 버리는 구간 (Overlap 방지)
    "validation": ("2024-04-01", "2024-09-30"), # ★ 4월 1일부터 시작
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test":       ("2025-01-01", None),  # None → 현재까지
}

# 고정 분할 정책(요청사항)을 코드 수준에서 잠급니다.
EXPECTED_SPLIT_POLICY = {
    "train": ("2021-01-01", "2023-12-31"),
    "embargo": ("2024-01-01", "2024-03-31"),
    "validation": ("2024-04-01", "2024-09-30"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test": ("2025-01-01", None),
}

STRIDE = 5  # 5일 간격 (≈ 1주)
N_MODELS = 5  # offset 0~4

# 반환 구조체
DataSplit = namedtuple("DataSplit", [
    "train",           # Train DataFrame (전체)
    "val",             # Validation DataFrame (전체)
    "test",            # Test DataFrame (전체, 샘플링 안 함)
    "final_train",     # Train ∪ Validation (전체)
    "feature_cols",    # Feature 컬럼 목록
])

StrideSplit = namedtuple("StrideSplit", [
    "offset",           # 시작 오프셋 (0~4)
    "train",            # 스트라이드 샘플링된 Train
    "val",              # 스트라이드 샘플링된 Validation
    "final_train",      # 스트라이드 샘플링된 Train ∪ Val
])


def _assert_no_leakage(train_df, val_df, test_df, final_train_df):
    """
    데이터 누수 방지를 위해 날짜 경계/교집합/격리 구간 포함 여부를 검증합니다.
    """
    # 1) 시간 순서 경계 검증
    if len(train_df) > 0 and len(val_df) > 0:
        assert train_df.index.max() < val_df.index.min(), \
            "Leakage detected: Train 기간이 Validation과 겹치거나 역전되었습니다."
    if len(val_df) > 0 and len(test_df) > 0:
        assert val_df.index.max() < test_df.index.min(), \
            "Leakage detected: Validation 기간이 Test와 겹치거나 역전되었습니다."

    # 2) 집합 교집합 검증
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert len(train_idx & val_idx) == 0, "Leakage detected: Train/Validation 인덱스 교집합 존재"
    assert len(train_idx & test_idx) == 0, "Leakage detected: Train/Test 인덱스 교집합 존재"
    assert len(val_idx & test_idx) == 0, "Leakage detected: Validation/Test 인덱스 교집합 존재"

    # 3) 격리 구간 포함 여부 검증
    emb_start, emb_end = SPLIT_CONFIG["embargo"]
    gap_start, gap_end = SPLIT_CONFIG["golden_gap"]

    assert len(train_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Train에 embargo 구간 포함"
    assert len(val_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Validation에 embargo 구간 포함"
    assert len(test_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Test에 embargo 구간 포함"
    assert len(final_train_df.loc[emb_start:emb_end]) == 0, "Leakage detected: Final Train에 embargo 구간 포함"

    assert len(train_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Train에 golden gap 포함"
    assert len(val_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Validation에 golden gap 포함"
    assert len(test_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Test에 golden gap 포함"
    assert len(final_train_df.loc[gap_start:gap_end]) == 0, "Leakage detected: Final Train에 golden gap 포함"


def _assert_split_policy_locked():
    """
    분할 규칙이 합의된 고정 정책과 일치하는지 검증합니다.
    """
    for key, expected in EXPECTED_SPLIT_POLICY.items():
        actual = SPLIT_CONFIG.get(key)
        assert actual == expected, (
            f"Split policy mismatch: {key}={actual}, expected={expected}"
        )


def _print_split_policy():
    """현재 적용 중인 분할 경계를 명시적으로 출력합니다."""
    print("\n" + "=" * 70)
    print("4-0. 고정 분할 정책 확인")
    print("=" * 70)
    print(f"  Train      : {SPLIT_CONFIG['train'][0]} ~ {SPLIT_CONFIG['train'][1]}")
    print(f"  Embargo    : {SPLIT_CONFIG['embargo'][0]} ~ {SPLIT_CONFIG['embargo'][1]}")
    print(f"  Validation : {SPLIT_CONFIG['validation'][0]} ~ {SPLIT_CONFIG['validation'][1]}")
    print(f"  Golden Gap : {SPLIT_CONFIG['golden_gap'][0]} ~ {SPLIT_CONFIG['golden_gap'][1]}")
    print(f"  Test       : {SPLIT_CONFIG['test'][0]} ~ 현재")
    print("=" * 70)


def split_dataset(drop_features=None):
    """
    generate_target()에서 피처+타겟 DataFrame을 받아 5단계 분할을 수행합니다.

    Args:
        drop_features: 피처 컬럼에서 제외할 컬럼명 리스트.

    Returns:
        DataSplit namedtuple
    """
    # ── 데이터 로드 ──
    df = generate_target()

    # ── 피처 컬럼 추출 (Target 컬럼 제외) ──
    exclude_cols = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class", "Alpha_Diff"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    requested_drop = list(drop_features or [])
    unknown_drop = [c for c in requested_drop if c not in feature_cols]
    if unknown_drop:
        raise ValueError(
            f"drop_features에 존재하지 않는 컬럼이 포함되어 있습니다: {unknown_drop}"
        )
    if requested_drop:
        drop_set = set(requested_drop)
        feature_cols = [c for c in feature_cols if c not in drop_set]
        print(f"\n  drop_features 적용: {requested_drop}")
    print(f"  최종 feature_cols 수: {len(feature_cols)}")

    # ── 타겟 미실현(최근 60일) 제거 ──
    df_valid = df.dropna(subset=["Target_Class"])

    # ── 구간 분할 (Raw Data) ──
    s = SPLIT_CONFIG
    _assert_split_policy_locked()
    _print_split_policy()
    train_df = df_valid.loc[s["train"][0]:s["train"][1]].copy()
    val_df = df_valid.loc[s["validation"][0]:s["validation"][1]].copy()
    test_df = df_valid.loc[s["test"][0]:].copy()

    # ──────────────────────────────────────────────────────────────
    #  [Data Leakage 방지] 단일 스케일러 전략
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4-1. StandardScaling 적용 (Data Leakage 방지)")
    print("=" * 70)

    # Train 데이터에만 스케일러를 fit 하여 누수를 방지합니다.
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Final Train은 (Train + Validation)을 합치되, 이미 동일 스케일러로 변환된 데이터만 사용합니다.
    final_train_df = pd.concat([train_df, val_df], axis=0).sort_index()

    print("  ✅ Scaling Completed (Train-Fit 단일 스케일러 적용)")

    # ── Leakage 방지 검증
    _assert_no_leakage(train_df, val_df, test_df, final_train_df)
    print("  ✅ Leakage Check Passed (날짜 경계/교집합/격리구간)")

    # ── 결과 구성 ──
    split = DataSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        final_train=final_train_df,
        feature_cols=feature_cols,
    )

    print_split_summary(split)
    return split


def get_stride_splits(split):
    """
    5일 간격 스트라이드 샘플링으로 N_MODELS개의 독립적인 학습 세트를 생성합니다.

    각 모델은 offset이 다른 행만 사용:
      모델 0: 0, 5, 10, 15, ... 번째 행
      모델 1: 1, 6, 11, 16, ... 번째 행
      ...
      모델 4: 4, 9, 14, 19, ... 번째 행

    Args:
        split: DataSplit namedtuple

    Returns:
        List[StrideSplit] (N_MODELS개)
    """
    stride_splits = []

    for offset in range(N_MODELS):
        # 1. Tuning용 (단일 스케일러로 변환된 split.train/split.val 기준)
        train_sampled = split.train.iloc[offset::STRIDE]
        val_sampled = split.val.iloc[offset::STRIDE]
        
        # 2. Final Fit용 (동일 단일 스케일러가 적용된 split.final_train 기준)
        refit_sampled = split.final_train.iloc[offset::STRIDE]

        ss = StrideSplit(
            offset=offset,
            train=train_sampled,
            val=val_sampled,
            final_train=refit_sampled,
        )
        stride_splits.append(ss)

    print_stride_summary(stride_splits, split.feature_cols)
    return stride_splits


def print_split_summary(split):
    """분할 결과를 요약 출력합니다."""
    print("\n" + "=" * 70)
    print("4. 데이터셋 분할 (5단계)")
    print("=" * 70)

    datasets = [
        ("Train",      split.train),
        ("Validation", split.val),
        ("Test",       split.test),
    ]

    for name, df in datasets:
        n = len(df)
        if n > 0:
            period = f"{df.index.min().date()} ~ {df.index.max().date()}"
            n_win = int(df["Target_Class"].sum())
            n_lose = n - n_win
            ratio = df["Target_Class"].mean() * 100
            print(f"  {name:12s}: {n:>5d}건 | {period} | "
                  f"Win {n_win} ({ratio:.1f}%) / Lose {n_lose} ({100-ratio:.1f}%)")
        else:
            print(f"  {name:12s}: 0건")

    print(f"\n  Final Train (Train ∪ Val) : {len(split.final_train)}건")
    print(f"  피처 수                    : {len(split.feature_cols)}개")
    print("=" * 70)


def print_stride_summary(stride_splits, feature_cols):
    """스트라이드 샘플링 결과를 요약 출력합니다."""
    print("\n" + "=" * 70)
    print(f"  스트라이드 앙상블 ({N_MODELS}개 모델, STRIDE={STRIDE})")
    print("=" * 70)

    for ss in stride_splits:
        n_tr = len(ss.train)
        n_val = len(ss.val)
        n_ref = len(ss.final_train)
        tr_ratio = ss.train["Target_Class"].mean() * 100
        print(f"  모델 {ss.offset}: Train {n_tr:>4d}건 | Val {n_val:>3d}건 | "
              f"Refit {n_ref:>4d}건 | Win {tr_ratio:.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    split = split_dataset()
    stride_splits = get_stride_splits(split)
