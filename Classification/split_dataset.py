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

import sys, os
from collections import namedtuple

# 모듈 경로 설정
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from common import pd, np
from sklearn.preprocessing import StandardScaler
from generate_target import generate_target

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
    "train_weights",   # ★ Train Sample Weights
    "val_weights",     # ★ Validation Sample Weights
    "final_train_weights", # ★ Final Train Sample Weights
    "feature_cols",    # Feature 컬럼 목록
    "scale_pos_weight", # Class 불균형 보정 비율 (Train 기준)
])

StrideSplit = namedtuple("StrideSplit", [
    "offset",           # 시작 오프셋 (0~4)
    "train",            # 스트라이드 샘플링된 Train
    "val",              # 스트라이드 샘플링된 Validation
    "final_train",      # 스트라이드 샘플링된 Train ∪ Val
    "train_weights",    # ★ Train Sample Weights
    "val_weights",      # ★ Validation Sample Weights
    "final_train_weights", # ★ Final Train Sample Weights
    "scale_pos_weight", # 해당 스트라이드의 scale_pos_weight
])


def _calc_spw(y):
    """scale_pos_weight 자동 산출"""
    n_neg = int((y == 0).sum())
    n_pos = max(int((y == 1).sum()), 1)
    return float(n_neg) / n_pos


def calculate_sample_weight(df):
    """
    샘플 가중치 계산 함수
    
    전략: "확실한 놈만 팬다"
    - AAPL과 SP500의 3개월 수익률 차이(Alpha)가 클수록 높은 가중치 부여
    - 방향성(Target_Class)이 명확한 날을 더 중요하게 학습하도록 유도
    """
    # 1. Alpha 절댓값 계산 (이미 Alpha_Diff 컬럼이 있다면 사용 가능하지만, 안전하게 다시 계산)
    #    Target_AAPL_3M, Target_SP500_3M 컬럼 필수
    if "Target_AAPL_3M" not in df.columns or "Target_SP500_3M" not in df.columns:
        return np.ones(len(df)) # 컬럼 없으면 가중치 1.0 (기본값)

    alpha_abs = np.abs(df["Target_AAPL_3M"] - df["Target_SP500_3M"])

    # 2. 가중치 스케일링 (Min-Max 정규화 후 +1)
    #    최소 가중치 1.0, 최대 가중치 2.0~3.0 정도가 되도록 설정
    #    너무 큰 가중치는 과적합 유발 가능성 있음
    w_min = alpha_abs.min()
    w_max = alpha_abs.max()
    
    if w_max == w_min:
        return np.ones(len(df))

    # 1.0 ~ 3.0 사이로 스케일링
    weights = 1.0 + 2.0 * (alpha_abs - w_min) / (w_max - w_min)
    
    return weights.values


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


def split_dataset():
    """
    generate_target()에서 피처+타겟 DataFrame을 받아 5단계 분할을 수행합니다.

    Returns:
        DataSplit namedtuple
    """
    # ── 데이터 로드 ──
    df = generate_target()

    # ── 피처 컬럼 추출 (Target 컬럼 제외) ──
    exclude_cols = ["Target_AAPL_3M", "Target_SP500_3M", "Target_Class", "Alpha_Diff"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

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

    # ── scale_pos_weight 자동 산출 (Train + Validation 기준) ──
    # ★ Final Fit 시에는 Train + Val 합쳐서 학습하므로, 비중도 합친 데이터 기준이어야 함
    spw = _calc_spw(final_train_df["Target_Class"])

    # ── Baseline 정책: 샘플 가중치 비활성화(모두 1.0)
    train_weights = pd.Series(np.ones(len(train_df), dtype=float), index=train_df.index)
    val_weights = pd.Series(np.ones(len(val_df), dtype=float), index=val_df.index)
    final_train_weights = pd.Series(np.ones(len(final_train_df), dtype=float), index=final_train_df.index)

    # ── Leakage 방지 검증
    _assert_no_leakage(train_df, val_df, test_df, final_train_df)
    print("  ✅ Leakage Check Passed (날짜 경계/교집합/격리구간)")

    # ── 결과 구성 ──
    split = DataSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        final_train=final_train_df,
        train_weights=train_weights,
        val_weights=val_weights,
        final_train_weights=final_train_weights,
        feature_cols=feature_cols,
        scale_pos_weight=spw,
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
        
        # Weights도 똑같이 샘플링
        train_w_sampled = split.train_weights.iloc[offset::STRIDE]
        val_w_sampled = split.val_weights.iloc[offset::STRIDE]

        # 2. Final Fit용 (동일 단일 스케일러가 적용된 split.final_train 기준)
        refit_sampled = split.final_train.iloc[offset::STRIDE]
        refit_w_sampled = split.final_train_weights.iloc[offset::STRIDE]

        # ★ Final Fit 기준 (Train + Val)
        spw = _calc_spw(refit_sampled["Target_Class"])

        ss = StrideSplit(
            offset=offset,
            train=train_sampled,
            val=val_sampled,
            final_train=refit_sampled,
            train_weights=train_w_sampled,
            val_weights=val_w_sampled,
            final_train_weights=refit_w_sampled,
            scale_pos_weight=spw,
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
    print(f"  scale_pos_weight (자동)    : {split.scale_pos_weight:.4f}")
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
              f"Refit {n_ref:>4d}건 | Win {tr_ratio:.1f}% | SPW {ss.scale_pos_weight:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    split = split_dataset()
    stride_splits = get_stride_splits(split)
