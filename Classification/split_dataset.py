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
from generate_target import generate_target

# ══════════════════════════════════════════════════════════════
#  구간 정의
# ══════════════════════════════════════════════════════════════
SPLIT_CONFIG = {
    "burn_in":    ("2015-01-01", "2020-12-31"),
    "train":      ("2021-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2024-09-30"),
    "golden_gap": ("2024-10-01", "2024-12-31"),
    "test":       ("2025-01-01", None),  # None → 현재까지
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
    "scale_pos_weight", # Class 불균형 보정 비율 (Train 기준)
])

StrideSplit = namedtuple("StrideSplit", [
    "offset",           # 시작 오프셋 (0~4)
    "train",            # 스트라이드 샘플링된 Train
    "val",              # 스트라이드 샘플링된 Validation
    "final_train",      # 스트라이드 샘플링된 Train ∪ Val
    "scale_pos_weight", # 해당 스트라이드의 scale_pos_weight
])


def _calc_spw(y):
    """scale_pos_weight 자동 산출"""
    n_neg = int((y == 0).sum())
    n_pos = max(int((y == 1).sum()), 1)
    return float(n_neg) / n_pos


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

    # ── 구간 분할 ──
    s = SPLIT_CONFIG
    train_df = df_valid.loc[s["train"][0]:s["train"][1]]
    val_df = df_valid.loc[s["validation"][0]:s["validation"][1]]
    test_df = df_valid.loc[s["test"][0]:]

    # ── Final Refit용 (Train ∪ Validation) ──
    final_train_df = pd.concat([train_df, val_df])

    # ── scale_pos_weight 자동 산출 (Train + Validation 기준) ──
    # ★ Final Fit 시에는 Train + Val 합쳐서 학습하므로, 비중도 합친 데이터 기준이어야 함
    spw = _calc_spw(final_train_df["Target_Class"])

    # ── 결과 구성 ──
    split = DataSplit(
        train=train_df,
        val=val_df,
        test=test_df,
        final_train=final_train_df,
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
        train_sampled = split.train.iloc[offset::STRIDE]
        val_sampled = split.val.iloc[offset::STRIDE]
        refit_sampled = pd.concat([train_sampled, val_sampled])

        # ★ Final Fit 기준 (Train + Val)
        spw = _calc_spw(refit_sampled["Target_Class"])

        ss = StrideSplit(
            offset=offset,
            train=train_sampled,
            val=val_sampled,
            final_train=refit_sampled,
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
