"""
전처리(Preprocessing) 패키지 공개 인터페이스 모듈.

마스터 데이터셋 생성, 타깃 생성, 시계열 분할 함수를
외부 파이프라인이 공통 경로에서 호출할 수 있도록 집약합니다.
"""

from .generate_target import generate_target
from .split_dataset import N_MODELS, get_stride_splits, split_dataset

__all__ = [
    "generate_target",
    "split_dataset",
    "get_stride_splits",
    "N_MODELS",
]
