"""
모델 공통 유틸리티 패키지.

현재는 Permutation 기반 중요도 계산 함수만 공개하며,
개별 모델(XGB/SVM/LogReg/RF)에서 동일한 방식으로
ΔIC 중요도 산출을 재사용할 수 있도록 진입점을 통일합니다.
"""

from .importance import compute_permutation_importance_ic

__all__ = ["compute_permutation_importance_ic"]
