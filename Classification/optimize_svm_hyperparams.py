"""
SVM 하이퍼파라미터 튜닝 진입점 래퍼.

기존 실행 경로 호환성을 유지하면서 내부 구현은 models/svm/optimize.py를 사용합니다.
"""

from Classification.models.svm.optimize import optimize

__all__ = ["optimize"]


if __name__ == "__main__":
    optimize()
