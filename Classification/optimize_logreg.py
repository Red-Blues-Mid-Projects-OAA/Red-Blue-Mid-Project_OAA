"""
LogisticRegression 최적화 진입점 래퍼.

기존 실행 경로 호환성을 유지하면서 내부 구현은 models/logreg/optimize.py를 사용합니다.
"""

try:
    from Classification.models.logreg.optimize import optimize
except ModuleNotFoundError:
    from models.logreg.optimize import optimize

__all__ = ["optimize"]


if __name__ == "__main__":
    optimize(profile="balanced", n_trials=100)
