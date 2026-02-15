"""
LogisticRegression 파이프라인 진입점 래퍼.

기존 실행 경로 호환성을 유지하면서 내부 구현은 models/logreg/pipeline.py를 사용합니다.
"""

from Classification.models.logreg.pipeline import run_pipeline

__all__ = ["run_pipeline"]


if __name__ == "__main__":
    run_pipeline(auto_optimize=False, optimize_profile="balanced")
