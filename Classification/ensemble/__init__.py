"""
이 파일은 ensemble 패키지의 공개 진입점을 모아 외부 import 경로를 단순하게 유지하는 초기화 파일입니다.
상위 배치나 실험 코드에서는 이 파일을 통해 앙상블 실행 함수만 안정적으로 가져가며, 내부 구현 파일 분리와 무관하게 동일한 진입점을 유지할 수 있습니다.
"""

__all__ = ["run_equal_weight_ensemble"]

def run_equal_weight_ensemble(*args, **kwargs):
    """equal 비중 ensemble 작업 전체를 순서대로 실행합니다."""
    from Classification.ensemble.ensemble import run_equal_weight_ensemble as _run

    return _run(*args, **kwargs)
