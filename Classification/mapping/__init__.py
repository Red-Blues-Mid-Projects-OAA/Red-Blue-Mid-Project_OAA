"""
이 파일은 매핑 패키지의 공개 진입점을 모아 외부 import 경로를 단순하게 유지하는 초기화 파일입니다.
상위 배치에서는 이 파일을 통해 매핑 실행 함수를 직접 가져가며, 하위 모듈 재구성 시에도 호출 코드를 바꾸지 않도록 완충 역할을 합니다.
"""

__all__ = ["run_mapping"]

def run_mapping(*args, **kwargs):
    """매핑 작업 전체를 순서대로 실행합니다."""
    from Classification.mapping.mapping import run_mapping as _run

    return _run(*args, **kwargs)
