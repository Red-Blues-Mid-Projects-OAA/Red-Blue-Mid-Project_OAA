# 한글 주석: 이 패키지의 공개 모듈/함수 사용을 위한 초기화 파일입니다.
"""Alpha mapping package."""

__all__ = ["run_mapping"]


def run_mapping(*args, **kwargs):
    from Classification.mapping.mapping import run_mapping as _run

    return _run(*args, **kwargs)
