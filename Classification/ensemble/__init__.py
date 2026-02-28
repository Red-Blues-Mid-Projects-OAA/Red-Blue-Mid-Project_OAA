# 한글 주석: 이 패키지의 공개 모듈/함수 사용을 위한 초기화 파일입니다.
"""Equal-weight ensemble package."""

__all__ = ["run_equal_weight_ensemble"]


def run_equal_weight_ensemble(*args, **kwargs):
    from Classification.ensemble.ensemble import run_equal_weight_ensemble as _run

    return _run(*args, **kwargs)
