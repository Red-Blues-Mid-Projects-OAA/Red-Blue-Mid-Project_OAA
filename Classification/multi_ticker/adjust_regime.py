"""
이 파일은 보정 시장 국면 관련 작업을 담당합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from DB.adjust_regime import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy

    runpy.run_module("DB.adjust_regime", run_name="__main__")
