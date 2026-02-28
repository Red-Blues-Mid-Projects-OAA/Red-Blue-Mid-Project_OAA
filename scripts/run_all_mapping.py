# 한글 주석: 이 파일은 run_all_mapping 관련 로직을 담당합니다.
"""Compatibility wrapper for DB.run_all_mapping."""

from DB.run_all_mapping import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("DB.run_all_mapping", run_name="__main__")
