"""
이 파일은 공통 패키지를 한 번에 불러오기 쉽게 만드는 초기화 파일입니다. 공통으로 노출할 모듈이나 기본 설정을 정리할 때 사용합니다.
이 초기화 파일은 하위 모듈의 공개 함수와 상수를 한곳에 모아 외부 import 경로를 단순하게 유지하고, 내부 폴더 구조가 바뀌어도 상위 호출부 수정 범위를 줄이는 역할을 맡습니다.
"""

from datetime import datetime, timedelta
import builtins
import locale
import os
import sys

import numpy as np
import oracledb
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

def _hangul_ratio(text: str) -> float:
    """문자열 안에 한글이 얼마나 들어 있는지 비율로 계산합니다."""
    if not text:
        return 0.0
    count = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
    return count / len(text)

def _try_repair_mojibake(text: str) -> str:
    """
    윈도우에서 UTF-8 한글이 cp949로 잘못 해석된 대표적인 깨짐 패턴을 복구합니다.
    예: 깨진 한글 문자열을 원래의 자연스러운 한글 표현으로 되돌립니다.
    """
    if not text:
        return text

    try:
        repaired = text.encode("cp949").decode("utf-8")
    except Exception:
        return text

    # 복구 결과가 원문보다 분명히 더 자연스러운 한글일 때만 교체합니다.
    if _hangul_ratio(repaired) >= _hangul_ratio(text) + 0.20 and "\ufffd" not in repaired:
        return repaired
    return text

def _configure_console_encoding() -> None:
    """
    오래된 윈도우 cp949 터미널을 망가뜨리지 않으면서 출력 인코딩을 최대한 안정적으로 맞춥니다.

    - 대화형 비 UTF 터미널에서는 현재 인코딩을 그대로 둡니다.
    - 리다이렉션 출력이나 UTF 터미널에서는 로그가 깨지지 않게 UTF-8을 우선 사용합니다.
    """

    streams = [getattr(sys, "stdout", None), getattr(sys, "stderr", None)]

    interactive_non_utf_terminal = any(
        stream is not None
        and hasattr(stream, "isatty")
        and stream.isatty()
        and "utf" not in str(getattr(stream, "encoding", "")).lower()
        for stream in streams
    )

    preferred = str(locale.getpreferredencoding(False) or "").lower()
    if interactive_non_utf_terminal and "utf" not in preferred:
        return

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in streams:
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

def _patch_print_for_mojibake_repair() -> None:
    """print 호출 시 깨진 한글 문자열을 한 번 더 복구하도록 출력 함수를 감쌉니다."""
    if getattr(builtins, "_common_print_patched", False):
        return

    original_print = builtins.print

    def patched_print(*args, **kwargs):
        """문자열 인자를 출력하기 전에 가능한 경우 깨진 한글을 복구합니다."""
        fixed_args = [
            _try_repair_mojibake(arg) if isinstance(arg, str) else arg
            for arg in args
        ]

        if "sep" in kwargs and isinstance(kwargs["sep"], str):
            kwargs["sep"] = _try_repair_mojibake(kwargs["sep"])
        if "end" in kwargs and isinstance(kwargs["end"], str):
            kwargs["end"] = _try_repair_mojibake(kwargs["end"])

        return original_print(*fixed_args, **kwargs)

    builtins.print = patched_print
    builtins._common_print_patched = True

_configure_console_encoding()
_patch_print_for_mojibake_repair()

__all__ = [
    "pd",
    "np",
    "os",
    "sys",
    "yf",
    "datetime",
    "timedelta",
    "oracledb",
    "load_dotenv",
]
