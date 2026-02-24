"""Shared imports and console/output normalization utilities."""

from datetime import datetime, timedelta
import builtins
import os
import sys

import numpy as np
import oracledb
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv


def _hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    count = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
    return count / len(text)


def _try_repair_mojibake(text: str) -> str:
    """
    Repair common Windows mojibake where UTF-8 Korean text was decoded as cp949.
    Example: "紐⑤뱺" -> "모든".
    """
    if not text:
        return text

    try:
        repaired = text.encode("cp949").decode("utf-8")
    except Exception:
        return text

    # Keep original unless repaired text is clearly better Korean text.
    if _hangul_ratio(repaired) >= _hangul_ratio(text) + 0.20 and "\ufffd" not in repaired:
        return repaired
    return text


def _configure_console_encoding() -> None:
    # Enforce UTF-8 output for redirected logs and terminal output.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _patch_print_for_mojibake_repair() -> None:
    if getattr(builtins, "_common_print_patched", False):
        return

    original_print = builtins.print

    def patched_print(*args, **kwargs):
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
