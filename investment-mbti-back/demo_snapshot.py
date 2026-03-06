"""
Helpers for reading/writing the backend demo snapshot payload.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = _PROJECT_ROOT / "investment-mbti-back" / "data"
SNAPSHOT_PATH = SNAPSHOT_DIR / "demo_snapshot.json"


def _normalize_value(value):
    if value is None:
        return None

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, str):
        return value

    if pd.isna(value):
        return None

    return value


def normalize_payload(obj):
    if isinstance(obj, dict):
        return {str(key): normalize_payload(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [normalize_payload(value) for value in obj]

    if isinstance(obj, tuple):
        return [normalize_payload(value) for value in obj]

    return _normalize_value(obj)


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return normalize_payload(records)


def save_demo_snapshot(payload: dict, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(normalize_payload(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def load_demo_snapshot(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else SNAPSHOT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Demo snapshot not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))
