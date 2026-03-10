"""
이 파일은 데모 화면에서 쓸 수 있는 예시 스냅샷 데이터를 준비하거나 변환합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = _PROJECT_ROOT / "investment-mbti-back" / "data"
SNAPSHOT_PATH = SNAPSHOT_DIR / "demo_snapshot.json"

def _normalize_value(value):
    """value 값을 서로 비교하기 쉽게 정규화합니다."""
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
    """payload 값을 서로 비교하기 쉽게 정규화합니다."""
    if isinstance(obj, dict):
        return {str(key): normalize_payload(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [normalize_payload(value) for value in obj]

    if isinstance(obj, tuple):
        return [normalize_payload(value) for value in obj]

    return _normalize_value(obj)

def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    """데이터프레임을 API 응답용 레코드 목록으로 바꿉니다."""
    if df is None or df.empty:
        return []
    records = df.to_dict(orient="records")
    return normalize_payload(records)

def save_demo_snapshot(payload: dict, path: Path | None = None) -> Path:
    """데모 스냅샷를 저장합니다."""
    target = Path(path) if path is not None else SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(normalize_payload(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target

def load_demo_snapshot(path: Path | None = None) -> dict:
    """데모 스냅샷 데이터를 메모리로 불러옵니다."""
    target = Path(path) if path is not None else SNAPSHOT_PATH
    if not target.exists():
        raise FileNotFoundError(f"Demo snapshot not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))

def should_force_demo_snapshot() -> bool:
    """force 데모 스냅샷 여부를 판단해 반환합니다."""
    source = str(os.getenv("BACKEND_DATA_SOURCE", "")).strip().lower()
    return source in {"snapshot", "demo", "demo_snapshot", "json"}
