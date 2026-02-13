"""
모델 공통 설정/경로 관리 모듈.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
XGB_ARTIFACT_DIR = ARTIFACTS_DIR / "xgb"
SVM_ARTIFACT_DIR = ARTIFACTS_DIR / "svm"

# 목표 게이트 (소수 비율 단위)
TARGET_ACC_MIN = 0.52
TARGET_IC_MIN = 0.05
TARGET_GAP_MAX = 0.25
# 팀 합의 정책: Accuracy는 참고 지표, 게이트 통과는 IC/GAP 우선
REQUIRE_ACCURACY_GATE = False

# 파라미터 파일 경로
XGB_PARAMS_ARTIFACT_PATH = XGB_ARTIFACT_DIR / "xgb_best_params.json"
XGB_PARAMS_LEGACY_PATHS = [BASE_DIR / "xgb_best_params.json"]

SVM_PARAMS_ARTIFACT_PATH = SVM_ARTIFACT_DIR / "best_svm_params.json"
SVM_PARAMS_LEGACY_PATHS = [BASE_DIR / "best_svm_params.json"]

# 결과 이미지 경로
XGB_RESULT_ARTIFACT_PATH = XGB_ARTIFACT_DIR / "xgb_classifier_result.png"
XGB_RESULT_LEGACY_PATHS = [BASE_DIR / "xgb_classifier_result.png"]

SVM_RESULT_ARTIFACT_PATH = SVM_ARTIFACT_DIR / "svm_classifier_result.png"
SVM_RESULT_LEGACY_PATHS = [BASE_DIR / "svm_classifier_result.png"]


def ensure_artifact_dirs() -> None:
    """아티팩트 디렉터리를 생성합니다."""
    XGB_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SVM_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_read_path(primary: Path, fallbacks: Iterable[Path]) -> Path | None:
    """읽기용 경로를 우선순위(primary -> fallbacks)로 찾습니다."""
    for candidate in [primary, *fallbacks]:
        if candidate.exists():
            return candidate
    return None


def load_json_with_fallback(primary: Path, fallbacks: Iterable[Path]) -> tuple[dict | None, Path | None]:
    """JSON 파일을 우선순위로 읽고 (데이터, 사용경로)를 반환합니다."""
    path = resolve_read_path(primary, fallbacks)
    if path is None:
        return None, None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f), path


def save_json_with_legacy(payload: dict, artifact_path: Path, legacy_paths: Iterable[Path]) -> None:
    """표준 경로에 저장하고 레거시 경로도 동기화합니다."""
    ensure_artifact_dirs()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    for legacy_path in legacy_paths:
        with legacy_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
