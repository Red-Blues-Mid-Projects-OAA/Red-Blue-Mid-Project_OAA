"""
모델 공통 설정/경로 관리 모듈.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
XGB_ARTIFACT_DIR = ARTIFACTS_DIR / "xgb"
SVM_ARTIFACT_DIR = ARTIFACTS_DIR / "svm"
LOGREG_ARTIFACT_DIR = ARTIFACTS_DIR / "logreg"
RF_ARTIFACT_DIR = ARTIFACTS_DIR / "rf"
ENSEMBLE_ARTIFACT_DIR = ARTIFACTS_DIR / "ensemble"

# 목표 게이트 (소수 비율 단위)
TARGET_ACC_MIN = 0.52
TARGET_IC_MIN = 0.05
TARGET_IC_HALF_MIN = 0.05
TARGET_GAP_MAX = 0.25
# 강화 정책: Accuracy와 IC 안정성을 게이트 필수 조건으로 사용
REQUIRE_ACCURACY_GATE = True
REQUIRE_IC_STABILITY_GATE = True

# permutation importance 공통 설정
PERM_IMPORTANCE_REPEATS = 20
PERM_IMPORTANCE_SEED = 42
PERM_IMPORTANCE_TOPK_TABLE = 8

# 파라미터 파일 경로
XGB_PARAMS_ARTIFACT_PATH = XGB_ARTIFACT_DIR / "xgb_best_params.json"
XGB_PARAMS_LEGACY_PATHS = []

SVM_PARAMS_ARTIFACT_PATH = SVM_ARTIFACT_DIR / "best_svm_params.json"
SVM_PARAMS_LEGACY_PATHS = []

LOGREG_PARAMS_ARTIFACT_PATH = LOGREG_ARTIFACT_DIR / "logreg_best_params.json"
LOGREG_PARAMS_LEGACY_PATHS = []

RF_PARAMS_ARTIFACT_PATH = RF_ARTIFACT_DIR / "rf_best_params.json"
RF_PARAMS_LEGACY_PATHS = []

# 결과 이미지 경로
XGB_RESULT_ARTIFACT_PATH = XGB_ARTIFACT_DIR / "xgb_classifier_result.png"
XGB_RESULT_LEGACY_PATHS = []

SVM_RESULT_ARTIFACT_PATH = SVM_ARTIFACT_DIR / "svm_classifier_result.png"
SVM_RESULT_LEGACY_PATHS = []

LOGREG_RESULT_ARTIFACT_PATH = LOGREG_ARTIFACT_DIR / "logreg_classifier_result.png"
LOGREG_RESULT_LEGACY_PATHS = []

RF_RESULT_ARTIFACT_PATH = RF_ARTIFACT_DIR / "rf_classifier_result.png"
RF_RESULT_LEGACY_PATHS = []

ENSEMBLE_RESULT_PATH = ENSEMBLE_ARTIFACT_DIR / "ensemble.json"


def ensure_artifact_dirs() -> None:
    """아티팩트 디렉터리를 생성합니다."""
    XGB_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SVM_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    LOGREG_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RF_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_json_artifact_only(artifact_path: Path) -> tuple[dict | None, Path | None]:
    """아티팩트 표준 경로의 JSON 파일을 읽고 (데이터, 사용경로)를 반환합니다."""
    if not artifact_path.exists():
        return None, None
    with artifact_path.open("r", encoding="utf-8") as f:
        return json.load(f), artifact_path


def save_json_artifact_only(payload: dict, artifact_path: Path) -> None:
    """아티팩트 표준 경로에만 JSON 파일을 저장합니다."""
    ensure_artifact_dirs()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
