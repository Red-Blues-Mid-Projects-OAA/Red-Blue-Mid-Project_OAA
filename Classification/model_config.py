"""
이 파일은 모델 학습에 공통으로 쓰는 경로, 파라미터, 설정값을 정리한 설정 모음입니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MULTI_TICKER_ARTIFACT_DIR = ARTIFACTS_DIR / "multi_ticker"

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

def ticker_to_slug(ticker: str) -> str:
    """파일/디렉터리/테이블명 안전 slug를 반환합니다. 예: BRK-A -> BRK_A"""
    raw = str(ticker).strip().upper()
    slug = re.sub(r"[^A-Z0-9]+", "_", raw).strip("_")
    if not slug:
        raise ValueError(f"유효하지 않은 ticker 입니다: {ticker}")
    return slug

def get_ticker_artifact_dir(ticker: str) -> Path:
    """종목별 아티팩트 루트 경로를 반환합니다."""
    return ARTIFACTS_DIR / ticker_to_slug(ticker)

def get_model_artifact_dir(model: str, ticker: str) -> Path:
    """종목별 모델 아티팩트 디렉터리를 반환합니다."""
    return get_ticker_artifact_dir(ticker) / str(model).lower()

def get_model_params_path(model: str, ticker: str) -> Path:
    """종목별 모델 파라미터 JSON 경로를 반환합니다."""
    model_key = str(model).lower()
    dir_path = get_model_artifact_dir(model_key, ticker)
    filename_map = {
        "xgb": "xgb_best_params.json",
        "svm": "best_svm_params.json",
        "logreg": "logreg_best_params.json",
        "rf": "rf_best_params.json",
    }
    if model_key not in filename_map:
        raise ValueError(f"지원하지 않는 model 입니다: {model}")
    return dir_path / filename_map[model_key]

def get_model_result_path(model: str, ticker: str) -> Path:
    """종목별 모델 result.png 경로를 반환합니다."""
    model_key = str(model).lower()
    dir_path = get_model_artifact_dir(model_key, ticker)
    filename_map = {
        "xgb": "xgb_classifier_result.png",
        "svm": "svm_classifier_result.png",
        "logreg": "logreg_classifier_result.png",
        "rf": "rf_classifier_result.png",
    }
    if model_key not in filename_map:
        raise ValueError(f"지원하지 않는 model 입니다: {model}")
    return dir_path / filename_map[model_key]

def get_model_metrics_path(model: str, ticker: str) -> Path:
    """종목별 모델 성능 metrics JSON 경로를 반환합니다."""
    model_key = str(model).lower()
    dir_path = get_model_artifact_dir(model_key, ticker)
    filename_map = {
        "xgb": "xgb_metrics.json",
        "svm": "svm_metrics.json",
        "logreg": "logreg_metrics.json",
        "rf": "rf_metrics.json",
    }
    if model_key not in filename_map:
        raise ValueError(f"지원하지 않는 model 입니다: {model}")
    return dir_path / filename_map[model_key]

def get_ensemble_result_path(ticker: str) -> Path:
    """종목별 ensemble 결과 JSON 경로를 반환합니다."""
    return get_ticker_artifact_dir(ticker) / "ensemble" / "ensemble.json"

def get_mapping_result_path(ticker: str) -> Path:
    """종목별 mapping 결과 JSON 경로를 반환합니다."""
    return get_ticker_artifact_dir(ticker) / "mapping" / "mapping.json"

def ensure_artifact_dirs() -> None:
    """아티팩트 디렉터리를 생성합니다."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_TICKER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
