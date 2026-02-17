"""
XGB/SVM/LogReg/RF 4모델 게이트 검증 스크립트.

정책:
1) validate 실행 중 DB 소스 업데이트는 수행하지 않음
2) 공통 split을 1회 생성해 4개 모델이 공유
3) LogReg/RF stale 파라미터 감지 시 1회 자동 재튜닝 후 재실행
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

from Classification.model_gate import evaluate_gate, print_gate_result
from Classification.Preprocessing.split_dataset import split_dataset
from Classification.models.logreg.optimize import optimize as optimize_logreg
from Classification.models.logreg.pipeline import run_pipeline as run_logreg_pipeline
from Classification.models.rf.optimize import optimize as optimize_rf
from Classification.models.rf.pipeline import run_pipeline as run_rf_pipeline
from Classification.models.svm.pipeline import run_pipeline as run_svm_pipeline
from Classification.models.xgb.pipeline import run_pipeline as run_xgb_pipeline

STALE_PATTERNS = (
    "메타데이터 누락",
    "feature_hash 불일치",
    "objective_version 불일치",
    "cv_mode 불일치",
    "data_end_date 구버전",
    "파라미터 아티팩트가 현재 정책과 불일치",
)
RETRY_N_TRIALS = 100


def _make_fail_metrics():
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "ic": 0.0,
        "ic_p_value": 1.0,
        "gap": 1.0,
        "ic_first": float("nan"),
        "ic_second": float("nan"),
        "overall_pass": False,
    }


def _is_stale_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    return any(pattern in error_text for pattern in STALE_PATTERNS)


def _status_from_error(error_text: str | None) -> str:
    if error_text is None:
        return "normal"
    return "error_non_stale"


def _execute_gate_only(model_name, run_fn):
    """모델을 1회 실행하고 게이트를 판정합니다. 실패 시 에러를 기록합니다."""
    print("\n" + "=" * 70)
    print(f"[{model_name}] 실행")
    print("=" * 70)
    try:
        metrics = run_fn()
        gate = evaluate_gate(metrics)
        print_gate_result(model_name, gate)
        return metrics, gate, None
    except Exception as e:
        print(f"  [ERROR] {model_name} 실행 실패: {e}")
        fail_metrics = _make_fail_metrics()
        gate = evaluate_gate(fail_metrics)
        print_gate_result(model_name, gate)
        return fail_metrics, gate, str(e)


def _execute_with_one_retune(model_name, run_fn, optimize_fn):
    """
    stale 파라미터 오류일 때만 1회 재튜닝 후 동일 실행을 재시도합니다.
    """
    metrics, gate, error = _execute_gate_only(model_name, run_fn)
    if error is None:
        return metrics, gate, None, "normal"

    if not _is_stale_error(error):
        return metrics, gate, error, "error_non_stale"

    print("\n" + "=" * 70)
    print(f"[{model_name}] stale 파라미터 감지 -> 1회 자동 재튜닝")
    print("=" * 70)
    try:
        optimize_fn(
            profile="balanced",
            n_trials=RETRY_N_TRIALS,
            auto_update=False,
            persist_total_features_on_update=False,
        )
    except Exception as opt_error:
        combined = f"{error} | retune_failed: {opt_error}"
        print(f"  [ERROR] {model_name} 재튜닝 실패: {opt_error}")
        return metrics, gate, combined, "retuned_once_failed"

    print(f"  {model_name} 재실행 시작...")
    retry_metrics, retry_gate, retry_error = _execute_gate_only(model_name, run_fn)
    if retry_error is None:
        return retry_metrics, retry_gate, None, "retuned_once_success"

    return retry_metrics, retry_gate, retry_error, "retuned_once_failed"


def validate_all_models():
    """XGB/SVM/LogReg/RF 전체 게이트를 검증합니다."""
    print("\n" + "=" * 70)
    print("[공통 Split 준비] validate 정책: no-update")
    print("=" * 70)
    try:
        shared_split = split_dataset(
            auto_update=False,
            persist_total_features_on_update=False,
        )
    except Exception as e:
        error_msg = f"shared split 생성 실패: {e}"
        print(f"  [ERROR] {error_msg}")
        fail_metrics = _make_fail_metrics()
        fail_gate = evaluate_gate(fail_metrics)
        return {
            "overall_pass": False,
            "xgb": {"metrics": fail_metrics, "gate": fail_gate, "error": error_msg, "status": "error_non_stale"},
            "svm": {"metrics": fail_metrics, "gate": fail_gate, "error": error_msg, "status": "error_non_stale"},
            "logreg": {"metrics": fail_metrics, "gate": fail_gate, "error": error_msg, "status": "error_non_stale"},
            "rf": {"metrics": fail_metrics, "gate": fail_gate, "error": error_msg, "status": "error_non_stale"},
        }

    xgb_run = lambda: run_xgb_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
        split_override=shared_split,
    )
    svm_run = lambda: run_svm_pipeline(
        return_metrics=True,
        split_override=shared_split,
    )
    logreg_run = lambda: run_logreg_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
        split_override=shared_split,
    )
    rf_run = lambda: run_rf_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
        split_override=shared_split,
    )

    xgb_metrics, xgb_gate, xgb_error = _execute_gate_only("XGBoost", xgb_run)
    xgb_status = _status_from_error(xgb_error)

    svm_metrics, svm_gate, svm_error = _execute_gate_only("SVM", svm_run)
    svm_status = _status_from_error(svm_error)

    logreg_metrics, logreg_gate, logreg_error, logreg_status = _execute_with_one_retune(
        "LogisticRegression",
        logreg_run,
        optimize_logreg,
    )
    rf_metrics, rf_gate, rf_error, rf_status = _execute_with_one_retune(
        "RandomForest",
        rf_run,
        optimize_rf,
    )

    overall_pass = (
        xgb_gate["pass_all"]
        and svm_gate["pass_all"]
        and logreg_gate["pass_all"]
        and rf_gate["pass_all"]
    )

    print("\n" + "=" * 70)
    print("[전체 게이트 요약]")
    print("=" * 70)
    print(
        f"  XGBoost : {'PASS' if xgb_gate['pass_all'] else 'FAIL'} | "
        f"error={'YES' if xgb_error else 'NO'} | status={xgb_status}"
    )
    print(
        f"  SVM     : {'PASS' if svm_gate['pass_all'] else 'FAIL'} | "
        f"error={'YES' if svm_error else 'NO'} | status={svm_status}"
    )
    print(
        f"  LogReg  : {'PASS' if logreg_gate['pass_all'] else 'FAIL'} | "
        f"error={'YES' if logreg_error else 'NO'} | status={logreg_status}"
    )
    print(
        f"  RF      : {'PASS' if rf_gate['pass_all'] else 'FAIL'} | "
        f"error={'YES' if rf_error else 'NO'} | status={rf_status}"
    )
    print(f"  Overall : {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    return {
        "overall_pass": overall_pass,
        "xgb": {
            "metrics": xgb_metrics,
            "gate": xgb_gate,
            "error": xgb_error,
            "status": xgb_status,
        },
        "svm": {
            "metrics": svm_metrics,
            "gate": svm_gate,
            "error": svm_error,
            "status": svm_status,
        },
        "logreg": {
            "metrics": logreg_metrics,
            "gate": logreg_gate,
            "error": logreg_error,
            "status": logreg_status,
        },
        "rf": {
            "metrics": rf_metrics,
            "gate": rf_gate,
            "error": rf_error,
            "status": rf_status,
        },
    }


if __name__ == "__main__":
    result = validate_all_models()
    sys.exit(0 if result["overall_pass"] else 1)
