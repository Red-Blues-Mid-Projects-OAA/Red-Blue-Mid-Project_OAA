"""
XGB/SVM/LogReg/RF 4모델 게이트 검증 스크립트.

동작:
1) 현재 파라미터로 1회 실행 (재튜닝 없음)
2) 모델별 PASS/FAIL 및 오류를 집계
3) 전체 PASS/FAIL 반환
"""

from __future__ import annotations

import sys

from Classification.model_gate import evaluate_gate, print_gate_result
from Classification.models.logreg.pipeline import run_pipeline as run_logreg_pipeline
from Classification.models.rf.pipeline import run_pipeline as run_rf_pipeline
from Classification.models.svm.pipeline import run_pipeline as run_svm_pipeline
from Classification.models.xgb.pipeline import run_pipeline as run_xgb_pipeline


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
        fail_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "ic": 0.0,
            "ic_p_value": 1.0,
            "gap": 1.0,
            "ic_first": float("nan"),
            "ic_second": float("nan"),
            "overall_pass": False,
        }
        gate = evaluate_gate(fail_metrics)
        print_gate_result(model_name, gate)
        return fail_metrics, gate, str(e)


def validate_all_models():
    """XGB/SVM/LogReg/RF 전체 게이트를 검증합니다."""
    xgb_run = lambda: run_xgb_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
    )
    svm_run = lambda: run_svm_pipeline(return_metrics=True)
    logreg_run = lambda: run_logreg_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
    )
    rf_run = lambda: run_rf_pipeline(
        auto_optimize=False,
        optimize_profile="balanced",
        return_metrics=True,
    )

    xgb_metrics, xgb_gate, xgb_error = _execute_gate_only("XGBoost", xgb_run)
    svm_metrics, svm_gate, svm_error = _execute_gate_only("SVM", svm_run)
    logreg_metrics, logreg_gate, logreg_error = _execute_gate_only("LogisticRegression", logreg_run)
    rf_metrics, rf_gate, rf_error = _execute_gate_only("RandomForest", rf_run)

    overall_pass = (
        xgb_gate["pass_all"]
        and svm_gate["pass_all"]
        and logreg_gate["pass_all"]
        and rf_gate["pass_all"]
    )

    print("\n" + "=" * 70)
    print("[전체 게이트 요약]")
    print("=" * 70)
    print(f"  XGBoost : {'PASS' if xgb_gate['pass_all'] else 'FAIL'} | error={'YES' if xgb_error else 'NO'}")
    print(f"  SVM     : {'PASS' if svm_gate['pass_all'] else 'FAIL'} | error={'YES' if svm_error else 'NO'}")
    print(f"  LogReg  : {'PASS' if logreg_gate['pass_all'] else 'FAIL'} | error={'YES' if logreg_error else 'NO'}")
    print(f"  RF      : {'PASS' if rf_gate['pass_all'] else 'FAIL'} | error={'YES' if rf_error else 'NO'}")
    print(f"  Overall : {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    return {
        "overall_pass": overall_pass,
        "xgb": {
            "metrics": xgb_metrics,
            "gate": xgb_gate,
            "error": xgb_error,
        },
        "svm": {
            "metrics": svm_metrics,
            "gate": svm_gate,
            "error": svm_error,
        },
        "logreg": {
            "metrics": logreg_metrics,
            "gate": logreg_gate,
            "error": logreg_error,
        },
        "rf": {
            "metrics": rf_metrics,
            "gate": rf_gate,
            "error": rf_error,
        },
    }


if __name__ == "__main__":
    result = validate_all_models()
    sys.exit(0 if result["overall_pass"] else 1)
