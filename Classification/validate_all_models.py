"""
XGB/SVM/LogReg/RF 4모델 게이트 검증 스크립트.

동작:
1) 현재 파라미터로 1차 실행
2) 실패 시 1회 재튜닝 후 재실행
3) 모델별 PASS/FAIL + 전체 PASS/FAIL 반환
"""

from __future__ import annotations

import sys

from Classification.model_gate import evaluate_gate, print_gate_result
from Classification.models.logreg.optimize import optimize as optimize_logreg
from Classification.models.logreg.pipeline import run_pipeline as run_logreg_pipeline
from Classification.models.rf.optimize import optimize as optimize_rf
from Classification.models.rf.pipeline import run_pipeline as run_rf_pipeline
from Classification.models.svm.optimize import optimize as optimize_svm
from Classification.models.svm.pipeline import run_pipeline as run_svm_pipeline
from Classification.models.xgb.optimize import optimize as optimize_xgb
from Classification.models.xgb.pipeline import run_pipeline as run_xgb_pipeline


def _execute_with_single_retry(model_name, run_fn, optimize_fn=None, optimize_kwargs=None):
    """모델 실행 후 실패 시 1회 재튜닝/재실행을 수행합니다."""
    print("\n" + "=" * 70)
    print(f"[{model_name}] 1차 실행")
    print("=" * 70)
    metrics = run_fn()
    gate = evaluate_gate(metrics)
    print_gate_result(model_name, gate)

    if gate["pass_all"] or optimize_fn is None:
        return metrics, gate, False

    print("\n" + "=" * 70)
    print(f"[{model_name}] 재튜닝 후 재실행")
    print("=" * 70)
    kwargs = optimize_kwargs or {}
    optimize_fn(**kwargs)

    metrics_retry = run_fn()
    gate_retry = evaluate_gate(metrics_retry)
    print_gate_result(model_name, gate_retry)
    return metrics_retry, gate_retry, True


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

    xgb_metrics, xgb_gate, xgb_retried = _execute_with_single_retry(
        model_name="XGBoost",
        run_fn=xgb_run,
        optimize_fn=optimize_xgb,
        optimize_kwargs={"profile": "balanced", "n_trials": 100},
    )

    svm_metrics, svm_gate, svm_retried = _execute_with_single_retry(
        model_name="SVM",
        run_fn=svm_run,
        optimize_fn=optimize_svm,
        optimize_kwargs={},
    )

    logreg_metrics, logreg_gate, logreg_retried = _execute_with_single_retry(
        model_name="LogisticRegression",
        run_fn=logreg_run,
        optimize_fn=optimize_logreg,
        optimize_kwargs={"profile": "balanced", "n_trials": 100},
    )

    rf_metrics, rf_gate, rf_retried = _execute_with_single_retry(
        model_name="RandomForest",
        run_fn=rf_run,
        optimize_fn=optimize_rf,
        optimize_kwargs={"profile": "balanced", "n_trials": 100},
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
    print(f"  XGBoost : {'PASS' if xgb_gate['pass_all'] else 'FAIL'} | retried={xgb_retried}")
    print(f"  SVM     : {'PASS' if svm_gate['pass_all'] else 'FAIL'} | retried={svm_retried}")
    print(f"  LogReg  : {'PASS' if logreg_gate['pass_all'] else 'FAIL'} | retried={logreg_retried}")
    print(f"  RF      : {'PASS' if rf_gate['pass_all'] else 'FAIL'} | retried={rf_retried}")
    print(f"  Overall : {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 70)

    return {
        "overall_pass": overall_pass,
        "xgb": {
            "metrics": xgb_metrics,
            "gate": xgb_gate,
            "retried": xgb_retried,
        },
        "svm": {
            "metrics": svm_metrics,
            "gate": svm_gate,
            "retried": svm_retried,
        },
        "logreg": {
            "metrics": logreg_metrics,
            "gate": logreg_gate,
            "retried": logreg_retried,
        },
        "rf": {
            "metrics": rf_metrics,
            "gate": rf_gate,
            "retried": rf_retried,
        },
    }


if __name__ == "__main__":
    result = validate_all_models()
    sys.exit(0 if result["overall_pass"] else 1)
