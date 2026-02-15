"""
모델 성능 게이트 판정 모듈.
"""

from __future__ import annotations

from dataclasses import dataclass

from Classification.model_config import (
    TARGET_ACC_MIN,
    TARGET_GAP_MAX,
    TARGET_IC_MIN,
    REQUIRE_ACCURACY_GATE,
)


@dataclass(frozen=True)
class GateThresholds:
    accuracy_min: float = TARGET_ACC_MIN
    ic_min: float = TARGET_IC_MIN
    gap_max: float = TARGET_GAP_MAX
    require_accuracy: bool = REQUIRE_ACCURACY_GATE


def evaluate_gate(metrics: dict, thresholds: GateThresholds | None = None) -> dict:
    """단일 모델 지표에 대해 PASS/FAIL을 계산합니다."""
    t = thresholds or GateThresholds()

    accuracy = float(metrics["accuracy"])
    ic = float(metrics["ic"])
    gap = float(metrics["gap"])

    acc_pass = accuracy >= t.accuracy_min
    ic_pass = ic >= t.ic_min
    gap_pass = gap <= t.gap_max
    pass_all = (ic_pass and gap_pass and acc_pass) if t.require_accuracy else (ic_pass and gap_pass)

    return {
        "accuracy": accuracy,
        "ic": ic,
        "gap": gap,
        "acc_pass": acc_pass,
        "ic_pass": ic_pass,
        "gap_pass": gap_pass,
        "pass_all": pass_all,
        "thresholds": {
            "accuracy_min": t.accuracy_min,
            "ic_min": t.ic_min,
            "gap_max": t.gap_max,
            "require_accuracy": t.require_accuracy,
        },
    }


def print_gate_result(model_name: str, gate: dict) -> None:
    """모델별 게이트 판정 결과를 출력합니다."""
    print("\n" + "=" * 70)
    print(f"[{model_name}] 목표 지표 PASS/FAIL")
    print("=" * 70)
    print(
        f"  Accuracy >= {gate['thresholds']['accuracy_min']*100:.0f}% : "
        f"{'PASS' if gate['acc_pass'] else 'FAIL'} ({gate['accuracy']*100:.2f}%)"
    )
    print(
        f"  IC >= {gate['thresholds']['ic_min']:.2f}       : "
        f"{'PASS' if gate['ic_pass'] else 'FAIL'} ({gate['ic']:+.4f})"
    )
    print(
        f"  Gap <= {gate['thresholds']['gap_max']*100:.0f}%p    : "
        f"{'PASS' if gate['gap_pass'] else 'FAIL'} ({gate['gap']*100:.2f}%p)"
    )
    print(
        f"  Accuracy Gate Required       : "
        f"{'YES' if gate['thresholds']['require_accuracy'] else 'NO (참고지표)'}"
    )
    print(f"  Overall                      : {'PASS' if gate['pass_all'] else 'FAIL'}")
    print("=" * 70)
