"""
이 파일은 모델 결과를 바로 사용할지 추가 검증할지 판단하는 게이트 규칙을 정의합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from Classification.model_config import (
    REQUIRE_ACCURACY_GATE,
    REQUIRE_IC_STABILITY_GATE,
    TARGET_ACC_MIN,
    TARGET_GAP_MAX,
    TARGET_IC_HALF_MIN,
    TARGET_IC_MIN,
)

@dataclass(frozen=True)
class GateThresholds:
    """게이트 Thresholds 관련 상태와 동작을 한곳에 묶어 관리하는 클래스입니다."""
    accuracy_min: float = TARGET_ACC_MIN
    ic_min: float = TARGET_IC_MIN
    ic_half_min: float = TARGET_IC_HALF_MIN
    gap_max: float = TARGET_GAP_MAX
    require_accuracy: bool = REQUIRE_ACCURACY_GATE
    require_ic_stability: bool = REQUIRE_IC_STABILITY_GATE

def evaluate_gate(metrics: dict, thresholds: GateThresholds | None = None) -> dict:
    """단일 모델 지표에 대해 PASS/FAIL을 계산합니다."""
    t = thresholds or GateThresholds()

    accuracy = float(metrics["accuracy"])
    ic = float(metrics["ic"])
    gap_base = float(metrics.get("gap", 0.0))
    gap_signed = float(metrics.get("gap_signed", gap_base))
    gap_abs = float(metrics.get("gap_abs", abs(gap_signed)))
    ic_first = float(metrics.get("ic_first", float("nan")))
    ic_second = float(metrics.get("ic_second", float("nan")))

    acc_pass = accuracy >= t.accuracy_min
    ic_pass = ic >= t.ic_min
    gap_pass = gap_abs <= t.gap_max
    ic_first_pass = (not math.isnan(ic_first)) and (ic_first >= t.ic_half_min)
    ic_second_pass = (not math.isnan(ic_second)) and (ic_second >= t.ic_half_min)
    stability_pass = ic_first_pass and ic_second_pass

    pass_all = ic_pass and gap_pass
    if t.require_accuracy:
        pass_all = pass_all and acc_pass
    if t.require_ic_stability:
        pass_all = pass_all and stability_pass

    return {
        "accuracy": accuracy,
        "ic": ic,
        "gap": gap_abs,  # backward-compatible alias
        "gap_signed": gap_signed,
        "gap_abs": gap_abs,
        "ic_first": ic_first,
        "ic_second": ic_second,
        "acc_pass": acc_pass,
        "ic_pass": ic_pass,
        "gap_pass": gap_pass,
        "ic_first_pass": ic_first_pass,
        "ic_second_pass": ic_second_pass,
        "stability_pass": stability_pass,
        "pass_all": pass_all,
        "thresholds": {
            "accuracy_min": t.accuracy_min,
            "ic_min": t.ic_min,
            "ic_half_min": t.ic_half_min,
            "gap_max": t.gap_max,
            "require_accuracy": t.require_accuracy,
            "require_ic_stability": t.require_ic_stability,
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
        f"{'PASS' if gate['gap_pass'] else 'FAIL'} "
        f"(abs={gate['gap_abs']*100:.2f}%p, signed={gate['gap_signed']*100:.2f}%p)"
    )
    print(
        f"  IC(전반기) >= {gate['thresholds']['ic_half_min']:.2f} : "
        f"{'PASS' if gate['ic_first_pass'] else 'FAIL'} ({gate['ic_first']:+.4f})"
    )
    print(
        f"  IC(후반기) >= {gate['thresholds']['ic_half_min']:.2f} : "
        f"{'PASS' if gate['ic_second_pass'] else 'FAIL'} ({gate['ic_second']:+.4f})"
    )
    print(
        f"  Accuracy Gate Required       : "
        f"{'YES' if gate['thresholds']['require_accuracy'] else 'NO (참고지표)'}"
    )
    print(
        f"  IC Stability Gate Required   : "
        f"{'YES' if gate['thresholds']['require_ic_stability'] else 'NO'}"
    )
    print(
        f"  IC Stability Overall         : "
        f"{'PASS' if gate['stability_pass'] else 'FAIL'}"
    )
    print(f"  Overall                      : {'PASS' if gate['pass_all'] else 'FAIL'}")
    print("=" * 70)
