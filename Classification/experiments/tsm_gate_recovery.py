"""
TSM 전용 4모델+앙상블 하드게이트 복구 실험 오케스트레이터.

Execution:
  python3 -m Classification.experiments.tsm_gate_recovery
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
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

from Classification.ensemble.ensemble import run_equal_weight_ensemble
from Classification.mapping.mapping import run_mapping
from Classification.model_config import (
    get_ensemble_result_path,
    get_model_metrics_path,
    get_model_params_path,
    get_ticker_artifact_dir,
    load_json_artifact_only,
    save_json_artifact_only,
)
from Classification.Preprocessing.split_dataset import split_dataset
from Classification.models.logreg.optimize import optimize as optimize_logreg
from Classification.models.logreg.pipeline import run_pipeline as run_logreg
from Classification.models.rf.optimize import optimize as optimize_rf
from Classification.models.rf.pipeline import run_pipeline as run_rf
from Classification.models.svm.optimize import optimize as optimize_svm
from Classification.models.svm.pipeline import run_pipeline as run_svm
from Classification.models.xgb.optimize import optimize as optimize_xgb
from Classification.models.xgb.pipeline import run_pipeline as run_xgb

TICKER = "TSM"
BENCHMARK = "SP500"
PROFILE = "tsm_gatehard_v1"
N_TRIALS = 100

ACC_MIN = 0.52
IC_MIN = 0.05
GAP_MAX = 0.25

MODELS = ("xgb", "svm", "rf", "logreg")

MODEL_RUNNERS = {
    "xgb": run_xgb,
    "svm": run_svm,
    "rf": run_rf,
    "logreg": run_logreg,
}

MODEL_OPTIMIZERS = {
    "xgb": optimize_xgb,
    "svm": optimize_svm,
    "rf": optimize_rf,
    "logreg": optimize_logreg,
}


def _json_load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(v, default=0.0) -> float:
    try:
        if v is None:
            return float(default)
        x = float(v)
        if math.isnan(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def _extract_ensemble_metrics(payload: dict) -> dict:
    return {
        "accuracy": _to_float(payload.get("accuracy_ref", 0.0)),
        "precision": _to_float(payload.get("precision_ref", 0.0)),
        "ic": _to_float(payload.get("ic_full", 0.0)),
        "ic_p_value": _to_float(payload.get("ic_pvalue", 1.0), default=1.0),
        "gap": _to_float(payload.get("gap_ref", 1.0), default=1.0),
        "ic_first": _to_float(payload.get("ic_first_half", 0.0)),
        "ic_second": _to_float(payload.get("ic_second_half", 0.0)),
    }


def _hard_gate(metrics: dict) -> dict:
    acc = _to_float(metrics.get("accuracy", 0.0))
    ic = _to_float(metrics.get("ic", 0.0))
    gap = _to_float(metrics.get("gap", 1.0), default=1.0)

    acc_ok = acc >= ACC_MIN
    ic_ok = ic >= IC_MIN
    gap_ok = gap <= GAP_MAX
    return {
        "accuracy_ok": acc_ok,
        "ic_ok": ic_ok,
        "gap_ok": gap_ok,
        "pass": bool(acc_ok and ic_ok and gap_ok),
    }


def _build_gate_summary(model_metrics: dict, ensemble_metrics: dict) -> dict:
    by_model = {name: _hard_gate(model_metrics[name]) for name in MODELS}
    by_model["ensemble"] = _hard_gate(ensemble_metrics)
    overall_pass = all(v["pass"] for v in by_model.values())
    return {
        "hard_gate": {
            "accuracy_min": ACC_MIN,
            "ic_min": IC_MIN,
            "gap_max": GAP_MAX,
        },
        "by_model": by_model,
        "overall_pass": bool(overall_pass),
    }


def _run_models_on_split(split, optimize_profile: str) -> dict:
    out = {}
    for model_name in MODELS:
        run_fn = MODEL_RUNNERS[model_name]
        metrics = run_fn(
            auto_optimize=False,
            optimize_profile=optimize_profile,
            return_metrics=True,
            ticker=TICKER,
            benchmark=BENCHMARK,
            save_plot=False,
            compute_importance=False,
            split_override=split,
        )
        out[model_name] = metrics
    return out


def _run_stage_optimizations(strategy_stage: str) -> None:
    for model_name in MODELS:
        optimize_fn = MODEL_OPTIMIZERS[model_name]
        optimize_fn(
            profile=PROFILE,
            n_trials=N_TRIALS,
            ticker=TICKER,
            benchmark=BENCHMARK,
            auto_update=False,
            persist_total_features_on_update=False,
            feature_source_mode="db_first",
            strategy_stage=strategy_stage,
        )


def _collect_stage_param_meta() -> dict:
    meta = {}
    for model_name in MODELS:
        params_path = get_model_params_path(model_name, TICKER)
        data, _ = load_json_artifact_only(params_path)
        if data is None:
            meta[model_name] = {"error": "params_missing", "path": str(params_path)}
            continue
        meta[model_name] = {
            "params_path": str(params_path),
            "profile": data.get("profile"),
            "objective_version": data.get("objective_version"),
            "strategy_stage": data.get("strategy_stage"),
            "direction_mode": data.get("direction_mode"),
            "recency_weight_lambda": data.get("recency_weight_lambda"),
            "class_weight_mode": data.get("class_weight_mode"),
            "gate_objective_version": data.get("gate_objective_version"),
        }
    return meta


def _load_existing_baseline() -> tuple[dict, dict] | tuple[None, None]:
    model_metrics = {}
    for model_name in MODELS:
        path = get_model_metrics_path(model_name, TICKER)
        data = _json_load(path)
        if data is None:
            return None, None
        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            return None, None
        model_metrics[model_name] = {
            "accuracy": _to_float(metrics.get("accuracy", 0.0)),
            "precision": _to_float(metrics.get("precision", 0.0)),
            "ic": _to_float(metrics.get("ic", 0.0)),
            "ic_p_value": _to_float(metrics.get("ic_p_value", 1.0), default=1.0),
            "gap": _to_float(metrics.get("gap", 1.0), default=1.0),
            "ic_first": _to_float(metrics.get("ic_first", 0.0)),
            "ic_second": _to_float(metrics.get("ic_second", 0.0)),
        }

    ens_data = _json_load(get_ensemble_result_path(TICKER))
    if ens_data is None:
        return None, None

    ensemble_metrics = _extract_ensemble_metrics(ens_data)
    return model_metrics, ensemble_metrics


def _build_runtime_baseline() -> tuple[dict, dict]:
    split = split_dataset(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    model_metrics = _run_models_on_split(split, optimize_profile="balanced")
    ensemble_payload = run_equal_weight_ensemble(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        optimize_profile="balanced",
        save_json=False,
    )
    ensemble_metrics = _extract_ensemble_metrics(ensemble_payload)
    return model_metrics, ensemble_metrics


def _metric_delta(base: dict, final: dict) -> dict:
    keys = ["accuracy", "precision", "ic", "gap", "ic_first", "ic_second"]
    out = {}
    for k in keys:
        b = base.get(k)
        f = final.get(k)
        if b is None or f is None:
            out[f"delta_{k}"] = None
        else:
            out[f"delta_{k}"] = _to_float(f) - _to_float(b)
    return out


def run_recovery(save_recovery_artifacts: bool = False) -> dict:
    print("=" * 80)
    print("TSM Gate Recovery: Stage 0/1/2/3")
    print("=" * 80)

    recovery_dir = get_ticker_artifact_dir(TICKER) / "recovery"

    # Stage 0: baseline snapshot
    baseline_model_metrics, baseline_ensemble_metrics = _load_existing_baseline()
    baseline_source = "artifacts"
    if baseline_model_metrics is None or baseline_ensemble_metrics is None:
        print("[Stage0] 기존 metrics artifact가 없어 runtime baseline을 생성합니다.")
        baseline_model_metrics, baseline_ensemble_metrics = _build_runtime_baseline()
        baseline_source = "runtime"

    stage0_gate = _build_gate_summary(baseline_model_metrics, baseline_ensemble_metrics)
    stage0_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER,
        "benchmark": BENCHMARK,
        "source": baseline_source,
        "model_metrics": baseline_model_metrics,
        "ensemble_metrics": baseline_ensemble_metrics,
        "gate": stage0_gate,
    }
    if save_recovery_artifacts:
        save_json_artifact_only(stage0_payload, recovery_dir / "stage0_baseline.json")

    # Stage 1: profile + class_weight 중심 튜닝
    print("\n[Stage1] tsm_gatehard_v1 / stage1 최적화 시작")
    _run_stage_optimizations("stage1")

    split_stage1 = split_dataset(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    stage1_model_metrics = _run_models_on_split(split_stage1, optimize_profile=PROFILE)
    stage1_ensemble_payload = run_equal_weight_ensemble(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        optimize_profile=PROFILE,
        save_json=True,
    )
    stage1_ensemble_metrics = _extract_ensemble_metrics(stage1_ensemble_payload)
    stage1_gate = _build_gate_summary(stage1_model_metrics, stage1_ensemble_metrics)

    # Stage 2: recency/direction 강화 튜닝
    print("\n[Stage2] tsm_gatehard_v1 / stage2 최적화 시작")
    _run_stage_optimizations("stage2")
    stage2_param_meta = _collect_stage_param_meta()

    # Stage 3: stage2 파라미터 고정 최종 테스트 1회
    print("\n[Stage3] stage2 고정 파라미터 최종 평가")
    split_final = split_dataset(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    final_model_metrics = _run_models_on_split(split_final, optimize_profile=PROFILE)
    final_ensemble_payload = run_equal_weight_ensemble(
        ticker=TICKER,
        benchmark=BENCHMARK,
        auto_update=False,
        persist_total_features_on_update=False,
        optimize_profile=PROFILE,
        save_json=True,
    )
    final_mapping_payload = run_mapping(ticker=TICKER, benchmark=BENCHMARK)

    final_ensemble_metrics = _extract_ensemble_metrics(final_ensemble_payload)
    final_gate = _build_gate_summary(final_model_metrics, final_ensemble_metrics)

    stage_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER,
        "benchmark": BENCHMARK,
        "profile": PROFILE,
        "n_trials": N_TRIALS,
        "fixed_policy": {
            "threshold": 0.5,
            "target_alpha_margin": 0.01,
            "split_policy": "train/embargo/validation/golden_gap/test fixed",
            "selection_policy": "validation_only",
        },
        "stage0_baseline": {
            "source": baseline_source,
            "gate": stage0_gate,
        },
        "stage1": {
            "model_metrics": stage1_model_metrics,
            "ensemble_metrics": stage1_ensemble_metrics,
            "gate": stage1_gate,
        },
        "stage2": {
            "params_meta": stage2_param_meta,
        },
        "stage3_final": {
            "model_metrics": final_model_metrics,
            "ensemble_metrics": final_ensemble_metrics,
            "gate": final_gate,
            "mapping": {
                "path": str(get_ticker_artifact_dir(TICKER) / "mapping" / "mapping.json"),
                "E_alpha_3M_log": final_mapping_payload.get("E_alpha_3M_log"),
                "E_alpha_3M_simple": final_mapping_payload.get("E_alpha_3M_simple"),
            },
        },
    }
    if save_recovery_artifacts:
        save_json_artifact_only(stage_summary, recovery_dir / "stage_summary.json")

    delta_by_model = {}
    for model_name in MODELS:
        delta_by_model[model_name] = {
            **_metric_delta(baseline_model_metrics[model_name], final_model_metrics[model_name]),
            "baseline_gate": stage0_gate["by_model"][model_name],
            "final_gate": final_gate["by_model"][model_name],
        }

    delta_by_model["ensemble"] = {
        **_metric_delta(baseline_ensemble_metrics, final_ensemble_metrics),
        "baseline_gate": stage0_gate["by_model"]["ensemble"],
        "final_gate": final_gate["by_model"]["ensemble"],
    }

    renegotiation = []
    if not final_gate["overall_pass"]:
        renegotiation = [
            {
                "option": "A",
                "description": "IC/GAP 고정, 정확도는 ensemble만 하드게이트",
            },
            {
                "option": "B",
                "description": "Accuracy/GAP 고정, IC 최소값을 0.00으로 완화",
            },
            {
                "option": "C",
                "description": "현재 목표 유지, 단 threshold 튜닝(0.5 고정 해제) 허용",
            },
        ]

    recovery_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": TICKER,
        "benchmark": BENCHMARK,
        "profile": PROFILE,
        "n_trials": N_TRIALS,
        "fixed_constraints": {
            "threshold": 0.5,
            "alpha_margin": 0.01,
            "hard_gate": {
                "accuracy_min": ACC_MIN,
                "ic_min": IC_MIN,
                "gap_max": GAP_MAX,
            },
        },
        "baseline": {
            "source": baseline_source,
            "model_metrics": baseline_model_metrics,
            "ensemble_metrics": baseline_ensemble_metrics,
            "gate": stage0_gate,
        },
        "final": {
            "model_metrics": final_model_metrics,
            "ensemble_metrics": final_ensemble_metrics,
            "gate": final_gate,
        },
        "delta": delta_by_model,
        "renegotiation_options": renegotiation,
    }
    if save_recovery_artifacts:
        save_json_artifact_only(recovery_report, recovery_dir / "recovery_report.json")

    print("\n" + "=" * 80)
    print("TSM Recovery Completed")
    print("=" * 80)
    for name in [*MODELS, "ensemble"]:
        g = final_gate["by_model"][name]
        print(
            f"  {name.upper():8s} | "
            f"acc={g['accuracy_ok']} ic={g['ic_ok']} gap={g['gap_ok']} "
            f"=> pass={g['pass']}"
        )
    print(f"  Overall hard-gate pass: {final_gate['overall_pass']}")
    if save_recovery_artifacts:
        print(f"  stage0_baseline.json : {recovery_dir / 'stage0_baseline.json'}")
        print(f"  stage_summary.json   : {recovery_dir / 'stage_summary.json'}")
        print(f"  recovery_report.json : {recovery_dir / 'recovery_report.json'}")
    else:
        print("  recovery artifact 저장: 비활성(default)")

    return recovery_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TSM gate recovery experiment")
    parser.add_argument(
        "--save-recovery-artifacts",
        action="store_true",
        help="Save recovery comparison JSON files under artifacts/TSM/recovery",
    )
    args = parser.parse_args()
    run_recovery(save_recovery_artifacts=bool(args.save_recovery_artifacts))
