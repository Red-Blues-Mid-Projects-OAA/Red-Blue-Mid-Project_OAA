"""
멀티티커 배치 오케스트레이터.

Execution:
  python3 -m Classification.multi_ticker.run_all_tickers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

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

from common import pd
from DB import StockDBManager, TICKERS
from Classification.ensemble.ensemble import build_equal_weight_from_probas
from Classification.mapping.mapping import run_mapping
from Classification.model_config import (
    MULTI_TICKER_ARTIFACT_DIR,
    get_ensemble_result_path,
    get_mapping_result_path,
    get_model_metrics_path,
    get_model_params_path,
    get_model_result_path,
    save_json_artifact_only,
    ticker_to_slug,
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

MIN_TEST_SAMPLES = 200
LOG_E_ALPHA_WARN_THRESHOLD = math.log(1.15)
DEFAULT_N_TRIALS_BY_MODEL = {
    "xgb": 100,
    "svm": 100,
    "rf": 100,
    "logreg": 100,
}
STALE_KEYWORDS = [
    "메타데이터 누락",
    "feature_hash 불일치",
    "objective_version 불일치",
    "cv_mode 불일치",
    "data_end_date 구버전",
    "파라미터 아티팩트가 현재 정책과 불일치",
]

ELIGIBILITY_MIN_TRAIN = 30
ELIGIBILITY_MIN_VAL = 30
ELIGIBILITY_MIN_TEST = 200

_WORKER_SP500_LOGRET_CACHE = None


def _normalize_for_json(obj):
    """json.dumps 전에 datetime/date 등 비직렬화 타입을 정규화합니다."""
    if isinstance(obj, dict):
        return {k: _normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_for_json(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def _is_stale_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(k in msg for k in STALE_KEYWORDS)


def _run_model_with_optional_retune(
    model_name,
    run_fn,
    run_kwargs,
    optimize_fn=None,
    optimize_kwargs=None,
    stale_policy: str = "retune_once",
):
    """
    stale 파라미터 에러 처리 정책:
      - retune_once: 1회 재튜닝 후 재시도
      - skip: stale면 즉시 예외
      - force: auto_optimize=True로 1회 강제 재실행
    """
    status = "normal"
    try:
        result = run_fn(**run_kwargs)
        return result, status
    except Exception as e:
        if optimize_fn is None or not _is_stale_error(e):
            raise

        if stale_policy == "skip":
            raise RuntimeError(f"[{model_name}] stale detected and skipped: {e}") from e

        if stale_policy == "force":
            print(f"  [{model_name}] stale 감지 → auto_optimize 강제 1회 재실행")
            forced_kwargs = dict(run_kwargs)
            forced_kwargs["auto_optimize"] = True
            result = run_fn(**forced_kwargs)
            return result, "forced_auto_optimize"

        print(f"  [{model_name}] stale 감지 → 1회 재튜닝 실행")
        optimize_fn(**(optimize_kwargs or {}))
        try:
            result = run_fn(**run_kwargs)
            status = "retuned_once_success"
            return result, status
        except Exception:
            status = "retuned_once_failed"
            raise


def _get_model_metric_warnings(model: str, metrics: dict) -> list[str]:
    warnings = []
    gap_abs = float(
        metrics.get("gap_abs", abs(float(metrics.get("gap_signed", metrics.get("gap", 0.0)))))
    )
    if gap_abs > 0.25:
        warnings.append(f"{model} train_test_gap > 0.25")
    if bool(metrics.get("ic_degenerate", False)):
        warnings.append(f"{model} ic_degenerate=true")
    return warnings


def _is_fail_fast(metrics: dict, model_name: str, ticker: str) -> bool:
    """
    Fail-Fast 조건 판정: 세 가지 조건이 모두 동시에 충족되면 True.
    - Test Accuracy ≤ 25%
    - Test IC < 0
    - Train-Test Gap(abs) ≥ 30%
    """
    test_acc = float(metrics.get("accuracy", 1.0))
    test_ic = float(metrics.get("ic", 0.0))
    gap_abs = float(
        metrics.get("gap_abs", abs(float(metrics.get("gap_signed", metrics.get("gap", 0.0)))))
    )
    triggered = (test_acc <= 0.25) and (test_ic < 0.0) and (gap_abs >= 0.30)
    if triggered:
        print(
            f"  ⚡ [FAIL-FAST] {ticker}/{model_name}: "
            f"Acc={test_acc:.1%}, IC={test_ic:+.4f}, Gap={gap_abs:.1%}"
        )
    return triggered


def _save_model_metrics_json(
    *,
    ticker: str,
    benchmark: str,
    model: str,
    metrics: dict,
    model_status: str,
    split,
    optimize_profile: str,
    n_trials: int,
    force_retune_all: bool,
    warnings: list[str],
) -> None:
    metrics_path = get_model_metrics_path(model, ticker)
    params_path = get_model_params_path(model, ticker)
    result_plot_path = get_model_result_path(model, ticker)
    test_start = None
    test_end = None
    n_test_samples = 0
    if hasattr(split, "test") and split.test is not None and len(split.test) > 0:
        n_test_samples = int(len(split.test))
        test_start = split.test.index.min()
        test_end = split.test.index.max()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "ticker_slug": ticker_to_slug(ticker),
        "benchmark": benchmark,
        "model": str(model).lower(),
        "run_context": {
            "source": "multi_ticker.run_all_tickers",
            "optimize_profile": optimize_profile,
            "n_trials": int(n_trials),
            "force_retune_all": bool(force_retune_all),
            "model_status": model_status,
            "params_path": str(params_path),
            "result_plot_path": str(result_plot_path),
        },
        "data_context": {
            "n_test_samples": n_test_samples,
            "test_period_start": test_start,
            "test_period_end": test_end,
        },
        "metrics": {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "ic": float(metrics.get("ic", 0.0)),
            "ic_p_value": float(metrics.get("ic_p_value", 1.0)),
            "gap": float(metrics.get("gap_abs", abs(float(metrics.get("gap_signed", metrics.get("gap", 1.0)))))),
            "gap_signed": float(metrics.get("gap_signed", metrics.get("gap", 0.0))),
            "gap_abs": float(metrics.get("gap_abs", abs(float(metrics.get("gap_signed", metrics.get("gap", 1.0)))))),
            "ic_first": float(metrics.get("ic_first", float("nan"))),
            "ic_second": float(metrics.get("ic_second", float("nan"))),
            "proba_std_test": float(metrics.get("proba_std_test", 0.0)),
            "proba_unique_test": int(metrics.get("proba_unique_test", 0)),
            "ic_degenerate": bool(metrics.get("ic_degenerate", False)),
            "overall_pass": bool(metrics.get("overall_pass", False)),
        },
        "warnings": list(warnings),
    }
    save_json_artifact_only(_normalize_for_json(payload), metrics_path)


def _coverage_precheck(tickers=None):
    db = StockDBManager()
    db.connect()
    try:
        report = db.get_ticker_coverage_report_bulk(tickers or TICKERS)
    finally:
        db.close()
    return report


def _force_retune_all_models_for_ticker(
    ticker: str,
    benchmark: str,
    optimize_profile: str,
    n_trials_by_model: dict[str, int],
):
    print(f"  [{ticker}] force_retune_all=True -> 4모델 재최적화 시작")
    optimize_xgb(
        profile=optimize_profile,
        n_trials=int(n_trials_by_model["xgb"]),
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    optimize_svm(
        profile=optimize_profile,
        n_trials=int(n_trials_by_model["svm"]),
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    optimize_rf(
        profile=optimize_profile,
        n_trials=int(n_trials_by_model["rf"]),
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    optimize_logreg(
        profile=optimize_profile,
        n_trials=int(n_trials_by_model["logreg"]),
        ticker=ticker,
        benchmark=benchmark,
        auto_update=False,
        persist_total_features_on_update=False,
        feature_source_mode="db_first",
    )
    print(f"  [{ticker}] force 재최적화 완료")


def _get_worker_sp500_logret_series():
    global _WORKER_SP500_LOGRET_CACHE
    if _WORKER_SP500_LOGRET_CACHE is not None:
        return _WORKER_SP500_LOGRET_CACHE
    db = StockDBManager()
    db.connect()
    try:
        _WORKER_SP500_LOGRET_CACHE = db.fetch_sp500_log_returns()
    finally:
        db.close()
    return _WORKER_SP500_LOGRET_CACHE


def _get_ticker_logret_series(ticker: str):
    db = StockDBManager()
    db.connect()
    try:
        return db.fetch_log_returns_by_ticker(ticker)
    finally:
        db.close()


def _check_eligibility(split) -> tuple[bool, str]:
    train_n = int(len(split.train))
    val_n = int(len(split.val))
    test_n = int(len(split.test))

    if train_n < ELIGIBILITY_MIN_TRAIN:
        return False, f"train 샘플 부족({train_n} < {ELIGIBILITY_MIN_TRAIN})"
    if val_n < ELIGIBILITY_MIN_VAL:
        return False, f"validation 샘플 부족({val_n} < {ELIGIBILITY_MIN_VAL})"
    if test_n < ELIGIBILITY_MIN_TEST:
        return False, f"test 샘플 부족({test_n} < {ELIGIBILITY_MIN_TEST})"

    y_train_unique = split.train["Target_Class"].dropna().astype(int).nunique()
    y_val_unique = split.val["Target_Class"].dropna().astype(int).nunique()
    if y_train_unique < 2:
        return False, f"train 단일 클래스({y_train_unique})"
    if y_val_unique < 2:
        return False, f"validation 단일 클래스({y_val_unique})"

    return True, "ok"


def _build_success_row(
    *,
    ticker,
    slug,
    ens,
    mapping,
    warnings,
    model_status,
    force_retune_all,
    mode,
    model_metrics=None,
):
    n_test_samples = int(ens.get("n_test_samples", 0))
    ic_full = float(ens.get("ic_full", 0.0))
    ic_pvalue = float(ens.get("ic_pvalue", 1.0))
    reliability_tag = "LOW" if (ic_full <= 0.0 or ic_pvalue >= 0.05) else "OK"

    if n_test_samples < MIN_TEST_SAMPLES:
        warnings.append(f"n_test_samples<{MIN_TEST_SAMPLES}")
    if reliability_tag == "LOW":
        warnings.append("신뢰도 낮음(IC<=0 또는 p>=0.05)")

    ens_gap_signed = float(ens.get("gap_signed_ref", ens.get("gap_ref", 0.0)))
    ens_gap_abs = float(ens.get("gap_abs_ref", abs(ens_gap_signed)))
    ens_ic_degenerate = bool(ens.get("ic_degenerate", False))
    if ens_gap_abs > 0.25:
        warnings.append("ensemble train_test_gap_abs > 0.25")
    if ens_ic_degenerate:
        warnings.append("ensemble ic_degenerate=true")

    te_3m = None
    e_alpha_log = None
    if mapping is not None:
        signal_raw = float(mapping.get("signal_raw", 0.0))
        if abs(signal_raw) > 0.95:
            warnings.append("abs(signal_raw) > 0.95")
        e_alpha_log = float(mapping.get("E_alpha_3M_log", 0.0))
        if abs(e_alpha_log) > LOG_E_ALPHA_WARN_THRESHOLD:
            warnings.append("abs(E_alpha_3M_log) > ln(1.15)")
        for msg in mapping.get("warnings", []):
            if "E_alpha_3M_simple" in msg:
                continue
            if msg not in warnings:
                warnings.append(msg)
        te_3m = mapping.get("TE_3M")

    row = {
        "ticker": ticker,
        "ticker_slug": slug,
        "status": "success",
        "mode": mode,
        "reliability_tag": reliability_tag,
        "p_latest": ens.get("latest_future_prediction", {}).get("p_ens"),
        "ic_full": ens.get("ic_full"),
        "ic_pvalue": ens.get("ic_pvalue"),
        "gap_signed": ens_gap_signed,
        "gap_abs": ens_gap_abs,
        "ic_degenerate": ens_ic_degenerate,
        "te_3m": te_3m,
        "e_alpha_3m_log": e_alpha_log,
        "test_acc": ens.get("accuracy_ref"),
        "test_period_start": ens.get("test_start"),
        "test_period_end": ens.get("test_end"),
        "n_test_samples": n_test_samples,
        "warnings": warnings,
        "model_status": model_status,
        "force_retune_all": force_retune_all,
    }
    if model_metrics is not None:
        row["model_metrics"] = model_metrics
        row["model_diagnostics"] = {
            name: {
                "gap_signed": m.get("gap_signed"),
                "gap_abs": m.get("gap_abs"),
                "ic_degenerate": m.get("ic_degenerate"),
            }
            for name, m in model_metrics.items()
        }
    return row


def _execute_single_ticker(
    idx: int,
    ticker: str,
    coverage: dict,
    *,
    benchmark: str,
    mode: str,
    with_mapping: bool,
    force_retune_all: bool,
    optimize_profile: str,
    trial_map: dict[str, int],
    stale_policy_fast: str,
) -> dict[str, Any]:
    ticker = str(ticker).upper()
    slug = ticker_to_slug(ticker)

    print("\n" + "=" * 80)
    print(f"[{ticker}] 배치 실행 시작 (mode={mode})")
    print("=" * 80)

    if int(coverage.get("stock_rows", 0)) == 0 or int(coverage.get("logret_rows", 0)) == 0 or int(coverage.get("sp500_rows", 0)) == 0:
        reason = (
            "coverage 부족 "
            f"(STOCK_DATA={coverage.get('stock_rows', 0)}, "
            f"LOG_RETURNS={coverage.get('logret_rows', 0)}, SP500_DATA={coverage.get('sp500_rows', 0)})"
        )
        return {
            "idx": idx,
            "row": {
                "ticker": ticker,
                "ticker_slug": slug,
                "status": "skipped",
                "reason": reason,
                "warnings": [reason],
                "model_status": {},
            },
        }

    if mode == "fast" and not force_retune_all:
        ens_path = get_ensemble_result_path(ticker)
        if ens_path.exists() and (not with_mapping or get_mapping_result_path(ticker).exists()):
            return {
                "idx": idx,
                "row": {
                    "ticker": ticker,
                    "ticker_slug": slug,
                    "status": "skipped_existing",
                    "reason": "기존 산출물 존재",
                    "warnings": ["existing artifact skip"],
                    "model_status": {},
                },
            }

    warnings = []
    model_status = {}

    try:
        if force_retune_all:
            _force_retune_all_models_for_ticker(
                ticker=ticker,
                benchmark=benchmark,
                optimize_profile=optimize_profile,
                n_trials_by_model=trial_map,
            )

        sp500_logret = _get_worker_sp500_logret_series()
        ticker_logret = _get_ticker_logret_series(ticker)

        split = split_dataset(
            ticker=ticker,
            benchmark=benchmark,
            auto_update=False,
            persist_total_features_on_update=False,
            feature_source_mode="db_first",
            cached_ticker_logret=ticker_logret,
            cached_sp500_logret=sp500_logret,
        )

        eligible, reason = _check_eligibility(split)
        if not eligible:
            return {
                "idx": idx,
                "row": {
                    "ticker": ticker,
                    "ticker_slug": slug,
                    "status": "not_eligible",
                    "reason": reason,
                    "warnings": [reason],
                    "model_status": model_status,
                },
            }

        # run kwargs 공통
        xgb_run_kwargs = {
            "auto_optimize": False,
            "optimize_profile": optimize_profile,
            "ticker": ticker,
            "benchmark": benchmark,
            "save_plot": False,
            "compute_importance": False,
            "split_override": split,
        }
        svm_run_kwargs = {
            "auto_optimize": False,
            "optimize_profile": optimize_profile,
            "ticker": ticker,
            "benchmark": benchmark,
            "save_plot": False,
            "compute_importance": False,
            "split_override": split,
        }
        rf_run_kwargs = {
            "auto_optimize": False,
            "optimize_profile": optimize_profile,
            "ticker": ticker,
            "benchmark": benchmark,
            "save_plot": False,
            "compute_importance": False,
            "split_override": split,
        }
        logreg_run_kwargs = {
            "auto_optimize": False,
            "optimize_profile": optimize_profile,
            "ticker": ticker,
            "benchmark": benchmark,
            "save_plot": False,
            "compute_importance": False,
            "split_override": split,
        }

        stale_policy = stale_policy_fast if mode == "fast" else "retune_once"

        if mode == "full":
            xgb_run_kwargs.update({"return_metrics": True, "return_details": True})
            svm_run_kwargs.update({"return_metrics": True, "return_details": True})
            rf_run_kwargs.update({"return_metrics": True, "return_details": True})
            logreg_run_kwargs.update({"return_metrics": True, "return_details": True})
        else:
            xgb_run_kwargs.update({"return_metrics": False})
            svm_run_kwargs.update({"return_metrics": False})
            rf_run_kwargs.update({"return_metrics": False})
            logreg_run_kwargs.update({"return_metrics": False})

        xgb_result, xgb_status = _run_model_with_optional_retune(
            "XGB",
            run_xgb,
            xgb_run_kwargs,
            optimize_fn=optimize_xgb,
            optimize_kwargs={
                "profile": optimize_profile,
                "n_trials": int(trial_map["xgb"]),
                "ticker": ticker,
                "benchmark": benchmark,
                "auto_update": False,
                "persist_total_features_on_update": False,
                "feature_source_mode": "db_first",
            },
            stale_policy=stale_policy,
        )
        model_status["xgb"] = xgb_status

        if mode == "full":
            xgb_metrics, xgb_models, p_xgb, _ = xgb_result
            _save_model_metrics_json(
                ticker=ticker,
                benchmark=benchmark,
                model="xgb",
                metrics=xgb_metrics,
                model_status=xgb_status,
                split=split,
                optimize_profile=optimize_profile,
                n_trials=int(trial_map["xgb"]),
                force_retune_all=force_retune_all,
                warnings=_get_model_metric_warnings("xgb", xgb_metrics),
            )
            if _is_fail_fast(xgb_metrics, "XGB", ticker):
                return {
                    "idx": idx,
                    "row": {
                        "ticker": ticker,
                        "ticker_slug": slug,
                        "status": "fail_fast_skipped",
                        "skipped_after": "xgb",
                        "reason": "XGB fail-fast 조건 충족",
                        "warnings": warnings + ["fail_fast: XGB"],
                        "model_status": model_status,
                    },
                }
        else:
            xgb_models, p_xgb, _ = xgb_result
            xgb_metrics = None

        svm_result, svm_status = _run_model_with_optional_retune(
            "SVM",
            run_svm,
            svm_run_kwargs,
            optimize_fn=optimize_svm,
            optimize_kwargs={
                "profile": optimize_profile,
                "n_trials": int(trial_map["svm"]),
                "ticker": ticker,
                "benchmark": benchmark,
                "auto_update": False,
                "persist_total_features_on_update": False,
                "feature_source_mode": "db_first",
            },
            stale_policy=stale_policy,
        )
        model_status["svm"] = svm_status

        if mode == "full":
            svm_metrics, svm_models, p_svm, _ = svm_result
            _save_model_metrics_json(
                ticker=ticker,
                benchmark=benchmark,
                model="svm",
                metrics=svm_metrics,
                model_status=svm_status,
                split=split,
                optimize_profile=optimize_profile,
                n_trials=int(trial_map["svm"]),
                force_retune_all=force_retune_all,
                warnings=_get_model_metric_warnings("svm", svm_metrics),
            )
            if _is_fail_fast(svm_metrics, "SVM", ticker):
                return {
                    "idx": idx,
                    "row": {
                        "ticker": ticker,
                        "ticker_slug": slug,
                        "status": "fail_fast_skipped",
                        "skipped_after": "svm",
                        "reason": "SVM fail-fast 조건 충족",
                        "warnings": warnings + ["fail_fast: SVM"],
                        "model_status": model_status,
                    },
                }
        else:
            svm_models, p_svm, _ = svm_result
            svm_metrics = None

        rf_result, rf_status = _run_model_with_optional_retune(
            "RF",
            run_rf,
            rf_run_kwargs,
            optimize_fn=optimize_rf,
            optimize_kwargs={
                "profile": optimize_profile,
                "n_trials": int(trial_map["rf"]),
                "ticker": ticker,
                "benchmark": benchmark,
                "auto_update": False,
                "persist_total_features_on_update": False,
                "feature_source_mode": "db_first",
            },
            stale_policy=stale_policy,
        )
        model_status["rf"] = rf_status

        if mode == "full":
            rf_metrics, rf_models, p_rf, _ = rf_result
            _save_model_metrics_json(
                ticker=ticker,
                benchmark=benchmark,
                model="rf",
                metrics=rf_metrics,
                model_status=rf_status,
                split=split,
                optimize_profile=optimize_profile,
                n_trials=int(trial_map["rf"]),
                force_retune_all=force_retune_all,
                warnings=_get_model_metric_warnings("rf", rf_metrics),
            )
            if _is_fail_fast(rf_metrics, "RF", ticker):
                return {
                    "idx": idx,
                    "row": {
                        "ticker": ticker,
                        "ticker_slug": slug,
                        "status": "fail_fast_skipped",
                        "skipped_after": "rf",
                        "reason": "RF fail-fast 조건 충족",
                        "warnings": warnings + ["fail_fast: RF"],
                        "model_status": model_status,
                    },
                }
        else:
            rf_models, p_rf, _ = rf_result
            rf_metrics = None

        logreg_result, logreg_status = _run_model_with_optional_retune(
            "LOGREG",
            run_logreg,
            logreg_run_kwargs,
            optimize_fn=optimize_logreg,
            optimize_kwargs={
                "profile": optimize_profile,
                "n_trials": int(trial_map["logreg"]),
                "ticker": ticker,
                "benchmark": benchmark,
                "auto_update": False,
                "persist_total_features_on_update": False,
                "feature_source_mode": "db_first",
            },
            stale_policy=stale_policy,
        )
        model_status["logreg"] = logreg_status

        if mode == "full":
            logreg_metrics, logreg_models, p_logreg, _ = logreg_result
            _save_model_metrics_json(
                ticker=ticker,
                benchmark=benchmark,
                model="logreg",
                metrics=logreg_metrics,
                model_status=logreg_status,
                split=split,
                optimize_profile=optimize_profile,
                n_trials=int(trial_map["logreg"]),
                force_retune_all=force_retune_all,
                warnings=_get_model_metric_warnings("logreg", logreg_metrics),
            )
            if _is_fail_fast(logreg_metrics, "LOGREG", ticker):
                return {
                    "idx": idx,
                    "row": {
                        "ticker": ticker,
                        "ticker_slug": slug,
                        "status": "fail_fast_skipped",
                        "skipped_after": "logreg",
                        "reason": "LOGREG fail-fast 조건 충족",
                        "warnings": warnings + ["fail_fast: LOGREG"],
                        "model_status": model_status,
                    },
                }

            for model_name, metrics in [
                ("xgb", xgb_metrics),
                ("svm", svm_metrics),
                ("rf", rf_metrics),
                ("logreg", logreg_metrics),
            ]:
                gap_abs = float(
                    metrics.get("gap_abs", abs(float(metrics.get("gap_signed", metrics.get("gap", 0.0)))))
                )
                if gap_abs > 0.25:
                    warnings.append(f"{model_name} train_test_gap > 0.25")
                if bool(metrics.get("ic_degenerate", False)):
                    warnings.append(f"{model_name} ic_degenerate=true")
        else:
            logreg_models, p_logreg, _ = logreg_result
            logreg_metrics = None

        ens = build_equal_weight_from_probas(
            split=split,
            p_xgb=p_xgb,
            p_svm=p_svm,
            p_rf=p_rf,
            p_logreg=p_logreg,
            xgb_models=xgb_models,
            svm_models=svm_models,
            rf_models=rf_models,
            logreg_models=logreg_models,
            ticker=ticker,
            benchmark=benchmark,
            optimize_profile=optimize_profile,
            save_json=True,
            ticker_logret_series=ticker_logret,
            sp500_logret_series=sp500_logret,
        )

        mapping_obj = None
        if with_mapping:
            mapping_obj = run_mapping(ticker=ticker, benchmark=benchmark)

        model_metrics = None
        if mode == "full":
            model_metrics = {
                "xgb": xgb_metrics,
                "svm": svm_metrics,
                "rf": rf_metrics,
                "logreg": logreg_metrics,
            }

        row = _build_success_row(
            ticker=ticker,
            slug=slug,
            ens=ens,
            mapping=mapping_obj,
            warnings=warnings,
            model_status=model_status,
            force_retune_all=force_retune_all,
            mode=mode,
            model_metrics=model_metrics,
        )
        print(f"  [{ticker}] success | p_latest={row['p_latest']}, ic={float(row['ic_full']):+.4f}")
        return {"idx": idx, "row": row}

    except Exception as e:
        print(f"  [{ticker}] error: {e}")
        return {
            "idx": idx,
            "row": {
                "ticker": ticker,
                "ticker_slug": slug,
                "status": "error",
                "error": str(e),
                "warnings": warnings + [str(e)],
                "model_status": model_status,
            },
        }


def run_all_tickers(
    force_retune_all: bool = False,
    optimize_profile: str = "balanced",
    n_trials_by_model: dict[str, int] | None = None,
    mode: str = "fast",
    workers: int = 8,
    with_mapping: bool = False,
    stale_policy_fast: str = "retune_once",
    tickers: list[str] | None = None,
):
    benchmark = "SP500"
    mode = str(mode).lower().strip()
    if mode not in {"fast", "full"}:
        raise ValueError(f"mode must be fast|full: {mode}")
    if stale_policy_fast not in {"retune_once", "skip", "force"}:
        raise ValueError(f"stale_policy_fast must be retune_once|skip|force: {stale_policy_fast}")

    if mode == "full":
        with_mapping = True

    MULTI_TICKER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trial_map = {**DEFAULT_N_TRIALS_BY_MODEL, **(n_trials_by_model or {})}

    ticker_list = [str(t).upper() for t in (tickers or TICKERS)]
    coverage_report = _coverage_precheck(ticker_list)
    coverage_map = {str(r.get("ticker", "")).upper(): r for r in coverage_report}

    tasks = []
    for idx, ticker in enumerate(ticker_list):
        ticker_up = str(ticker).upper()
        tasks.append((idx, ticker_up, coverage_map.get(ticker_up, {})))

    results_by_idx = {}
    max_workers = max(1, int(workers or 1))

    if max_workers == 1:
        for idx, ticker, cov in tasks:
            out = _execute_single_ticker(
                idx,
                ticker,
                cov,
                benchmark=benchmark,
                mode=mode,
                with_mapping=with_mapping,
                force_retune_all=force_retune_all,
                optimize_profile=optimize_profile,
                trial_map=trial_map,
                stale_policy_fast=stale_policy_fast,
            )
            results_by_idx[out["idx"]] = out["row"]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(
                    _execute_single_ticker,
                    idx,
                    ticker,
                    cov,
                    benchmark=benchmark,
                    mode=mode,
                    with_mapping=with_mapping,
                    force_retune_all=force_retune_all,
                    optimize_profile=optimize_profile,
                    trial_map=trial_map,
                    stale_policy_fast=stale_policy_fast,
                ): idx
                for idx, ticker, cov in tasks
            }
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    out = fut.result()
                    results_by_idx[out["idx"]] = out["row"]
                except Exception as e:
                    ticker = tasks[idx][1]
                    results_by_idx[idx] = {
                        "ticker": ticker,
                        "ticker_slug": ticker_to_slug(ticker),
                        "status": "error",
                        "error": f"worker exception: {e}",
                        "warnings": [str(e)],
                        "model_status": {},
                    }

    summary = [results_by_idx[i] for i in sorted(results_by_idx.keys())]

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "benchmark": benchmark,
        "mode": mode,
        "workers": max_workers,
        "with_mapping": bool(with_mapping),
        "stale_policy_fast": stale_policy_fast,
        "coverage_report": coverage_report,
        "results": summary,
    }
    summary_payload = _normalize_for_json(summary_payload)

    summary_path = MULTI_TICKER_ARTIFACT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for r in summary:
        rows.append(
            {
                "ticker": r.get("ticker"),
                "ticker_slug": r.get("ticker_slug"),
                "p_latest": r.get("p_latest"),
                "ic_full": r.get("ic_full"),
                "ic_pvalue": r.get("ic_pvalue"),
                "gap_signed": r.get("gap_signed"),
                "gap_abs": r.get("gap_abs"),
                "ic_degenerate": r.get("ic_degenerate"),
                "te_3m": r.get("te_3m"),
                "e_alpha_3m_log": r.get("e_alpha_3m_log"),
                "test_acc": r.get("test_acc"),
                "test_period_start": r.get("test_period_start"),
                "test_period_end": r.get("test_period_end"),
                "n_test_samples": r.get("n_test_samples"),
                "status": r.get("status"),
                "reliability_tag": r.get("reliability_tag", "NA"),
                "warnings": " | ".join(r.get("warnings", [])),
            }
        )

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard.columns = [str(c).strip() for c in leaderboard.columns]
    if not leaderboard.empty and "e_alpha_3m_log" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by="e_alpha_3m_log", ascending=False, na_position="last")

    leaderboard_path = MULTI_TICKER_ARTIFACT_DIR / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\n" + "=" * 80)
    print("멀티티커 배치 완료")
    print(f"  mode       : {mode}")
    print(f"  workers    : {max_workers}")
    print(f"  mapping    : {with_mapping}")
    print(f"  summary    : {summary_path}")
    print(f"  leaderboard: {leaderboard_path}")
    print("=" * 80)
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-ticker model/ensemble/mapping batch")
    parser.add_argument("--force-retune-all", action="store_true", help="Retune all 4 models for each ticker before run")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "regularized"], help="Optimization profile")
    parser.add_argument("--xgb-trials", type=int, default=100, help="XGB optimize trial count")
    parser.add_argument("--svm-trials", type=int, default=100, help="SVM optimize trial count")
    parser.add_argument("--rf-trials", type=int, default=100, help="RF optimize trial count")
    parser.add_argument("--logreg-trials", type=int, default=100, help="LogReg optimize trial count")
    parser.add_argument("--mode", default="fast", choices=["fast", "full"], help="Execution mode")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker count")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker subset (optional)")
    parser.add_argument("--with-mapping", action="store_true", help="Run mapping in fast mode")
    parser.add_argument(
        "--stale-policy-fast",
        default="retune_once",
        choices=["retune_once", "skip", "force"],
        help="Stale handling policy in fast mode",
    )
    args = parser.parse_args()

    run_all_tickers(
        force_retune_all=bool(args.force_retune_all),
        optimize_profile=args.profile,
        n_trials_by_model={
            "xgb": args.xgb_trials,
            "svm": args.svm_trials,
            "rf": args.rf_trials,
            "logreg": args.logreg_trials,
        },
        mode=args.mode,
        workers=args.workers,
        with_mapping=bool(args.with_mapping),
        stale_policy_fast=args.stale_policy_fast,
        tickers=[t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else None,
    )
