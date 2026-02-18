"""
10티커 배치 오케스트레이터.

Execution:
  python3 -m Classification.multi_ticker.run_all_tickers
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
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

from common import pd
from DB import StockDBManager, TICKERS
from Classification.ensemble.ensemble import run_equal_weight_ensemble
from Classification.mapping.mapping import run_mapping
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR, ticker_to_slug
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


def _run_model_with_optional_retune(model_name, run_fn, run_kwargs, optimize_fn=None, optimize_kwargs=None):
    """
    stale 파라미터 에러가 발생하면 1회 재튜닝 후 재시도합니다.
    """
    status = "normal"
    try:
        metrics = run_fn(**run_kwargs)
        return metrics, status
    except Exception as e:
        if optimize_fn is None or not _is_stale_error(e):
            raise
        print(f"  [{model_name}] stale 감지 → 1회 재튜닝 실행")
        optimize_fn(**(optimize_kwargs or {}))
        try:
            metrics = run_fn(**run_kwargs)
            status = "retuned_once_success"
            return metrics, status
        except Exception:
            status = "retuned_once_failed"
            raise


def _coverage_precheck():
    db = StockDBManager()
    db.connect()
    try:
        report = db.get_ticker_coverage_report(TICKERS)
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


def run_all_tickers(
    force_retune_all: bool = False,
    optimize_profile: str = "balanced",
    n_trials_by_model: dict[str, int] | None = None,
):
    benchmark = "SP500"
    MULTI_TICKER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trial_map = {**DEFAULT_N_TRIALS_BY_MODEL, **(n_trials_by_model or {})}

    coverage_report = _coverage_precheck()
    coverage_map = {r["ticker"]: r for r in coverage_report}

    summary = []
    for ticker in TICKERS:
        ticker = str(ticker).upper()
        slug = ticker_to_slug(ticker)
        cov = coverage_map.get(ticker, {})

        print("\n" + "=" * 80)
        print(f"[{ticker}] 배치 실행 시작")
        print("=" * 80)

        if int(cov.get("stock_rows", 0)) == 0 or int(cov.get("logret_rows", 0)) == 0 or int(cov.get("sp500_rows", 0)) == 0:
            reason = "coverage 부족(STOCK_DATA/LOG_RETURNS/SP500_DATA)"
            summary.append(
                {
                    "ticker": ticker,
                    "ticker_slug": slug,
                    "status": "skipped",
                    "reason": reason,
                    "warnings": [reason],
                }
            )
            print(f"  [SKIP] {reason}")
            continue

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

            split = split_dataset(
                ticker=ticker,
                benchmark=benchmark,
                auto_update=False,
                persist_total_features_on_update=True,
                feature_source_mode="db_first",
            )

            xgb_metrics, xgb_status = _run_model_with_optional_retune(
                "XGB",
                run_xgb,
                {
                    "auto_optimize": False,
                    "optimize_profile": "balanced",
                    "return_metrics": True,
                    "ticker": ticker,
                    "benchmark": benchmark,
                    "save_plot": False,
                    "compute_importance": False,
                    "split_override": split,
                },
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
            )
            model_status["xgb"] = xgb_status

            svm_metrics, svm_status = _run_model_with_optional_retune(
                "SVM",
                run_svm,
                {
                    "auto_optimize": False,
                    "optimize_profile": optimize_profile,
                    "return_metrics": True,
                    "ticker": ticker,
                    "benchmark": benchmark,
                    "save_plot": False,
                    "compute_importance": False,
                    "split_override": split,
                },
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
            )
            model_status["svm"] = svm_status

            rf_metrics, rf_status = _run_model_with_optional_retune(
                "RF",
                run_rf,
                {
                    "auto_optimize": False,
                    "optimize_profile": "balanced",
                    "return_metrics": True,
                    "ticker": ticker,
                    "benchmark": benchmark,
                    "save_plot": False,
                    "compute_importance": False,
                    "split_override": split,
                },
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
            )
            model_status["rf"] = rf_status

            logreg_metrics, logreg_status = _run_model_with_optional_retune(
                "LOGREG",
                run_logreg,
                {
                    "auto_optimize": False,
                    "optimize_profile": "balanced",
                    "return_metrics": True,
                    "ticker": ticker,
                    "benchmark": benchmark,
                    "save_plot": False,
                    "compute_importance": False,
                    "split_override": split,
                },
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
            )
            model_status["logreg"] = logreg_status

            # Gap 경고(모델별)
            for model_name, metrics in [
                ("xgb", xgb_metrics),
                ("svm", svm_metrics),
                ("rf", rf_metrics),
                ("logreg", logreg_metrics),
            ]:
                if float(metrics.get("gap", 0.0)) > 0.25:
                    warnings.append(f"{model_name} train_test_gap > 0.25")

            ens = run_equal_weight_ensemble(
                ticker=ticker,
                benchmark=benchmark,
                auto_update=False,
                persist_total_features_on_update=False,
                save_json=True,
            )
            mapping = run_mapping(ticker=ticker, benchmark=benchmark)

            n_test_samples = int(ens.get("n_test_samples", 0))
            if n_test_samples < MIN_TEST_SAMPLES:
                warnings.append(f"n_test_samples<{MIN_TEST_SAMPLES}")

            ic_full = float(ens.get("ic_full", 0.0))
            ic_pvalue = float(ens.get("ic_pvalue", 1.0))
            reliability_tag = "LOW" if (ic_full <= 0.0 or ic_pvalue >= 0.05) else "OK"
            if reliability_tag == "LOW":
                warnings.append("신뢰도 낮음(IC<=0 또는 p>=0.05)")

            signal_raw = float(mapping.get("signal_raw", 0.0))
            if abs(signal_raw) > 0.95:
                warnings.append("abs(signal_raw) > 0.95")

            e_alpha_log = float(mapping.get("E_alpha_3M_log", 0.0))
            if abs(e_alpha_log) > LOG_E_ALPHA_WARN_THRESHOLD:
                warnings.append("abs(E_alpha_3M_log) > ln(1.15)")

            for msg in mapping.get("warnings", []):
                # summary/leaderboard는 로그수익률 기준 경고만 유지
                if "E_alpha_3M_simple" in msg:
                    continue
                if msg not in warnings:
                    warnings.append(msg)

            row = {
                "ticker": ticker,
                "ticker_slug": slug,
                "status": "success",
                "reliability_tag": reliability_tag,
                "p_latest": ens.get("latest_future_prediction", {}).get("p_ens"),
                "ic_full": ens.get("ic_full"),
                "ic_pvalue": ens.get("ic_pvalue"),
                "te_3m": mapping.get("TE_3M"),
                "e_alpha_3m_log": mapping.get("E_alpha_3M_log"),
                "test_acc": ens.get("accuracy_ref"),
                "test_period_start": ens.get("test_start"),
                "test_period_end": ens.get("test_end"),
                "n_test_samples": n_test_samples,
                "warnings": warnings,
                "model_status": model_status,
                "force_retune_all": force_retune_all,
                "model_metrics": {
                    "xgb": xgb_metrics,
                    "svm": svm_metrics,
                    "rf": rf_metrics,
                    "logreg": logreg_metrics,
                },
            }
            summary.append(row)
            print(f"  [{ticker}] success | p_latest={row['p_latest']}, ic={row['ic_full']:+.4f}")
        except Exception as e:
            summary.append(
                {
                    "ticker": ticker,
                    "ticker_slug": slug,
                    "status": "error",
                    "error": str(e),
                    "warnings": warnings + [str(e)],
                    "model_status": model_status,
                }
            )
            print(f"  [{ticker}] error: {e}")

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "benchmark": benchmark,
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
    if not leaderboard.empty and "e_alpha_3m_log" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(by="e_alpha_3m_log", ascending=False, na_position="last")

    leaderboard_path = MULTI_TICKER_ARTIFACT_DIR / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print("\n" + "=" * 80)
    print("멀티티커 배치 완료")
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
    )
