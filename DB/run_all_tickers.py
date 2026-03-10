"""
이 파일은 여러 티커를 대상으로 필요한 배치 작업을 한 번에 순서대로 실행합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
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
    "파라미터 아티팩트가 현재 정책과 불일치",
]

ELIGIBILITY_MIN_TRAIN = 30
ELIGIBILITY_MIN_VAL = 30
ELIGIBILITY_MIN_TEST = 200

DEFAULT_INELIGIBLE_CACHE_PATH = MULTI_TICKER_ARTIFACT_DIR / "ineligible_permanent_skip.json"
SCALER_ZERO_SAMPLE_KEYWORDS = [
    "Found array with 0 sample(s)",
    "minimum of 1 is required by StandardScaler",
    "shape=(0,",
]

_WORKER_DB = None
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

def _to_iso_date(value) -> str | None:
    """날짜 값을 ISO 형식 문자열로 안전하게 변환합니다."""
    if value is None:
        return None
    try:
        return pd.to_datetime(value).date().isoformat()
    except Exception:
        return None

def _parse_date_like(value):
    """date like를 읽기 쉬운 형태로 해석합니다."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        return ts.date()
    except Exception:
        return None

def _load_ensemble_snapshot_date(ticker: str) -> tuple[str | None, bool]:
    """ensemble 스냅샷 date 데이터를 메모리로 불러옵니다."""
    ens_path = get_ensemble_result_path(ticker)
    if not ens_path.exists():
        return None, False
    try:
        payload = json.loads(ens_path.read_text(encoding="utf-8"))
    except Exception:
        return None, True
    return _to_iso_date(payload.get("data_snapshot_end_date")), True

def _decide_refresh(
    *,
    refresh_policy: str,
    latest_feature_date: str | None,
    ensemble_snapshot_date: str | None,
    artifact_exists: bool,
) -> str:
    """기존 산출물을 재생성할지 그대로 사용할지 판단합니다."""
    if refresh_policy == "always":
        return "refresh_forced"
    if not artifact_exists:
        return "missing_artifact"
    if refresh_policy == "never":
        return "fresh_skip"

    latest_dt = _parse_date_like(latest_feature_date)
    snapshot_dt = _parse_date_like(ensemble_snapshot_date)
    if snapshot_dt is None:
        return "refresh_required"
    if latest_dt is None:
        # 최신일 조회 실패 시 보수적으로 기존 산출물을 사용
        return "fresh_skip"
    if latest_dt > snapshot_dt:
        return "refresh_required"
    return "fresh_skip"

def _load_ineligible_cache(path: Path) -> dict[str, Any]:
    """ineligible 캐시 데이터를 메모리로 불러옵니다."""
    if not path.exists():
        return {"version": 1, "updated_at": None, "tickers": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"version": 1, "updated_at": None, "tickers": {}}
        payload.setdefault("version", 1)
        payload.setdefault("updated_at", None)
        payload.setdefault("tickers", {})
        if not isinstance(payload["tickers"], dict):
            payload["tickers"] = {}
        normalized = {}
        for key, value in payload["tickers"].items():
            normalized[str(key).upper()] = value if isinstance(value, dict) else {"reason": str(value)}
        payload["tickers"] = normalized
        return payload
    except Exception:
        return {"version": 1, "updated_at": None, "tickers": {}}

def _save_ineligible_cache(path: Path, payload: dict[str, Any]) -> None:
    """ineligible 캐시를 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(_normalize_for_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")

def _should_mark_permanent_ineligible(row: dict[str, Any]) -> tuple[bool, str]:
    """mark permanent ineligible 여부를 판단해 반환합니다."""
    status = str(row.get("status", "")).strip().lower()
    reason = str(row.get("reason") or row.get("error") or "")
    warnings_joined = " | ".join(str(w) for w in row.get("warnings", []))
    merged = f"{reason} | {warnings_joined}"

    if status == "not_eligible":
        return True, reason or "not_eligible"

    if status == "error" and any(k in merged for k in SCALER_ZERO_SAMPLE_KEYWORDS):
        return True, "scaler_zero_sample"

    return False, ""

def _is_stale_error(exc: Exception) -> bool:
    """stale error 여부를 판단해 반환합니다."""
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
    """모델 metric warnings 정보를 조회해 반환합니다."""
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
    """모델 metrics json를 저장합니다."""
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
    """티커별 데이터 적재 범위를 배치 실행 전에 미리 점검합니다."""
    db = StockDBManager()
    db.connect(ensure_tables=False, quiet=True)
    try:
        report = db.get_ticker_coverage_report_bulk(tickers or TICKERS)
    finally:
        db.close()
    return report

def _master_feature_latest_dates_precheck(tickers=None):
    """티커별 통합 피처의 최신 날짜를 배치 실행 전에 미리 조회합니다."""
    db = StockDBManager()
    db.connect(ensure_tables=False, quiet=True)
    try:
        db.ensure_runtime_indexes()
        return db.get_master_features_latest_dates_bulk(tickers or TICKERS)
    finally:
        db.close()

def _force_retune_all_models_for_ticker(
    ticker: str,
    benchmark: str,
    optimize_profile: str,
    n_trials_by_model: dict[str, int],
):
    """한 티커에 대해 모든 모델 재튜닝을 강제로 수행합니다."""
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

def _init_worker_runtime():
    """워커 프로세스에서 재사용할 DB 연결과 캐시를 초기화합니다."""
    global _WORKER_DB, _WORKER_SP500_LOGRET_CACHE
    if _WORKER_DB is None:
        _WORKER_DB = StockDBManager()
        _WORKER_DB.connect(ensure_tables=False, quiet=True)
    if _WORKER_SP500_LOGRET_CACHE is None:
        _WORKER_SP500_LOGRET_CACHE = _WORKER_DB.fetch_sp500_log_returns()

def _close_worker_runtime():
    """워커 프로세스에서 열어 둔 DB 연결과 캐시를 정리합니다."""
    global _WORKER_DB, _WORKER_SP500_LOGRET_CACHE
    if _WORKER_DB is not None:
        _WORKER_DB.close()
    _WORKER_DB = None
    _WORKER_SP500_LOGRET_CACHE = None

def _get_worker_db() -> StockDBManager:
    """worker 데이터베이스 정보를 조회해 반환합니다."""
    if _WORKER_DB is None:
        _init_worker_runtime()
    return _WORKER_DB

def _get_worker_sp500_logret_series():
    """worker S&P 500 logret series 정보를 조회해 반환합니다."""
    global _WORKER_SP500_LOGRET_CACHE
    if _WORKER_SP500_LOGRET_CACHE is None:
        _WORKER_SP500_LOGRET_CACHE = _get_worker_db().fetch_sp500_log_returns()
    return _WORKER_SP500_LOGRET_CACHE

def _get_ticker_logret_series(ticker: str):
    """티커 logret series 정보를 조회해 반환합니다."""
    return _get_worker_db().fetch_log_returns_by_ticker(ticker)

def _get_ticker_master_features(ticker: str):
    """티커 통합 피처 정보를 조회해 반환합니다."""
    return _get_worker_db().fetch_master_features(ticker)

def _check_eligibility(split) -> tuple[bool, str]:
    """학습에 필요한 최소 샘플 조건을 만족하는지 검사합니다."""
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
    refresh_decision,
    latest_feature_date,
    ensemble_snapshot_date,
    ineligible_cached,
    model_metrics=None,
):
    """success row 결과를 여러 데이터를 바탕으로 조합해 만듭니다."""
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
        "refresh_decision": refresh_decision,
        "latest_feature_date": latest_feature_date,
        "ensemble_snapshot_date": ensemble_snapshot_date,
        "ineligible_cached": bool(ineligible_cached),
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
    refresh_decision: str,
    latest_feature_date: str | None,
    ensemble_snapshot_date: str | None,
    ineligible_cached: bool,
) -> dict[str, Any]:
    """execute single 티커 관련 처리를 담당하는 함수입니다."""
    ticker = str(ticker).upper()
    slug = ticker_to_slug(ticker)

    print("\n" + "=" * 80)
    print(f"[{ticker}] 배치 실행 시작 (mode={mode})")
    print("=" * 80)

    base_meta = {
        "refresh_decision": refresh_decision,
        "latest_feature_date": latest_feature_date,
        "ensemble_snapshot_date": ensemble_snapshot_date,
        "ineligible_cached": bool(ineligible_cached),
    }

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
                **base_meta,
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

        worker_db = _get_worker_db()
        sp500_logret = _get_worker_sp500_logret_series()
        ticker_logret = _get_ticker_logret_series(ticker)
        master_df = _get_ticker_master_features(ticker)

        split = split_dataset(
            ticker=ticker,
            benchmark=benchmark,
            auto_update=False,
            persist_total_features_on_update=False,
            feature_source_mode="db_first",
            cached_ticker_logret=ticker_logret,
            cached_sp500_logret=sp500_logret,
            master_df_override=master_df,
            db=worker_db,
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
                    **base_meta,
                },
            }

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
                        **base_meta,
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
                        **base_meta,
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
                        **base_meta,
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
                        **base_meta,
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
            master_df_override=master_df,
            db=worker_db,
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
            refresh_decision=refresh_decision,
            latest_feature_date=latest_feature_date,
            ensemble_snapshot_date=ensemble_snapshot_date,
            ineligible_cached=ineligible_cached,
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
                **base_meta,
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
    refresh_policy: str = "freshness",
    ineligible_cache_path: str | None = None,
    respect_ineligible_cache: bool = True,
    tickers: list[str] | None = None,
):
    """전체 티커 목록 작업 전체를 순서대로 실행합니다."""
    benchmark = "SP500"
    mode = str(mode).lower().strip()
    refresh_policy = str(refresh_policy).lower().strip()
    if mode not in {"fast", "full"}:
        raise ValueError(f"mode must be fast|full: {mode}")
    if stale_policy_fast not in {"retune_once", "skip", "force"}:
        raise ValueError(f"stale_policy_fast must be retune_once|skip|force: {stale_policy_fast}")
    if refresh_policy not in {"freshness", "always", "never"}:
        raise ValueError(f"refresh_policy must be freshness|always|never: {refresh_policy}")

    if mode == "full":
        with_mapping = True

    MULTI_TICKER_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    trial_map = {**DEFAULT_N_TRIALS_BY_MODEL, **(n_trials_by_model or {})}

    ticker_list = [str(t).upper() for t in (tickers or TICKERS)]
    coverage_report = _coverage_precheck(ticker_list)
    coverage_map = {str(r.get("ticker", "")).upper(): r for r in coverage_report}
    latest_feature_map = _master_feature_latest_dates_precheck(ticker_list)

    cache_path = Path(ineligible_cache_path) if ineligible_cache_path else DEFAULT_INELIGIBLE_CACHE_PATH
    cache_payload = _load_ineligible_cache(cache_path)
    cached_tickers = set(cache_payload.get("tickers", {}).keys()) if respect_ineligible_cache else set()

    tasks = []
    results_by_idx = {}

    for idx, ticker in enumerate(ticker_list):
        ticker_up = str(ticker).upper()
        cov = coverage_map.get(ticker_up, {})
        latest_feature_date = _to_iso_date(latest_feature_map.get(ticker_up))

        if ticker_up in cached_tickers:
            reason = str(cache_payload.get("tickers", {}).get(ticker_up, {}).get("reason", "permanent ineligible"))
            results_by_idx[idx] = {
                "ticker": ticker_up,
                "ticker_slug": ticker_to_slug(ticker_up),
                "status": "skipped_permanent_ineligible",
                "reason": reason,
                "warnings": [f"permanent ineligible cache: {reason}"],
                "model_status": {},
                "refresh_decision": "ineligible_cached",
                "latest_feature_date": latest_feature_date,
                "ensemble_snapshot_date": None,
                "ineligible_cached": True,
            }
            continue

        refresh_decision = "refresh_forced" if (force_retune_all or mode == "full") else "refresh_required"
        ensemble_snapshot_date = None

        if mode == "fast" and not force_retune_all:
            ensemble_snapshot_date, artifact_exists = _load_ensemble_snapshot_date(ticker_up)
            mapping_ready = (not with_mapping) or get_mapping_result_path(ticker_up).exists()
            refresh_decision = _decide_refresh(
                refresh_policy=refresh_policy,
                latest_feature_date=latest_feature_date,
                ensemble_snapshot_date=ensemble_snapshot_date,
                artifact_exists=artifact_exists,
            )
            if refresh_decision == "fresh_skip" and not mapping_ready:
                refresh_decision = "refresh_required"

            if refresh_decision == "fresh_skip" and artifact_exists and mapping_ready:
                results_by_idx[idx] = {
                    "ticker": ticker_up,
                    "ticker_slug": ticker_to_slug(ticker_up),
                    "status": "skipped_existing_fresh",
                    "reason": "기존 산출물이 최신 DB 기준으로 신선함",
                    "warnings": ["freshness skip"],
                    "model_status": {},
                    "refresh_decision": "fresh_skip",
                    "latest_feature_date": latest_feature_date,
                    "ensemble_snapshot_date": ensemble_snapshot_date,
                    "ineligible_cached": False,
                }
                continue

        tasks.append(
            (
                idx,
                ticker_up,
                cov,
                refresh_decision,
                latest_feature_date,
                ensemble_snapshot_date,
                False,
            )
        )

    max_workers = max(1, int(workers or 1))

    if max_workers == 1:
        _init_worker_runtime()
        try:
            for idx, ticker, cov, refresh_decision, latest_feature_date, ensemble_snapshot_date, ineligible_cached in tasks:
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
                    refresh_decision=refresh_decision,
                    latest_feature_date=latest_feature_date,
                    ensemble_snapshot_date=ensemble_snapshot_date,
                    ineligible_cached=ineligible_cached,
                )
                results_by_idx[out["idx"]] = out["row"]
        finally:
            _close_worker_runtime()
    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_runtime) as ex:
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
                    refresh_decision=refresh_decision,
                    latest_feature_date=latest_feature_date,
                    ensemble_snapshot_date=ensemble_snapshot_date,
                    ineligible_cached=ineligible_cached,
                ): idx
                for idx, ticker, cov, refresh_decision, latest_feature_date, ensemble_snapshot_date, ineligible_cached in tasks
            }
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    out = fut.result()
                    results_by_idx[out["idx"]] = out["row"]
                except Exception as e:
                    ticker = next((t[1] for t in tasks if t[0] == idx), ticker_list[idx])
                    results_by_idx[idx] = {
                        "ticker": ticker,
                        "ticker_slug": ticker_to_slug(ticker),
                        "status": "error",
                        "error": f"worker exception: {e}",
                        "warnings": [str(e)],
                        "model_status": {},
                        "refresh_decision": "refresh_required",
                        "latest_feature_date": _to_iso_date(latest_feature_map.get(ticker)),
                        "ensemble_snapshot_date": None,
                        "ineligible_cached": False,
                    }

    summary = [results_by_idx[i] for i in sorted(results_by_idx.keys())]

    if respect_ineligible_cache:
        cache_changed = False
        ticker_cache = dict(cache_payload.get("tickers", {}))
        for row in summary:
            ticker = str(row.get("ticker", "")).upper()
            if not ticker:
                continue
            mark, reason = _should_mark_permanent_ineligible(row)
            if not mark:
                continue
            prev = ticker_cache.get(ticker, {})
            ticker_cache[ticker] = {
                "reason": reason or prev.get("reason") or "permanent ineligible",
                "last_seen": datetime.now().isoformat(timespec="seconds"),
                "status": row.get("status"),
            }
            cache_changed = True
        if cache_changed:
            cache_payload["tickers"] = ticker_cache
            _save_ineligible_cache(cache_path, cache_payload)

    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_payload = {
        "generated_at": generated_at,
        "benchmark": benchmark,
        "mode": mode,
        "workers": max_workers,
        "with_mapping": bool(with_mapping),
        "stale_policy_fast": stale_policy_fast,
        "refresh_policy": refresh_policy,
        "ineligible_cache_path": str(cache_path),
        "respect_ineligible_cache": bool(respect_ineligible_cache),
        "coverage_report": coverage_report,
        "latest_feature_dates": {k: _to_iso_date(v) for k, v in latest_feature_map.items()},
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
                "refresh_decision": r.get("refresh_decision"),
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
    parser.add_argument("--refresh-policy", default="freshness", choices=["freshness", "always", "never"], help="Refresh policy for existing ensemble artifacts")
    parser.add_argument("--ineligible-cache-path", default=str(DEFAULT_INELIGIBLE_CACHE_PATH), help="Permanent ineligible cache JSON path")
    parser.add_argument("--respect-ineligible-cache", action=argparse.BooleanOptionalAction, default=True, help="Respect permanent ineligible cache before running models")
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
        refresh_policy=args.refresh_policy,
        ineligible_cache_path=args.ineligible_cache_path,
        respect_ineligible_cache=bool(args.respect_ineligible_cache),
        tickers=[t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else None,
    )
