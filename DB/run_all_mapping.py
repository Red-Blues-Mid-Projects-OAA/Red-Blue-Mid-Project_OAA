"""
이 파일은 분류 결과와 보조 지표를 연결하는 매핑 작업을 일괄 실행합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "DB").is_dir() and (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from DB import StockDBManager, TICKERS
from Classification.capm.capm import run_capm_single
from Classification.mapping.mapping import run_mapping_from_inputs
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR
from Classification.model_config import (
    get_ensemble_result_path,
    get_mapping_result_path,
    load_json_artifact_only,
    save_json_artifact_only,
)

def _normalize_tickers(tickers=None):
    """티커 목록 값을 서로 비교하기 쉽게 정규화합니다."""
    base = tickers if tickers is not None else TICKERS
    return [str(t).strip().upper() for t in base if str(t).strip()]

def load_ensemble_payloads(tickers: list[str]) -> dict[str, dict | None]:
    """ensemble payloads 데이터를 메모리로 불러옵니다."""
    payloads: dict[str, dict | None] = {}
    for ticker in tickers:
        data, _ = load_json_artifact_only(get_ensemble_result_path(ticker))
        payloads[ticker] = data
    return payloads

def evaluate_hard_gate_from_payload(payload: dict | None) -> tuple[bool, str]:
    """
    엄격한 3중 OAA Hard Gate 평가:
    1. Ensemble Test Accuracy >= 52% (0.52)
    2. Ensemble IC_full >= 0.05
    3. Ensemble Train-Test Gap Abs <= 25%p (0.25)
    """
    if payload is None:
        return False, "ensemble missing"

    try:
        acc = float(payload.get("accuracy_ref", 0.0))
        ic = float(payload.get("ic_full", -1.0))
        gap = float(payload.get("gap_abs_ref", payload.get("gap_ref", 1.0)))
    except Exception:
        return False, "ensemble invalid: numeric parse failed"

    if acc >= 0.52 and ic >= 0.05 and gap <= 0.25:
        return True, f"Acc={acc:.1%}, IC={ic:.4f}, Gap={gap:.1%}"
    return False, f"Failed: Acc={acc:.1%}, IC={ic:.4f}, Gap={gap:.1%}"

def _extract_mapping_inputs(payload: dict) -> tuple[float, float, str | None]:
    """매핑 계산에 필요한 입력값만 골라 정리합니다."""
    if payload is None:
        raise ValueError("ensemble payload is None")

    p_block = payload.get("latest_future_prediction", {})
    if not isinstance(p_block, dict):
        raise ValueError("latest_future_prediction 블록이 없습니다.")

    if "p_ens" not in p_block:
        raise ValueError("latest_future_prediction.p_ens 누락")
    if "ic_full" not in payload:
        raise ValueError("ensemble.ic_full 누락")

    p_latest = float(p_block["p_ens"])
    ic_full = float(payload["ic_full"])
    latest_trade_date = p_block.get("trade_date")
    return p_latest, ic_full, latest_trade_date

def run_all_mapping(tickers: list[str] | None = None):
    """전체 매핑 작업 전체를 순서대로 실행합니다."""
    ticker_list = _normalize_tickers(tickers)
    final_output_path = Path(MULTI_TICKER_ARTIFACT_DIR) / "final_expected_returns.csv"

    print("=" * 80)
    print("Final Aggregation: Strict OAA Mapping & CAPM Fallback")
    print(f"Target tickers: {len(ticker_list)}")
    print("=" * 80)

    ensemble_map = load_ensemble_payloads(ticker_list)

    results = []
    db = StockDBManager()
    db.connect()

    sp500_df = pd.DataFrame()
    stock_returns_df = pd.DataFrame()
    sp500_logret_series = pd.Series(dtype=float)

    try:
        # 배치 성능을 위해 DB는 1회 연결 후 공통 시계열을 한 번만 로드합니다.
        sp500_data = db.fetch_sp500_data()
        if not sp500_data.empty:
            sp500_df = sp500_data.loc["2024-01-01":].copy()
            sp500_df.rename(columns={"LOG_RETURN": "Rm"}, inplace=True)
            sp500_logret_series = pd.to_numeric(sp500_data["LOG_RETURN"], errors="coerce")
            sp500_logret_series.index = pd.to_datetime(sp500_logret_series.index)

        stock_returns_df = db.fetch_log_returns()
        if not stock_returns_df.empty:
            stock_returns_df.columns = stock_returns_df.columns.astype(str).str.upper()
            stock_returns_df.index = pd.to_datetime(stock_returns_df.index)

        for i, ticker in enumerate(ticker_list, 1):
            payload = ensemble_map.get(ticker)
            passed, reason = evaluate_hard_gate_from_payload(payload)

            try:
                if passed:
                    if stock_returns_df.empty or sp500_logret_series.empty:
                        raise ValueError("Mapping required data is missing from DB")
                    if ticker not in stock_returns_df.columns:
                        raise ValueError(f"Ticker {ticker} not found in LOG_RETURNS columns")

                    p_latest, ic_full, latest_trade_date = _extract_mapping_inputs(payload)
                    ticker_series = pd.to_numeric(stock_returns_df[ticker], errors="coerce")

                    res = run_mapping_from_inputs(
                        ticker=ticker,
                        benchmark="SP500",
                        p_latest=p_latest,
                        ic_full=ic_full,
                        latest_trade_date=latest_trade_date,
                        ticker_logret_series=ticker_series,
                        sp500_logret_series=sp500_logret_series,
                        save_json=True,
                        verbose=False,
                    )

                    res["Return_Type"] = "Grinold-Kahn"
                    save_json_artifact_only(res, get_mapping_result_path(ticker))

                    results.append(
                        {
                            "Ticker": ticker,
                            "Gate_Passed": True,
                            "Expected_Return_3M": float(res.get("E_alpha_3M_log", 0.0)),
                            "Return_Type": "Grinold-Kahn",
                            "Warnings": "",
                        }
                    )
                    print(f"[{i}/{len(ticker_list)}] {ticker}: GK done")
                else:
                    if sp500_df.empty or stock_returns_df.empty:
                        raise ValueError("CAPM required data is missing from DB")

                    res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                    res["Return_Type"] = "CAPM"
                    save_json_artifact_only(res, get_mapping_result_path(ticker))

                    results.append(
                        {
                            "Ticker": ticker,
                            "Gate_Passed": False,
                            "Expected_Return_3M": float(res.get("expected_capm_return_3m_log", 0.0)),
                            "Return_Type": "CAPM",
                            "Warnings": reason,
                        }
                    )
                    print(f"[{i}/{len(ticker_list)}] {ticker}: CAPM fallback ({reason})")
            except Exception as e:
                print(f"[{i}/{len(ticker_list)}] {ticker}: Error - {e}")
                results.append(
                    {
                        "Ticker": ticker,
                        "Gate_Passed": passed,
                        "Expected_Return_3M": 0.0,
                        "Return_Type": "Error",
                        "Warnings": str(e),
                    }
                )
    finally:
        db.close()

    final_df = pd.DataFrame(results)
    final_df.to_csv(final_output_path, index=False)

    print("\n" + "=" * 80)
    print("Final Aggregation Complete")
    print(f"  - Total Processed: {len(results)}")
    print(f"  - Output Saved: {final_output_path}")
    print("=" * 80)
    return final_df

def _parse_args():
    """CLI 인자를 파싱합니다. --help 호출 시 DB 접속 없이 종료됩니다."""
    parser = argparse.ArgumentParser(
        description="Run final expected return aggregation (GK mapping + CAPM fallback)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker subset (optional)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    ticker_subset = None
    if args.tickers:
        ticker_subset = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    run_all_mapping(tickers=ticker_subset)
