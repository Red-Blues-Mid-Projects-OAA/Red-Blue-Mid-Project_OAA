import os
import sys
import json
from pathlib import Path

import pandas as pd

from DB import StockDBManager, TICKERS
from Classification.capm.capm import run_capm_single
from Classification.mapping.mapping import run_mapping
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR
from Classification.model_config import (
    get_ensemble_result_path,
    get_mapping_result_path,
    load_json_artifact_only,
    save_json_artifact_only
)


def evaluate_hard_gate(ticker: str) -> tuple[bool, str]:
    """
    엄격한 3중 OAA Hard Gate 평가:
    1. Ensemble Test Accuracy >= 52% (0.52)
    2. Ensemble IC_full >= 0.05
    3. Ensemble Train-Test Gap Abs <= 25%p (0.25)
    """
    ens_path = get_ensemble_result_path(ticker)
    if not os.path.exists(ens_path):
        return False, "ensemble.json missing"
    
    data, _ = load_json_artifact_only(ens_path)
    if not data:
        return False, "Failed to load ensemble.json"
        
    acc = data.get("accuracy_ref", 0.0)
    ic = data.get("ic_full", -1.0)
    gap = data.get("gap_abs_ref", 1.0)
    
    if acc >= 0.52 and ic >= 0.05 and gap <= 0.25:
        return True, f"Acc={acc:.1%}, IC={ic:.4f}, Gap={gap:.1%}"
    else:
        return False, f"Failed: Acc={acc:.1%}, IC={ic:.4f}, Gap={gap:.1%}"


def run_all_mapping():
    final_output_path = Path(MULTI_TICKER_ARTIFACT_DIR) / "final_expected_returns.csv"

    print("=" * 80)
    print("Final Aggregation: Strict OAA Mapping & CAPM Fallback")
    print("=" * 80)

    results = []
    db = StockDBManager()
    db.connect()

    sp500_df = pd.DataFrame()
    stock_returns_df = pd.DataFrame()

    try:
        sp500_data = db.fetch_sp500_data()
        if not sp500_data.empty:
            sp500_df = sp500_data.loc["2024-01-01":].copy()
            sp500_df.rename(columns={"LOG_RETURN": "Rm"}, inplace=True)

        stock_returns_df = db.fetch_log_returns()

        for i, ticker in enumerate(TICKERS, 1):
            ticker = str(ticker).upper()
            
            passed, reason = evaluate_hard_gate(ticker)
            
            try:
                if passed:
                    print(f"[{i}/{len(TICKERS)}] {ticker}: Gate Passed ({reason}) -> Grinold-Kahn Mapping")
                    res = run_mapping(ticker=ticker, benchmark="SP500")
                    
                    # 명시적으로 Type 기록 후 덮어쓰기
                    res["Return_Type"] = "Grinold-Kahn"
                    save_json_artifact_only(res, get_mapping_result_path(ticker))
                    
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": True,
                        "Expected_Return_3M": float(res.get("E_alpha_3M_log", 0.0)),
                        "Return_Type": "Grinold-Kahn",
                        "Warnings": ""
                    })
                else:
                    print(f"[{i}/{len(TICKERS)}] {ticker}: Gate Failed ({reason}) -> CAPM Fallback")
                    if sp500_df.empty or stock_returns_df.empty:
                        raise ValueError("CAPM required data is missing from DB")

                    res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                    
                    # CAPM도 동일하게 mapping.json 생성 및 덮어쓰기
                    res["Return_Type"] = "CAPM"
                    save_json_artifact_only(res, get_mapping_result_path(ticker))
                    
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": False,
                        "Expected_Return_3M": float(res.get("expected_capm_return_3m_log", 0.0)),
                        "Return_Type": "CAPM",
                        "Warnings": reason
                    })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
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


if __name__ == "__main__":
    run_all_mapping()
