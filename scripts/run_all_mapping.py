import sys
import os
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from DB import StockDBManager, TICKERS
from Classification.mapping.mapping import run_mapping
from Classification.capm.capm import run_capm_single
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR

def run_all_mapping():
    leaderboard_path = os.path.join(MULTI_TICKER_ARTIFACT_DIR, "leaderboard.csv")
    final_output_path = os.path.join(MULTI_TICKER_ARTIFACT_DIR, "final_expected_returns.csv")
    
    print("=" * 80)
    print("🚀 Final Aggregation: Mapping & CAPM Fallback 🚀")
    print("=" * 80)

    results = []

    db = StockDBManager()
    db.connect()
    
    sp500_df = pd.DataFrame()
    stock_returns_df = pd.DataFrame()
    
    try:
        sp500_data = db.fetch_sp500_data()
        if not sp500_data.empty:
            sp500_df = sp500_data.loc['2024-01-01':].copy()
            sp500_df.rename(columns={'LOG_RETURN': 'Rm'}, inplace=True)
            
        stock_returns_df = db.fetch_log_returns()
        
        # 3. Process each ticker
        for i, ticker in enumerate(TICKERS):
            ticker = ticker.upper()
            
            try:
                # Try Grinold-Kahn mapping first
                res = run_mapping(ticker)
                warnings = res.get("warnings", 0)
                
                if warnings == 0:
                    print(f"[{i+1}/{len(TICKERS)}] {ticker}: Gate Passed -> Grinold-Kahn Mapping")
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": True,
                        "Expected_Return_3M": res.get("E_alpha_3M_log", 0.0),
                        "Return_Type": "Grinold-Kahn",
                        "Warnings": ""
                    })
                else:
                    print(f"[{i+1}/{len(TICKERS)}] {ticker}: Gate Failed (Warnings={warnings}) -> CAPM Fallback")
                    if sp500_df.empty or stock_returns_df.empty:
                        raise ValueError("CAPM required data is missing from DB")
                        
                    capm_res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": False,
                        "Expected_Return_3M": capm_res.get("expected_capm_return_3m_log", 0.0),
                        "Return_Type": "CAPM",
                        "Warnings": f"GK Warnings: {warnings}"
                    })
            except Exception as e:
                # If GK fails (e.g., ensemble.json missing), fallback to CAPM
                try:
                    print(f"[{i+1}/{len(TICKERS)}] {ticker}: Error in GK ({e}) -> CAPM Fallback")
                    capm_res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": False,
                        "Expected_Return_3M": capm_res.get("expected_capm_return_3m_log", 0.0),
                        "Return_Type": "CAPM",
                        "Warnings": str(e)
                    })
                except Exception as capm_e:
                    print(f"❌ Error processing {ticker} (Both GK & CAPM failed): {capm_e}")
                    results.append({
                        "Ticker": ticker,
                        "Gate_Passed": False,
                        "Expected_Return_3M": 0.0,
                        "Return_Type": "Error",
                        "Warnings": f"GK Error: {e} | CAPM Error: {capm_e}"
                    })
    finally:
        db.close()

    # 4. Save Final CSV
    final_df = pd.DataFrame(results)
    final_df.to_csv(final_output_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"🎉 Final Aggregation Complete! 🎉")
    print(f"  - Total Processed: {len(results)}")
    print(f"  - Output Saved: {final_output_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_mapping()
