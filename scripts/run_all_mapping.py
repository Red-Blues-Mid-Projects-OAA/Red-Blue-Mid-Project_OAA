from pathlib import Path

import pandas as pd

from DB import StockDBManager, TICKERS
from Classification.capm.capm import run_capm_single
from Classification.mapping.mapping import run_mapping
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR


def _build_pass_map(leaderboard_df):
    pass_map = {}
    for _, row in leaderboard_df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        warning_val = row.get("warnings")
        passed = pd.isna(warning_val) or str(warning_val).strip() == ""
        pass_map[ticker] = passed
    return pass_map


def run_all_mapping():
    leaderboard_path = Path(MULTI_TICKER_ARTIFACT_DIR) / "leaderboard.csv"
    final_output_path = Path(MULTI_TICKER_ARTIFACT_DIR) / "final_expected_returns.csv"

    print("=" * 80)
    print("Final Aggregation: Mapping & CAPM Fallback")
    print("=" * 80)

    if not leaderboard_path.exists():
        print(f"Leaderboard not found: {leaderboard_path}")
        return

    lb_df = pd.read_csv(leaderboard_path)
    lb_df.columns = lb_df.columns.str.strip()
    pass_map = _build_pass_map(lb_df)

    for ticker in TICKERS:
        upper_ticker = str(ticker).upper()
        if upper_ticker not in pass_map:
            pass_map[upper_ticker] = False

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
            passed = bool(pass_map.get(ticker, False))
            try:
                if passed:
                    print(f"[{i}/{len(TICKERS)}] {ticker}: Gate Passed -> Grinold-Kahn Mapping")
                    res = run_mapping(ticker=ticker, benchmark="SP500")
                    results.append(
                        {
                            "Ticker": ticker,
                            "Gate_Passed": True,
                            "Expected_Return_3M": float(res.get("E_alpha_3M_log", 0.0)),
                            "Return_Type": "Grinold-Kahn",
                            "Warnings": "",
                        }
                    )
                else:
                    print(f"[{i}/{len(TICKERS)}] {ticker}: Gate Failed -> CAPM Fallback")
                    if sp500_df.empty or stock_returns_df.empty:
                        raise ValueError("CAPM required data is missing from DB")

                    res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                    results.append(
                        {
                            "Ticker": ticker,
                            "Gate_Passed": False,
                            "Expected_Return_3M": float(res.get("expected_capm_return_3m_log", 0.0)),
                            "Return_Type": "CAPM",
                            "Warnings": "",
                        }
                    )
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
