import logging
from pathlib import Path

from common import np, pd
from DB import StockDBManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CAPM_Fallback")

_THIS_DIR = Path(__file__).resolve().parent
_CLASSIFICATION_DIR = _THIS_DIR.parent
LEADERBOARD_PATH = _CLASSIFICATION_DIR / "artifacts" / "multi_ticker" / "leaderboard.csv"
CAPM_RESULTS_PATH = _THIS_DIR / "capm_results.csv"


def get_failed_tickers(leaderboard_path=LEADERBOARD_PATH):
    """
    leaderboard의 warnings 컬럼을 기준으로 Gate 실패 티커를 추출합니다.
    """
    leaderboard_path = Path(leaderboard_path)
    if not leaderboard_path.exists():
        logger.error(f"Leaderboard file not found at {leaderboard_path}")
        return []

    df = pd.read_csv(leaderboard_path)
    df.columns = df.columns.str.strip()

    failed_tickers = []
    if "warnings" in df.columns:
        # warnings가 비어 있지 않은 티커를 fallback 대상으로 간주합니다.
        mask = df["warnings"].notna() & (df["warnings"].astype(str).str.strip() != "")
        failed_tickers = df[mask]["ticker"].astype(str).str.strip().tolist()
    else:
        logger.warning("Could not find 'warnings' column in leaderboard.")

    logger.info(f"Target Tickers for CAPM Fallback (Warnings exist): {failed_tickers}")
    return failed_tickers


def run_capm_single(
    ticker: str,
    db: StockDBManager,
    sp500_df: pd.DataFrame,
    stock_returns_df: pd.DataFrame,
):
    """
    단일 티커의 CAPM 3개월 기대수익률(log)을 계산합니다.
    """
    _ = db
    Rf_annual = 0.037
    Rf_3m = Rf_annual * (60 / 252)
    Rf_daily = Rf_annual / 252

    E_Rm_annual = 0.105
    E_Rm_3m = E_Rm_annual * (60 / 252)
    latest_date = sp500_df.index[-1]

    if ticker not in stock_returns_df.columns:
        raise ValueError(f"Ticker {ticker} not found in DB columns.")

    stock_series = stock_returns_df[ticker].rename("Ri")
    merged_df = pd.merge(sp500_df[["Rm"]], stock_series, left_index=True, right_index=True, how="inner")
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        raise ValueError(f"No overlapping data between {ticker} and S&P 500.")

    merged_df["Ri_excess"] = merged_df["Ri"] - Rf_daily
    merged_df["Rm_excess"] = merged_df["Rm"] - Rf_daily

    cov_ols = merged_df["Ri_excess"].cov(merged_df["Rm_excess"])
    var_ols = merged_df["Rm_excess"].var()
    beta_ols = cov_ols / var_ols if var_ols != 0 else 0

    cov_ewma_series = merged_df["Ri_excess"].ewm(alpha=0.06, adjust=False).cov(merged_df["Rm_excess"])
    var_ewma_series = merged_df["Rm_excess"].ewm(alpha=0.06, adjust=False).var()
    beta_ewma_series = cov_ewma_series / var_ewma_series
    beta_ewma = beta_ewma_series.iloc[-1]

    beta_final = (0.7 * beta_ols) + (0.3 * beta_ewma)

    E_Ri_3m_simple = Rf_3m + beta_final * (E_Rm_3m - Rf_3m)
    E_Ri_3m_log = np.log(1 + E_Ri_3m_simple)

    return {
        "ticker": ticker,
        "beta_ols": beta_ols,
        "beta_ewma": beta_ewma,
        "beta_final": beta_final,
        "expected_market_return_3m": E_Rm_3m,
        "expected_capm_return_3m_log": E_Ri_3m_log,
        "risk_free_rate_3m": Rf_3m,
        "market_risk_premium_3m": E_Rm_3m - Rf_3m,
        "date": latest_date.strftime("%Y-%m-%d"),
    }


def calculate_capm_for_failed_tickers():
    """
    Gate 실패 티커들에 대해 CAPM 결과를 단일 CSV로 저장합니다.
    """
    tickers = get_failed_tickers()
    if not tickers:
        logger.info("No failed tickers found. Exiting CAPM fallback module.")
        return

    db = StockDBManager()
    db.connect()

    results = []
    try:
        sp500_df = db.fetch_sp500_data()
        if sp500_df.empty:
            logger.error("Failed to fetch S&P 500 data.")
            return

        sp500_df = sp500_df.loc["2024-01-01":].copy()
        sp500_df.rename(columns={"LOG_RETURN": "Rm"}, inplace=True)

        stock_returns_df = db.fetch_log_returns()
        if stock_returns_df.empty:
            logger.error("Failed to fetch stock returns data.")
            return

        for ticker in tickers:
            try:
                res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                logger.info(
                    f"Ticker: {ticker} | OLS: {res['beta_ols']:.4f} | "
                    f"EWMA: {res['beta_ewma']:.4f} | Final: {res['beta_final']:.4f}"
                )
                logger.info(f"[{ticker}] CAPM(3M) Log Return: {res['expected_capm_return_3m_log']:.6f}")
                results.append(res)
            except Exception as e:
                logger.error(f"Error calculating CAPM for {ticker}: {e}", exc_info=True)

        if results:
            pd.DataFrame(results).to_csv(CAPM_RESULTS_PATH, index=False)
            logger.info(f"All target tickers saved to a single file: {CAPM_RESULTS_PATH}")
        else:
            logger.warning("No successful calculations were generated to save.")
    finally:
        db.close()
        logger.info("DB Connection closed.")


if __name__ == "__main__":
    calculate_capm_for_failed_tickers()
