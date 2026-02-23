import logging

# 먼저 경로를 추가해야 common과 DB 모듈을 임포트할 수 있습니다.
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CLASSIFICATION_DIR = os.path.dirname(_THIS_DIR)
_ROOT_DIR = os.path.dirname(_CLASSIFICATION_DIR)
sys.path.insert(0, _ROOT_DIR)

# common 패키지를 활용한 깔끔한 import
from common import pd, np
from DB.stock_db_manager import StockDBManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CAPM_Fallback")

LEADERBOARD_PATH = os.path.join(_CLASSIFICATION_DIR, "artifacts", "multi_ticker", "leaderboard.csv")
CAPM_RESULTS_PATH = os.path.join(_THIS_DIR, "capm_results.csv")

def get_failed_tickers(leaderboard_path=LEADERBOARD_PATH):
    """
    Leaderboard를 읽어 Hard Gate를 통과하지 못한 종목 리스트 추출.
    기준: 'warnings' 컬럼이 비어있지 않은 종목들은 모두 CAPM Fallback 대상.
    """
    if not os.path.exists(leaderboard_path):
        logger.error(f"Leaderboard file not found at {leaderboard_path}")
        return []

    df = pd.read_csv(leaderboard_path)
    df.columns = df.columns.str.strip()
    
    failed_tickers = []
    
    if "warnings" in df.columns:
        # warnings가 컬럼에 채워져 있으면(비어있지 않으면) Gate 통과 실패로 간주
        mask = df["warnings"].notna() & (df["warnings"].str.strip() != "")
        failed_tickers = df[mask]["ticker"].str.strip().tolist()
    else:
        logger.warning("Could not find 'warnings' column in leaderboard.")

    logger.info(f"Target Tickers for CAPM Fallback (Warnings exist): {failed_tickers}")
    return failed_tickers

def run_capm_single(ticker: str, db: StockDBManager, sp500_df: pd.DataFrame, stock_returns_df: pd.DataFrame):
    """
    단일 종목에 대해 CAPM 기대수익률을 계산하여 딕셔너리로 반환.
    """
    # 무위험 수익률 3개월 변환 (Rf = 연환산 3.7%)
    Rf_annual = 0.037
    Rf_3m = Rf_annual * (60 / 252)
    Rf_daily = Rf_annual / 252
    
    # E(Rm) 3개월 기대수익률 산출 (하드코딩된 연평균 10.5% 사용)
    E_Rm_annual = 0.105
    E_Rm_3m = E_Rm_annual * (60 / 252)
    latest_date = sp500_df.index[-1]
    
    if ticker not in stock_returns_df.columns:
        raise ValueError(f"Ticker {ticker} not found in DB columns.")
        
    stock_series = stock_returns_df[ticker].rename('Ri')
    merged_df = pd.merge(sp500_df[['Rm']], stock_series, left_index=True, right_index=True, how='inner')
    merged_df.dropna(inplace=True)
    
    if merged_df.empty:
        raise ValueError(f"No overlapping data between {ticker} and S&P 500.")
        
    merged_df['Ri_excess'] = merged_df['Ri'] - Rf_daily
    merged_df['Rm_excess'] = merged_df['Rm'] - Rf_daily
    
    # 1. OLS Beta (장기/전체 기간)
    cov_ols = merged_df['Ri_excess'].cov(merged_df['Rm_excess'])
    var_ols = merged_df['Rm_excess'].var()
    beta_ols = cov_ols / var_ols if var_ols != 0 else 0
    
    # 2. EWMA Beta (단기/최근 민감도)
    cov_ewma_series = merged_df['Ri_excess'].ewm(alpha=0.06, adjust=False).cov(merged_df['Rm_excess'])
    var_ewma_series = merged_df['Rm_excess'].ewm(alpha=0.06, adjust=False).var()
    beta_ewma_series = cov_ewma_series / var_ewma_series
    beta_ewma = beta_ewma_series.iloc[-1]
    
    # 3. Shrinkage 결합 (70% OLS + 30% EWMA)
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
        "date": latest_date.strftime("%Y-%m-%d")
    }

def calculate_capm_for_failed_tickers():
    """
    Gate를 통과하지 못한 종목들에 대해 CAPM 기반 3개월 기대수익률을 계산하고 
    단일 CSV 파일(capm_results.csv)로 통합 저장.
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
            
        sp500_df = sp500_df.loc['2024-01-01':].copy()
        sp500_df.rename(columns={'LOG_RETURN': 'Rm'}, inplace=True)

        stock_returns_df = db.fetch_log_returns()
        if stock_returns_df.empty:
            logger.error("Failed to fetch stock returns data.")
            return

        for ticker in tickers:
            try:
                res = run_capm_single(ticker, db, sp500_df, stock_returns_df)
                
                logger.info(f"Ticker: {ticker} | OLS: {res['beta_ols']:.4f} | EWMA: {res['beta_ewma']:.4f} | Final: {res['beta_final']:.4f}")
                logger.info(f"[{ticker}] CAPM(3M) Log Return: {res['expected_capm_return_3m_log']:.6f}")
                results.append(res)
                
            except Exception as e:
                logger.error(f"Error calculating CAPM for {ticker}: {e}", exc_info=True)

        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(CAPM_RESULTS_PATH, index=False)
            logger.info(f"All target tickers saved to a single file: {CAPM_RESULTS_PATH}")
        else:
            logger.warning("No successful calculations were generated to save.")

    finally:
        db.close()
        logger.info("DB Connection closed.")



if __name__ == "__main__":
    calculate_capm_for_failed_tickers()
