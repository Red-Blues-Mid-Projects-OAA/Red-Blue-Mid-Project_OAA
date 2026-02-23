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
        # Step 2: S&P 500 데이터 로드
        logger.info("Fetching S&P 500 data from DB...")
        sp500_df = db.fetch_sp500_data()
        if sp500_df.empty:
            logger.error("Failed to fetch S&P 500 data.")
            return
            
        # 2015-01-05 이후 데이터 필터링
        sp500_df = sp500_df.loc['2015-01-05':].copy()
        sp500_df.rename(columns={'LOG_RETURN': 'Rm'}, inplace=True)

        # Step 3: E(Rm) 3개월 기대수익률 산출
        sp500_mean_ewma = sp500_df['Rm'].ewm(alpha=0.06, adjust=False).mean()
        latest_date = sp500_df.index[-1]
        
        daily_E_Rm = sp500_mean_ewma.iloc[-1]
        E_Rm_3m = daily_E_Rm * 60  # 3개월 (60 거래일) 변환
        
        logger.info(f"Latest Date: {latest_date.date()}, E(Rm) Daily: {daily_E_Rm:.6f}, E(Rm) 3M: {E_Rm_3m:.6f}")

        # 개별 종목 수익률 로드
        logger.info("Fetching stock log returns from DB...")
        stock_returns_df = db.fetch_log_returns()
        if stock_returns_df.empty:
            logger.error("Failed to fetch stock returns data.")
            return

        # 무위험 수익률 3개월 변환 (Rf = 연환산 3.7%)
        Rf_annual = 0.037
        Rf_3m = Rf_annual * (60 / 252)

        for ticker in tickers:
            try:
                if ticker not in stock_returns_df.columns:
                    logger.warning(f"Ticker {ticker} not found in DB columns. Skipping.")
                    continue
                
                # S&P500과 Date 기준 Inner Join
                stock_series = stock_returns_df[ticker].rename('Ri')
                merged_df = pd.merge(sp500_df[['Rm']], stock_series, left_index=True, right_index=True, how='inner')
                merged_df.dropna(inplace=True)
                
                if merged_df.empty:
                    logger.warning(f"No overlapping data between {ticker} and S&P 500. Skipping.")
                    continue
                
                # Step 4: EWMA 기반 Beta (β) 산출
                cov_ewma = merged_df['Ri'].ewm(alpha=0.06, adjust=False).cov(merged_df['Rm'])
                var_ewma = merged_df['Rm'].ewm(alpha=0.06, adjust=False).var()
                
                beta_series = cov_ewma / var_ewma
                beta_latest = beta_series.iloc[-1]
                
                # Step 5: CAPM 3개월 기대수익률 계산
                # E(Ri) = Rf + β * (E(Rm) - Rf)
                E_Ri_3m = Rf_3m + beta_latest * (E_Rm_3m - Rf_3m)
                
                logger.info(f"[{ticker}] Beta: {beta_latest:.4f} | E(Ri)_3M: {E_Ri_3m:.6f} | E(Rm)_3M: {E_Rm_3m:.6f} | Rf_3M: {Rf_3m:.6f}")
                
                # Step 6: 결과 수집
                results.append({
                    "ticker": ticker,
                    "beta": beta_latest,
                    "expected_market_return_3m": E_Rm_3m,
                    "expected_capm_return_3m": E_Ri_3m,
                    "risk_free_rate_3m": Rf_3m,
                    "market_risk_premium_3m": E_Rm_3m - Rf_3m,
                    "date": latest_date.strftime("%Y-%m-%d")
                })
                
            except Exception as e:
                logger.error(f"Error calculating CAPM for {ticker}: {e}", exc_info=True)

        # 수집된 결과를 단일 CSV로 저장
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
