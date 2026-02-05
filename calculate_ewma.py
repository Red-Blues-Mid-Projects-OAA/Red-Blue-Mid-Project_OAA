from common import pd, np
from stock_db_manager import StockDBManager

def calculate_ewma_covariance(lambda_val=0.94):
    """
    Oracle DB에서 로그 수익률 데이터를 가져와 EWMA 공분산 행렬 및 연율화된 변동성을 계산하고 DB에 저장하는 함수입니다.
    
    Args:
        lambda_val (float): Decay Factor (기본값: 0.94)
    """
    print("EWMA 공분산 및 변동성 계산 프로세스 시작...")
    
    # 1. DB 연결 및 로그 수익률 데이터 로드
    db_manager = StockDBManager()
    db_manager.connect()
    
    try:
        # 한글 주석 필수: DB에서 로그 수익률 데이터(Pivot 형태) 가져오기
        log_returns = db_manager.fetch_log_returns()
        
        if log_returns.empty:
            print("로그 수익률 데이터가 없어 계산을 중단합니다. calculate_log_returns.py를 먼저 실행해주세요.")
            return

        print(f"로그 수익률 데이터 로드 완료: {log_returns.shape}")
        
        # 2. EWMA 공분산 행렬 계산 (pandas ewm().cov() 활용)
        # alpha = 1 - lambda
        ewma_cov_series = log_returns.ewm(alpha=(1 - lambda_val)).cov()
        
        # 3. 가장 마지막 날짜(최신)의 일별 공분산 행렬 추출
        num_stocks = len(log_returns.columns)
        latest_daily_cov = ewma_cov_series.tail(num_stocks)
        
        # 최신 날짜 추출 (MultiIndex의 첫 번째 레벨에서 가져옴)
        latest_date = latest_daily_cov.index.get_level_values(0)[0]
        
        # 인덱스 정리를 위해 날짜 레벨 제거 (행: Ticker, 열: Ticker 형태가 됨)
        latest_daily_cov_matrix = latest_daily_cov.droplevel(0)
        
        print(f"\n[{latest_date.date()}] 최신 일별 공분산 행렬:")
        print(latest_daily_cov_matrix)
        
        # 4. 연율화 (Annualization)
        # 일별 공분산에 252(거래일수)를 곱하여 연율화된 공분산 행렬 도출
        annualized_cov = latest_daily_cov_matrix * 252
        
        print("\n연율화된 공분산 행렬 (Annualized Covariance Matrix):")
        print(annualized_cov)
        
        # 5. 연율화된 변동성(Volatility) 도출
        # 공분산 행렬의 대각 성분(분산)에 루트를 씌워 표준편차(변동성) 계산
        daily_variance = latest_daily_cov_matrix.to_numpy().diagonal()
        annualized_variance = daily_variance * 252
        annualized_volatility = np.sqrt(annualized_variance)
        
        volatility_series = pd.Series(annualized_volatility, index=log_returns.columns, name="Annualized Volatility")
        
        print("\n[연율화된 변동성 저장]")
        for ticker, value in volatility_series.items():
            print(f"{ticker}: {value:.4f}")
            # 한글 주석 필수: 통계 테이블에 'ANNUAL_VOLATILITY'로 저장
            db_manager.insert_stock_stats(ticker, "ANNUAL_VOLATILITY", value)
            
        # 6. 연율화된 공분산 행렬 DB 저장
        # 한글 주석 필수: EWMA 공분산 행렬 테이블에 적재
        print(f"\nEWMA 공분산 행렬 ({latest_date.date()}) 저장 시작...")
        db_manager.insert_ewma_covariance(latest_date, annualized_cov)
            
    except Exception as e:
        print(f"계산 중 오류 발생: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    calculate_ewma_covariance()
