from common import pd, np
from DB import StockDBManager

def calculate_ewma_covariance(lambda_val=0.94):
    """
    Oracle DB에서 로그 수익률 데이터를 가져와 EWMA 공분산 행렬을 계산하고 DB에 저장하는 함수입니다.
    
    Args:
        lambda_val (float): Decay Factor (기본값: 0.94)
    """
    print("EWMA 공분산 및 변동성 계산 프로세스 시작...")
    
    # 1. DB 연결 및 로그 수익률 데이터 로드
    db_manager = StockDBManager()
    db_manager.connect()
    
    try:
        # DB에서 로그 수익률 데이터(Pivot 형태) 가져오기
        log_returns = db_manager.fetch_log_returns()
        
        if log_returns.empty:
            print("로그 수익률 데이터가 없어 계산을 중단합니다. DB/calculate_log_returns.py를 먼저 실행해주세요.")
            return

        print(f"로그 수익률 데이터 로드 완료: {log_returns.shape}")
        
        # 2. EWMA 공분산 행렬 계산
        # pandas의 ewm().cov()는 감쇠계수 alpha를 사용하므로,
        # 금융 관행의 람다(lambda)와 관계를 alpha = 1 - lambda로 변환합니다.
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

        
        # 4. 일별 EWMA 공분산 행렬 DB 저장
        # 한글 주석 필수: EWMA 공분산 행렬 테이블에 적재 (일별 공분산 그대로 저장)
        print(f"\nEWMA 공분산 행렬 ({latest_date.date()}) 저장 시작...")
        db_manager.insert_ewma_covariance(latest_date, latest_daily_cov_matrix)

    finally:
        db_manager.close()

if __name__ == "__main__":
    calculate_ewma_covariance()
