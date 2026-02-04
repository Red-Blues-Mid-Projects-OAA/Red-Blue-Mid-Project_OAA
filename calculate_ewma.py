from common import pd, np, os, DATASETS_DIR

def calculate_ewma_covariance(input_file_path, lambda_val=0.94):
    """
    일별 로그 수익률 데이터를 사용하여 EWMA 공분산 행렬을 계산하고 저장하는 함수입니다.
    
    Args:
        input_file_path (str): 일별 로그 수익률 데이터 CSV 파일 경로
        lambda_val (float): Decay Factor (기본값: 0.94)
    """
    # 1. 로그 수익률 데이터 로드 (Date를 인덱스로 설정)
    log_returns = pd.read_csv(input_file_path)
    log_returns['Date'] = pd.to_datetime(log_returns['Date'])
    log_returns.set_index('Date', inplace=True)
    
    print(f"데이터 로드 완료: {log_returns.shape}")
    
    # 2. EWMA 공분산 행렬 계산 (pandas ewm().cov() 활용)
    # alpha = 1 - lambda
    # span 등을 사용할 수도 있지만, 재귀적 수식에 가장 부합하는 alpha 지정 방식 사용
    ewma_cov_series = log_returns.ewm(alpha=(1 - lambda_val)).cov()
    
    # 3. 가장 마지막 날짜(최신)의 일별 공분산 행렬 추출
    # ewm().cov()는 (날짜 수 * 종목 수) x 종목 수 형태의 MultiIndex DataFrame을 반환합니다.
    # tail()을 사용하여 마지막 시점의 종목 수만큼의 행을 가져옵니다.
    num_stocks = len(log_returns.columns)
    latest_daily_cov = ewma_cov_series.tail(num_stocks)
    
    print("\n최신 일별 공분산 행렬 (Latest Daily Covariance Matrix):")
    print(latest_daily_cov)
    
    # 4. 연율화 (Annualization)
    # 일별 공분산에 252(거래일수)를 곱하여 연율화된 공분산 행렬 도출
    annualized_cov = latest_daily_cov * 252
    
    print("\n연율화된 공분산 행렬 (Annualized Covariance Matrix):")
    print(annualized_cov)
    
    # 5. 연율화된 변동성(Volatility) 도출
    # 공분산 행렬의 대각 성분(분산)에 루트를 씌워 표준편차(변동성) 계산
    # droplevel을 통해 Date 인덱스 계층 제거 후 계산
    daily_variance = latest_daily_cov.droplevel(0).to_numpy().diagonal()
    annualized_variance = daily_variance * 252
    annualized_volatility = np.sqrt(annualized_variance)
    
    # 변동성 결과를 보기 좋게 Series로 변환
    volatility_series = pd.Series(annualized_volatility, index=log_returns.columns, name="Annualized Volatility")
    
    print("\n최송 연율화된 변동성 (Annualized Volatility):")
    print(volatility_series)
    
    # (1) 최신 일별 공분산 행렬 저장
    daily_cov_path = os.path.join(DATASETS_DIR, "latest_daily_ewma_covariance.csv")
    latest_daily_cov.to_csv(daily_cov_path)
    print(f"\n파일 저장 완료: {daily_cov_path}")
    
    # (2) 연율화된 공분산 행렬 저장
    annual_cov_path = os.path.join(DATASETS_DIR, "annualized_ewma_covariance.csv")
    annualized_cov.to_csv(annual_cov_path)
    print(f"파일 저장 완료: {annual_cov_path}")
    
    # (3) 연율화된 변동성 저장
    vol_path = os.path.join(DATASETS_DIR, "annualized_ewma_volatility.csv")
    volatility_series.to_csv(vol_path)
    print(f"파일 저장 완료: {vol_path}")

if __name__ == "__main__":
    # 파일 경로 설정
    입력_파일명 = "top_10_stocks_log_returns.csv"
    입력_경로 = os.path.join(DATASETS_DIR, 입력_파일명)
    
    try:
        calculate_ewma_covariance(입력_경로)
    except Exception as e:
        print(f"오류 발생: {e}")
