from common import pd, np, os, DATASETS_DIR

def calculate_and_save_log_returns(input_file_path, output_file_path):
    """
    주가 데이터 CSV 파일로부터 일별 로그 수익률을 계산하여 저장하는 함수입니다.
    
    Args:
        input_file_path (str): 원본 주가 데이터 파일 경로
        output_file_path (str): 계산된 로그 수익률을 저장할 파일 경로
    """
    # 1. 원본 데이터 로드 (Date 컬럼을 인덱스로 설정)
    주가_데이터 = pd.read_csv(input_file_path)
    주가_데이터['Date'] = pd.to_datetime(주가_데이터['Date'])
    주가_데이터.set_index('Date', inplace=True)
    
    # 2. 로그 수익률 계산: ln(P_t / P_{t-1})
    # numpy의 log 함수를 사용하여 벡터화 연산 수행
    로그_수익률 = np.log(주가_데이터 / 주가_데이터.shift(1))
    
    # 3. 첫 번째 행은 수익률을 계산할 수 없으므로(NaN) 제거
    로그_수익률.dropna(inplace=True)
    
    # 4. 결과를 CSV 파일로 저장
    로그_수익률.to_csv(output_file_path)
    print(f"로그 수익률 데이터가 성공적으로 저장되었습니다: {output_file_path}")

if __name__ == "__main__":
    # 파일 경로 설정
    입력_파일명 = "top_10_stocks_2015_to_present.csv"
    출력_파일명 = "top_10_stocks_log_returns.csv"
    
    # datasets 폴더 기준 절대 경로 생성
    입력_경로 = os.path.join(DATASETS_DIR, 입력_파일명)
    출력_경로 = os.path.join(DATASETS_DIR, 출력_파일명)
    
    # 함수 실행
    calculate_and_save_log_returns(입력_경로, 출력_경로)
