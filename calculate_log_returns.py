from common import pd, np
from stock_db_manager import StockDBManager

def calculate_and_save_log_returns():
    """
    Oracle DB에서 주가 데이터를 가져와 일별 로그 수익률 및 연평균 수익률을 계산하고 DB에 저장하는 함수입니다.
    """
    print("로그 수익률 계산 프로세스 시작...")
    
    # 1. DB 연결 및 데이터 로드
    db_manager = StockDBManager()
    db_manager.connect()
    
    try:
        # 한글 주석 필수: DB에서 주가 데이터(Pivot 형태) 가져오기
        주가_데이터 = db_manager.fetch_prices()
        
        if 주가_데이터.empty:
            print("데이터가 없어 계산을 중단합니다.")
            return

        print(f"주가 데이터 로드 완료: {주가_데이터.shape}")

        # 2. 로그 수익률 계산: ln(P_t / P_{t-1})
        # numpy의 log 함수를 사용하여 벡터화 연산 수행
        로그_수익률 = np.log(주가_데이터 / 주가_데이터.shift(1))
        
        # 3. 첫 번째 행은 수익률을 계산할 수 없으므로(NaN) 제거
        로그_수익률.dropna(inplace=True)
        
        # 4. 로그 수익률 데이터 DB 저장
        print(f"로그 수익률 데이터 {len(로그_수익률)}건 저장 시작...")
        db_manager.insert_log_returns(로그_수익률)
        
    except Exception as e:
        print(f"계산 중 오류 발생: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    calculate_and_save_log_returns()
