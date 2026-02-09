
from common import yf, pd, datetime, timedelta
from stock_db_manager import StockDBManager

# 한글 주석 필수: 업데이트할 종목 리스트
TICKERS = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSM', 'TSLA', 'AVGO', 'BRK-A']

def update_stock_data():
    """
    모든 종목에 대해 DB의 최신 날짜를 확인하고, 가장 늦은 날짜 이후의 데이터를 일괄 다운로드하여 업데이트하는 함수
    """
    db_manager = StockDBManager()
    
    try:
        db_manager.connect()
        
        # 1. 각 종목별 최신 날짜 조회하여 업데이트 시작일 결정
        # 가장 보수적으로(데이터가 가장 옛날에 멈춘 종목 기준으로) 시작일을 잡아야 누락 없이 채울 수 있음
        # 하지만 이미 모든 종목이 동기화되어 있다고 가정하면, MIN(MAX(Date)) + 1일을 시작일로 설정 가능
        # 여기서는 각 종목별로 최신 날짜를 확인하고, 가장 오래된 '최신 날짜'를 기준으로 잡음
        
        min_latest_date = None
        
        print("각 종목의 최신 데이터 날짜를 확인합니다...")
        for ticker in TICKERS:
            latest_date = db_manager.get_latest_date(ticker)
            if latest_date is None:
                print(f"[{ticker}] 데이터가 없습니다. 전체 초기 적재가 필요할 수 있습니다.")
                min_latest_date = None
                break # 하나라도 데이터가 없으면 전체 로드 로직으로 가는 게 안전
            
            # datetime -> date 변환
            latest_date = latest_date.date()
            
            if min_latest_date is None or latest_date < min_latest_date:
                min_latest_date = latest_date
        
        start_date = None
        if min_latest_date:
            start_date = (min_latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"\n모든 종목의 공통 업데이트 시작일: {start_date} (최소 최신 날짜 + 1일)")
        else:
            print("\n일부 또는 전체 종목의 데이터가 없어 2015-01-01부터 전체 다운로드를 시도합니다.")
            start_date = "2015-01-01"
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 한글 주석 필수: 시작일이 오늘보다 미래이거나 같으면 업데이트 불필요
        if start_date >= end_date:
            print("이미 모든 데이터가 최신 상태입니다. (Skip)")
            return

        # 2. yfinance를 통해 일괄 데이터 다운로드
        print(f"\n[{start_date} ~ {end_date}] 전체 종목 데이터 일괄 다운로드 중...")
        
        # 한글 주석 필수: auto_adjust=True로 수정종가 사용, ['Close']만 선택
        data = yf.download(TICKERS, start=start_date, end=end_date, auto_adjust=True)['Close']
        
        if data.empty:
            print("업데이트할 데이터가 없습니다 (휴장일 등).")
            return

        # 3. 데이터 전처리 및 적재
        # fetch_stock_data.py와 동일하게 insert_data는 DataFrame을 인자로 받음
        # 단, insert_data 내부에서 stack() 처리를 하므로 multi-column DataFrame을 그대로 넘기면 됨
        
        # 한글 주석 필수: 데이터를 날짜(인덱스) 기준으로 오름차순 정렬
        data = data.sort_index()
        
        print("\n다운로드된 데이터 예시 (First 5 rows):")
        print(data.head())
        
        # 한글 주석 필수: DB 적재
        # insert_data 메서드 내부에서 이미 stack() 및 정렬 로직이 구현되어 있으므로 그대로 전달
        db_manager.insert_data(data)
        
    except Exception as e:
        print(f"업데이트 프로세스 중 심각한 오류 발생: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    update_stock_data()
