from common import yf, pd, datetime
from stock_db_manager import StockDBManager


# 티커 정의

tickers = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSM', 'TSLA', 'AVGO', 'BRK-A']

# 데이터 범위 설정

start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"Fetching data for: {', '.join(tickers)}")
print(f"Data range: {start_date} to {end_date} (Exclusive)")

try:
    # yfinance를 통해 수정 종가 데이터 다운로드
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    

    
    
    # 데이터가 비어있는지 확인
    if data.empty:
        print("데이터를 불러오지 못했습니다. 연결상태나 종목코드를 다시 확인 하세요")
    else:
        # 상위5개행 출력

        print("\nFirst 5 rows of fetched data:")
        print(data.head())
        
        # DB 접속 생성자 생성 및 테이블(STOCK_DATA) 생성

        print("\nOracle DB에 접속 및 테이블 생성을 시작합니다.")
        db_manager = StockDBManager()
        db_manager.connect()
        
        # 초기 적재 시 테이블을 깔끔하게 비우고(Truncate) 시작하여
        # 데이터가 뒤죽박죽 섞이는 것을 방지하고 엄격한 정렬 순서로 적재되도록 함
        db_manager.truncate_table()
        
        db_manager.insert_data(data)
        db_manager.reorganize_stock_data()
        db_manager.close()
        print("모든 데이터 저장 완료")
        
except Exception as e:
    print(f"An error occurred: {e}")
