from common import yf, pd, datetime
from stock_db_manager import StockDBManager

# Define tickers
# Note: Yahoo Finance uses 'BRK-A' for Berkshire Hathaway Class A
tickers = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSM', 'TSLA', 'AVGO', 'BRK-A']

# Define date range
start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"Fetching data for: {', '.join(tickers)}")
print(f"Data range: {start_date} to {end_date} (Exclusive)")

try:
    # yfinance를 통해 수정 종가 데이터 다운로드
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # 데이터를 날짜(인덱스) 기준으로 오름차순 정렬 (연-월-일 순)하여 적재 준비
    data = data.sort_index()
    
    # Check if data is empty
    if data.empty:
        print("No data fetched. Please check your internet connection or ticker symbols.")
    else:
        # Display first few rows
        print("\nFirst 5 rows of fetched data:")
        print(data.head())
        
        # 한글 주석 필수: DB 매니저 초기화 및 데이터 저장
        print("\nOracle DB에 데이터를 저장을 시작합니다...")
        db_manager = StockDBManager()
        db_manager.connect()
        
        # 한글 주석 필수: 초기 적재 시 테이블을 깔끔하게 비우고(Truncate) 시작하여
        # 데이터가 뒤죽박죽 섞이는 것을 방지하고 엄격한 정렬 순서로 적재되도록 함
        db_manager.truncate_table()
        
        db_manager.insert_data(data)
        db_manager.close()
        print("모든 데이터 저장 완료")
        
except Exception as e:
    print(f"An error occurred: {e}")
