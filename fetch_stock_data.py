from common import datetime, yf
from stock_db_manager import StockDBManager


DEFAULT_TICKERS = [
    "NVDA",
    "GOOGL",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "TSM",
    "TSLA",
    "AVGO",
    "BRK-A",
]


def fetch_stock_data(tickers=None, start_date="2015-01-01", end_date=None):
    """
    지정한 티커의 전체 기간 데이터를 다운로드해 DB에 적재한다.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching data for: {', '.join(tickers)}")
    print(f"Data range: {start_date} to {end_date} (Exclusive)")

    db_manager = StockDBManager()
    try:
        # yfinance를 통해 수정 종가 데이터 다운로드
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

        # 데이터가 비어있는지 확인
        if data.empty:
            print("데이터를 불러오지 못했습니다. 연결상태나 종목코드를 다시 확인 하세요")
            return False

        print("\nFirst 5 rows of fetched data:")
        print(data.head())

        print("\nOracle DB에 접속 및 테이블 생성을 시작합니다.")
        db_manager.connect()

        # 초기 적재 시 테이블을 비우고 시작
        db_manager.truncate_table()
        db_manager.insert_data(data)
        db_manager.reorganize_stock_data()
        print("모든 데이터 저장 완료")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        db_manager.close()


if __name__ == "__main__":
    fetch_stock_data()
