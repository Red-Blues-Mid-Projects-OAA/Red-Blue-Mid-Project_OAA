
from common import yf, pd, datetime, timedelta
from stock_db_manager import StockDBManager

# 한글 주석 필수: 업데이트할 종목 리스트
TICKERS = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSM', 'TSLA', 'AVGO', 'BRK-A']

def update_stock_data():
    """
    각 종목별로 DB의 최신 날짜를 확인하고, 그 이후의 데이터를 가져와 업데이트하는 함수
    """
    db_manager = StockDBManager()
    
    try:
        db_manager.connect()
        
        for ticker in TICKERS:
            # 한글 주석 필수: DB에서 해당 종목의 최신 날짜 조회
            latest_date = db_manager.get_latest_date(ticker)
            
            start_date = None
            if latest_date:
                # 한글 주석 필수: 최신 날짜 다음 날부터 조회
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"[{ticker}] 최신 데이터 날짜: {latest_date.date()} -> 업데이트 시작일: {start_date}")
            else:
                # 한글 주석 필수: 데이터가 없으면 2015년부터 전체 조회
                start_date = "2015-01-01"
                print(f"[{ticker}] 데이터 없음 -> 전체 데이터 다운로드 시작 ({start_date})")
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # 한글 주석 필수: 시작일이 오늘보다 미래이거나 같으면 업데이트 불필요 (장 마감 전이라도 오늘 포함 여부 주의)
            # yfinance는 end 날짜를 포함하지 않음 (start <= date < end)
            # 따라서 start_date < end_date 여야 함
            if start_date >= end_date:
                print(f"[{ticker}] 이미 최신 데이터입니다. (Skip)")
                continue

            try:
                # 한글 주석 필수: 데이터 다운로드
                print(f"[{ticker}] {start_date} ~ {end_date} 데이터 다운로드 중...")
                data = yf.download([ticker], start=start_date, end=end_date)
                
                if data.empty:
                    print(f"[{ticker}] 업데이트할 데이터가 없습니다.")
                    continue

                # 한글 주석 필수: 'Close' 컬럼만 추출 (단일 종목이라도 MultiIndex일 수 있음)
                if 'Close' in data:
                    close_data = data['Close']
                else:
                    # 구조가 다를 경우 전체 사용 시도 (Close가 Series로 올 수 있음)
                    close_data = data
                
                # 한글 주석 필수: 데이터를 날짜(인덱스) 기준으로 오름차순 정렬
                close_data = close_data.sort_index()
                
                # insert_data 메서드는 DataFrame(컬럼=Ticker, 인덱스=Date) 형태를 기대하므로 형식 맞춤
                # 단일 종목 다운로드 시 Series로 오거나 DataFrame(Date, Close) 형태일 수 있음.
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=ticker)
                
                # yf.download([ticker])의 경우 컬럼이 해당 ticker 이름이어야 StockDBManager가 인식
                # 하지만 yf 결과의 컬럼명이 'Close'이거나 (Ticker, Close) 멀티인덱스일 수 있음.
                # 가장 안전하게 Rename 또는 재구성
                if isinstance(close_data, pd.DataFrame):
                    # 만약 컬럼이 Ticker 이름이 아니라면 변경 (Close -> Ticker)
                    if ticker not in close_data.columns:
                        close_data.columns = [ticker]
                
                # 한글 주석 필수: DB 적재
                db_manager.insert_data(close_data)
                
            except Exception as e:
                print(f"[{ticker}] 데이터 다운로드 또는 저장 실패: {e}")
                
    except Exception as e:
        print(f"업데이트 프로세스 중 심각한 오류 발생: {e}")
    finally:
        db_manager.close()

if __name__ == "__main__":
    update_stock_data()
