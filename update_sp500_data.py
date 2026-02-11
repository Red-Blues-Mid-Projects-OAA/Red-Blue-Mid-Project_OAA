from common import yf, pd, datetime, timedelta, np
from stock_db_manager import StockDBManager

def update_sp500_data():
    """
    S&P 500 지수(^GSPC) 데이터를 다운로드하고 로그 수익률을 계산하여 DB에 저장
    """
    print("S&P 500의 최신 데이터 날짜를 확인합니다.")
    
    db_manager = StockDBManager()
    
    try:
        db_manager.connect()
        
        # 1. 최신 데이터 날짜 확인 후 업데이트 시작일 결정
        latest_date = db_manager.get_latest_sp500_date()
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        if latest_date:
            # 시작일이 오늘보다 미래이거나 같으면 업데이트 불필요
            if (latest_date.date() + timedelta(days=1)) >= datetime.now().date():
                print("이미 모든 데이터가 최신 상태입니다. (Skip)")
                return
            
            # 업데이트가 필요한 경우, 수익률 계산을 위해 마지막 날짜부터 요청
            fetch_start_date = latest_date.strftime('%Y-%m-%d')
            print(f"최신 데이터 날짜: {latest_date.date()} -> 이후 날짜부터 다운로드를 진행합니다.")
        else:
            # 데이터가 아예 없는 경우
            print("기존 데이터가 없어 2015-01-01부터 전체 다운로드를 시도합니다.")
            fetch_start_date = "2015-01-01"
        
        # 2. yfinance를 사용하여 S&P 500 데이터 다운로드
        print(f"{fetch_start_date} ~ {end_date} S&P 500 데이터 다운로드 중...")
        ticker = "^GSPC"
        # auto_adjust=True로 수정종가 사용, ['Close']만 선택
        df = yf.download(ticker, start=fetch_start_date, end=end_date, auto_adjust=True, progress=True)
        
        # MultiIndex 처리 (일부 yfinance 버전에서는 단일 종목도 MultiIndex로 반환)
        if isinstance(df.columns, pd.MultiIndex):
            data = df['Close'][ticker]
        else:
            data = df['Close']
            
        if data.empty:
            print("업데이트할 데이터가 없습니다 (휴장일 등)")
            return

        # Series를 DataFrame으로 변환 (로그수익률 컬럼을 추가하기 위해)
        data = pd.DataFrame(data)
        data.columns = ['Close']
        
        # 3. 로그 수익률 계산
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # 첫 번째 행(NaN) 제거
        data.dropna(inplace=True)
        
        # 4. DB 저장
    
        if latest_date:
            # 날짜 형식 통일 (datetime -> date)
            latest_date_val = latest_date.date() if isinstance(latest_date, datetime) else latest_date
            # 이미 DB에 있는 날짜의 데이터는 제외하고 새로운 날짜의 데이터만 필터링 
            data_to_insert = data[data.index.date > latest_date_val].copy()
            
            if data_to_insert.empty:
                print("새로 추가할 데이터가 없습니다 (이미 최신 상태).")
                return
        else:
            data_to_insert = data.copy()

            
        print(f"로그 수익률 데이터 {len(data_to_insert)}건 저장 시작...")
        
        # 5. DB 저장
        db_manager.insert_sp500_data(data_to_insert)
        
    except Exception as e:
        print(f"업데이트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.close()

if __name__ == "__main__":
    update_sp500_data()
