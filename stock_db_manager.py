
import oracledb
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# 환경 변수 로드 (파일 위치 기준 상위 디렉토리의 .env 로드)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_BASE_DIR, ".env")
load_dotenv(_ENV_PATH, override=True)

class StockDBManager:
    """
    Oracle DB 연결 및 주식 데이터 관리 클래스
    """
    def __init__(self):
        # .env 파일에서 DB 연결 정보 가져오기
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        self.connection = None
        self.cursor = None

    def connect(self):
        """
        DB 연결 설정
        """
        try:
            # Oracle DB 연결 시도
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor() # 연결 다리
            print("Oracle DB에 성공적으로 연결되었습니다.")
            
            # 테이블이 없으면 생성
            self._create_table_if_not_exists()
            
        except oracledb.Error as e:
            print(f"DB 연결 실패: {e}")
            raise

    def _create_table_if_not_exists(self):
        """
        STOCK_DATA 테이블 생성 (없을 경우)
        """
        # 테이블 생성 쿼리 (Ticker, Date를 복합 기본키로 설정)
        create_stock_data_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE STOCK_DATA (
                TICKER VARCHAR2(10),
                TRADE_DATE DATE,
                CLOSE_PRICE NUMBER,
                VOLUME NUMBER,
                PRIMARY KEY (TRADE_DATE, TICKER)
            ) ORGANIZATION INDEX';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """
        
        # 로그 수익률 테이블 생성
        create_log_returns_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE LOG_RETURNS (
                TICKER VARCHAR2(10),
                TRADE_DATE DATE,
                LOG_RETURN NUMBER,
                PRIMARY KEY (TICKER, TRADE_DATE)
            )';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """

        # EWMA 공분산 행렬 테이블 생성
        create_ewma_cov_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE EWMA_COVARIANCE (
                CALC_DATE DATE,
                TICKER_X VARCHAR2(10),
                TICKER_Y VARCHAR2(10),
                COV_VALUE NUMBER,
                PRIMARY KEY (CALC_DATE, TICKER_X, TICKER_Y)
            )';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """

        # S&P 500 데이터 테이블 생성 
        create_sp500_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE SP500_DATA (
                TRADE_DATE DATE PRIMARY KEY,
                ADJ_CLOSE NUMBER,
                LOG_RETURN NUMBER
            ) ORGANIZATION INDEX';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """

        try:
            self.cursor.execute(create_stock_data_query)
            self.cursor.execute(create_log_returns_query)
            self.cursor.execute(create_ewma_cov_query)
            self.cursor.execute(create_sp500_query)
            
            # 변경 사항 커밋
            self.connection.commit()
            print("모든 DB 테이블(STOCK_DATA, LOG_RETURNS, EWMA_COVARIANCE)이 준비되었습니다.")
        except oracledb.Error as e:
            print(f"테이블 생성 중 오류 발생: {e}")

    def recreate_stock_data_table(self):
        """
        STOCK_DATA 테이블을 삭제하고 재생성
        """
        try:
            # 존재하면 삭제
            try:
                self.cursor.execute("DROP TABLE STOCK_DATA PURGE")
            except oracledb.Error:
                pass # 테이블이 없으면 무시
            
            # 재생성
            self._create_table_if_not_exists()
            print("STOCK_DATA 테이블이 성공적으로 재생성(Recreate) 되었습니다.")
        except oracledb.Error as e:
            print(f"테이블 재생성 실패: {e}")

    def truncate_log_returns(self):
        """
        LOG_RETURNS 테이블의 모든 데이터를 삭제 (정돈된 재적재용)
        """
        try:
            # 로그 수익률 테이블을 비우고 정돈된 상태로 다시 적재하기 위함입니다.
            self.cursor.execute("TRUNCATE TABLE LOG_RETURNS")
            print("LOG_RETURNS 테이블이 성공적으로 초기화(Truncate) 되었습니다.")
        except oracledb.Error as e:
            print(f"로그 수익률 테이블 초기화 실패: {e}")

    def reorganize_log_returns(self):
        """
        LOG_RETURNS 테이블을 TICKE, DATE 오름차순으로 정렬하여 재생성
        (물리적 저장 순서 보장 + 인덱스 최적화 효과)
        """
        try:
            print("로그 수익률 테이블 재구조화(Reorganization) 시작...")
            
            # 1. 정렬된 데이터를 가진 임시 테이블 생성 (CTAS)
            create_copy_query = """
            CREATE TABLE LOG_RETURNS_COPY AS
            SELECT * FROM LOG_RETURNS
            ORDER BY TRADE_DATE ASC, TICKER ASC
            """
            self.cursor.execute(create_copy_query)
            print("1. 정렬된 임시 테이블(LOG_RETURNS_COPY) 생성 완료")
            
            # 2. 기존 테이블 삭제
            self.cursor.execute("DROP TABLE LOG_RETURNS PURGE")
            print("2. 기존 LOG_RETURNS 테이블 삭제 완료")
            
            # 3. 임시 테이블 이름을 원본 이름으로 변경
            self.cursor.execute("ALTER TABLE LOG_RETURNS_COPY RENAME TO LOG_RETURNS")
            print("3. 테이블명 변경 완료 (COPY -> ORIG)")
            
            # 4. 기본키(PK) 및 인덱스 재설정
            # LOG_RETURNS는 (TICKER, TRADE_DATE) 복합키 사용
            add_pk_query = """
            ALTER TABLE LOG_RETURNS 
            ADD CONSTRAINT PK_LOG_RETURNS PRIMARY KEY (TICKER, TRADE_DATE)
            USING INDEX
            """
            self.cursor.execute(add_pk_query)
            print("4. PK(TICKER, TRADE_DATE) 제약조건 및 인덱스 재생성 완료")
            
            print("LOG_RETURNS 테이블 재구조화 완료!")
            
        except oracledb.Error as e:
            print(f"테이블 재구조화 실패: {e}")
            # 복구 로직이 필요하다면 추가 (여기서는 로그만 출력)

    def get_latest_date(self, ticker):
        """
        특정 종목의 DB상 가장 최신 날짜 조회
        """
        query = "SELECT MAX(TRADE_DATE) FROM STOCK_DATA WHERE TICKER = :ticker"
        try:
            self.cursor.execute(query, [ticker])
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0] # datetime 객체 반환
            return None
        except oracledb.Error as e:
            print(f"{ticker} 최신 날짜 조회 실패: {e}")
            return None

    def get_latest_sp500_date(self):
        """
        SP500_DATA 테이블에서 가장 최신 날짜 조회
        """
        query = "SELECT MAX(TRADE_DATE) FROM SP500_DATA"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except oracledb.Error as e:
            print(f"S&P 500 최신 날짜 조회 실패: {e}")
            return None

    def insert_data(self, df):
        """
        DataFrame 데이터를 DB에 삽입 (Upsert 방식: )
        이미 데이터가 있는 경우 업데이트하고, 없으면 새로 삽입합니다.
        """
        # MERGE 문을 사용하여 중복 데이터 발생 시 업데이트 처리 
        insert_query = """
        MERGE INTO STOCK_DATA d
        USING (SELECT :1 as TICKER, :2 as TRADE_DATE, :3 as CLOSE_PRICE, :4 as VOLUME FROM dual) s
        ON (d.TICKER = s.TICKER AND d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.CLOSE_PRICE = s.CLOSE_PRICE, d.VOLUME = s.VOLUME
        WHEN NOT MATCHED THEN
            INSERT (TICKER, TRADE_DATE, CLOSE_PRICE, VOLUME)
            VALUES (s.TICKER, s.TRADE_DATE, s.CLOSE_PRICE, s.VOLUME)
        """
        
        data_to_insert = []
        
        try:
            # yfinance MultiIndex 데이터 처리 (level 1이 Ticker라고 가정)
            if isinstance(df.columns, pd.MultiIndex):
                # Columns: (Price, Ticker) -> Stack Ticker to Index -> Columns: Price types
                df_processed = df.stack(level=1).reset_index()
                
                # Date 컬럼 찾기
                date_col = df_processed.columns[0]
                # Ticker 컬럼 찾기 (보통 'Ticker' 혹은 'level_1')
                ticker_col = df_processed.columns[1]
                
                # Close, Volume 컬럼명 확보
                close_col = df_processed.columns[2]
                vol_col = df_processed.columns[-1]
                
                # 벡터화 연산으로 리스트 생성 (Performance Optimization)
                # 1. Date 변환: to_pydatetime().date()는 벡터화가 어려우므로 리스트 컴프리헨션 사용하되, dt 접근자 활용
                dates = df_processed[date_col].dt.date.tolist()
                tickers = df_processed[ticker_col].astype(str).tolist()                
                closes = df_processed[close_col].astype(float).tolist()
                volumes = df_processed[vol_col].astype(float).tolist()
                
                data_to_insert = list(zip(tickers, dates, closes, volumes))
            
            else:
                raise ValueError("데이터프레임의 컬럼이 예상과 다릅니다.")

            if data_to_insert:
                # executemany를 사용하여 대량 삽입 성능 향상
                # MERGE 문(Upsert)을 사용하므로 데이터가 중복되어도 안전하게 날짜순으로 들어갑니다.
                self.cursor.executemany(insert_query, data_to_insert)
                self.connection.commit()
                print(f"{len(data_to_insert)}개의 데이터가 날짜 오름차순으로 DB에 성공적으로 저장되었습니다.")
            else:
                print("저장할 데이터가 없습니다.")
                
        except Exception as e:
            print(f"데이터 삽입 실패: {e}")
            # 에러 발생 시 롤백하지 않고 오류 출력

    def fetch_prices(self):
        """
        DB에서 전체 주가 데이터를 가져와 Pivot된 DataFrame으로 반환
        Index: Date, Columns: Ticker
        """
        query = "SELECT TICKER, TRADE_DATE, CLOSE_PRICE FROM STOCK_DATA ORDER BY TRADE_DATE, TICKER"
        try:
            # 데이터 가져오기
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            if not rows:
                print("저장된 주가 데이터가 없습니다.")
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame(rows, columns=['TICKER', 'TRADE_DATE', 'CLOSE_PRICE'])
            
            # Pivot하여 사용하기 편한 형태(행: 날짜, 열: 종목)로 변환
            pivot_df = df.pivot(index='TRADE_DATE', columns='TICKER', values='CLOSE_PRICE')
            pivot_df.index = pd.to_datetime(pivot_df.index)
            return pivot_df
            
        except oracledb.Error as e:
            print(f"주가 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_log_returns(self):
        """
        DB에서 로그 수익률 데이터를 가져와 Pivot된 DataFrame으로 반환
        """
        query = "SELECT TICKER, TRADE_DATE, LOG_RETURN FROM LOG_RETURNS ORDER BY TRADE_DATE, TICKER"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['TICKER', 'TRADE_DATE', 'LOG_RETURN'])
            pivot_df = df.pivot(index='TRADE_DATE', columns='TICKER', values='LOG_RETURN')
            return pivot_df
        except oracledb.Error as e:
            print(f"로그 수익률 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_ticker_data(self, ticker):
        """
        특정 티커의 주가 및 거래량 데이터를 가져와 반환
        """
        query = """
            SELECT TRADE_DATE, CLOSE_PRICE, VOLUME 
            FROM STOCK_DATA 
            WHERE TICKER = :ticker 
            ORDER BY TRADE_DATE
        """
        try:
            self.cursor.execute(query, [ticker])
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['Date', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
            return df
        except oracledb.Error as e:
            print(f"{ticker} 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def insert_log_returns(self, df):
        """
        로그 수익률 DataFrame(Index: Date, Cols: Ticker)을 DB에 저장 (Upsert)
        """
        upsert_query = """
        MERGE INTO LOG_RETURNS d
        USING (SELECT :1 as TICKER, :2 as TRADE_DATE, :3 as LOG_RETURN FROM dual) s
        ON (d.TICKER = s.TICKER AND d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.LOG_RETURN = s.LOG_RETURN
        WHEN NOT MATCHED THEN
            INSERT (TICKER, TRADE_DATE, LOG_RETURN)
            VALUES (s.TICKER, s.TRADE_DATE, s.LOG_RETURN)
        """
        
        data_to_insert = []
        try:
            # 날짜순 정렬을 위한 Stack 및 Sort
            df_long = df.stack().reset_index()
            df_long.columns = ['TRADE_DATE', 'TICKER', 'LOG_RETURN']
            df_long = df_long.sort_values(by=['TRADE_DATE', 'TICKER'])
            
            for _, row in df_long.iterrows():
                trade_date = row['TRADE_DATE'].to_pydatetime().date()
                data_to_insert.append((str(row['TICKER']), trade_date, float(row['LOG_RETURN'])))
            
            if data_to_insert:
                self.cursor.executemany(upsert_query, data_to_insert)
                self.connection.commit()
                print(f"{len(data_to_insert)}개의 로그 수익률 데이터 저장 완료.")
        except oracledb.Error as e:
            print(f"로그 수익률 저장 실패: {e}")
            self.connection.rollback()


    def insert_ewma_covariance(self, calc_date, cov_df):
        """
        EWMA 공분산 행렬 저장 (기존 해당 날짜 데이터 삭제 후 재적재)
        """
        insert_query = "INSERT INTO EWMA_COVARIANCE (CALC_DATE, TICKER_X, TICKER_Y, COV_VALUE) VALUES (:1, :2, :3, :4)"
        
        # 계산 날짜 포맷 (시간 제거)
        calc_date_val = calc_date.date()
        
        data_to_insert = []
       
        try:
            # 먼저 테이블의 모든 기존 데이터 삭제(Truncate)
            self.cursor.execute("TRUNCATE TABLE EWMA_COVARIANCE")
            # 공분산 행렬(이중 반복문을 돌며(Loop) 각 셀의 데이터 추출)(Ticker X, Ticker Y)
            # cov_df는 컬럼과 인덱스가 모두 Ticker인 대칭 행렬
            for ticker_x in cov_df.index:
                for ticker_y in cov_df.columns:
                    value = cov_df.loc[ticker_x, ticker_y]
                    data_to_insert.append((calc_date_val, str(ticker_x), str(ticker_y), float(value)))
            
            if data_to_insert:
                self.cursor.executemany(insert_query, data_to_insert)
                self.connection.commit()
                print(f"EWMA 공분산 행렬 ({calc_date_val}) {len(data_to_insert)}건 저장 완료.")
                
        except oracledb.Error as e:
            print(f"공분산 행렬 저장 실패: {e}")
            self.connection.rollback()

    def reorganize_stock_data(self):
        """
        STOCK_DATA 테이블을 TRADE_DATE, TICKER 순으로 정렬된 복사본으로 교체
        """
        try:
            print("STOCK_DATA 테이블 재구조화 시작...")
            
            # 1. 정렬된 데이터로 복사 테이블 생성 (CTAS)
            self.cursor.execute("""
                CREATE TABLE STOCK_DATA_COPY AS
                SELECT TICKER, TRADE_DATE, CLOSE_PRICE, VOLUME
                FROM STOCK_DATA
                ORDER BY TRADE_DATE ASC, TICKER ASC
            """)

            # 2. 기존 STOCK_DATA 테이블 삭제
            self.cursor.execute("DROP TABLE STOCK_DATA PURGE")

            # 3. 복사 테이블 이름을 STOCK_DATA로 변경
            self.cursor.execute("ALTER TABLE STOCK_DATA_COPY RENAME TO STOCK_DATA")
            
            # 4. 기본키(PK) 재설정 (CTAS는 제약조건을 복사하지 않으므로 직접 설정)
            self.cursor.execute("""
                ALTER TABLE STOCK_DATA 
                ADD CONSTRAINT PK_STOCK_DATA PRIMARY KEY (TRADE_DATE, TICKER)
                USING INDEX
            """)
            
            self.connection.commit()
            print("STOCK_DATA 테이블이 TRADE_DATE, TICKER 순으로 재정렬 및 PK 설정이 완료되었습니다.")

        except oracledb.Error as e:
            print(f"테이블 재정렬 실패: {e}")
            self.connection.rollback()

    def insert_sp500_data(self, df):
        """
        S&P 500 데이터(Date index, Columns: [Close, Log_Return])를 DB에 저장 (Upsert)
        """
        upsert_query = """
        MERGE INTO SP500_DATA d
        USING (SELECT :1 as TRADE_DATE, :2 as ADJ_CLOSE, :3 as LOG_RETURN FROM dual) s
        ON (d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.ADJ_CLOSE = s.ADJ_CLOSE, d.LOG_RETURN = s.LOG_RETURN
        WHEN NOT MATCHED THEN
            INSERT (TRADE_DATE, ADJ_CLOSE, LOG_RETURN)
            VALUES (s.TRADE_DATE, s.ADJ_CLOSE, s.LOG_RETURN)
        """
        
        data_to_insert = []
        try:
            # 날짜순 정렬
            df = df.sort_index()
            
            for date_idx, row in df.iterrows():
                trade_date = date_idx.to_pydatetime().date()
                adj_close = float(row['Close'])
                log_return = float(row['Log_Return']) if pd.notna(row['Log_Return']) else None
                
                data_to_insert.append((trade_date, adj_close, log_return))
            
            if data_to_insert:
                self.cursor.executemany(upsert_query, data_to_insert)
                self.connection.commit()
                print(f"S&P 500 데이터 {len(data_to_insert)}건 저장 완료.")
        except oracledb.Error as e:
            print(f"S&P 500 데이터 저장 실패: {e}")
            self.connection.rollback()

    def close(self):
        """
        리소스 해제
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("DB 연결이 종료되었습니다.")
