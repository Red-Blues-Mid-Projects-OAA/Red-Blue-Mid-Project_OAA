import math

if __package__ in (None, ""):
    from datetime import datetime
    import os

    import oracledb
    import pandas as pd
    from dotenv import load_dotenv
else:
    from common import datetime, load_dotenv, oracledb, os, pd

# 환경 변수 로드 (프로젝트 루트 .env 우선)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BASE_DIR)
_ENV_CANDIDATES = [
    os.path.join(_PROJECT_ROOT, ".env"),
    os.path.join(_BASE_DIR, ".env"),
]
for _env_path in _ENV_CANDIDATES:
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
        break

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
                HIGH_PRICE NUMBER,
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

        # MARKET_FEATURES 테이블 생성 (VIX, DXY 등 시장 지표)
        create_market_features_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE MARKET_FEATURES (
                INDICATOR  VARCHAR2(20),
                TRADE_DATE DATE,
                CLOSE_VALUE NUMBER,
                LOG_RETURN  NUMBER,
                PRIMARY KEY (TRADE_DATE, INDICATOR)
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
            self.cursor.execute(create_market_features_query)
            
            # 변경 사항 커밋
            self.connection.commit()
            print("모든 DB 테이블이 준비되었습니다.")
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

    def get_ticker_coverage_report(self, tickers):
        """
        다중 티커의 최소 커버리지를 점검해 리스트(dict) 형태로 반환합니다.
        반환 컬럼:
          - ticker
          - stock_rows
          - logret_rows
          - sp500_rows
        """
        normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not normalized:
            return []

        sp500_rows = 0
        try:
            self.cursor.execute("SELECT COUNT(*) FROM SP500_DATA")
            sp500_rows = int(self.cursor.fetchone()[0] or 0)
        except oracledb.Error:
            sp500_rows = 0

        report = []
        for ticker in normalized:
            stock_rows = 0
            logret_rows = 0
            try:
                self.cursor.execute(
                    "SELECT COUNT(*) FROM STOCK_DATA WHERE TICKER = :ticker",
                    {"ticker": ticker},
                )
                stock_rows = int(self.cursor.fetchone()[0] or 0)
                self.cursor.execute(
                    "SELECT COUNT(*) FROM LOG_RETURNS WHERE TICKER = :ticker",
                    {"ticker": ticker},
                )
                logret_rows = int(self.cursor.fetchone()[0] or 0)
            except oracledb.Error as e:
                print(f"{ticker} coverage query failed: {e}")

            report.append(
                {
                    "ticker": ticker,
                    "stock_rows": stock_rows,
                    "logret_rows": logret_rows,
                    "sp500_rows": sp500_rows,
                }
            )

        return report

    def insert_data(self, df):
        """
        DataFrame 데이터를 DB에 삽입 (Upsert 방식: )
        이미 데이터가 있는 경우 업데이트하고, 없으면 새로 삽입합니다.
        """
        # MERGE 문을 사용하여 중복 데이터 발생 시 업데이트 처리 
        insert_query = """
        MERGE INTO STOCK_DATA d
        USING (SELECT :1 as TICKER, :2 as TRADE_DATE, :3 as CLOSE_PRICE, :4 as HIGH_PRICE, :5 as VOLUME FROM dual) s
        ON (d.TICKER = s.TICKER AND d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.CLOSE_PRICE = s.CLOSE_PRICE, d.HIGH_PRICE = s.HIGH_PRICE, d.VOLUME = s.VOLUME
        WHEN NOT MATCHED THEN
            INSERT (TICKER, TRADE_DATE, CLOSE_PRICE, HIGH_PRICE, VOLUME)
            VALUES (s.TICKER, s.TRADE_DATE, s.CLOSE_PRICE, s.HIGH_PRICE, s.VOLUME)
        """
        
        data_to_insert = []
        
        try:
            # yfinance MultiIndex 데이터 처리 (level 1이 Ticker라고 가정)
            if isinstance(df.columns, pd.MultiIndex):
                # yfinance 멀티 인덱스 컬럼((가격종류, 티커))을
                # 행 인덱스(날짜, 티커) 형태로 재배치해 종목별 업서트 입력을
                # 벡터화로 처리하기 쉽게 만듭니다.
                try:
                    df_processed = df.stack(level=1, future_stack=True).reset_index()
                except TypeError:
                    df_processed = df.stack(level=1).reset_index()
                
                # Date 컬럼 찾기
                date_col = df_processed.columns[0]
                # Ticker 컬럼 찾기 (보통 'Ticker' 혹은 'level_1')
                ticker_col = df_processed.columns[1]
                
                # Close, High, Volume 컬럼명 확보 (컬럼명으로 직접 검색)
                col_names = [str(c) for c in df_processed.columns]
                close_col = next(c for c in df_processed.columns if str(c) == 'Close')
                high_col = next(c for c in df_processed.columns if str(c) == 'High')
                vol_col = next(c for c in df_processed.columns if str(c) == 'Volume')
                
                # 벡터화 연산으로 리스트 생성 (Performance Optimization)
                # 1. Date 변환: to_pydatetime().date()는 벡터화가 어려우므로 리스트 컴프리헨션 사용하되, dt 접근자 활용
                dates = df_processed[date_col].dt.date.tolist()
                tickers = df_processed[ticker_col].astype(str).tolist()                
                
                # Oracle DB는 np.nan을 받으면 DPY-4004 에러를 발생하므로 None으로 변환
                closes = [None if pd.isna(x) else float(x) for x in df_processed[close_col]]
                highs = [None if pd.isna(x) else float(x) for x in df_processed[high_col]]
                volumes = [None if pd.isna(x) else float(x) for x in df_processed[vol_col]]
                
                data_to_insert = list(zip(tickers, dates, closes, highs, volumes))
            
            else:
                raise ValueError("데이터프레임의 컬럼이 예상과 다릅니다.")

            if data_to_insert:
                # executemany를 사용하여 대량 삽입 성능 향상
                # MERGE 문(Upsert)을 사용하므로 데이터가 중복되어도 안전하게 날짜순으로 들어갑니다.
                batch_size = 10000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    self.cursor.executemany(insert_query, batch)
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
            SELECT TRADE_DATE, CLOSE_PRICE, HIGH_PRICE, VOLUME 
            FROM STOCK_DATA 
            WHERE TICKER = :ticker 
            ORDER BY TRADE_DATE
        """
        try:
            self.cursor.execute(query, [ticker])
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['Date', 'Close', 'High', 'Volume'])
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
                log_ret = row['LOG_RETURN']
                if pd.isna(log_ret):
                    continue
                try:
                    log_ret_val = float(log_ret)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(log_ret_val):
                    continue

                trade_date = row['TRADE_DATE'].to_pydatetime().date()
                data_to_insert.append((str(row['TICKER']), trade_date, log_ret_val))
            
            if data_to_insert:
                batch_size = 10000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    self.cursor.executemany(upsert_query, batch)
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
                batch_size = 10000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    self.cursor.executemany(insert_query, batch)
                self.connection.commit()
                print(f"[{calc_date_val}] EWMA 공분산 행렬 {len(data_to_insert)}건 저장 완료.")
                
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
                SELECT TICKER, TRADE_DATE, CLOSE_PRICE, HIGH_PRICE, VOLUME
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

    def fetch_sp500_data(self):
        """
        DB에서 S&P 500 데이터(TRADE_DATE, LOG_RETURN)를 가져와 DataFrame으로 반환
        """
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA ORDER BY TRADE_DATE"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows, columns=['TRADE_DATE', 'LOG_RETURN'])
            df.set_index('TRADE_DATE', inplace=True)
            return df
        except oracledb.Error as e:
            print(f"S&P 500 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def insert_market_features(self, indicator, df):
        """
        시장 지표 데이터(Date index, Columns: [Close, Log_Return])를 DB에 저장 (Upsert)
        indicator: 'VIX' 또는 'DXY'
        """
        upsert_query = """
        MERGE INTO MARKET_FEATURES d
        USING (SELECT :1 as INDICATOR, :2 as TRADE_DATE, :3 as CLOSE_VALUE, :4 as LOG_RETURN FROM dual) s
        ON (d.INDICATOR = s.INDICATOR AND d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.CLOSE_VALUE = s.CLOSE_VALUE, d.LOG_RETURN = s.LOG_RETURN
        WHEN NOT MATCHED THEN
            INSERT (INDICATOR, TRADE_DATE, CLOSE_VALUE, LOG_RETURN)
            VALUES (s.INDICATOR, s.TRADE_DATE, s.CLOSE_VALUE, s.LOG_RETURN)
        """

        data_to_insert = []
        try:
            df = df.sort_index()
            for date_idx, row in df.iterrows():
                trade_date = date_idx.to_pydatetime().date() if hasattr(date_idx, 'to_pydatetime') else date_idx
                close_val = float(row['Close'])
                log_ret = float(row['Log_Return']) if pd.notna(row['Log_Return']) else None
                data_to_insert.append((indicator, trade_date, close_val, log_ret))

            if data_to_insert:
                batch_size = 10000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    self.cursor.executemany(upsert_query, batch)
                self.connection.commit()
                print(f"{indicator} 데이터 {len(data_to_insert)}건 저장 완료.")
        except oracledb.Error as e:
            print(f"{indicator} 데이터 저장 실패: {e}")
            self.connection.rollback()

    def fetch_market_features(self, indicator):
        """
        DB에서 특정 시장 지표(VIX/DXY) 데이터를 조회하여 DataFrame으로 반환
        """
        query = """
            SELECT TRADE_DATE, CLOSE_VALUE, LOG_RETURN
            FROM MARKET_FEATURES
            WHERE INDICATOR = :indicator
            ORDER BY TRADE_DATE
        """
        try:
            self.cursor.execute(query, [indicator])
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['TRADE_DATE', 'Close', 'Log_Return'])
            df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
            df.set_index('TRADE_DATE', inplace=True)
            return df
        except oracledb.Error as e:
            print(f"{indicator} 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def get_latest_market_date(self, indicator):
        """
        특정 시장 지표의 DB상 가장 최신 날짜 조회
        """
        query = "SELECT MAX(TRADE_DATE) FROM MARKET_FEATURES WHERE INDICATOR = :indicator"
        try:
            self.cursor.execute(query, [indicator])
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except oracledb.Error as e:
            print(f"{indicator} 최신 날짜 조회 실패: {e}")
            return None

    def reorganize_sp500_data(self):
        """
        SP500_DATA 테이블을 TRADE_DATE 순으로 정렬된 복사본으로 교체
        (물리적 저장 순서 보장 + 인덱스 최적화 효과)
        """
        try:
            print("S&P 500 테이블 재구조화(Reorganization) 시작...")

            # 1. 정렬된 데이터를 가진 임시 테이블 생성 (CTAS)
            create_copy_query = """
            CREATE TABLE SP500_DATA_COPY AS
            SELECT * FROM SP500_DATA
            ORDER BY TRADE_DATE ASC
            """
            self.cursor.execute(create_copy_query)
            print("1. 정렬된 임시 테이블(SP500_DATA_COPY) 생성 완료")

            # 2. 기존 테이블 삭제
            self.cursor.execute("DROP TABLE SP500_DATA PURGE")
            print("2. 기존 SP500_DATA 테이블 삭제 완료")

            # 3. 임시 테이블 이름을 원본 이름으로 변경
            self.cursor.execute("ALTER TABLE SP500_DATA_COPY RENAME TO SP500_DATA")
            print("3. 테이블명 변경 완료 (COPY -> ORIG)")

            # 4. 기본키(PK) 및 인덱스 재설정
            add_pk_query = """
            ALTER TABLE SP500_DATA 
            ADD CONSTRAINT PK_SP500_DATA PRIMARY KEY (TRADE_DATE)
            USING INDEX
            """
            self.cursor.execute(add_pk_query)
            print("4. PK(TRADE_DATE) 제약조건 및 인덱스 재생성 완료")

            self.connection.commit()
            print("S&P 500 테이블 재구조화 완료!")

        except oracledb.Error as e:
            print(f"S&P 500 테이블 재구조화 실패: {e}")
            self.connection.rollback()

    def reorganize_market_features(self):
        """
        MARKET_FEATURES 테이블을 TRADE_DATE, INDICATOR 순으로 정렬된 복사본으로 교체
        """
        try:
            print("MARKET_FEATURES 테이블 재구조화 시작...")

            # 1. 정렬된 데이터로 복사 테이블 생성 (CTAS)
            self.cursor.execute("""
                CREATE TABLE MARKET_FEATURES_COPY AS
                SELECT INDICATOR, TRADE_DATE, CLOSE_VALUE, LOG_RETURN
                FROM MARKET_FEATURES
                ORDER BY TRADE_DATE ASC, INDICATOR ASC
            """)

            # 2. 기존 테이블 삭제
            self.cursor.execute("DROP TABLE MARKET_FEATURES PURGE")

            # 3. 복사 테이블 이름을 원본으로 변경
            self.cursor.execute("ALTER TABLE MARKET_FEATURES_COPY RENAME TO MARKET_FEATURES")

            # 4. 기본키(PK) 재설정
            self.cursor.execute("""
                ALTER TABLE MARKET_FEATURES
                ADD CONSTRAINT PK_MARKET_FEATURES PRIMARY KEY (TRADE_DATE, INDICATOR)
                USING INDEX
            """)

            self.connection.commit()
            print("MARKET_FEATURES 테이블이 TRADE_DATE, INDICATOR 순으로 재정렬 및 PK 설정이 완료되었습니다.")

        except oracledb.Error as e:
            print(f"MARKET_FEATURES 테이블 재정렬 실패: {e}")
            self.connection.rollback()


    def master_features_exists(self):
        """MASTER_FEATURES 테이블 존재 여부를 확인합니다."""
        self.cursor.execute(
            "SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = 'MASTER_FEATURES'"
        )
        return int(self.cursor.fetchone()[0] or 0) > 0

    def insert_master_features(self, df):
        """
        단일 MASTER_FEATURES 패널 테이블에 데이터를 업서트(Upsert) 합니다.
        DataFrame은 'TICKER', 'TRADE_DATE' 컬럼을 반드시 포함해야 합니다.
        """
        table_name = "MASTER_FEATURES"
        
        df = df.copy()
        if "TRADE_DATE" not in df.columns and "TICKER" not in df.columns:
            df.reset_index(inplace=True)

        cols = list(df.columns)
        if "TRADE_DATE" not in cols or "TICKER" not in cols:
            raise ValueError("MASTER_FEATURES 적재에는 TICKER와 TRADE_DATE 컬럼이 필요합니다.")
            
        # 1. 동적 테이블 생성/컬럼 추가
        col_defs = []
        for col in cols:
            col_type = "DATE" if col == "TRADE_DATE" else ("VARCHAR2(50)" if col == "TICKER" else "NUMBER")
            col_defs.append(f'"{col}" {col_type}')
            
        # PK 제약 추가
        col_defs.append("CONSTRAINT PK_MASTER_FEATURES PRIMARY KEY (TRADE_DATE, TICKER)")

        create_table_query = f"""
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE {table_name} ({", ".join(col_defs)})';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN RAISE; END IF;
        END;
        """

        try:
            self.cursor.execute(create_table_query)

            # 기존 컬럼 확인 및 누락 컬럼 추가
            self.cursor.execute(
                """
                SELECT COLUMN_NAME
                FROM USER_TAB_COLUMNS
                WHERE TABLE_NAME = :table_name
                """,
                {"table_name": table_name},
            )
            existing_cols = {row[0] for row in self.cursor.fetchall()}
            added_cols = []
            for col in cols:
                if col not in existing_cols and col not in ["TRADE_DATE", "TICKER"]:
                    self.cursor.execute(f'ALTER TABLE {table_name} ADD "{col}" NUMBER')
                    added_cols.append(col)
            if added_cols:
                self.connection.commit()
                print(f"{table_name} 누락 컬럼 추가: {added_cols}")

            feature_cols = [c for c in cols if c not in ["TRADE_DATE", "TICKER"]]

            select_parts = [":1 as TRADE_DATE", ":2 as TICKER"]
            update_parts = []
            insert_cols = ["TRADE_DATE", "TICKER"]
            insert_vals = ["s.TRADE_DATE", "s.TICKER"]

            for i, col in enumerate(feature_cols, 3):
                select_parts.append(f':{i} as "{col}"')
                update_parts.append(f'd."{col}" = s."{col}"')
                insert_cols.append(f'"{col}"')
                insert_vals.append(f's."{col}"')

            merge_query = f"""
            MERGE INTO {table_name} d
            USING (SELECT {", ".join(select_parts)} FROM dual) s
            ON (d.TRADE_DATE = s.TRADE_DATE AND d.TICKER = s.TICKER)
            WHEN MATCHED THEN
                UPDATE SET {", ".join(update_parts)}
            WHEN NOT MATCHED THEN
                INSERT ({", ".join(insert_cols)})
                VALUES ({", ".join(insert_vals)})
            """

            data_to_insert = []
            import pandas as pd
            import math
            for _, row in df.iterrows():
                trade_date = row["TRADE_DATE"]
                if hasattr(trade_date, "to_pydatetime"):
                    trade_date = trade_date.to_pydatetime().date()
                elif hasattr(trade_date, "date"):
                    trade_date = trade_date.date()

                row_data = [trade_date, str(row["TICKER"])]
                for col in feature_cols:
                    val = row[col]
                    numeric_val = None
                    if pd.notna(val):
                        try:
                            fval = float(val)
                            if math.isfinite(fval):
                                numeric_val = fval
                        except (TypeError, ValueError):
                            numeric_val = None
                    row_data.append(numeric_val)
                data_to_insert.append(tuple(row_data))

            if data_to_insert:
                # Chunking insertion to avoid bind variable limits
                batch_size = 10000
                for i in range(0, len(data_to_insert), batch_size):
                    batch = data_to_insert[i:i + batch_size]
                    self.cursor.executemany(merge_query, batch)
                self.connection.commit()
                print(f"{table_name} 데이터 {len(data_to_insert)}건 저장 완료.")
        except Exception as e:
            print(f"{table_name} 저장 실패: {e}")
            self.connection.rollback()
            raise

    def fetch_master_features(self, ticker=None):
        """
        MASTER_FEATURES에서 전체 또는 특정 티커의 데이터를 조회합니다.
        """
        if not self.master_features_exists():
            import pandas as pd
            return pd.DataFrame()

        query = "SELECT * FROM MASTER_FEATURES"
        params = {}
        if ticker:
            query += " WHERE TICKER = :ticker"
            params = {"ticker": ticker}
            
        query += " ORDER BY TRADE_DATE ASC, TICKER ASC"
            
        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            import pandas as pd
            if not rows:
                return pd.DataFrame()
            col_names = [d[0] for d in self.cursor.description]
            df = pd.DataFrame(rows, columns=col_names)
            df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
            return df
        except Exception as e:
            import pandas as pd
            print(f"MASTER_FEATURES 조회 실패: {e}")
            return pd.DataFrame()

    def reorganize_master_features(self):
        """TRADE_DATE, TICKER 오름차순 CTAS로 재구조화합니다."""
        if not self.master_features_exists():
            return
            
        try:
            print("MASTER_FEATURES 테이블 재구조화 시작...")
            self.cursor.execute(
                """
                CREATE TABLE MASTER_FEATURES_COPY AS
                SELECT * FROM MASTER_FEATURES
                ORDER BY TRADE_DATE ASC, TICKER ASC
                """
            )
            self.cursor.execute("DROP TABLE MASTER_FEATURES PURGE")
            self.cursor.execute("ALTER TABLE MASTER_FEATURES_COPY RENAME TO MASTER_FEATURES")
            self.cursor.execute(
                """
                ALTER TABLE MASTER_FEATURES
                ADD CONSTRAINT PK_MASTER_FEATURES PRIMARY KEY (TRADE_DATE, TICKER)
                USING INDEX
                """
            )
            self.connection.commit()
            print("MASTER_FEATURES 테이블 재구조화 완료!")
        except Exception as e:
            print(f"MASTER_FEATURES 재구조화 실패: {e}")
            self.connection.rollback()
            raise

    def truncate_master_features(self):
        """MASTER_FEATURES 테이블을 초기화합니다."""
        if not self.master_features_exists():
            return
        self.cursor.execute("TRUNCATE TABLE MASTER_FEATURES")
        self.connection.commit()

    def get_ticker_coverage_report(self, tickers):
        """
        각 종목별 DB 적재 현황을 요약하여 리포트를 반환합니다.
        MASTER_FEATURES 통합 테이블 기준으로 조회합니다.
        """
        report = []
        try:
            # SP500_DATA 총 행수 (공통)
            self.cursor.execute("SELECT COUNT(*) FROM SP500_DATA")
            sp500_total = self.cursor.fetchone()[0]

            # MASTER_FEATURES 종목별 행수 일괄 조회
            mf_counts = {}
            if self.master_features_exists():
                self.cursor.execute(
                    "SELECT TICKER, COUNT(*) FROM MASTER_FEATURES GROUP BY TICKER"
                )
                for row in self.cursor.fetchall():
                    mf_counts[row[0]] = row[1]

            for ticker in tickers:
                ticker = str(ticker).upper()

                # STOCK_DATA 행수
                self.cursor.execute(
                    "SELECT COUNT(*) FROM STOCK_DATA WHERE TICKER = :1",
                    [ticker],
                )
                stock_rows = self.cursor.fetchone()[0]

                # LOG_RETURNS 행수
                self.cursor.execute(
                    "SELECT COUNT(*) FROM LOG_RETURNS WHERE TICKER = :1",
                    [ticker],
                )
                logret_rows = self.cursor.fetchone()[0]

                report.append({
                    "ticker": ticker,
                    "stock_rows": stock_rows,
                    "logret_rows": logret_rows,
                    "sp500_rows": sp500_total,
                    "feature_rows": mf_counts.get(ticker, 0),
                })
        except Exception as e:
            print(f"Coverage report 생성 실패: {e}")

        return report

    def close(self):
        """
        리소스 해제
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("DB 연결이 종료되었습니다.")
