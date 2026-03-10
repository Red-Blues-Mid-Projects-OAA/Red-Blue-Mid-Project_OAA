"""
이 파일은 Oracle 데이터베이스 연결, 테이블 준비, 주요 조회와 저장 작업을 담당하는 관리자 클래스 모음입니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

import math

if __package__ in (None, ""):
    from datetime import datetime
    import os

    import numpy as np
    import oracledb
    import pandas as pd
    from dotenv import load_dotenv
else:
    from common import datetime, load_dotenv, np, oracledb, os, pd

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
        """객체가 처음 만들어질 때 필요한 기본 값과 연결 상태를 준비합니다."""
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        self.connection = None
        self.cursor = None
        self._quiet = False

    def connect(self, ensure_tables: bool = True, quiet: bool = False):
        """
        DB 연결 설정
        """
        self._quiet = bool(quiet)
        try:
            # Oracle DB 연결 시도
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor() # 연결 다리
            if not self._quiet:
                print("Oracle DB에 성공적으로 연결되었습니다.")

            # Step7 read 경로에서는 테이블 준비 쿼리를 생략해 연결 오버헤드를 줄입니다.
            if ensure_tables:
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

        # 모델 기대수익률 보정 결과를 단일 스냅샷으로 저장하는 테이블입니다.
        # API와 후속 배치는 이 테이블만 읽어 최신 추천 풀을 구성합니다.
        create_adjusted_expected_returns_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE ADJUSTED_EXPECTED_RETURNS (
                TICKER VARCHAR2(20) PRIMARY KEY,
                GATE_PASSED NUMBER(1),
                RETURN_TYPE VARCHAR2(30),
                ORIGINAL_E_RET NUMBER,
                E_TOTAL_3M NUMBER,
                REALIZED_3M NUMBER,
                GAP NUMBER,
                ADJ_WEIGHT NUMBER,
                VOL_3M NUMBER,
                ADJUSTMENT NUMBER,
                ADJUSTED_E_TOTAL NUMBER,
                ADJUSTMENT_APPLIED NUMBER(1),
                UPDATED_AT DATE
            )';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """

        # Legacy two-table schema cleanup (metrics/holdings -> unified snapshot table)
        drop_legacy_risk_metrics_query = """
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE RISK_LEVEL_PORTFOLIO_METRICS PURGE';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -942 THEN
                    RAISE;
                END IF;
        END;
        """

        drop_legacy_risk_holdings_query = """
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE RISK_LEVEL_PORTFOLIO_HOLDINGS PURGE';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -942 THEN
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
            self.cursor.execute(create_adjusted_expected_returns_query)
            self.cursor.execute(drop_legacy_risk_metrics_query)
            self.cursor.execute(drop_legacy_risk_holdings_query)
            
            # 변경 사항 커밋
            self.connection.commit()
            if not self._quiet:
                print("모든 DB 테이블이 준비되었습니다.")
        except oracledb.Error as e:
            print(f"테이블 생성 중 오류 발생: {e}")

    def ensure_runtime_indexes(self):
        """
        런타임 조회 성능에 필요한 인덱스를 idempotent 하게 보장합니다.
        """
        if self.cursor is None:
            raise RuntimeError("DB 연결 후 ensure_runtime_indexes()를 호출해야 합니다.")

        create_master_features_ticker_date_idx = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE INDEX IDX_MASTER_FEATURES_TICKER_DATE
                               ON MASTER_FEATURES (TICKER, TRADE_DATE)';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE NOT IN (-955, -942) THEN
                    RAISE;
                END IF;
        END;
        """
        try:
            self.cursor.execute(create_master_features_ticker_date_idx)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

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

    def get_latest_dates_map(self, tickers):
        """
        Return latest TRADE_DATE per ticker using grouped queries.
        Missing tickers are returned with None.
        """
        normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not normalized:
            return {}

        latest_map = {ticker: None for ticker in normalized}

        try:
            step = 900  # Oracle IN list limit guard (1000)
            for i in range(0, len(normalized), step):
                subset = normalized[i : i + step]
                bind_params = {f"t{j}": ticker for j, ticker in enumerate(subset)}
                placeholders = ", ".join(f":{k}" for k in bind_params.keys())
                query = f"""
                SELECT TICKER, MAX(TRADE_DATE) AS LATEST_DATE
                FROM STOCK_DATA
                WHERE TICKER IN ({placeholders})
                GROUP BY TICKER
                """
                self.cursor.execute(query, bind_params)
                for ticker, latest_date in self.cursor.fetchall():
                    latest_map[str(ticker).strip().upper()] = latest_date
        except Exception:
            # Fallback to per-ticker lookup for safety
            for ticker in normalized:
                latest_map[ticker] = self.get_latest_date(ticker)

        return latest_map

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

    def get_latest_log_returns_date(self):
        """
        LOG_RETURNS 테이블에서 가장 최신 TRADE_DATE 조회
        """
        query = "SELECT MAX(TRADE_DATE) FROM LOG_RETURNS"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except oracledb.Error as e:
            print(f"LOG_RETURNS 최신 날짜 조회 실패: {e}")
            return None

    def get_latest_ewma_cov_date(self):
        """
        EWMA_COVARIANCE 테이블에서 가장 최신 CALC_DATE 조회
        """
        query = "SELECT MAX(CALC_DATE) FROM EWMA_COVARIANCE"
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except oracledb.Error as e:
            print(f"EWMA_COVARIANCE 최신 날짜 조회 실패: {e}")
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

    def insert_data(self, df, commit=True, single_ticker=None):
        """
        Upsert market rows into STOCK_DATA.

        Args:
            df: yfinance DataFrame (MultiIndex or flat)
            commit: if True, commit inside this method

        Returns:
            Number of rows sent to executemany.
        """
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

        rows = []

        try:
            if isinstance(df.columns, pd.MultiIndex):
                # Handle both MultiIndex layouts:
                # 1) (Price, Ticker) and 2) (Ticker, Price)
                level0 = set(str(x) for x in df.columns.get_level_values(0))
                level1 = set(str(x) for x in df.columns.get_level_values(1))
                if ("Adj Close" in level0) or ("Close" in level0):
                    working_df = df
                elif ("Adj Close" in level1) or ("Close" in level1):
                    working_df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
                else:
                    print(
                        "insert_data: unsupported MultiIndex column layout. "
                        f"level0_sample={list(level0)[:6]}, level1_sample={list(level1)[:6]}"
                    )
                    return 0

                try:
                    df_processed = working_df.stack(level=1, future_stack=True).reset_index()
                except TypeError:
                    df_processed = working_df.stack(level=1).reset_index()

                date_col = df_processed.columns[0]
                ticker_col = df_processed.columns[1]
                col_names = [str(c) for c in df_processed.columns]

                if "Adj Close" in col_names:
                    close_col = next(c for c in df_processed.columns if str(c) == "Adj Close")
                else:
                    close_col = next((c for c in df_processed.columns if str(c) == "Close"), None)

                high_col = next((c for c in df_processed.columns if str(c) == "High"), None)
                vol_col = next((c for c in df_processed.columns if str(c) == "Volume"), None)

                dates = pd.to_datetime(df_processed[date_col], errors="coerce").dt.date.tolist()
                tickers = df_processed[ticker_col].astype(str).str.strip().str.upper().tolist()

                if close_col is not None:
                    close_series = pd.to_numeric(df_processed[close_col], errors="coerce")
                    close_series = close_series.where(np.isfinite(close_series), np.nan)
                    closes = [None if pd.isna(x) else float(x) for x in close_series]
                else:
                    closes = [None] * len(df_processed)

                if high_col is not None:
                    high_series = pd.to_numeric(df_processed[high_col], errors="coerce")
                    high_series = high_series.where(np.isfinite(high_series), np.nan)
                    highs = [None if pd.isna(x) else float(x) for x in high_series]
                else:
                    highs = [None] * len(df_processed)

                if vol_col is not None:
                    vol_series = pd.to_numeric(df_processed[vol_col], errors="coerce")
                    vol_series = vol_series.where(np.isfinite(vol_series), np.nan)
                    volumes = [None if pd.isna(x) else float(x) for x in vol_series]
                else:
                    volumes = [None] * len(df_processed)

                rows = list(zip(tickers, dates, closes, highs, volumes))
            else:
                # yfinance single-ticker download can be flat OHLCV columns.
                # Use caller-provided ticker hint to map this format correctly.
                if single_ticker and (("Adj Close" in df.columns) or ("Close" in df.columns)):
                    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
                    high_col = "High" if "High" in df.columns else None
                    vol_col = "Volume" if "Volume" in df.columns else None

                    dates = pd.to_datetime(df.index, errors="coerce").date.tolist()
                    tickers = [str(single_ticker).strip().upper()] * len(df)

                    close_series = pd.to_numeric(df[close_col], errors="coerce")
                    close_series = close_series.where(np.isfinite(close_series), np.nan)
                    closes = [None if pd.isna(x) else float(x) for x in close_series]

                    if high_col is not None:
                        high_series = pd.to_numeric(df[high_col], errors="coerce")
                        high_series = high_series.where(np.isfinite(high_series), np.nan)
                        highs = [None if pd.isna(x) else float(x) for x in high_series]
                    else:
                        highs = [None] * len(df)

                    if vol_col is not None:
                        vol_series = pd.to_numeric(df[vol_col], errors="coerce")
                        vol_series = vol_series.where(np.isfinite(vol_series), np.nan)
                        volumes = [None if pd.isna(x) else float(x) for x in vol_series]
                    else:
                        volumes = [None] * len(df)

                    rows = list(zip(tickers, dates, closes, highs, volumes))
                else:
                    df_flat = df.reset_index().melt(
                        id_vars=df.index.name or "Date",
                        var_name="Ticker",
                        value_name="Close",
                    )

                    date_col = df_flat.columns[0]
                    dates = pd.to_datetime(df_flat[date_col], errors="coerce").dt.date.tolist()
                    tickers = df_flat["Ticker"].astype(str).str.strip().str.upper().tolist()

                    close_series = pd.to_numeric(df_flat["Close"], errors="coerce")
                    close_series = close_series.where(np.isfinite(close_series), np.nan)
                    closes = [None if pd.isna(x) else float(x) for x in close_series]

                    highs = [None] * len(df_flat)
                    volumes = [None] * len(df_flat)
                    rows = list(zip(tickers, dates, closes, highs, volumes))

            raw_row_count = len(rows)
            rows = [
                row
                for row in rows
                if row[0]
                and (row[1] is not None)
                and (not pd.isna(row[1]))
                and not (row[2] is None and row[3] is None and row[4] is None)
            ]

            if not rows:
                print(
                    "No rows to insert. "
                    f"(raw={raw_row_count}, filtered=0, single_ticker={single_ticker})"
                )
                return 0

            batch_size = 10000
            for i in range(0, len(rows), batch_size):
                self.cursor.executemany(insert_query, rows[i : i + batch_size])

            if commit:
                self.connection.commit()

            print(f"Inserted/updated {len(rows)} rows into STOCK_DATA.")
            return len(rows)

        except Exception as e:
            if commit:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            print(f"insert_data failed: {e}")
            if not commit:
                raise
            return 0

    def fetch_prices(self, start_date=None):
        """
        DB에서 전체 주가 데이터를 가져와 Pivot된 DataFrame으로 반환
        Index: Date, Columns: Ticker
        """
        query = "SELECT TICKER, TRADE_DATE, CLOSE_PRICE FROM STOCK_DATA"
        params = {}
        if start_date is not None:
            query += " WHERE TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE, TICKER"
        try:
            # 데이터 가져오기
            if params:
                self.cursor.execute(query, params)
            else:
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

    def fetch_log_returns(self, start_date=None):
        """
        DB에서 로그 수익률 데이터를 가져와 Pivot된 DataFrame으로 반환
        """
        query = "SELECT TICKER, TRADE_DATE, LOG_RETURN FROM LOG_RETURNS"
        params = {}
        if start_date is not None:
            query += " WHERE TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE, TICKER"
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['TICKER', 'TRADE_DATE', 'LOG_RETURN'])
            # 중복 (TRADE_DATE, TICKER) 제거 - pivot 오류 방지
            df = df.drop_duplicates(subset=['TRADE_DATE', 'TICKER'], keep='last')
            pivot_df = df.pivot(index='TRADE_DATE', columns='TICKER', values='LOG_RETURN')
            return pivot_df
        except oracledb.Error as e:
            print(f"로그 수익률 데이터 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_log_returns_by_ticker(self, ticker, start_date=None):
        """
        단일 티커의 로그수익률 시계열을 조회해 Series(index=TRADE_DATE)로 반환.
        """
        query = """
            SELECT TRADE_DATE, LOG_RETURN
            FROM LOG_RETURNS
            WHERE TICKER = :ticker
        """
        params = {"ticker": str(ticker).upper()}
        if start_date is not None:
            query += " AND TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE"
        try:
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.Series(dtype=float, name="LOG_RETURN")
            df = pd.DataFrame(rows, columns=["TRADE_DATE", "LOG_RETURN"])
            df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
            return pd.to_numeric(df.set_index("TRADE_DATE")["LOG_RETURN"], errors="coerce")
        except oracledb.Error as e:
            print(f"{ticker} 로그수익률 조회 실패: {e}")
            return pd.Series(dtype=float, name="LOG_RETURN")

    def fetch_ticker_data(self, ticker, start_date=None):
        """
        특정 티커의 주가 및 거래량 데이터를 가져와 반환
        """
        query = """
            SELECT TRADE_DATE, CLOSE_PRICE, HIGH_PRICE, VOLUME 
            FROM STOCK_DATA 
            WHERE TICKER = :ticker
        """
        params = {"ticker": ticker}
        if start_date is not None:
            query += " AND TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE"
        try:
            self.cursor.execute(query, params)
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

    def fetch_ewma_covariance(self):
        """
        EWMA 공분산 행렬을 DB에서 로드합니다.
        가장 최근 CALC_DATE의 전체 공분산 데이터를 가져와
        (정렬된 ticker 리스트, numpy 2D 행렬) 튜플로 반환합니다.
        """
        try:
            # 가장 최근 계산 날짜 조회
            self.cursor.execute("SELECT MAX(CALC_DATE) FROM EWMA_COVARIANCE")
            latest_date = self.cursor.fetchone()[0]
            if latest_date is None:
                print("EWMA_COVARIANCE 테이블에 데이터가 없습니다.")
                return [], np.array([])

            # 해당 날짜의 전체 공분산 데이터 조회
            query = """
                SELECT TICKER_X, TICKER_Y, COV_VALUE
                FROM EWMA_COVARIANCE
                WHERE CALC_DATE = :1
                ORDER BY TICKER_X, TICKER_Y
            """
            self.cursor.execute(query, [latest_date])
            rows = self.cursor.fetchall()

            if not rows:
                print("EWMA 공분산 데이터를 가져오지 못했습니다.")
                return [], np.array([])

            # DataFrame으로 변환 후 pivot하여 정방행렬 생성
            df = pd.DataFrame(rows, columns=["TICKER_X", "TICKER_Y", "COV_VALUE"])
            cov_pivot = df.pivot(index="TICKER_X", columns="TICKER_Y", values="COV_VALUE")
            cov_pivot = cov_pivot.sort_index(axis=0).sort_index(axis=1)

            ticker_list = list(cov_pivot.index)
            cov_matrix = cov_pivot.values.astype(float)

            print(f"[DB] EWMA 공분산 행렬 로드 완료: {len(ticker_list)}×{len(ticker_list)} (날짜: {latest_date})")
            return ticker_list, cov_matrix

        except oracledb.Error as e:
            print(f"EWMA 공분산 조회 실패: {e}")
            return [], np.array([])

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

    def fetch_sp500_data(self, start_date=None):
        """
        DB에서 S&P 500 데이터(TRADE_DATE, LOG_RETURN)를 가져와 DataFrame으로 반환
        """
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA"
        params = {}
        if start_date is not None:
            query += " WHERE TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE"
        try:
            if params:
                self.cursor.execute(query, params)
            else:
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

    def fetch_sp500_log_returns(self, start_date=None):
        """
        SP500 로그수익률 시계열만 조회해 Series(index=TRADE_DATE)로 반환.
        """
        query = "SELECT TRADE_DATE, LOG_RETURN FROM SP500_DATA"
        params = {}
        if start_date is not None:
            query += " WHERE TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE"
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.Series(dtype=float, name="LOG_RETURN")
            df = pd.DataFrame(rows, columns=["TRADE_DATE", "LOG_RETURN"])
            df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
            return pd.to_numeric(df.set_index("TRADE_DATE")["LOG_RETURN"], errors="coerce")
        except oracledb.Error as e:
            print(f"SP500 로그수익률 조회 실패: {e}")
            return pd.Series(dtype=float, name="LOG_RETURN")

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

    def fetch_market_features(self, indicator, start_date=None):
        """
        DB에서 특정 시장 지표(VIX/DXY) 데이터를 조회하여 DataFrame으로 반환
        """
        query = """
            SELECT TRADE_DATE, CLOSE_VALUE, LOG_RETURN
            FROM MARKET_FEATURES
            WHERE INDICATOR = :indicator
        """
        params = {"indicator": indicator}
        if start_date is not None:
            query += " AND TRADE_DATE >= :start_date"
            params["start_date"] = pd.Timestamp(start_date).to_pydatetime().date()
        query += " ORDER BY TRADE_DATE"
        try:
            self.cursor.execute(query, params)
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

    def get_latest_master_features_date(self):
        """MASTER_FEATURES 테이블의 가장 최신 TRADE_DATE를 반환합니다."""
        if not self.master_features_exists():
            return None
        try:
            self.cursor.execute("SELECT MAX(TRADE_DATE) FROM MASTER_FEATURES")
            result = self.cursor.fetchone()
            if result and result[0]:
                return result[0]
            return None
        except Exception:
            return None

    def get_master_features_latest_dates_bulk(self, tickers):
        """
        MASTER_FEATURES에서 티커별 최신 TRADE_DATE를 집계 조회합니다.
        반환: {TICKER: latest_trade_date_or_None}
        """
        normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not normalized:
            return {}

        latest_map = {t: None for t in normalized}
        if not self.master_features_exists():
            return latest_map

        try:
            step = 900  # Oracle IN limit guard
            for i in range(0, len(normalized), step):
                subset = normalized[i : i + step]
                bind_params = {f"t{j}": ticker for j, ticker in enumerate(subset)}
                placeholders = ", ".join(f":{k}" for k in bind_params.keys())
                query = f"""
                SELECT TICKER, MAX(TRADE_DATE) AS LATEST_DATE
                FROM MASTER_FEATURES
                WHERE TICKER IN ({placeholders})
                GROUP BY TICKER
                """
                self.cursor.execute(query, bind_params)
                for ticker, latest_date in self.cursor.fetchall():
                    latest_map[str(ticker).strip().upper()] = latest_date
        except Exception as e:
            print(f"MASTER_FEATURES 최신일 bulk 조회 실패: {e}")

        return latest_map

    def get_ticker_coverage_report_bulk(self, tickers):
        """
        다중 티커 coverage를 집계 쿼리로 조회합니다 (N+1 제거).
        """
        normalized = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not normalized:
            return []

        placeholder = ",".join(f":{i+1}" for i in range(len(normalized)))
        report_map = {
            t: {
                "ticker": t,
                "stock_rows": 0,
                "logret_rows": 0,
                "sp500_rows": 0,
                "feature_rows": 0,
            }
            for t in normalized
        }
        try:
            self.cursor.execute("SELECT COUNT(*) FROM SP500_DATA")
            sp500_total = int(self.cursor.fetchone()[0] or 0)
            for t in normalized:
                report_map[t]["sp500_rows"] = sp500_total

            self.cursor.execute(
                f"SELECT TICKER, COUNT(*) FROM STOCK_DATA WHERE TICKER IN ({placeholder}) GROUP BY TICKER",
                normalized,
            )
            for ticker, cnt in self.cursor.fetchall():
                key = str(ticker).upper()
                if key in report_map:
                    report_map[key]["stock_rows"] = int(cnt or 0)

            self.cursor.execute(
                f"SELECT TICKER, COUNT(*) FROM LOG_RETURNS WHERE TICKER IN ({placeholder}) GROUP BY TICKER",
                normalized,
            )
            for ticker, cnt in self.cursor.fetchall():
                key = str(ticker).upper()
                if key in report_map:
                    report_map[key]["logret_rows"] = int(cnt or 0)

            if self.master_features_exists():
                self.cursor.execute(
                    f"SELECT TICKER, COUNT(*) FROM MASTER_FEATURES WHERE TICKER IN ({placeholder}) GROUP BY TICKER",
                    normalized,
                )
                for ticker, cnt in self.cursor.fetchall():
                    key = str(ticker).upper()
                    if key in report_map:
                        report_map[key]["feature_rows"] = int(cnt or 0)
        except Exception as e:
            print(f"Coverage report bulk 생성 실패: {e}")

        return [report_map[t] for t in normalized]

    def get_ticker_coverage_report(self, tickers):
        """
        하위호환 API: bulk coverage 조회 결과를 반환합니다.
        """
        return self.get_ticker_coverage_report_bulk(tickers)

    def upsert_adjusted_expected_returns_snapshot(self, df):
        """
        adjusted_expected_returns 스냅샷을 티커 기준으로 업서트하고,
        입력 데이터에 없는 기존 티커는 삭제해 DB와 완전 동기화합니다.
        """
        # 입력 컬럼 스키마를 명시적으로 검증합니다.
        required_cols = [
            "Ticker",
            "Gate_Passed",
            "Return_Type",
            "Original_E_Ret",
            "E_Total_3M",
            "Realized_3M",
            "Gap",
            "Adj_Weight",
            "Vol_3M",
            "Adjustment",
            "Adjusted_E_Total",
            "Adjustment_Applied",
        ]

        # 원본 DataFrame은 보존하고, DB 적재 전용 복사본에서만 컬럼명과 값을 정리합니다.
        normalized_df = df.copy()
        normalized_df.columns = [str(col).strip() for col in normalized_df.columns]
        # MERGE 문이 기대하는 모든 컬럼이 존재하는지 선검증해 부분 적재를 막습니다.
        missing_cols = [col for col in required_cols if col not in normalized_df.columns]
        if missing_cols:
            raise ValueError(
                f"[SYNC][VALIDATION] adjusted_expected_returns 필수 컬럼 누락: {missing_cols}"
            )

        # bool 컬럼은 Oracle NUMBER(1)로 변환합니다.
        def _to_bool_flag(value):
            """Python bool/문자/숫자 표현을 Oracle NUMBER(1) 플래그로 정규화합니다."""
            if pd.isna(value):
                return 0
            if isinstance(value, bool):
                return 1 if value else 0
            normalized = str(value).strip().lower()
            return 1 if normalized in {"1", "true", "t", "y", "yes"} else 0

        # 숫자 컬럼은 NaN/inf를 None으로 변환합니다.
        def _to_number(value):
            """숫자형 입력을 Oracle에 적재 가능한 float 또는 None으로 정규화합니다."""
            if pd.isna(value):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None

        merge_query = """
        MERGE INTO ADJUSTED_EXPECTED_RETURNS d
        USING (
            SELECT
                :1 AS TICKER,
                :2 AS GATE_PASSED,
                :3 AS RETURN_TYPE,
                :4 AS ORIGINAL_E_RET,
                :5 AS E_TOTAL_3M,
                :6 AS REALIZED_3M,
                :7 AS GAP,
                :8 AS ADJ_WEIGHT,
                :9 AS VOL_3M,
                :10 AS ADJUSTMENT,
                :11 AS ADJUSTED_E_TOTAL,
                :12 AS ADJUSTMENT_APPLIED
            FROM dual
        ) s
        ON (d.TICKER = s.TICKER)
        WHEN MATCHED THEN
            UPDATE SET
                d.GATE_PASSED = s.GATE_PASSED,
                d.RETURN_TYPE = s.RETURN_TYPE,
                d.ORIGINAL_E_RET = s.ORIGINAL_E_RET,
                d.E_TOTAL_3M = s.E_TOTAL_3M,
                d.REALIZED_3M = s.REALIZED_3M,
                d.GAP = s.GAP,
                d.ADJ_WEIGHT = s.ADJ_WEIGHT,
                d.VOL_3M = s.VOL_3M,
                d.ADJUSTMENT = s.ADJUSTMENT,
                d.ADJUSTED_E_TOTAL = s.ADJUSTED_E_TOTAL,
                d.ADJUSTMENT_APPLIED = s.ADJUSTMENT_APPLIED,
                d.UPDATED_AT = SYSDATE
        WHEN NOT MATCHED THEN
            INSERT (
                TICKER,
                GATE_PASSED,
                RETURN_TYPE,
                ORIGINAL_E_RET,
                E_TOTAL_3M,
                REALIZED_3M,
                GAP,
                ADJ_WEIGHT,
                VOL_3M,
                ADJUSTMENT,
                ADJUSTED_E_TOTAL,
                ADJUSTMENT_APPLIED,
                UPDATED_AT
            )
            VALUES (
                s.TICKER,
                s.GATE_PASSED,
                s.RETURN_TYPE,
                s.ORIGINAL_E_RET,
                s.E_TOTAL_3M,
                s.REALIZED_3M,
                s.GAP,
                s.ADJ_WEIGHT,
                s.VOL_3M,
                s.ADJUSTMENT,
                s.ADJUSTED_E_TOTAL,
                s.ADJUSTMENT_APPLIED,
                SYSDATE
            )
        """

        # records는 executemany MERGE에 바로 넣을 튜플 목록, ticker_set은 삭제 동기화 기준 집합입니다.
        records = []
        ticker_set = set()
        for _, row in normalized_df.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            if not ticker:
                continue
            ticker_set.add(ticker)
            # SQL 바인드 순서와 정확히 같은 튜플 순서로 적재 데이터를 만듭니다.
            records.append(
                (
                    ticker,
                    _to_bool_flag(row["Gate_Passed"]),
                    str(row["Return_Type"]).strip(),
                    _to_number(row["Original_E_Ret"]),
                    _to_number(row["E_Total_3M"]),
                    _to_number(row["Realized_3M"]),
                    _to_number(row["Gap"]),
                    _to_number(row["Adj_Weight"]),
                    _to_number(row["Vol_3M"]),
                    _to_number(row["Adjustment"]),
                    _to_number(row["Adjusted_E_Total"]),
                    _to_bool_flag(row["Adjustment_Applied"]),
                )
            )

        try:
            # 1) 스냅샷 업서트 단계
            if records:
                # Oracle 바인드 개수와 트랜잭션 부담을 고려해 대량 데이터는 배치 단위로 나눕니다.
                batch_size = 10000
                for index in range(0, len(records), batch_size):
                    batch = records[index:index + batch_size]
                    self.cursor.executemany(merge_query, batch)

            # 2) 스냅샷 삭제 동기화 단계
            if ticker_set:
                # 이번 입력에 없는 티커는 스냅샷에서 제거해 CSV/DB 간 완전 동기화를 유지합니다.
                ordered_tickers = sorted(ticker_set)
                bind_params = {f"t{idx}": ticker for idx, ticker in enumerate(ordered_tickers)}
                placeholders = ", ".join(f":{key}" for key in bind_params.keys())
                delete_query = (
                    f"DELETE FROM ADJUSTED_EXPECTED_RETURNS "
                    f"WHERE TICKER NOT IN ({placeholders})"
                )
                self.cursor.execute(delete_query, bind_params)
            else:
                self.cursor.execute("TRUNCATE TABLE ADJUSTED_EXPECTED_RETURNS")

            self.connection.commit()
            print(
                f"[SYNC][OK] ADJUSTED_EXPECTED_RETURNS 동기화 완료 "
                f"(업서트={len(records)}, 유지티커={len(ticker_set)})"
            )
        except Exception as e:
            self.connection.rollback()
            print(f"[SYNC][ERROR] ADJUSTED_EXPECTED_RETURNS 동기화 실패: {e}")
            raise

    def fetch_adjusted_expected_returns(self):
        """
        ADJUSTED_EXPECTED_RETURNS 스냅샷을 조회하여
        CSV와 동일한 컬럼 이름으로 반환합니다.
        """
        query = """
        SELECT
            TICKER AS "Ticker",
            GATE_PASSED AS "Gate_Passed",
            RETURN_TYPE AS "Return_Type",
            ORIGINAL_E_RET AS "Original_E_Ret",
            E_TOTAL_3M AS "E_Total_3M",
            REALIZED_3M AS "Realized_3M",
            GAP AS "Gap",
            ADJ_WEIGHT AS "Adj_Weight",
            VOL_3M AS "Vol_3M",
            ADJUSTMENT AS "Adjustment",
            ADJUSTED_E_TOTAL AS "Adjusted_E_Total",
            ADJUSTMENT_APPLIED AS "Adjustment_Applied",
            UPDATED_AT AS "Updated_At"
        FROM ADJUSTED_EXPECTED_RETURNS
        ORDER BY TICKER
        """
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()

            col_names = [desc[0] for desc in self.cursor.description]
            result_df = pd.DataFrame(rows, columns=col_names)

            # Oracle NUMBER(1) -> Python bool 복원
            for col in ["Gate_Passed", "Adjustment_Applied"]:
                if col in result_df.columns:
                    # 숫자 플래그를 Python bool로 돌려놓아 API/캐시 코드가 추가 변환 없이 바로 쓰도록 합니다.
                    result_df[col] = result_df[col].apply(
                        lambda value: bool(int(value)) if pd.notna(value) else False
                    )

            if "Updated_At" in result_df.columns:
                result_df["Updated_At"] = pd.to_datetime(result_df["Updated_At"])

            return result_df
        except Exception as e:
            print(f"[SYNC][ERROR] ADJUSTED_EXPECTED_RETURNS 조회 실패: {e}")
            return pd.DataFrame()

    def fetch_risk_level_portfolio_snapshot(self):
        """
        RISK_LEVEL_PORTFOLIO_SNAPSHOT 스냅샷을 조회합니다.
        - rank_no=0: 리스크 레벨 요약
        - rank_no>=1: 편입 종목
        """
        query = """
        SELECT
            RISK_LEVEL AS "Risk_Level",
            RANK_NO AS "Rank_No",
            RISK_LABEL AS "Risk_Label",
            LAMBDA_VALUE AS "Lambda_Value",
            PORTFOLIO_EXPECTED_RETURN_3M AS "Portfolio_Expected_Return_3M",
            PORTFOLIO_STD_60D AS "Portfolio_Std_60D",
            HOLDINGS_COUNT AS "Holdings_Count",
            TICKER AS "Ticker",
            STOCK_NAME AS "Stock_Name",
            WEIGHT_PCT AS "Weight_Pct",
            STOCK_EXPECTED_RETURN_3M AS "Stock_Expected_Return_3M",
            STOCK_SIGMA_EWMA_60D AS "Stock_Sigma_EWMA_60D",
            UPDATED_AT AS "Updated_At"
        FROM RISK_LEVEL_PORTFOLIO_SNAPSHOT
        ORDER BY RISK_LEVEL DESC, RANK_NO ASC
        """
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            if not rows:
                return pd.DataFrame()

            col_names = [desc[0] for desc in self.cursor.description]
            result_df = pd.DataFrame(rows, columns=col_names)
            if "Updated_At" in result_df.columns:
                result_df["Updated_At"] = pd.to_datetime(result_df["Updated_At"])
            return result_df
        except Exception as e:
            # ORA-00942(테이블 미존재) 포함, 조회 실패 시 빈 프레임 반환
            print(f"[SYNC][ERROR] RISK_LEVEL_PORTFOLIO_SNAPSHOT 조회 실패: {e}")
            return pd.DataFrame()

    def _ensure_risk_level_portfolio_snapshot_table(self):
        """Create snapshot table only when snapshot sync is actually requested."""
        create_risk_level_snapshot_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE RISK_LEVEL_PORTFOLIO_SNAPSHOT (
                RISK_LEVEL NUMBER(1),
                RANK_NO NUMBER,
                RISK_LABEL VARCHAR2(40),
                LAMBDA_VALUE NUMBER,
                PORTFOLIO_EXPECTED_RETURN_3M NUMBER,
                PORTFOLIO_STD_60D NUMBER,
                HOLDINGS_COUNT NUMBER,
                TICKER VARCHAR2(20),
                STOCK_NAME VARCHAR2(120),
                WEIGHT_PCT NUMBER,
                STOCK_EXPECTED_RETURN_3M NUMBER,
                STOCK_SIGMA_EWMA_60D NUMBER,
                UPDATED_AT DATE,
                CONSTRAINT PK_RISK_LEVEL_PORTFOLIO_SNAPSHOT PRIMARY KEY (RISK_LEVEL, RANK_NO)
            )';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """
        self.cursor.execute(create_risk_level_snapshot_query)

    def replace_risk_level_portfolio_snapshot(self, metrics_rows, holdings_rows):
        """
        4단계 리스크 포트폴리오 스냅샷을 단일 테이블에 교체 적재합니다.
        - rank_no=0: 리스크별 요약 행
        - rank_no>=1: 편입 종목 행
        """
        if not metrics_rows:
            raise ValueError("metrics_rows is empty")

        def _to_number(value):
            """number 관련 처리를 담당하는 함수입니다."""
            if pd.isna(value):
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            return numeric if math.isfinite(numeric) else None

        snapshot_insert_query = """
        INSERT INTO RISK_LEVEL_PORTFOLIO_SNAPSHOT (
            RISK_LEVEL,
            RANK_NO,
            RISK_LABEL,
            LAMBDA_VALUE,
            PORTFOLIO_EXPECTED_RETURN_3M,
            PORTFOLIO_STD_60D,
            HOLDINGS_COUNT,
            TICKER,
            STOCK_NAME,
            WEIGHT_PCT,
            STOCK_EXPECTED_RETURN_3M,
            STOCK_SIGMA_EWMA_60D,
            UPDATED_AT
        ) VALUES (
            :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, SYSDATE
        )
        """

        metric_map = {}
        for row in metrics_rows:
            risk_level = int(row.get("risk_level"))
            metric_map[risk_level] = {
                "risk_label": str(row.get("risk_label", "")).strip(),
                "lambda_value": _to_number(row.get("lambda_value")),
                "portfolio_return_3m": _to_number(row.get("expected_return_3m")),
                "portfolio_std_60d": _to_number(row.get("portfolio_std_60d")),
                "holdings_count": int(row.get("holdings_count", 0)),
            }

        holding_map = {}
        for row in holdings_rows:
            risk_level = int(row.get("risk_level"))
            rank_no = int(row.get("rank_no"))
            ticker = str(row.get("ticker", "")).strip().upper()
            stock_name = str(row.get("stock_name", ticker)).strip()[:120]
            weight_pct = _to_number(row.get("weight_pct"))
            stock_expected_return_3m = _to_number(row.get("expected_return_3m"))
            stock_sigma_ewma_60d = _to_number(row.get("sigma_ewma_60d"))
            if not ticker:
                continue
            holding_map.setdefault(risk_level, []).append(
                {
                    "rank_no": rank_no,
                    "ticker": ticker,
                    "stock_name": stock_name,
                    "weight_pct": weight_pct,
                    "stock_expected_return_3m": stock_expected_return_3m,
                    "stock_sigma_ewma_60d": stock_sigma_ewma_60d,
                }
            )

        snapshot_records = []
        for risk_level in sorted(metric_map.keys(), reverse=True):
            metric = metric_map[risk_level]

            # Summary row (rank_no=0)
            snapshot_records.append(
                (
                    risk_level,
                    0,
                    metric["risk_label"],
                    metric["lambda_value"],
                    metric["portfolio_return_3m"],
                    metric["portfolio_std_60d"],
                    metric["holdings_count"],
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            )

            risk_holdings = sorted(
                holding_map.get(risk_level, []),
                key=lambda item: item["rank_no"],
            )
            for holding in risk_holdings:
                snapshot_records.append(
                    (
                        risk_level,
                        holding["rank_no"],
                        metric["risk_label"],
                        metric["lambda_value"],
                        metric["portfolio_return_3m"],
                        metric["portfolio_std_60d"],
                        metric["holdings_count"],
                        holding["ticker"],
                        holding["stock_name"],
                        holding["weight_pct"],
                        holding["stock_expected_return_3m"],
                        holding["stock_sigma_ewma_60d"],
                    )
                )

        try:
            self._ensure_risk_level_portfolio_snapshot_table()
            self.cursor.execute("DELETE FROM RISK_LEVEL_PORTFOLIO_SNAPSHOT")
            if snapshot_records:
                self.cursor.executemany(snapshot_insert_query, snapshot_records)

            self.connection.commit()
            print(
                f"[SYNC][OK] RISK_LEVEL_PORTFOLIO snapshot synced "
                f"(rows={len(snapshot_records)})"
            )
        except Exception as e:
            self.connection.rollback()
            print(f"[SYNC][ERROR] RISK_LEVEL_PORTFOLIO snapshot sync failed: {e}")
            raise

    def close(self):
        """
        리소스 해제
        """
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            if not self._quiet:
                print("DB 연결이 종료되었습니다.")
