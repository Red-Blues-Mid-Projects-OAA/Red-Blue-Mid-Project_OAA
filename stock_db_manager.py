
import oracledb
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# 한글 주석 필수: 환경 변수 로드
load_dotenv()

class StockDBManager:
    """
    Oracle DB 연결 및 주식 데이터 관리 클래스
    """
    def __init__(self):
        # 한글 주석 필수: .env 파일에서 DB 연결 정보 가져오기
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
            # 한글 주석 필수: Oracle DB 연결 시도
            self.connection = oracledb.connect(
                user=self.user,
                password=self.password,
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor()
            print("Oracle DB에 성공적으로 연결되었습니다.")
            
            # 한글 주석 필수: 테이블이 없으면 생성
            self._create_table_if_not_exists()
            
        except oracledb.Error as e:
            print(f"DB 연결 실패: {e}")
            raise

    def _create_table_if_not_exists(self):
        """
        STOCK_DATA 테이블 생성 (없을 경우)
        """
        # 한글 주석 필수: 테이블 생성 쿼리 (Ticker, Date를 복합 기본키로 설정)
        create_table_query = """
        BEGIN
            EXECUTE IMMEDIATE 'CREATE TABLE STOCK_DATA (
                TICKER VARCHAR2(10),
                TRADE_DATE DATE,
                CLOSE_PRICE NUMBER,
                PRIMARY KEY (TICKER, TRADE_DATE)
            )';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -955 THEN
                    RAISE;
                END IF;
        END;
        """
        try:
            self.cursor.execute(create_table_query)
            # 한글 주석 필수: 변경 사항 커밋
            self.connection.commit()
            print("STOCK_DATA 테이블이 준비되었습니다.")
        except oracledb.Error as e:
            print(f"테이블 생성 중 오류 발생: {e}")

    def truncate_table(self):
        """
        테이블의 모든 데이터를 삭제 (초기화)
        """
        try:
            # 한글 주석 필수: TRUNCATE는 DDL이라 롤백 불가하지만 속도가 빠르고 공간을 즉시 반환함
            # 이 명령을 통해 테이블을 깨끗하게 비우고 새로 적재할 준비를 합니다.
            self.cursor.execute("TRUNCATE TABLE STOCK_DATA")
            print("STOCK_DATA 테이블이 성공적으로 초기화(Truncate) 되었습니다.")
        except oracledb.Error as e:
            print(f"테이블 초기화 실패: {e}")

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

    def insert_data(self, df):
        """
        DataFrame 데이터를 DB에 삽입 (Upsert 방식)
        이미 데이터가 있는 경우 업데이트하고, 없으면 새로 삽입합니다.
        """
        # 한글 주석 필수: MERGE 문을 사용하여 중복 데이터 발생 시 업데이트 처리 (ORA-00001 방지)
        insert_query = """
        MERGE INTO STOCK_DATA d
        USING (SELECT :1 as TICKER, :2 as TRADE_DATE, :3 as CLOSE_PRICE FROM dual) s
        ON (d.TICKER = s.TICKER AND d.TRADE_DATE = s.TRADE_DATE)
        WHEN MATCHED THEN
            UPDATE SET d.CLOSE_PRICE = s.CLOSE_PRICE
        WHEN NOT MATCHED THEN
            INSERT (TICKER, TRADE_DATE, CLOSE_PRICE)
            VALUES (s.TICKER, s.TRADE_DATE, s.CLOSE_PRICE)
        """
        
        data_to_insert = []
        
        try:
            # 한글 주석 필수: Wide format (Ticker가 컬럼)을 Long format (날짜, 티커, 가격)으로 변환
            # 이를 통해 모든 종목의 데이터를 날짜순으로 한꺼번에 정렬하여 적재할 수 있습니다.
            df_long = df.stack().reset_index()
            df_long.columns = ['TRADE_DATE', 'TICKER', 'CLOSE_PRICE']
            
            # 한글 주석 필수: 날짜(TRADE_DATE)와 티커(TICKER) 순으로 엄격하게 오름차순 정렬
            df_long = df_long.sort_values(by=['TRADE_DATE', 'TICKER'])
            
            for _, row in df_long.iterrows():
                # datetime -> date 변환 (시간 제거)
                trade_date = row['TRADE_DATE'].to_pydatetime().date()
                data_to_insert.append((str(row['TICKER']), trade_date, float(row['CLOSE_PRICE'])))
            
            if data_to_insert:
                # 한글 주석 필수: 최종 확인차 리스트에서도 날짜순 정렬 수행
                data_to_insert.sort(key=lambda x: (x[1], x[0]))
                
                # 한글 주석 필수: executemany를 사용하여 대량 삽입 성능 향상
                # MERGE 문(Upsert)을 사용하므로 데이터가 중복되어도 안전하게 날짜순으로 들어갑니다.
                self.cursor.executemany(insert_query, data_to_insert)
                self.connection.commit()
                print(f"{len(data_to_insert)}개의 데이터가 날짜 오름차순으로 DB에 성공적으로 저장되었습니다.")
            else:
                print("저장할 데이터가 없습니다.")
                
        except oracledb.Error as e:
            print(f"데이터 삽입 실패: {e}")
            # 한글 주석 필수: 에러 발생 시 롤백하지 않고 오류 출력 (일부 성공 가능성 배제, Transaction 단위)
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
