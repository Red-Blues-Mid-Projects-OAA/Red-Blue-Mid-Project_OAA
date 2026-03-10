"""
이 파일은 S&P 500 지수 데이터를 갱신해 비교 기준으로 사용할 수 있게 준비합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))
        
    from datetime import datetime, timedelta

    import numpy as np
    import pandas as pd
    import yfinance as yf

    from DB.stock_db_manager import StockDBManager
else:
    from common import yf, pd, datetime, timedelta, np
    from DB import StockDBManager

def update_sp500_data():
    """
    S&P 500 지수(^GSPC) 데이터를 다운로드하고 로그 수익률을 계산하여 DB에 저장
    """
    print("S&P 500의 최신 데이터 날짜를 확인합니다.")

    result = {
        "new_rows": 0,
        "updated_any": False,
        "status": "success",
        "start_date": None,
        "end_date": datetime.now().strftime("%Y-%m-%d"),
    }

    db_manager = StockDBManager()

    try:
        db_manager.connect()

        latest_date = db_manager.get_latest_sp500_date()
        end_date = datetime.now().strftime("%Y-%m-%d")
        result["end_date"] = end_date

        if latest_date:
            if (latest_date.date() + timedelta(days=1)) >= datetime.now().date():
                print("이미 모든 데이터가 최신 상태입니다. (Skip)")
                result["status"] = "skipped"
                return result

            fetch_start_date = latest_date.strftime("%Y-%m-%d")
            print(f"최신 데이터 날짜: {latest_date.date()} -> 이후 날짜부터 다운로드를 진행합니다.")
        else:
            print("기존 데이터가 없어 2015-01-01부터 전체 다운로드를 시도합니다.")
            fetch_start_date = "2015-01-01"

        result["start_date"] = fetch_start_date

        print(f"{fetch_start_date} ~ {end_date} S&P 500 데이터 다운로드 중...")
        ticker = "^GSPC"
        df = yf.download(ticker, start=fetch_start_date, end=end_date, auto_adjust=False, progress=True)

        if isinstance(df.columns, pd.MultiIndex):
            data = df["Adj Close"][ticker]
        else:
            data = df["Adj Close"]

        if data.empty:
            print("업데이트할 데이터가 없습니다 (휴장일 등)")
            result["status"] = "skipped"
            return result

        data = pd.DataFrame(data)
        data.columns = ["Close"] # DB column name expects 'Close' logic below to remain unchanged

        data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
        data.dropna(inplace=True)

        if latest_date:
            latest_date_val = latest_date.date() if isinstance(latest_date, datetime) else latest_date
            data_to_insert = data[data.index.date > latest_date_val].copy()

            if data_to_insert.empty:
                print("새로 추가할 데이터가 없습니다 (이미 최신 상태).")
                result["status"] = "skipped"
                return result
        else:
            data_to_insert = data.copy()

        print(f"로그 수익률 데이터 {len(data_to_insert)}건 저장 시작...")

        db_manager.insert_sp500_data(data_to_insert)
        db_manager.reorganize_sp500_data()

        result["new_rows"] = int(len(data_to_insert))
        result["updated_any"] = result["new_rows"] > 0
        return result

    except Exception as e:
        print(f"업데이트 중 오류 발생: {e}")
        result["status"] = "error"
        return result

    finally:
        db_manager.close()

if __name__ == "__main__":
    update_sp500_data()
