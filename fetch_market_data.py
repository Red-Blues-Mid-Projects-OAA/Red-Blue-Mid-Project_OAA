"""
시장 지표(VIX, DXY) 데이터 수집 및 Oracle DB 적재 모듈

- VIX 종가 : yfinance (^VIX)
- 달러 인덱스 종가 : FRED (DTWEXBGS)
- 각 지표의 로그 수익률을 계산하여 MARKET_FEATURES 테이블에 저장

실행: python fetch_market_data.py
"""

from common import pd, np, yf, datetime, timedelta, web
from stock_db_manager import StockDBManager

START_DATE = "2015-01-01"


def fetch_and_store_vix(db_manager):
    """
    yfinance에서 VIX 데이터를 다운로드하여 DB에 적재 (증분 업데이트)
    """
    print("=" * 60)
    print("[1] VIX 데이터 수집 및 DB 적재")
    print("=" * 60)

    # 최신 날짜 확인
    latest_date = db_manager.get_latest_market_date("VIX")

    if latest_date:
        if (latest_date.date() + timedelta(days=1)) >= datetime.now().date():
            print("VIX 데이터가 이미 최신 상태입니다. (Skip)")
            return
        fetch_start = latest_date.strftime('%Y-%m-%d')
        print(f"  최신 데이터: {latest_date.date()} → 이후 날짜부터 다운로드")
    else:
        fetch_start = START_DATE
        print(f"  기존 데이터 없음 → {START_DATE}부터 전체 다운로드")

    # yfinance에서 VIX 다운로드
    vix_raw = yf.download("^VIX", start=fetch_start, auto_adjust=True)
    vix_close = vix_raw["Close"]

    # MultiIndex 처리
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    vix_close.index = vix_close.index.tz_localize(None)
    vix_close = vix_close.dropna()

    df_vix = pd.DataFrame({"Close": vix_close})

    # 로그 수익률 계산
    df_vix["Log_Return"] = np.log(df_vix["Close"] / df_vix["Close"].shift(1))
    df_vix = df_vix.dropna()

    # 증분 필터링: 이미 DB에 있는 날짜 제외
    if latest_date:
        latest_date_val = latest_date.date() if isinstance(latest_date, datetime) else latest_date
        df_vix = df_vix[df_vix.index.date > latest_date_val]

    if df_vix.empty:
        print("새로 추가할 VIX 데이터가 없습니다.")
        return

    print(f"{len(df_vix)}건 적재 시작...")
    db_manager.insert_market_features("VIX", df_vix)


def fetch_and_store_dxy(db_manager):
    """
    FRED에서 달러 인덱스(DTWEXBGS) 데이터를 다운로드하여 DB에 적재 (증분 업데이트)
    """
    print("\n" + "=" * 60)
    print("[2] 달러 인덱스(DXY) 데이터 수집 및 DB 적재")
    print("=" * 60)

    # 최신 날짜 확인
    latest_date = db_manager.get_latest_market_date("DXY")
    end_date = datetime.now().strftime("%Y-%m-%d")

    if latest_date:
        if (latest_date.date() + timedelta(days=1)) >= datetime.now().date():
            print("DXY 데이터가 이미 최신 상태입니다. (Skip)")
            return
        fetch_start = latest_date.strftime('%Y-%m-%d')
        print(f"  최신 데이터: {latest_date.date()} → 이후 날짜부터 다운로드")
    else:
        fetch_start = START_DATE
        print(f"  기존 데이터 없음 → {START_DATE}부터 전체 다운로드")

    # FRED에서 달러 인덱스 다운로드
    dxy = web.DataReader("DTWEXBGS", "fred", fetch_start, end_date)
    dxy.columns = ["Close"]
    dxy = dxy.dropna()

    # 로그 수익률 계산
    dxy["Log_Return"] = np.log(dxy["Close"] / dxy["Close"].shift(1))
    dxy = dxy.dropna()

    # 증분 필터링
    if latest_date:
        latest_date_val = latest_date.date() if isinstance(latest_date, datetime) else latest_date
        dxy = dxy[dxy.index.date > latest_date_val]

    if dxy.empty:
        print("  새로 추가할 DXY 데이터가 없습니다.")
        return

    print(f"  {len(dxy)}건 적재 시작...")
    db_manager.insert_market_features("DXY", dxy)


def main():
    db_manager = StockDBManager()
    try:
        db_manager.connect()

        # 1. VIX 데이터 적재
        fetch_and_store_vix(db_manager)

        # 2. DXY 데이터 적재
        fetch_and_store_dxy(db_manager)

        # 3. 테이블 재구조화 (날짜순 정렬)
        db_manager.reorganize_market_features()

        # 4. 적재 결과 확인
        print("\n" + "=" * 60)
        print("적재 결과 확인")
        print("=" * 60)
        for indicator in ["VIX", "DXY"]:
            df = db_manager.fetch_market_features(indicator)
            if not df.empty:
                print(f"  {indicator}: {len(df)}건 | {df.index[0].date()} ~ {df.index[-1].date()}")
            else:
                print(f"  {indicator}: 데이터 없음")

    except Exception as e:
        print(f"오류 발생: {e}")
        
    finally:
        db_manager.close()


if __name__ == "__main__":
    main()
