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

    import pandas as pd
    import yfinance as yf

    from DB.stock_db_manager import StockDBManager
    from DB import TICKERS
else:
    from common import yf, pd, datetime, timedelta
    from DB import StockDBManager
    from DB import TICKERS

def _normalize_tickers(tickers):
    if tickers is None:
        return list(TICKERS)
    return [str(t).strip().upper() for t in tickers if str(t).strip()]


def _resolve_end_date(end_date):
    if end_date is None:
        return datetime.now().strftime("%Y-%m-%d")
    return str(end_date)


def _estimate_insert_rows(data):
    # 전체 MultiIndex DataFrame에서 삽입될 행 수를 추정합니다.
    if data.empty:
        return 0
    if isinstance(data.columns, pd.MultiIndex):
        # Adj Close 컬럼으로 유효 행 수 추정 (NaN 제외한 실제 데이터 수)
        if "Adj Close" in data.columns.get_level_values(0):
            return int(data["Adj Close"].count().sum())
        elif "Close" in data.columns.get_level_values(0):
            return int(data["Close"].count().sum())
        return int(data.stack(level=1).shape[0])
    return int(data.shape[0])


def _get_min_latest_date(db_manager, tickers):
    min_latest_date = None
    for ticker in tickers:
        latest_date = db_manager.get_latest_date(ticker)
        if latest_date is None:
            return None
        latest_date = latest_date.date()
        if min_latest_date is None or latest_date < min_latest_date:
            min_latest_date = latest_date
    return min_latest_date


def update_stock_data(
    mode="auto",
    tickers=None,
    start_date="2015-01-01",
    end_date=None,
    recreate_on_full=True,
):
    """
    주가 데이터 적재/업데이트 통합 엔트리.

    mode:
      - auto: 데이터 유무를 보고 full/incremental 자동 선택
      - full: 지정 기간 전체 재적재
      - incremental: DB 최신일 이후 구간만 추가 적재
    """
    selected_tickers = _normalize_tickers(tickers)
    if not selected_tickers:
        raise ValueError("tickers는 최소 1개 이상이어야 합니다.")

    requested_mode = str(mode).lower().strip()
    if requested_mode not in {"auto", "full", "incremental"}:
        raise ValueError("mode는 'auto', 'full', 'incremental' 중 하나여야 합니다.")

    end_date = _resolve_end_date(end_date)

    result = {
        "mode": requested_mode,
        "start_date": None,
        "end_date": end_date,
        "new_rows": 0,
        "updated_any": False,
        "tickers": selected_tickers,
        "status": "success",
    }

    db_manager = StockDBManager()
    try:
        db_manager.connect()

        effective_mode = requested_mode
        effective_start_date = start_date

        if requested_mode == "auto":
            min_latest_date = _get_min_latest_date(db_manager, selected_tickers)
            if min_latest_date is None:
                effective_mode = "full"
                effective_start_date = start_date
                print("일부/전체 종목 데이터가 없어 full 모드로 전체 적재를 수행합니다.")
            else:
                effective_mode = "incremental"
                effective_start_date = (min_latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
                print(
                    f"공통 업데이트 시작일: {effective_start_date} "
                    "(최소 최신 날짜 + 1일)"
                )

        elif requested_mode == "incremental":
            min_latest_date = _get_min_latest_date(db_manager, selected_tickers)
            if min_latest_date is None:
                print("일부 종목 데이터가 없어 incremental 요청을 full로 전환합니다.")
                effective_mode = "full"
                effective_start_date = start_date
            else:
                effective_start_date = (min_latest_date + timedelta(days=1)).strftime("%Y-%m-%d")

        result["mode"] = effective_mode
        result["start_date"] = effective_start_date

        start_ts = pd.Timestamp(effective_start_date).date()
        end_ts = pd.Timestamp(end_date).date()
        if start_ts >= end_ts:
            print("이미 모든 데이터가 최신 상태입니다. (Skip)")
            result["status"] = "skipped"
            return result

        print(f"\n[{effective_mode}] {effective_start_date} ~ {end_date} 데이터 다운로드")
        data = yf.download(
            selected_tickers,
            start=effective_start_date,
            end=end_date,
            auto_adjust=False, # Changed to False to explicitly fetch Adj Close
            progress=True,
        )

        if data.empty:
            print("적재할 데이터가 없습니다.")
            result["status"] = "skipped"
            return result

        data = data.sort_index()
        
        # auto_adjust=False로 다운받으므로 MultiIndex에
        # 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume' 컬럼이 모두 포함됩니다.
        # 전체 DataFrame을 그대로 insert_data에 전달하며,
        # insert_data 내부에서 'Adj Close' → CLOSE_PRICE 매핑을 수행합니다.
        print("\n다운로드 데이터 샘플 (상위 3행):")
        print(data.head(3))

        if effective_mode == "full" and recreate_on_full:
            db_manager.recreate_stock_data_table()

        estimated_rows = _estimate_insert_rows(data)
        db_manager.insert_data(data)

        if effective_mode == "full" and recreate_on_full:
            db_manager.reorganize_stock_data()

        result["new_rows"] = estimated_rows
        result["updated_any"] = estimated_rows > 0
        return result

    except Exception as e:
        print(f"주가 데이터 동기화 중 오류 발생: {e}")
        result["status"] = "error"
        return result
    finally:
        db_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="주가 데이터 full/incremental 동기화")
    parser.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"])
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    update_stock_data(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
    )
