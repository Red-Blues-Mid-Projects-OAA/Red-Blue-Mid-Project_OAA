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
    if data.empty:
        return 0
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            return int(data["Adj Close"].stack().shape[0])
        elif "Close" in data.columns.get_level_values(0):
            return int(data["Close"].stack().shape[0])
        return int(data.stack(level=1).shape[0])
    # flat dataframe (Adj Close only) -> rows x columns
    return int(data.shape[0] * data.shape[1])


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

        if effective_start_date >= end_date:
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
        
        # yfinance 0.2.x 이후 여러 종목을 다운받을 때 컬럼이 MultiIndex 이며,
        # 최상위 레벨에 'Adj Close', 하위 레벨에 종목 코드가 들어갑니다.
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                # 'Adj Close'만 추출하여 DataFrame 재구성
                data = data["Adj Close"].copy()
            else:
                # 만약 어떤 이유로 'Adj Close'가 없다면 'Close'로 진행
                data = data["Close"].copy()
                print("Warning: 'Adj Close' not found. Using 'Close' instead.")
        else:
            # 단일 종목일 경우
            pass # we shouldn't hit this with 300 tickers, but just in case
            
        print("\n다운로드 데이터 샘플 (상위 5행) - Adj Close만 추출:")
        print(data.head())

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
