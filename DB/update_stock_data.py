"""
이 파일은 개별 종목 시세 데이터를 새로 받아 데이터베이스에 반영합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

# 수정 이력 (주석 추가 전 라인 번호 기준)
# - 57줄: _build_ticker_start_dates 정리
# - 72줄: _trim_downloaded_data_by_ticker_start 유지/정리
# - 99줄: update_stock_data 리팩터링
#   * 최신일 조회 SQL 중복 제거 -> StockDBManager.get_latest_dates_map 사용
#   * _estimate_insert_rows 제거
#   * insert_data 반환값 기준으로 실제 반영 row 집계
# - 59줄: _resolve_download_end_date 추가 (yfinance end 배타 처리 보정)
# - 87줄: _trim_downloaded_data_by_ticker_start 멀티인덱스 레이아웃 양방향 대응
# - 128줄: _extract_single_ticker_from_batch 추가 (배치 0건 시 로컬 fallback 분리)
# - 214줄: active_tickers 조건 < -> <= 변경 (종료일 포함 의미로 정렬)
# - 56줄: end_date 기본값을 오늘 -> 어제로 변경 (미완성 일봉 요청 방지)

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
    """티커 목록 값을 서로 비교하기 쉽게 정규화합니다."""
    if tickers is None:
        source = list(TICKERS)
    else:
        source = [str(t).strip().upper() for t in tickers if str(t).strip()]

    deduped = []
    seen = set()
    for ticker in source:
        if ticker in seen:
            continue
        seen.add(ticker)
        deduped.append(ticker)
    return deduped

def _resolve_end_date(end_date):
    """resolve end date 관련 처리를 담당하는 함수입니다."""
    if end_date is None:
        # Use previous day by default to avoid querying incomplete "today" daily bars.
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return str(end_date)

def _resolve_download_end_date(end_date):
    """
    yfinance end is exclusive, so convert inclusive end_date -> exclusive end_date+1.
    """
    return (pd.Timestamp(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")

def _chunked(values, size):
    """chunked 관련 처리를 담당하는 함수입니다."""
    step = max(int(size), 1)
    for idx in range(0, len(values), step):
        yield values[idx : idx + step]

def _build_ticker_start_dates(effective_mode, selected_tickers, start_date, latest_map):
    """Build per-ticker start dates (YYYY-MM-DD)."""
    if effective_mode == "full":
        return {ticker: start_date for ticker in selected_tickers}

    starts = {}
    for ticker in selected_tickers:
        latest_dt = latest_map.get(ticker)
        if latest_dt is None:
            starts[ticker] = start_date
        else:
            starts[ticker] = (latest_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")
    return starts

def _trim_downloaded_data_by_ticker_start(data, ticker_start_map):
    """
    Trim rows before each ticker start date to reduce unnecessary upserts.
    """
    if data is None or data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        trimmed = data.copy()
        idx = pd.IndexSlice

        requested = set(str(t) for t in ticker_start_map.keys())
        level0 = set(str(t) for t in trimmed.columns.get_level_values(0))
        level1 = set(str(t) for t in trimmed.columns.get_level_values(1))
        if requested & level1:
            ticker_level = 1
        elif requested & level0:
            ticker_level = 0
        else:
            return trimmed

        for ticker, start_str in ticker_start_map.items():
            if ticker_level == 1 and ticker not in level1:
                continue
            if ticker_level == 0 and ticker not in level0:
                continue
            cutoff = pd.Timestamp(start_str)
            mask = trimmed.index < cutoff
            if mask.any():
                if ticker_level == 1:
                    trimmed.loc[mask, idx[:, ticker]] = float("nan")
                else:
                    trimmed.loc[mask, idx[ticker, :]] = float("nan")

        return trimmed.dropna(how="all")

    first_ticker = next(iter(ticker_start_map.keys()))
    cutoff = pd.Timestamp(ticker_start_map[first_ticker])
    return data.loc[data.index >= cutoff]

def _extract_single_ticker_from_batch(data, ticker):
    """
    Split one ticker view from a multi-ticker yfinance batch frame.
    Returns flat OHLCV columns if possible.
    """
    if data is None or data.empty:
        return data

    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    try:
        frame = data.xs(ticker, axis=1, level=1, drop_level=True)
    except Exception:
        try:
            frame = data.xs(ticker, axis=1, level=0, drop_level=True)
        except Exception:
            return pd.DataFrame(index=data.index)

    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame

def update_stock_data(
    mode="auto",
    tickers=None,
    start_date="2015-01-01",
    end_date=None,
    recreate_on_full=True,
    download_batch_size=80,
    show_progress=False,
):
    """
    Unified stock update entrypoint.

    mode:
      - auto: choose full/incremental from DB state
      - full: load full range
      - incremental: load per-ticker from latest+1
    """
    selected_tickers = _normalize_tickers(tickers)
    if not selected_tickers:
        raise ValueError("tickers must include at least one symbol")

    requested_mode = str(mode).lower().strip()
    if requested_mode not in {"auto", "full", "incremental"}:
        raise ValueError("mode must be one of: auto, full, incremental")

    user_provided_end_date = end_date is not None
    end_date = _resolve_end_date(end_date)
    download_end_date = _resolve_download_end_date(end_date)

    result = {
        "mode": requested_mode,
        "start_date": None,
        "end_date": end_date,
        "download_end_date": download_end_date,
        "user_provided_end_date": user_provided_end_date,
        "new_rows": 0,
        "updated_any": False,
        "tickers": selected_tickers,
        "status": "success",
    }

    db_manager = StockDBManager()
    try:
        db_manager.connect()

        # moved from local helper to DB layer to avoid duplicated SQL code
        latest_map = db_manager.get_latest_dates_map(selected_tickers)

        if requested_mode == "auto":
            effective_mode = (
                "full"
                if all(latest_map.get(ticker) is None for ticker in selected_tickers)
                else "incremental"
            )
        else:
            effective_mode = requested_mode

        ticker_start_map = _build_ticker_start_dates(
            effective_mode,
            selected_tickers,
            start_date,
            latest_map,
        )

        end_ts = pd.Timestamp(end_date).date()
        active_tickers = [
            ticker
            for ticker, ticker_start in ticker_start_map.items()
            if pd.Timestamp(ticker_start).date() <= end_ts
        ]

        if not active_tickers:
            print("No new data range to fetch. Skipping.")
            result["status"] = "skipped"
            result["mode"] = effective_mode
            return result

        result["mode"] = effective_mode
        result["start_date"] = min(ticker_start_map[ticker] for ticker in active_tickers)

        if effective_mode == "full" and recreate_on_full:
            db_manager.recreate_stock_data_table()

        total_rows = 0

        for ticker_batch in _chunked(sorted(active_tickers), download_batch_size):
            batch_start = min(ticker_start_map[ticker] for ticker in ticker_batch)
            print(
                f"\n[{effective_mode}] {batch_start} ~ {end_date} "
                f"({len(ticker_batch)} tickers)"
            )

            data = yf.download(
                ticker_batch,
                start=batch_start,
                end=download_end_date,
                auto_adjust=False,
                progress=show_progress,
                threads=True,
            )

            if data.empty:
                continue

            data = data.sort_index()
            batch_starts = {ticker: ticker_start_map[ticker] for ticker in ticker_batch}
            data = _trim_downloaded_data_by_ticker_start(data, batch_starts)
            if data is None or data.empty:
                continue

            ticker_hint = ticker_batch[0] if len(ticker_batch) == 1 else None
            inserted_rows = db_manager.insert_data(
                data,
                commit=False,
                single_ticker=ticker_hint,
            )
            if not inserted_rows:
                print(f"[WARN] no rows inserted for batch: {ticker_batch}")
                if len(ticker_batch) > 1:
                    fallback_rows = 0
                    for ticker in ticker_batch:
                        single_start = ticker_start_map[ticker]
                        single_data = _extract_single_ticker_from_batch(data, ticker)
                        if single_data is None or single_data.empty:
                            single_data = yf.download(
                                ticker,
                                start=single_start,
                                end=download_end_date,
                                auto_adjust=False,
                                progress=False,
                                threads=False,
                            )
                        if single_data is None or single_data.empty:
                            continue
                        single_data = single_data.sort_index()
                        single_data = _trim_downloaded_data_by_ticker_start(
                            single_data,
                            {ticker: single_start},
                        )
                        if single_data is None or single_data.empty:
                            continue

                        try:
                            fallback_rows += int(
                                db_manager.insert_data(
                                    single_data,
                                    commit=False,
                                    single_ticker=ticker,
                                )
                                or 0
                            )
                        except Exception as retry_error:
                            print(
                                f"[WARN] fallback insert failed for {ticker}: "
                                f"{retry_error}"
                            )

                    if fallback_rows > 0:
                        print(
                            f"[INFO] fallback single-ticker insert rows: "
                            f"{fallback_rows}"
                        )
                        inserted_rows = fallback_rows
            total_rows += int(inserted_rows or 0)

        if total_rows > 0:
            db_manager.connection.commit()

        if effective_mode == "full" and recreate_on_full:
            db_manager.reorganize_stock_data()

        result["new_rows"] = total_rows
        result["updated_any"] = total_rows > 0
        if total_rows == 0:
            result["status"] = "skipped"
            if pd.Timestamp(end_date).date() >= datetime.now().date():
                result["reason"] = (
                    "No completed daily bars for end_date yet. "
                    "Try previous trading day."
                )
                print(
                    "[INFO] No completed daily bars for end_date yet. "
                    "Try previous trading day."
                )

        return result

    except Exception as e:
        try:
            if db_manager.connection:
                db_manager.connection.rollback()
        except Exception:
            pass
        print(f"update_stock_data failed: {e}")
        result["status"] = "error"
        return result
    finally:
        db_manager.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync stock data (full/incremental)")
    parser.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"])
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    sync_result = update_stock_data(
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        download_batch_size=args.batch_size,
        show_progress=args.progress,
    )
    print(f"sync_result={sync_result}")
