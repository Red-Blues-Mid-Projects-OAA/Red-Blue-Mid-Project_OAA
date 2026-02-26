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

    import numpy as np
    import pandas as pd

    from DB.stock_db_manager import StockDBManager
else:
    from common import np, pd
    from DB import StockDBManager


def calculate_and_save_log_returns(mode="auto", lookback_days=10, reorganize_on_full=True):
    """
    STOCK_DATA에서 로그 수익률을 계산해 LOG_RETURNS에 저장합니다.

    mode:
      - auto: 최신 날짜를 비교해 full/incremental 자동 선택
      - full: LOG_RETURNS 전체 재계산
      - incremental: 최신 날짜 이후만 계산/적재
    """
    print("로그 수익률 계산 프로세스 시작...")

    db_manager = StockDBManager()
    db_manager.connect()

    try:
        requested_mode = str(mode).lower().strip()
        if requested_mode not in {"auto", "full", "incremental"}:
            raise ValueError("mode는 auto/full/incremental 중 하나여야 합니다.")

        db_manager.cursor.execute("SELECT MAX(TRADE_DATE) FROM STOCK_DATA")
        stock_latest = db_manager.cursor.fetchone()[0]
        if stock_latest is None:
            print("STOCK_DATA가 비어있어 로그 수익률 계산을 종료합니다.")
            return {"status": "skipped", "reason": "stock_data_empty"}

        log_latest = db_manager.get_latest_log_returns_date()
        effective_mode = requested_mode

        if requested_mode == "auto":
            if log_latest is None:
                effective_mode = "full"
            elif stock_latest > log_latest:
                effective_mode = "incremental"
            else:
                print(
                    f"이미 최신입니다. (STOCK_DATA 최신: {stock_latest.date()}, "
                    f"LOG_RETURNS 최신: {log_latest.date()})"
                )
                return {
                    "status": "skipped",
                    "mode": "auto",
                    "stock_latest_date": stock_latest.strftime("%Y-%m-%d"),
                    "log_latest_date": log_latest.strftime("%Y-%m-%d"),
                }
        elif requested_mode == "incremental":
            if log_latest is None:
                print("LOG_RETURNS가 비어 있어 full 모드로 전환합니다.")
                effective_mode = "full"
            elif stock_latest <= log_latest:
                print(
                    f"이미 최신입니다. (STOCK_DATA 최신: {stock_latest.date()}, "
                    f"LOG_RETURNS 최신: {log_latest.date()})"
                )
                return {
                    "status": "skipped",
                    "mode": "incremental",
                    "stock_latest_date": stock_latest.strftime("%Y-%m-%d"),
                    "log_latest_date": log_latest.strftime("%Y-%m-%d"),
                }

        if effective_mode == "full":
            prices_df = db_manager.fetch_prices()
            if prices_df.empty:
                print("주가 데이터가 없어 계산을 중단합니다.")
                return {"status": "skipped", "reason": "price_empty"}

            print(f"주가 데이터 로드 완료: {prices_df.shape}")
            log_returns_df = np.log(prices_df / prices_df.shift(1))

            print("\n[DB 정리] 기존 LOG_RETURNS 데이터를 초기화합니다...")
            db_manager.truncate_log_returns()

            row_count = int(log_returns_df.count().sum())
            print(f"로그 수익률 데이터 {row_count}건 저장 시작...")
            db_manager.insert_log_returns(log_returns_df)

            if reorganize_on_full:
                db_manager.reorganize_log_returns()

            return {
                "status": "success",
                "mode": "full",
                "inserted_rows": row_count,
                "latest_date": stock_latest.strftime("%Y-%m-%d"),
            }

        # incremental 모드
        warmup_days = max(int(lookback_days), 1)
        start_date = (pd.Timestamp(log_latest) - pd.Timedelta(days=warmup_days)).date()
        prices_df = db_manager.fetch_prices(start_date=start_date)
        if prices_df.empty:
            print("증분 계산 대상 주가 데이터가 없습니다.")
            return {
                "status": "skipped",
                "mode": "incremental",
                "reason": "price_empty",
            }

        print(
            f"증분 계산용 주가 데이터 로드 완료: {prices_df.shape} "
            f"(start={start_date}, latest_log={log_latest.date()})"
        )

        log_returns_df = np.log(prices_df / prices_df.shift(1))
        new_log_returns_df = log_returns_df[log_returns_df.index > pd.Timestamp(log_latest)]
        new_rows = int(new_log_returns_df.count().sum())

        if new_rows == 0:
            print("새로 저장할 로그 수익률이 없습니다.")
            return {
                "status": "skipped",
                "mode": "incremental",
                "stock_latest_date": stock_latest.strftime("%Y-%m-%d"),
                "log_latest_date": log_latest.strftime("%Y-%m-%d"),
                "inserted_rows": 0,
            }

        print(f"로그 수익률 증분 데이터 {new_rows}건 저장 시작...")
        db_manager.insert_log_returns(new_log_returns_df)
        return {
            "status": "success",
            "mode": "incremental",
            "stock_latest_date": stock_latest.strftime("%Y-%m-%d"),
            "log_latest_date": log_latest.strftime("%Y-%m-%d"),
            "inserted_rows": new_rows,
        }
    except Exception as e:
        print(f"계산 중 오류 발생: {e}")
        return {"status": "error", "error": str(e)}
    finally:
        db_manager.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LOG_RETURNS 계산/저장")
    parser.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"])
    parser.add_argument("--lookback-days", type=int, default=10, help="증분 계산 워밍업 기간(일)")
    parser.add_argument(
        "--skip-reorganize-on-full",
        action="store_true",
        default=False,
        help="full 모드에서 reorganize_log_returns 생략",
    )
    args = parser.parse_args()

    calculate_and_save_log_returns(
        mode=args.mode,
        lookback_days=args.lookback_days,
        reorganize_on_full=not args.skip_reorganize_on_full,
    )
