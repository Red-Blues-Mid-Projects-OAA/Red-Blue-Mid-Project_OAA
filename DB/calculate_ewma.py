"""
이 파일은 최근 데이터에 더 큰 가중치를 두는 EWMA 통계를 계산합니다.
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

    import numpy as np
    import pandas as pd

    from DB.stock_db_manager import StockDBManager
else:
    from common import pd, np
    from DB import StockDBManager

def calculate_ewma_covariance(lambda_val=0.94, mode="auto", lookback_days=756):
    """
    LOG_RETURNS 기반 EWMA 공분산 행렬을 계산해 EWMA_COVARIANCE에 저장합니다.

    mode:
      - auto: 최신 날짜 비교 후 full/incremental 자동 선택
      - full: 전체 LOG_RETURNS 기반 계산
      - incremental: 최신 EWMA 계산일 이후 데이터가 있을 때만 계산
    """
    print("EWMA 공분산 계산 프로세스 시작...")

    requested_mode = str(mode).lower().strip()
    if requested_mode not in {"auto", "full", "incremental"}:
        raise ValueError("mode는 auto/full/incremental 중 하나여야 합니다.")

    db_manager = StockDBManager()
    db_manager.connect()

    try:
        latest_log_date = db_manager.get_latest_log_returns_date()
        if latest_log_date is None:
            print("LOG_RETURNS가 비어 있어 EWMA 계산을 종료합니다.")
            return {"status": "skipped", "reason": "log_returns_empty"}

        latest_cov_date = db_manager.get_latest_ewma_cov_date()
        effective_mode = requested_mode

        if requested_mode == "auto":
            if latest_cov_date is None:
                effective_mode = "full"
            elif latest_log_date > latest_cov_date:
                effective_mode = "incremental"
            else:
                print(
                    f"이미 최신입니다. (LOG_RETURNS 최신: {latest_log_date.date()}, "
                    f"EWMA 최신: {latest_cov_date.date()})"
                )
                return {
                    "status": "skipped",
                    "mode": "auto",
                    "log_latest_date": latest_log_date.strftime("%Y-%m-%d"),
                    "ewma_latest_date": latest_cov_date.strftime("%Y-%m-%d"),
                }
        elif requested_mode == "incremental":
            if latest_cov_date is None:
                print("EWMA_COVARIANCE가 비어 있어 full 모드로 전환합니다.")
                effective_mode = "full"
            elif latest_log_date <= latest_cov_date:
                print(
                    f"이미 최신입니다. (LOG_RETURNS 최신: {latest_log_date.date()}, "
                    f"EWMA 최신: {latest_cov_date.date()})"
                )
                return {
                    "status": "skipped",
                    "mode": "incremental",
                    "log_latest_date": latest_log_date.strftime("%Y-%m-%d"),
                    "ewma_latest_date": latest_cov_date.strftime("%Y-%m-%d"),
                }

        if effective_mode == "full":
            log_returns = db_manager.fetch_log_returns()
            range_desc = "전체"
        else:
            warmup_days = max(int(lookback_days), 1)
            start_date = (pd.Timestamp(latest_cov_date) - pd.Timedelta(days=warmup_days)).date()
            log_returns = db_manager.fetch_log_returns(start_date=start_date)
            range_desc = f"start={start_date}"

        if log_returns.empty:
            print("EWMA 계산 대상 LOG_RETURNS 데이터가 없습니다.")
            return {
                "status": "skipped",
                "mode": effective_mode,
                "reason": "log_returns_empty_after_filter",
            }

        if log_returns.shape[0] < 2:
            print("EWMA 계산에 필요한 최소 행(2개)이 부족합니다.")
            return {
                "status": "skipped",
                "mode": effective_mode,
                "reason": "insufficient_rows",
                "rows": int(log_returns.shape[0]),
            }

        print(f"LOG_RETURNS 로드 완료: {log_returns.shape} ({range_desc})")

        ewma_cov_series = log_returns.ewm(alpha=(1 - lambda_val)).cov()

        num_stocks = len(log_returns.columns)
        latest_daily_cov = ewma_cov_series.tail(num_stocks)
        latest_date = latest_daily_cov.index.get_level_values(0)[0]
        latest_daily_cov_matrix = latest_daily_cov.droplevel(0)

        if latest_cov_date is not None and latest_date <= latest_cov_date:
            print(
                f"이미 최신 공분산입니다. (계산일: {latest_date.date()}, "
                f"기존 EWMA 최신: {latest_cov_date.date()})"
            )
            return {
                "status": "skipped",
                "mode": effective_mode,
                "log_latest_date": latest_log_date.strftime("%Y-%m-%d"),
                "ewma_latest_date": latest_cov_date.strftime("%Y-%m-%d"),
            }

        print(f"\n[{latest_date.date()}] 최신 일별 공분산 행렬 계산 완료")
        print(f"EWMA 공분산 행렬 ({latest_date.date()}) 저장 시작...")
        db_manager.insert_ewma_covariance(latest_date, latest_daily_cov_matrix)

        return {
            "status": "success",
            "mode": effective_mode,
            "saved_calc_date": latest_date.strftime("%Y-%m-%d"),
            "log_latest_date": latest_log_date.strftime("%Y-%m-%d"),
            "ewma_prev_latest_date": (
                latest_cov_date.strftime("%Y-%m-%d") if latest_cov_date is not None else None
            ),
            "matrix_shape": [int(latest_daily_cov_matrix.shape[0]), int(latest_daily_cov_matrix.shape[1])],
        }
    finally:
        db_manager.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EWMA 공분산 계산/저장")
    parser.add_argument("--lambda-val", type=float, default=0.94, help="EWMA lambda 값")
    parser.add_argument("--mode", default="auto", choices=["auto", "full", "incremental"])
    parser.add_argument("--lookback-days", type=int, default=756, help="증분 계산 워밍업 기간(일)")
    args = parser.parse_args()

    calculate_ewma_covariance(
        lambda_val=args.lambda_val,
        mode=args.mode,
        lookback_days=args.lookback_days,
    )
