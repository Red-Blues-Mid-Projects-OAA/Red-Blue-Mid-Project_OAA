"""
이 파일은 데모용 스냅샷을 외부 파일이나 다른 저장소 형식으로 내보냅니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    _BOOT_PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _BOOT_PROJECT_ROOT is not None and str(_BOOT_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_BOOT_PROJECT_ROOT))

import argparse
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import pandas as pd

from DB.stock_db_manager import StockDBManager

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "investment-mbti-back"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from demo_snapshot import SNAPSHOT_PATH, dataframe_to_records, save_demo_snapshot

def _pivot_log_returns_to_records(df: pd.DataFrame) -> list[dict]:
    """로그 수익률 피벗 테이블을 직렬화 가능한 레코드 목록으로 바꿉니다."""
    if df is None or df.empty:
        return []

    working = df.copy()
    working.index = pd.to_datetime(working.index)
    working.index.name = "TRADE_DATE"
    melted = (
        working.reset_index()
        .melt(id_vars="TRADE_DATE", var_name="TICKER", value_name="LOG_RETURN")
        .dropna(subset=["LOG_RETURN"])
        .sort_values(["TRADE_DATE", "TICKER"])
    )
    return dataframe_to_records(melted)

def _sp500_to_records(df: pd.DataFrame) -> list[dict]:
    """S&P 500 데이터를 직렬화 가능한 레코드 목록으로 바꿉니다."""
    if df is None or df.empty:
        return []

    working = df.copy()
    working.index = pd.to_datetime(working.index)
    working.index.name = "TRADE_DATE"
    normalized = working.reset_index().sort_values("TRADE_DATE")
    return dataframe_to_records(normalized)

def _covariance_payload(ticker_list, cov_matrix) -> dict:
    """공분산 행렬을 JSON으로 내보낼 수 있는 형태로 정리합니다."""
    if not ticker_list or cov_matrix is None:
        return {"ticker_list": [], "matrix": []}

    return {
        "ticker_list": [str(t).strip().upper() for t in ticker_list],
        "matrix": cov_matrix.tolist(),
    }

def export_demo_snapshot(lookback_days: int = 450, output_path: Path | None = None) -> Path:
    """데모 스냅샷를 외부 형식으로 내보냅니다."""
    db = StockDBManager()
    connected = False

    try:
        db.connect(ensure_tables=False)
        connected = True

        start_date = datetime.now(timezone.utc).date() - timedelta(days=int(lookback_days))

        adjusted_df = db.fetch_adjusted_expected_returns()
        risk_snapshot_df = db.fetch_risk_level_portfolio_snapshot()
        log_returns_df = db.fetch_log_returns(start_date=start_date)
        sp500_df = db.fetch_sp500_data(start_date=start_date)
        ewma_ticker_list, ewma_cov_matrix = db.fetch_ewma_covariance()

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_days": int(lookback_days),
            "source": "oracle_db",
            "adjusted_expected_returns": dataframe_to_records(adjusted_df),
            "risk_level_portfolio_snapshot": dataframe_to_records(risk_snapshot_df),
            "log_returns": _pivot_log_returns_to_records(log_returns_df),
            "sp500_data": _sp500_to_records(sp500_df),
            "ewma_covariance": _covariance_payload(ewma_ticker_list, ewma_cov_matrix),
            "meta": {
                "adjusted_expected_returns_count": int(len(adjusted_df)),
                "risk_level_portfolio_snapshot_count": int(len(risk_snapshot_df)),
                "log_returns_count": int(
                    0
                    if log_returns_df is None or log_returns_df.empty
                    else int(log_returns_df.count().sum())
                ),
                "sp500_data_count": int(0 if sp500_df is None else len(sp500_df)),
                "ewma_covariance_ticker_count": int(len(ewma_ticker_list)),
            },
        }

        target = save_demo_snapshot(payload, path=output_path or SNAPSHOT_PATH)
        print(
            "[DEMO_SNAPSHOT][OK] "
            f"path={target}, adjusted={payload['meta']['adjusted_expected_returns_count']}, "
            f"risk_rows={payload['meta']['risk_level_portfolio_snapshot_count']}, "
            f"log_returns={payload['meta']['log_returns_count']}, "
            f"sp500={payload['meta']['sp500_data_count']}, "
            f"ewma={payload['meta']['ewma_covariance_ticker_count']}"
        )
        return target
    finally:
        if connected:
            db.close()

def main():
    """메인 관련 처리를 담당하는 함수입니다."""
    parser = argparse.ArgumentParser(description="Export backend demo snapshot JSON")
    parser.add_argument("--lookback-days", type=int, default=450)
    parser.add_argument("--output", type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()

    export_demo_snapshot(lookback_days=args.lookback_days, output_path=args.output)

if __name__ == "__main__":
    main()
