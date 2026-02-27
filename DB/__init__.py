"""
DB 패키지 공개 인터페이스.

외부 호출부에서 DB 내부 구현 세부사항을 직접 참조하지 않도록
핵심 엔트리포인트(관리자 객체, 업데이트/계산 함수)를
이 모듈 단일 경로로 노출합니다.
"""

from DB.stock_db_manager import StockDBManager

import json
import os

_TOP300_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_top300.json")

def _load_tickers():
    if os.path.exists(_TOP300_JSON_PATH):
        try:
            with open(_TOP300_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback if json is missing
    from DB.utils.sp500_scraper import scrape_sp500_top300_by_weight
    return scrape_sp500_top300_by_weight()

TICKERS = _load_tickers()


def update_stock_data(*args, **kwargs):
    from DB.update_stock_data import update_stock_data as _update_stock_data

    return _update_stock_data(*args, **kwargs)


def update_sp500_data(*args, **kwargs):
    from DB.update_sp500_data import update_sp500_data as _update_sp500_data

    return _update_sp500_data(*args, **kwargs)


def update_market_data(*args, **kwargs):
    from DB.update_market_data import update_market_data as _update_market_data

    return _update_market_data(*args, **kwargs)


def calculate_and_save_log_returns(*args, **kwargs):
    from DB.calculate_log_returns import (
        calculate_and_save_log_returns as _calculate_and_save_log_returns,
    )

    return _calculate_and_save_log_returns(*args, **kwargs)


def sync_risk_level_portfolio_snapshot(*args, **kwargs):
    from DB.update_risk_level_portfolio_snapshot import (
        sync_risk_level_portfolio_snapshot as _sync_risk_level_portfolio_snapshot,
    )

    return _sync_risk_level_portfolio_snapshot(*args, **kwargs)


def calculate_ewma_covariance(*args, **kwargs):
    from DB.calculate_ewma import calculate_ewma_covariance as _calculate_ewma_covariance

    return _calculate_ewma_covariance(*args, **kwargs)

__all__ = [
    "StockDBManager",
    "TICKERS",
    "update_stock_data",
    "update_sp500_data",
    "update_market_data",
    "calculate_and_save_log_returns",
    "calculate_ewma_covariance",
    "sync_risk_level_portfolio_snapshot",
]
