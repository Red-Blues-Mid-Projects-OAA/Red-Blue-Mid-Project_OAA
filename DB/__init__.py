"""
이 파일은 DB 패키지를 한 번에 불러오기 쉽게 만드는 초기화 파일입니다. 공통으로 노출할 모듈이나 기본 설정을 정리할 때 사용합니다.
이 초기화 파일은 하위 모듈의 공개 함수와 상수를 한곳에 모아 외부 import 경로를 단순하게 유지하고, 내부 폴더 구조가 바뀌어도 상위 호출부 수정 범위를 줄이는 역할을 맡습니다.
"""

from DB.stock_db_manager import StockDBManager

import json
import os

_TOP300_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp500_top300.json")

def _load_tickers():
    """티커 목록 데이터를 메모리로 불러옵니다."""
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
    """종목 데이터 데이터를 최신 상태로 갱신합니다."""
    from DB.update_stock_data import update_stock_data as _update_stock_data

    return _update_stock_data(*args, **kwargs)

def update_sp500_data(*args, **kwargs):
    """S&P 500 데이터 데이터를 최신 상태로 갱신합니다."""
    from DB.update_sp500_data import update_sp500_data as _update_sp500_data

    return _update_sp500_data(*args, **kwargs)

def update_market_data(*args, **kwargs):
    """시장 데이터 데이터를 최신 상태로 갱신합니다."""
    from DB.update_market_data import update_market_data as _update_market_data

    return _update_market_data(*args, **kwargs)

def calculate_and_save_log_returns(*args, **kwargs):
    """save 로그 수익률 값을 계산합니다."""
    from DB.calculate_log_returns import (
        calculate_and_save_log_returns as _calculate_and_save_log_returns,
    )

    return _calculate_and_save_log_returns(*args, **kwargs)

def sync_risk_level_portfolio_snapshot(*args, **kwargs):
    """sync 위험 단계 포트폴리오 스냅샷 관련 처리를 담당하는 함수입니다."""
    from DB.update_risk_level_portfolio_snapshot import (
        sync_risk_level_portfolio_snapshot as _sync_risk_level_portfolio_snapshot,
    )

    return _sync_risk_level_portfolio_snapshot(*args, **kwargs)

def calculate_ewma_covariance(*args, **kwargs):
    """ewma covariance 값을 계산합니다."""
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
