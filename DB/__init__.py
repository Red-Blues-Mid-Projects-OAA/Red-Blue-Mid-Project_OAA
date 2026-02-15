"""DB 패키지 공개 인터페이스."""

from DB.stock_db_manager import StockDBManager

TICKERS = [
    "NVDA",
    "GOOGL",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "TSM",
    "TSLA",
    "AVGO",
    "BRK-A",
]


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
]
