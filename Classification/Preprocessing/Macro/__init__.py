"""
거시지표(Macro) 전처리 공개 인터페이스 모듈.

DXY, VIX, S&P500 모멘텀 등 외생 변수 피처를 수집/정렬하는
거시 데이터 전처리 함수를 외부에 제공합니다.
"""

from .macro import get_market_features

__all__ = ["get_market_features"]
