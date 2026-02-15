"""
변동성(Volatility) 전처리 공개 인터페이스 모듈.

EWMA 기반 일별 변동성, 이동평균 변동성, 시장과의 상관계수 등
리스크 성격의 피처 생성 함수를 외부로 노출합니다.
"""

from .volatility import calculate_risk_features

__all__ = ["calculate_risk_features"]
