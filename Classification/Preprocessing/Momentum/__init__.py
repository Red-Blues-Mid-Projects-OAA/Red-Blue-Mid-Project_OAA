"""
모멘텀(Momentum) 전처리 공개 인터페이스 모듈.

로그수익률, 이격도, 52주 고점 근접도, RSI 등
가격 기반 모멘텀 피처 생성 함수를 외부에 노출합니다.
"""

from .momentum import calculate_features

__all__ = ["calculate_features"]
