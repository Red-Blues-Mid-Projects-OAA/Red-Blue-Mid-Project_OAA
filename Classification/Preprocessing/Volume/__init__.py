"""
거래량(Volume) 전처리 공개 인터페이스 모듈.

거래량 비율, OBV 기반 변화율 등 거래량 파생 피처 생성 함수를
패키지 외부에서 일관된 경로로 호출할 수 있도록 노출합니다.
"""

from .volume import calculate_aapl_volume_analysis

__all__ = ["calculate_aapl_volume_analysis"]
