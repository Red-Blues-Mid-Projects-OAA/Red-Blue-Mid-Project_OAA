"""
프로젝트 전반에서 공통적으로 사용되는 외부 라이브러리 및 모듈을 통합 관리하는 패키지입니다.
매번 개별 파일에서 import하는 번거로움을 줄이고 관리를 일원화합니다.
"""

# 외부 라이브러리 통합 임포트
import pandas as pd
import numpy as np
import os
import yfinance as yf
from datetime import datetime, timedelta

# 패키지 수준에서 편리하게 접근할 수 있도록 노출
__all__ = ['pd', 'np', 'os', 'yf', 'datetime', 'timedelta']
