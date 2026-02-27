"""
실제 데이터 기반 종목 추천 모듈.

adjusted_expected_returns.csv에서 필터링된 종목을 대상으로,
리스크 타입별 λ를 이용한 개별 유틸리티 스코어로 종목을 정렬하여 추천합니다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트 경로 설정 (investment-mbti-back에서 실행 시)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DB import StockDBManager
from risk_profile import get_lambda_by_level

# ──────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────
# adjusted_expected_returns.csv 경로
_CSV_PATH = _PROJECT_ROOT / "Classification" / "artifacts" / "multi_ticker" / "adjusted_expected_returns.csv"

# 예측 Horizon (거래일 기준, 일별 공분산 → 3개월 스케일링용)
HORIZON_DAYS = 60

# 리스크 타입별 λ 값 (risk_profile.py 로직과 동기화된 동적 매핑)
LAMBDA_MAP = {i: get_lambda_by_level(i) for i in range(1, 5)}

# 종목 한글 이름 매핑 (주요 종목)
TICKER_NAME_MAP = {
    "NVDA": "엔비디아", "AAPL": "애플", "MSFT": "마이크로소프트", "AMZN": "아마존",
    "GOOGL": "알파벳 A", "GOOG": "알파벳 C", "META": "메타", "AVGO": "브로드컴",
    "TSLA": "테슬라", "BRK-B": "버크셔 해서웨이", "WMT": "월마트", "LLY": "일라이릴리",
    "JPM": "JP모건", "V": "비자", "XOM": "엑슨모빌", "JNJ": "존슨앤드존슨",
    "MA": "마스터카드", "COST": "코스트코", "HD": "홈디포", "PG": "프록터앤갬블",
    "NFLX": "넷플릭스", "BAC": "뱅크오브아메리카", "AMD": "AMD", "ABBV": "애브비",
    "KO": "코카콜라", "PEP": "펩시코", "MRK": "머크", "MCD": "맥도날드",
    "CSCO": "시스코", "INTC": "인텔", "PFE": "화이자", "TMO": "써모피셔",
    "UNH": "유나이티드헬스", "ORCL": "오라클", "CRM": "세일즈포스", "GS": "골드만삭스",
    "RTX": "RTX", "GE": "GE에어로스페이스", "CAT": "캐터필러", "BA": "보잉",
    "DIS": "월트디즈니", "NEE": "넥스트에라에너지", "LMT": "록히드마틴", "QCOM": "퀄컴",
    "CVX": "쉐브론", "WFC": "웰스파고", "T": "AT&T", "GILD": "길리어드",
    "MO": "알트리아", "VZ": "버라이즌", "PM": "필립모리스", "IBM": "IBM",
    "COIN": "코인베이스", "PLTR": "팔란티어", "HOOD": "로빈후드", "TSLA": "테슬라",
}

# ──────────────────────────────────────────────────────────────────
# 글로벌 캐시 (FastAPI startup 시 1회 로드)
# ──────────────────────────────────────────────────────────────────
_cache = {
    "adj_returns_df": None,       # 필터링된 adjusted returns DataFrame
    "cov_ticker_list": None,      # EWMA 공분산 행렬의 ticker 순서
    "cov_matrix": None,           # EWMA 공분산 numpy 행렬
    "ticker_to_cov_idx": None,    # ticker → 행렬 인덱스 매핑
    "variance_map": None,         # ticker → σ²_i (대각선 원소)
    "precalculated_pools": {},    # 🚀 성능 최적화: risk_level(1~4)별로 사전 계산된 (정렬 완료된) 종목 리스트 캐싱
}


def load_cache():
    """
    서버 시작 시 1회 호출하여 전체 데이터를 메모리에 캐싱합니다.
    CSV와 DB 공분산 행렬을 로드하고 필터링합니다.
    """
    print("\n[Cache] 실제 데이터 캐싱 시작...")

    # 1. Adjusted Returns CSV 로드 및 필터링
    if not _CSV_PATH.exists():
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {_CSV_PATH}")

    df = pd.read_csv(_CSV_PATH)
    df.columns = df.columns.str.strip()

    # 필터 조건: Gate_Passed=True AND E_Total_3M > 0 AND Adjusted_E_Total > 0
    filtered = df[
        (df["Gate_Passed"] == True) &
        (df["E_Total_3M"] > 0) &
        (df["Adjusted_E_Total"] > 0)
    ].copy()
    filtered["Ticker"] = filtered["Ticker"].str.strip().str.upper()
    _cache["adj_returns_df"] = filtered
    print(f"  CSV 필터링 완료: {len(df)}종목 → {len(filtered)}종목 (Gate_Passed + 양수 수익률)")

    # 2. EWMA 공분산 행렬 로드 (DB 우선)
    db = StockDBManager()
    db.connect()
    try:
        ticker_list, cov_matrix = db.fetch_ewma_covariance()
    finally:
        db.close()

    if len(ticker_list) > 0:
        # 일별 공분산 → 3개월(60거래일) 기준으로 스케일링
        cov_matrix_scaled = cov_matrix * HORIZON_DAYS
        _cache["cov_ticker_list"] = ticker_list
        _cache["cov_matrix"] = cov_matrix_scaled
        _cache["ticker_to_cov_idx"] = {t: i for i, t in enumerate(ticker_list)}

        # 개별 종목 3개월 분산 (스케일링된 대각선 원소) 추출
        variance_map = {}
        for i, t in enumerate(ticker_list):
            variance_map[t] = float(cov_matrix_scaled[i, i])
        _cache["variance_map"] = variance_map
        print(f"  EWMA 공분산 행렬 캐싱 완료: {len(ticker_list)}×{len(ticker_list)} (×{HORIZON_DAYS} 스케일링 적용)")
    else:
        # DB에 데이터가 없는 경우 CSV의 Vol_3M을 분산 대체값으로 사용
        print("  ⚠️ DB 공분산 없음 → CSV Vol_3M² 를 분산 대체값으로 사용")
        variance_map = {}
        for _, row in df.iterrows():
            t = str(row["Ticker"]).strip().upper()
            vol = float(row.get("Vol_3M", 0.1))
            variance_map[t] = vol ** 2
        _cache["variance_map"] = variance_map

    # 🚀 O(1) 조회를 위한 리스크 레벨(1~4)별 추천 풀(Pool) 사전 계산
    print("  🚀 [Performance] 리스크 레벨별 추천 풀 사전 계산 중...")
    _cache["precalculated_pools"] = {}
    
    # 1. 대상 종목 기본 정보 구성
    stocks_base = []
    for _, row in filtered.iterrows():
        t = str(row["Ticker"]).strip().upper()
        e_ri = float(row["Adjusted_E_Total"])
        vol_3m = float(row.get("Vol_3M", 0.0))
        sigma_sq = _cache["variance_map"].get(t, 0.01)
        sigma = sigma_sq ** 0.5
        stocks_base.append({
            "ticker": t,
            "e_return": e_ri,
            "vol_3m": vol_3m,
            "variance": sigma_sq,
            "sigma": sigma,
        })
        
    # sigma_ewma_60d (sigma) 기준 오름차순 정렬
    stocks_base.sort(key=lambda x: x["sigma"])
    total_count = len(stocks_base)
    
    # 위험도 점수 계산용 정규화 변수
    all_sigmas = [s["sigma"] for s in stocks_base]
    sigma_min = min(all_sigmas) if all_sigmas else 0
    sigma_max = max(all_sigmas) if all_sigmas else 1
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1
    import math

    # 레벨 1~4까지 반복 계산
    for lvl in range(1, 5):
        lam = LAMBDA_MAP.get(lvl, get_lambda_by_level(2))
        
        # 레벨별로 풀 복사 후 점수 계산
        level_stocks = []
        for s in stocks_base:
            score = s["e_return"] - 0.5 * lam * s["variance"]
            level_stocks.append({**s, "score": score})
            
        # 리스크 등급에 따른 Pool 컷오프
        if lvl == 4:
            pool_size = int(total_count * 0.25)
        elif lvl == 3:
            pool_size = int(total_count * 0.50)
        elif lvl == 2:
            pool_size = int(total_count * 0.75)
        else:
            pool_size = total_count
            
        # 최소 10개는 보장
        pool_size = max(pool_size, 10)
        
        pool = level_stocks[:pool_size]
        
        # Pool 내에서 목적함수(score) 기준으로 내림차순 정렬
        pool.sort(key=lambda x: x["score"], reverse=True)
        
        # 프론트 호환 규격 포맷팅
        formatted_pool = []
        for rank, st in enumerate(pool, 1):
            risk_score = int((st["sigma"] - sigma_min) / sigma_range * 100)
            risk_score = max(0, min(100, risk_score))
            
            log_ret = st["e_return"]
            simple_ret = round((math.exp(log_ret) - 1) * 100, 2)
            
            formatted_pool.append({
                "rank": rank,
                "ticker": st["ticker"],
                "name": TICKER_NAME_MAP.get(st["ticker"], st["ticker"]),
                "expectedReturn3M": round(st["e_return"] * 100, 2),
                "expectedReturn3M_simple": simple_ret,
                "sigma_ewma_60d": round(st["sigma"], 4),
                "risk_score": risk_score,
                "score": round(st["score"], 4),
            })
            
        _cache["precalculated_pools"][lvl] = formatted_pool

    print("[Cache] 데이터 캐싱 및 사전 분석 완료!\n")


def get_recommended_stocks(risk_level: int, top_n: int = 10) -> list[dict]:
    """
    Vol_3M 기준으로 오름차순 정렬하여 누적 사분위수(Cumulative Quartile) 풀을 형성하고,
    해당 풀 내에서 개별 목적함수(Utility Score)로 순위를 매겨 추천합니다.

    풀 크기 (누적):
      - Level 4 (거북이, 안전 지향): 상위 25% (Vol_3M 최저)
      - Level 3 (강아지, 신중함)  : 상위 50%
      - Level 2 (사자, 균형)     : 상위 75%
      - Level 1 (독수리, 공격적)  : 전체 100%

    목적함수 (스코어):
      score_i = E(R_i) - (1/2) * λ * σ_ewma(60일)²

    Args:
        risk_level: 1(독수리) ~ 4(거북이)
        top_n: 추천 종목 수 (기본 10)

    Returns:
        추천 종목 딕셔너리 리스트 (프론트엔드 호환 형식)
    """
    if not _cache.get("precalculated_pools"):
        raise RuntimeError("캐시가 초기화되지 않았습니다. 백엔드 서버를 재시작해주세요.")
        
    pool = _cache["precalculated_pools"].get(risk_level, [])
    
    # 요청한 개수만큼 잘라서 O(1) 반환
    return pool[:top_n]


def get_mvo_inputs(selected_tickers: list[str]) -> tuple:
    """
    사용자가 선택한 종목들에 대한 MVO 입력 데이터를 반환합니다.

    Returns:
        (mu, cov_sub, ticker_names)
        - mu: 선택 종목의 adjusted expected returns (numpy array)
        - cov_sub: 선택 종목의 EWMA 공분산 sub-matrix (numpy 2D array)
        - ticker_names: 정렬된 종목 리스트
    """
    df = _cache["adj_returns_df"]
    ticker_to_idx = _cache.get("ticker_to_cov_idx")
    cov_matrix = _cache.get("cov_matrix")

    # adjusted returns 딕셔너리 (전체 CSV 기준, 필터링되지 않은 종목도 포함 가능)
    all_returns = {}
    full_df = pd.read_csv(_CSV_PATH)
    full_df.columns = full_df.columns.str.strip()
    for _, row in full_df.iterrows():
        t = str(row["Ticker"]).strip().upper()
        all_returns[t] = float(row["Adjusted_E_Total"])

    # 선택된 종목 중 데이터가 존재하는 것만 추출
    valid_tickers = [t for t in selected_tickers if t in all_returns]

    # 기대수익률 벡터
    mu = np.array([all_returns[t] for t in valid_tickers])

    # 공분산 sub-matrix 추출
    if cov_matrix is not None and ticker_to_idx is not None:
        # DB 공분산 사용 가능 → sub-matrix 추출
        idx_list = []
        final_tickers = []
        for t in valid_tickers:
            if t in ticker_to_idx:
                idx_list.append(ticker_to_idx[t])
                final_tickers.append(t)

        idx_arr = np.array(idx_list)
        cov_sub = cov_matrix[np.ix_(idx_arr, idx_arr)]
        mu = np.array([all_returns[t] for t in final_tickers])
        return mu, cov_sub, final_tickers
    else:
        # DB 공분산 없음 → 분산 대각행렬로 대체
        variance_map = _cache.get("variance_map", {})
        n = len(valid_tickers)
        cov_sub = np.zeros((n, n))
        for i, t in enumerate(valid_tickers):
            cov_sub[i, i] = variance_map.get(t, 0.01)
        return mu, cov_sub, valid_tickers
