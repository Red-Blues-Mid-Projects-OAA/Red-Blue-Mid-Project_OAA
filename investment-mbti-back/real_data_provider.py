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

# ──────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────
# adjusted_expected_returns.csv 경로
_CSV_PATH = _PROJECT_ROOT / "Classification" / "artifacts" / "multi_ticker" / "adjusted_expected_returns.csv"

# 예측 Horizon (거래일 기준, 일별 공분산 → 3개월 스케일링용)
HORIZON_DAYS = 60

# 리스크 타입별 λ 값 (risk_profile.py와 동기화)
LAMBDA_MAP = {
    4: 23.10,  # 거북이 (안전 지향)
    3: 16.57,  # 강아지 (신중한 탐험가)
    2: 10.04,  # 사자 (균형 잡힌)
    1: 3.50,   # 독수리 (공격적)
}

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

    print("[Cache] 데이터 캐싱 완료!\n")


def get_recommended_stocks(risk_level: int, top_n: int = 10) -> list[dict]:
    """
    리스크 등급(1~4)에 해당하는 λ를 이용하여
    개별 종목 유틸리티 스코어 기반 상위 종목을 추천합니다.

    score_i = E(R_i) - (1/2) * λ * σ²_i

    Args:
        risk_level: 1(독수리) ~ 4(거북이)
        top_n: 추천 종목 수 (기본 10)

    Returns:
        추천 종목 딕셔너리 리스트 (프론트엔드 호환 형식)
    """
    lam = LAMBDA_MAP.get(risk_level, 10.04)
    df = _cache["adj_returns_df"]
    variance_map = _cache["variance_map"]

    if df is None or variance_map is None:
        raise RuntimeError("캐시가 초기화되지 않았습니다. load_cache()를 먼저 호출하세요.")

    # 각 종목별 유틸리티 스코어 계산
    scores = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        e_ri = float(row["Adjusted_E_Total"])  # 기대수익률
        sigma_sq = variance_map.get(ticker, 0.01)  # 분산 (σ²_i)

        # score_i = E(R_i) - (1/2) * λ * σ²_i
        score = e_ri - 0.5 * lam * sigma_sq
        scores.append({
            "ticker": ticker,
            "e_return": e_ri,
            "variance": sigma_sq,
            "score": score,
        })

    # 스코어 내림차순 정렬 → 상위 top_n 선택
    scores.sort(key=lambda x: x["score"], reverse=True)
    top_stocks = scores[:top_n]

    # 변동성 등급 분류 (분산 기반 사분위수)
    all_vars = [s["variance"] for s in scores]
    q25 = np.percentile(all_vars, 25)
    q50 = np.percentile(all_vars, 50)
    q75 = np.percentile(all_vars, 75)

    # 리스크 타입별 색상 (현재 프론트엔드 호환)
    color_map = {4: "#10b981", 3: "#3b82f6", 2: "#8b5cf6", 1: "#ef4444"}
    color = color_map.get(risk_level, "#8b5cf6")

    # 프론트엔드 호환 형식으로 변환
    result = []
    for rank, s in enumerate(top_stocks, 1):
        var = s["variance"]
        if var <= q25:
            vol_label = "Very Low"
        elif var <= q50:
            vol_label = "Low"
        elif var <= q75:
            vol_label = "Medium"
        else:
            vol_label = "High"

        result.append({
            "rank": rank,
            "ticker": s["ticker"],
            "name": TICKER_NAME_MAP.get(s["ticker"], s["ticker"]),
            "volatility": vol_label,
            "color": color,
            "expectedReturn3M": round(s["e_return"] * 100, 2),  # 퍼센트 변환
            "score": round(s["score"], 6),
        })

    return result


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
