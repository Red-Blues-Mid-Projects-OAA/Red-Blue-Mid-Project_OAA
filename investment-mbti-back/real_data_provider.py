"""
실제 데이터 기반 종목 추천 모듈.

ADJUSTED_EXPECTED_RETURNS DB 스냅샷을 대상으로,
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
}


def _parse_bool(value) -> bool:
    """불리언/문자/숫자 입력을 Python bool로 정규화합니다."""
    # 한글 주석: DB에서 들어오는 다양한 bool 표현을 단일 규칙으로 처리합니다.
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "y", "yes"}


def _normalize_adjusted_returns_df(df: pd.DataFrame) -> pd.DataFrame:
    """adjusted_expected_returns 표준 컬럼 형태로 DataFrame을 정규화합니다."""
    # 한글 주석: 필수 컬럼 누락을 초기에 차단해 이후 로직 오류를 방지합니다.
    required_cols = [
        "Ticker",
        "Gate_Passed",
        "Return_Type",
        "Original_E_Ret",
        "E_Total_3M",
        "Realized_3M",
        "Gap",
        "Adj_Weight",
        "Vol_3M",
        "Adjustment",
        "Adjusted_E_Total",
        "Adjustment_Applied",
    ]

    normalized_df = df.copy()
    normalized_df.columns = [str(c).strip() for c in normalized_df.columns]
    missing_cols = [c for c in required_cols if c not in normalized_df.columns]
    if missing_cols:
        raise ValueError(f"adjusted returns 필수 컬럼 누락: {missing_cols}")

    # 한글 주석: 티커 키 정규화(공백 제거 + 대문자)로 DB 키를 일치시킵니다.
    normalized_df["Ticker"] = normalized_df["Ticker"].astype(str).str.strip().str.upper()
    normalized_df["Gate_Passed"] = normalized_df["Gate_Passed"].apply(_parse_bool)
    normalized_df["Adjustment_Applied"] = normalized_df["Adjustment_Applied"].apply(_parse_bool)
    return normalized_df


def _build_all_returns_map(df: pd.DataFrame) -> dict[str, float]:
    """전체 티커의 Adjusted_E_Total 맵을 생성합니다."""
    # 한글 주석: optimize 단계에서 매 호출마다 파일 재로드하지 않도록 캐시 맵을 만듭니다.
    all_returns = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        value = pd.to_numeric(row.get("Adjusted_E_Total"), errors="coerce")
        if ticker and pd.notna(value):
            all_returns[ticker] = float(value)
    return all_returns


def _build_variance_map_from_df(df: pd.DataFrame) -> dict[str, float]:
    """DB 공분산이 없을 때 사용할 대체 분산 맵(Vol_3M^2)을 생성합니다."""
    variance_map = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        vol = pd.to_numeric(row.get("Vol_3M"), errors="coerce")
        if pd.notna(vol):
            variance_map[ticker] = float(vol) ** 2
        else:
            variance_map[ticker] = 0.01
    return variance_map


def _filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """추천 후보 필터(Gate 통과 + 양수 수익률)를 적용합니다."""
    # 한글 주석: 기존 서비스 규칙을 그대로 유지합니다.
    gate_mask = df["Gate_Passed"].apply(_parse_bool)
    e_total = pd.to_numeric(df["E_Total_3M"], errors="coerce")
    adjusted = pd.to_numeric(df["Adjusted_E_Total"], errors="coerce")
    filtered_df = df[(gate_mask) & (e_total > 0) & (adjusted > 0)].copy()
    filtered_df["Ticker"] = filtered_df["Ticker"].astype(str).str.strip().str.upper()
    filtered_df["Gate_Passed"] = filtered_df["Gate_Passed"].apply(_parse_bool)
    filtered_df["Adjustment_Applied"] = filtered_df["Adjustment_Applied"].apply(_parse_bool)
    return filtered_df


def get_cache_sync_status() -> dict:
    """캐시 동기화 상태를 확인하기 위한 진단 정보를 반환합니다."""
    # 한글 주석: 문제가 발생했을 때 바로 원인 단계를 확인할 수 있게 합니다.
    filtered_df = _cache.get("adj_returns_df")
    all_returns = _cache.get("all_returns_map")
    return {
        "snapshot_source": _cache.get("snapshot_source"),
        "last_stage": _cache.get("last_stage"),
        "last_error": _cache.get("last_error"),
        "filtered_count": int(len(filtered_df)) if filtered_df is not None else 0,
        "all_returns_count": int(len(all_returns)) if all_returns is not None else 0,
    }


def reload_cache() -> dict:
    """문제 확인 후 수동으로 캐시 동기화를 다시 실행합니다."""
    # 한글 주석: 재가동 시 즉시 현재 상태를 반환합니다.
    load_cache()
    return get_cache_sync_status()


def load_cache():
    """
    서버 시작 시 1회 호출되어 전체 데이터를 메모리에 캐싱합니다.
    ADJUSTED_EXPECTED_RETURNS 스냅샷을 DB에서 직접 조회해 사용합니다.
    """
    print("\n[Cache] 실제 데이터 캐싱 시작...")

    # 한글 주석: 이전 캐시 상태를 먼저 초기화합니다.
    _cache["adj_returns_df"] = None
    _cache["cov_ticker_list"] = None
    _cache["cov_matrix"] = None
    _cache["ticker_to_cov_idx"] = None
    _cache["variance_map"] = None
    _cache["all_returns_map"] = None
    _cache["snapshot_source"] = None
    _cache["last_error"] = None
    _cache["last_stage"] = None

    source_df = pd.DataFrame()
    source_name = "DB"
    ticker_list = []
    cov_matrix = np.array([])

    # 한글 주석: DB 스냅샷과 EWMA 공분산을 직접 조회합니다.
    db = StockDBManager()
    db_connected = False
    try:
        db.connect()
        db_connected = True

        try:
            db_snapshot_df = db.fetch_adjusted_expected_returns()
            if not db_snapshot_df.empty:
                source_df = _normalize_adjusted_returns_df(db_snapshot_df)
                print(f"  [SYNC][DB_READ][OK] {len(source_df)}종목")
            else:
                _cache["last_stage"] = "db_snapshot_empty"
                _cache["last_error"] = "ADJUSTED_EXPECTED_RETURNS가 비어 있습니다."
                print("  [SYNC][DB_READ][ERROR] 스냅샷 테이블이 비어 있습니다.")
        except Exception as e:
            _cache["last_stage"] = "db_read"
            _cache["last_error"] = str(e)
            print(f"  [SYNC][DB_READ][ERROR] {e}")

        try:
            ticker_list, cov_matrix = db.fetch_ewma_covariance()
        except Exception as e:
            _cache["last_stage"] = "ewma_cov"
            _cache["last_error"] = str(e)
            print(f"  [SYNC][EWMA][ERROR] {e}")
    except Exception as e:
        _cache["last_stage"] = "db_connect"
        _cache["last_error"] = str(e)
        print(f"  [SYNC][DB_CONNECT][ERROR] {e}")
    finally:
        if db_connected:
            db.close()

    if source_df.empty:
        stage = _cache.get("last_stage")
        error = _cache.get("last_error")
        raise RuntimeError(
            f"adjusted expected returns DB 조회 실패 (stage={stage}, error={error})"
        )

    filtered = _filter_candidates(source_df)
    _cache["adj_returns_df"] = filtered
    _cache["all_returns_map"] = _build_all_returns_map(source_df)
    _cache["snapshot_source"] = source_name

    print(f"  [SYNC][SOURCE] {source_name} 사용, 후보 {len(source_df)} -> {len(filtered)}")

    if len(ticker_list) > 0 and cov_matrix.size > 0:
        # 한글 주석: 일간 공분산을 60거래일 기준으로 스케일링해 기존 계산식을 유지합니다.
        cov_matrix_scaled = cov_matrix * HORIZON_DAYS
        _cache["cov_ticker_list"] = ticker_list
        _cache["cov_matrix"] = cov_matrix_scaled
        _cache["ticker_to_cov_idx"] = {t: i for i, t in enumerate(ticker_list)}

        variance_map = {}
        for i, ticker in enumerate(ticker_list):
            variance_map[ticker] = float(cov_matrix_scaled[i, i])
        _cache["variance_map"] = variance_map
        print(f"  [SYNC][EWMA][OK] {len(ticker_list)}x{len(ticker_list)}")
    else:
        _cache["variance_map"] = _build_variance_map_from_df(source_df)
        print("  [SYNC][EWMA][FALLBACK] Vol_3M 기반 분산 사용")

    print("[Cache] 데이터 캐싱 완료!\n")


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
    lam = LAMBDA_MAP.get(risk_level, get_lambda_by_level(2))
    df = _cache["adj_returns_df"]
    variance_map = _cache["variance_map"]

    if df is None or variance_map is None:
        raise RuntimeError("캐시가 초기화되지 않았습니다. load_cache()를 먼저 호출하세요.")

    # 1. 대상 종목 데이터 구성
    stocks = []
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        e_ri = float(row["Adjusted_E_Total"])     # E(R_i)
        vol_3m = float(row.get("Vol_3M", 0.0))    # (참고용) 과거 3개월 변동성
        sigma_sq = variance_map.get(ticker, 0.01) # 목적함수 계산용 EWMA 분산(60일 스케일링)
        
        # 각 종목의 Utility Score 계산
        score = e_ri - 0.5 * lam * sigma_sq
        
        stocks.append({
            "ticker": ticker,
            "e_return": e_ri,
            "vol_3m": vol_3m,
            "variance": sigma_sq,
            "sigma": sigma_sq ** 0.5,
            "score": score,
        })

    # 2. sigma_ewma_60d (sigma) 기준 오름차순 정렬 (최신 변동성 낮은 순으로 컷오프 통일)
    stocks.sort(key=lambda x: x["sigma"])

    # 3. 리스크 등급에 따른 Pool 잘라내기
    total_count = len(stocks)
    if risk_level == 4:
        pool_size = max(int(total_count * 0.25), top_n)
    elif risk_level == 3:
        pool_size = max(int(total_count * 0.50), top_n)
    elif risk_level == 2:
        pool_size = max(int(total_count * 0.75), top_n)
    else: # risk_level == 1
        pool_size = total_count

    pool = stocks[:pool_size]

    # 4. Pool 내에서 목적함수(score) 기준으로 내림차순 정렬
    pool.sort(key=lambda x: x["score"], reverse=True)
    top_stocks = pool[:top_n]

    # 5. 프론트엔드 호환 형식으로 변환
    vol_labels = {4: "Very Low", 3: "Low", 2: "Medium", 1: "High"}
    color_map = {4: "#10b981", 3: "#3b82f6", 2: "#8b5cf6", 1: "#ef4444"}
    
    color = color_map.get(risk_level, "#8b5cf6")
    vol_label = vol_labels.get(risk_level, "Medium")

    result = []
    for rank, s in enumerate(top_stocks, 1):
        result.append({
            "rank": rank,
            "ticker": s["ticker"],
            "name": TICKER_NAME_MAP.get(s["ticker"], s["ticker"]),
            "volatility": vol_label,
            "color": color,
            "expectedReturn3M": round(s["e_return"] * 100, 2),  # 퍼센트 변환
            "sigma_ewma_60d": round(s["sigma"], 4),
            "score": round(s["score"], 4),
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

    # adjusted returns 딕셔너리 (DB 스냅샷 기준, 필터링되지 않은 종목도 포함 가능)
    all_returns = _cache.get("all_returns_map")
    if df is None or all_returns is None:
        raise RuntimeError("캐시가 초기화되지 않았습니다. load_cache()를 먼저 호출하세요.")

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
