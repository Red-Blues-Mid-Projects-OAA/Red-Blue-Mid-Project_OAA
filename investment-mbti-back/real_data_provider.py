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

# 종목 한글 이름 매핑 — sp500_top300_kr.json 에서 동적 로드
import json as _json

_KR_NAME_JSON = _PROJECT_ROOT / "DB" / "sp500_top300_kr.json"

def _load_ticker_name_map() -> dict:
    """sp500_top300_kr.json 파일에서 종목 한글명 매핑을 로드합니다."""
    try:
        with open(_KR_NAME_JSON, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return {}

TICKER_NAME_MAP = _load_ticker_name_map()

# ──────────────────────────────────────────────────────────────────
# 글로벌 캐시 (FastAPI startup 시 1회 로드)
# ──────────────────────────────────────────────────────────────────
_cache = {
    "adj_returns_df": None,       # 필터링된 adjusted returns DataFrame
    "full_returns_df": None,      # 전체 (미필터) adjusted returns DataFrame
    "cov_ticker_list": None,      # EWMA 공분산 행렬의 ticker 순서
    "cov_matrix": None,           # EWMA 공분산 numpy 행렬
    "ticker_to_cov_idx": None,    # ticker → 행렬 인덱스 매핑
    "variance_map": None,         # ticker → σ²_i (대각선 원소)
    "risk_snapshot_stage": None,
    "risk_snapshot_error": None,
    "risk_snapshot_metrics_count": 0,
    "risk_snapshot_holdings_count": 0,
    "risk_snapshot_by_level": {},
}

RISK_LEVEL_LABEL_MAP = {
    4: "Very Low Risk",
    3: "Low Risk",
    2: "Medium Risk",
    1: "High Risk",
}


def normalize_risk_score(ewma_std_60d: float, min_ewma_std_60d: float, max_ewma_std_60d: float, eps: float = 1e-8) -> int:
    """
    개별 종목 EWMA 표준편차를 0~100 Min-Max 정규화 점수로 변환합니다.
    전체 variance_map의 EWMA 표준편차 분포(전 종목) 기준 min/max를 사용합니다.
    """
    score = int((ewma_std_60d - min_ewma_std_60d) / max(max_ewma_std_60d - min_ewma_std_60d, eps) * 100)
    return max(0, min(100, score))


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



def _build_risk_snapshot_by_level(df: pd.DataFrame) -> dict[int, dict]:
    """RISK_LEVEL_PORTFOLIO_SNAPSHOT 조회 결과를 레벨별 캐시 구조로 변환합니다."""
    if df is None or df.empty:
        return {}

    vol_labels = {4: "Very Low", 3: "Low", 2: "Medium", 1: "High"}
    color_map = {4: "#10b981", 3: "#3b82f6", 2: "#8b5cf6", 1: "#ef4444"}

    result: dict[int, dict] = {}
    ordered = df.sort_values(["Risk_Level", "Rank_No"], ascending=[False, True])

    for _, row in ordered.iterrows():
        try:
            risk_level = int(row["Risk_Level"])
            rank_no = int(row["Rank_No"])
        except Exception:
            continue

        bucket = result.setdefault(risk_level, {"summary": None, "holdings": []})

        if rank_no == 0:
            bucket["summary"] = {
                "risk_level": risk_level,
                "risk_label": str(row.get("Risk_Label") or RISK_LEVEL_LABEL_MAP.get(risk_level, f"Level {risk_level}")),
                "lambda_value": float(row.get("Lambda_Value") or 0.0),
                "expected_portfolio_return_3m": float(row.get("Portfolio_Expected_Return_3M") or 0.0),
                "portfolio_std_60d": float(row.get("Portfolio_Std_60D") or 0.0),
                "holdings_count": int(row.get("Holdings_Count") or 0),
                "updated_at": row.get("Updated_At"),
            }
            continue

        ticker = str(row.get("Ticker") or "").strip().upper()
        if not ticker:
            continue

        holding = {
            "rank": rank_no,
            "ticker": ticker,
            "name": str(row.get("Stock_Name") or TICKER_NAME_MAP.get(ticker, ticker)),
            "volatility": vol_labels.get(risk_level, "Medium"),
            "color": color_map.get(risk_level, "#8b5cf6"),
            "expectedReturn3M": round(float(row.get("Stock_Expected_Return_3M") or 0.0), 2),
            "sigma_ewma_60d": round(float(row.get("Stock_Sigma_EWMA_60D") or 0.0), 4),
            "weight": round(float(row.get("Weight_Pct") or 0.0), 2),
        }
        bucket["holdings"].append(holding)

    for risk_level in result:
        result[risk_level]["holdings"] = sorted(
            result[risk_level]["holdings"],
            key=lambda item: int(item.get("rank", 9999)),
        )

    return result

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
        "risk_snapshot_stage": _cache.get("risk_snapshot_stage"),
        "risk_snapshot_error": _cache.get("risk_snapshot_error"),
        "risk_snapshot_metrics_count": int(_cache.get("risk_snapshot_metrics_count") or 0),
        "risk_snapshot_holdings_count": int(_cache.get("risk_snapshot_holdings_count") or 0),
        "risk_snapshot_levels": sorted(list((_cache.get("risk_snapshot_by_level") or {}).keys()), reverse=True),
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
    _cache["full_returns_df"] = None
    _cache["cov_ticker_list"] = None
    _cache["cov_matrix"] = None
    _cache["ticker_to_cov_idx"] = None
    _cache["variance_map"] = None
    _cache["all_returns_map"] = None
    _cache["snapshot_source"] = None
    _cache["last_error"] = None
    _cache["last_stage"] = None
    _cache["risk_snapshot_stage"] = None
    _cache["risk_snapshot_error"] = None
    _cache["risk_snapshot_metrics_count"] = 0
    _cache["risk_snapshot_holdings_count"] = 0
    _cache["risk_snapshot_by_level"] = {}

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
    _cache["full_returns_df"] = source_df
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

    # risk snapshot 계산/적재는 DB 전용 모듈로 위임 후, 조회 캐시에 반영
    try:
        from DB import sync_risk_level_portfolio_snapshot

        snapshot_status = sync_risk_level_portfolio_snapshot(top_n=10, raise_on_error=False)
        _cache["risk_snapshot_stage"] = snapshot_status.get("stage")
        _cache["risk_snapshot_error"] = snapshot_status.get("error")

        snapshot_db = StockDBManager()
        snapshot_connected = False
        try:
            snapshot_db.connect()
            snapshot_connected = True
            snapshot_df = snapshot_db.fetch_risk_level_portfolio_snapshot()
        finally:
            if snapshot_connected:
                snapshot_db.close()

        risk_snapshot_by_level = _build_risk_snapshot_by_level(snapshot_df)
        _cache["risk_snapshot_by_level"] = risk_snapshot_by_level

        _cache["risk_snapshot_metrics_count"] = int(
            sum(1 for data in risk_snapshot_by_level.values() if data.get("summary") is not None)
        )
        _cache["risk_snapshot_holdings_count"] = int(
            sum(len(data.get("holdings", [])) for data in risk_snapshot_by_level.values())
        )

        if not risk_snapshot_by_level:
            _cache["risk_snapshot_stage"] = "empty"
            _cache["risk_snapshot_error"] = "RISK_LEVEL_PORTFOLIO_SNAPSHOT가 비어 있습니다."
    except Exception as e:
        _cache["risk_snapshot_stage"] = "error"
        _cache["risk_snapshot_error"] = str(e)
        _cache["risk_snapshot_metrics_count"] = 0
        _cache["risk_snapshot_holdings_count"] = 0
        _cache["risk_snapshot_by_level"] = {}
        print(f"  [SYNC][RISK_SNAPSHOT][ERROR] {e}")

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
      score_i = E(R_i) - (1/2) * λ * Var_i(EWMA 60일 분산)

    Args:
        risk_level: 1(독수리) ~ 4(거북이)
        top_n: 추천 종목 수 (기본 10)

    Returns:
        추천 종목 딕셔너리 리스트 (프론트엔드 호환 형식)
    """
    risk_level = int(risk_level)

    # 0. DB 스냅샷 우선 사용 (웹 응답 DB-only)
    risk_snapshot_by_level = _cache.get("risk_snapshot_by_level") or {}
    if risk_level in risk_snapshot_by_level:
        holdings = list(risk_snapshot_by_level[risk_level].get("holdings", []))
        if top_n is not None and int(top_n) > 0:
            holdings = holdings[: int(top_n)]

        # 전체 variance_map 기준 sigma min/max 계산 (비교 가능성 유지)
        variance_map = _cache.get("variance_map") or {}
        all_sigmas = [v ** 0.5 for v in variance_map.values()] if variance_map else []
        sigma_min = min(all_sigmas) if all_sigmas else 0
        sigma_max = max(all_sigmas) if all_sigmas else 1

        for holding in holdings:
            # 한글 종목명 매핑 (sp500_top300_kr.json 기준)
            ticker = holding.get("ticker", "")
            if not holding.get("name") or holding.get("name") == ticker:
                holding["name"] = TICKER_NAME_MAP.get(ticker, ticker)

            try:
                ewma_std_60d = float(holding.get("sigma_ewma_60d", 0.0) or 0.0)
            except (TypeError, ValueError):
                ewma_std_60d = 0.0

            holding["risk_score"] = normalize_risk_score(ewma_std_60d, sigma_min, sigma_max)

            try:
                expected_log_pct = float(holding.get("expectedReturn3M", 0.0) or 0.0)
            except (TypeError, ValueError):
                expected_log_pct = 0.0
            holding.setdefault(
                "expectedReturn3M_simple",
                round(float((np.exp(expected_log_pct / 100.0) - 1.0) * 100.0), 2),
            )
            holding.setdefault("historical_returns", {"1M": 0, "3M": 0, "6M": 0, "12M": 0})

        return holdings

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
        ewma_variance_60d = variance_map.get(ticker, 0.01)  # 목적함수 계산용 EWMA 분산(60일 스케일링)
        
        # 각 종목의 Utility Score 계산
        # score_i = E(R_i) - 0.5 * λ * Var_i(EWMA 60일 분산)
        # 여기서 패널티 항은 표준편차가 아닌 분산입니다.
        score = e_ri - 0.5 * lam * ewma_variance_60d
        
        stocks.append({
            "ticker": ticker,
            "e_return": e_ri,
            "vol_3m": vol_3m,
            "variance": ewma_variance_60d,
            "sigma": ewma_variance_60d ** 0.5,
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
    # MinMaxScaler 기반 개별 종목 위험도 점수 (0~100) 산출
    # 전체 종목의 sigma_ewma_60d 분포를 기준으로 정규화
    all_sigmas = [v ** 0.5 for v in variance_map.values()]
    sigma_min = min(all_sigmas) if all_sigmas else 0
    sigma_max = max(all_sigmas) if all_sigmas else 1

    import math
    result = []
    for rank, s in enumerate(top_stocks, 1):
        risk_score = normalize_risk_score(s["sigma"], sigma_min, sigma_max)

        # 로그수익률 → 단순수익률 변환: simple = (e^r - 1) * 100
        log_ret = s["e_return"]
        simple_ret = round((math.exp(log_ret) - 1) * 100, 2)

        result.append({
            "rank": rank,
            "ticker": s["ticker"],
            "name": TICKER_NAME_MAP.get(s["ticker"], s["ticker"]),
            "expectedReturn3M": round(s["e_return"] * 100, 2),        # 로그수익률 (백엔드 참조용)
            "expectedReturn3M_simple": simple_ret,                     # 단순수익률 (프론트엔드 표시용)
            "sigma_ewma_60d": round(s["sigma"], 4),
            "risk_score": risk_score,                                  # 0~100 위험도 점수
            "score": round(s["score"], 4),
        })

    return result


def get_risk_level_portfolio_summary(risk_level: int) -> dict | None:
    """리스크 레벨별 포트폴리오 요약(스냅샷 rank_no=0 행)을 반환합니다."""
    risk_snapshot_by_level = _cache.get("risk_snapshot_by_level") or {}
    data = risk_snapshot_by_level.get(int(risk_level))
    if not data:
        return None
    return data.get("summary")


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


# ──────────────────────────────────────────────────────────────────
# 전체 300개 종목 데이터 및 포트폴리오 스코어 산출
# ──────────────────────────────────────────────────────────────────

# sp500_top300.json 에서 시가총액 순위 로드
_SP500_JSON = _PROJECT_ROOT / "DB" / "sp500_top300.json"

def _load_market_cap_rank() -> dict[str, int]:
    """sp500_top300.json 순서 기반 시가총액 순위(1-based)를 반환합니다."""
    try:
        with open(_SP500_JSON, encoding="utf-8") as f:
            tickers = _json.load(f)
        return {str(t).strip().upper(): i + 1 for i, t in enumerate(tickers)}
    except Exception:
        return {}

MARKET_CAP_RANK = _load_market_cap_rank()


def get_all_300_stocks() -> list[dict]:
    """
    전체 300개 종목의 return_rank, risk_rank, market_cap_rank 를 반환합니다.
    - return_rank: Realized_3M (최근 3개월 실현수익률) 내림차순 순위 (1위 = 수익률 최고)
    - risk_rank: EWMA std(√60일) 오름차순 순위 (1위 = 위험도 최저, 가장 안전)
    - market_cap_rank: sp500_top300.json 순서(1~300)
    """
    full_df = _cache.get("full_returns_df")
    variance_map = _cache.get("variance_map") or {}

    if full_df is None:
        raise RuntimeError("캐시가 초기화되지 않았습니다. load_cache()를 먼저 호출하세요.")

    # 1. Realized_3M 수집 (수익률 순위 계산용)
    realized_3m_values = {}
    for _, row in full_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        val_3m = pd.to_numeric(row.get("Realized_3M"), errors="coerce")
        if pd.notna(val_3m):
            realized_3m_values[ticker] = float(val_3m)

    # 2. EWMA sigma 수집
    sigma_values = {}
    for ticker, var_val in variance_map.items():
        sigma_values[ticker] = var_val ** 0.5

    # 3. sp500_top300 종목 목록
    target_tickers = list(MARKET_CAP_RANK.keys())

    # 4. 1M/3M/6M/12M 누적 수익률 (DashboardResult와 동일 로직 사용)
    from chart_data_provider import get_historical_returns
    historical_returns = get_historical_returns(target_tickers)

    # 5. Return 순위: 수익률 내림차순 (1위 = 최고 수익률)
    return_sorted = sorted(
        [(t, realized_3m_values.get(t, float('-inf'))) for t in target_tickers],
        key=lambda x: x[1], reverse=True
    )
    return_rank_map = {t: rank for rank, (t, _) in enumerate(return_sorted, 1)}

    # 6. Risk 순위: 변동성 오름차순 (1위 = 가장 낮은 위험)
    risk_sorted = sorted(
        [(t, sigma_values.get(t, float('inf'))) for t in target_tickers],
        key=lambda x: x[1]
    )
    risk_rank_map = {t: rank for rank, (t, _) in enumerate(risk_sorted, 1)}

    # 7. 결과 생성
    result = []
    for ticker in sorted(target_tickers, key=lambda t: MARKET_CAP_RANK.get(t, 999)):
        r_data = historical_returns.get(ticker, {"1M": 0, "3M": 0, "6M": 0, "12M": 0})
        result.append({
            "ticker": ticker,
            "name": TICKER_NAME_MAP.get(ticker, ticker),
            "return_rank": return_rank_map.get(ticker, 300),
            "risk_rank": risk_rank_map.get(ticker, 300),
            "market_cap_rank": MARKET_CAP_RANK.get(ticker, 300),
            "returns": r_data
        })

    return result


def calculate_portfolio_scores(selected_tickers: list[str]) -> dict:
    """
    선택 종목의 다각화 비율(Diversification Ratio)을 반영하여 포트폴리오 스코어를 산출합니다.

    - return_pct: 선택 종목들의 개별 수익률 순위의 평균을 0~100 점수로 환산 (100점이 가장 높음)
    - risk_pct_naive: 선택 종목들의 개별 리스크 순위의 평균을 0~100 점수로 환산 (100점이 가장 위험함)
    - risk_pct: risk_pct_naive * (실제 포트폴리오 변동성 / 평균 개별 변동성) [다각화 비율 적용]
    - diversification_benefit: risk_pct_naive - risk_pct

    Returns:
        {"return_pct": int, "risk_pct": int, "risk_pct_naive": int, "diversification_benefit": int}
    """
    tickers = [t.strip().upper() for t in selected_tickers]
    n = len(tickers)
    if n == 0:
        return {"return_pct": 0, "risk_pct": 0, "risk_pct_naive": 0, "diversification_benefit": 0}

    # 1) 개별 종목들의 rank 정보 획득
    all_stocks = get_all_300_stocks()
    stock_map = {s["ticker"]: s for s in all_stocks}

    sum_return_rank = 0
    sum_risk_rank = 0
    valid_count = 0

    for t in tickers:
        if t in stock_map:
            sum_return_rank += stock_map[t]["return_rank"]
            sum_risk_rank += stock_map[t]["risk_rank"]
            valid_count += 1

    if valid_count == 0:
        return {"return_pct": 50, "risk_pct": 50, "risk_pct_naive": 50, "diversification_benefit": 0}

    avg_return_rank = sum_return_rank / valid_count
    avg_risk_rank = sum_risk_rank / valid_count

    # 순위 기반 점수 환산 처리
    # Return: 1위(가장 높음) -> 100점, 300위 -> 20점 (기본 점수 20점 부여)
    return_pct = max(0, min(100, int(100 - ((avg_return_rank - 1) / 299 * 80))))
    
    # Risk Naive: 1위(가장 안전함) -> 20점 (위험이 0인 주식은 없으므로 기본 위험 20점 부여), 300위 -> 100점
    risk_pct_naive = max(0, min(100, int(20 + ((avg_risk_rank - 1) / 299 * 80))))

    # 2) 실제 변동성 및 개별 변동성 평균 계산 (Diversification Ratio 용도)
    variance_map = _cache.get("variance_map") or {}
    cov_matrix = _cache.get("cov_matrix")
    ticker_to_idx = _cache.get("ticker_to_cov_idx") or {}

    sigmas = []
    for t in tickers:
        var_val = variance_map.get(t)
        if var_val is not None:
            sigmas.append(var_val ** 0.5)

    avg_sigma = sum(sigmas) / max(len(sigmas), 1) if sigmas else 0

    actual_sigma = 0
    if cov_matrix is not None and ticker_to_idx:
        valid_idx = []
        for t in tickers:
            if t in ticker_to_idx:
                valid_idx.append(ticker_to_idx[t])
        
        if valid_idx:
            idx_arr = np.array(valid_idx)
            cov_sub = cov_matrix[np.ix_(idx_arr, idx_arr)]
            n_valid = len(valid_idx)
            w_vec = np.ones(n_valid) / n_valid
            port_var = float(w_vec @ cov_sub @ w_vec)
            actual_sigma = max(port_var, 0) ** 0.5

    # 3) 다각화 비율(DR) = 실제 포트폴리오 위험 / 개별 위험의 단순 평균
    if avg_sigma > 0 and actual_sigma > 0:
        dr = actual_sigma / avg_sigma
        # 안전장치: 이론상 동일비중 포트폴리오의 DR은 0~1 사이 (음의 상관관계 포함)
        dr = max(0.0, min(1.0, dr))
        risk_pct = int(risk_pct_naive * dr)
    else:
        risk_pct = risk_pct_naive

    diversification_benefit = max(0, risk_pct_naive - risk_pct)

    return {
        "return_pct": return_pct,
        "risk_pct": risk_pct,
        "risk_pct_naive": risk_pct_naive,
        "diversification_benefit": diversification_benefit,
    }

