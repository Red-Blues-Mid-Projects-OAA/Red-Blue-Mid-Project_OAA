"""
이 파일은 실제 종목 데이터와 포트폴리오 계산용 데이터를 읽어 와 API가 바로 쓰기 좋은 형태로 가공합니다.
이 모듈은 DB 스냅샷 또는 데모 스냅샷을 메모리 캐시에 올린 뒤, 추천 종목/전체 종목/최적화 입력처럼 FastAPI 응답에서 바로 소비할 구조로 다시 정리하는 역할을 맡습니다.
상단에서는 캐시와 공통 매핑을 준비하고, 중간 함수들은 스냅샷 정규화와 fallback 계산을 담당하며, 하단 공개 함수들은 API 엔드포인트가 직접 호출하는 조회 인터페이스를 제공합니다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 백엔드 하위 폴더에서 실행해도 `DB`, `risk_profile` 같은 상위 패키지를 바로 import할 수 있도록
# 프로젝트 루트 경로를 먼저 sys.path에 주입합니다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from DB import StockDBManager
from demo_snapshot import load_demo_snapshot, should_force_demo_snapshot
from risk_profile import get_lambda_by_level

# ──────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────
# 예측 Horizon (거래일 기준, 일별 공분산 → 3개월 스케일링용)
HORIZON_DAYS = 60

# 리스크 타입별 λ 값 (risk_profile.py 로직과 동기화된 동적 매핑)
LAMBDA_MAP = {i: get_lambda_by_level(i) for i in range(1, 5)}

# 티커를 프론트엔드 친화적인 한글 이름으로 보여 주기 위한 정적 매핑 파일 경로입니다.
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
    "risk_snapshot_stage": None,  # 위험단계 스냅샷 동기화가 어디까지 진행됐는지 나타내는 상태 문자열
    "risk_snapshot_error": None,  # 위험단계 스냅샷 동기화에서 마지막으로 발생한 오류 메시지
    "risk_snapshot_metrics_count": 0,  # rank_no=0 요약 행으로 계산한 레벨별 메트릭 개수
    "risk_snapshot_holdings_count": 0,  # 스냅샷에서 읽어 온 전체 보유 종목 행 개수
    "risk_snapshot_by_level": {},  # risk_level -> {"summary": ..., "holdings": [...]} 캐시
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
    # 분모가 0에 가까운 구간에서도 폭주하지 않도록 eps로 최소 범위를 보장합니다.
    denominator = max(max_ewma_std_60d - min_ewma_std_60d, eps)
    # 선형 정규화 결과를 0~100 정수 점수로 잘라 프론트엔드 막대/배지에 재사용합니다.
    score = int((ewma_std_60d - min_ewma_std_60d) / denominator * 100)
    return max(0, min(100, score))

def _parse_bool(value) -> bool:
    """불리언/문자/숫자 입력을 Python bool로 정규화합니다."""
    # DB에서 들어오는 다양한 bool 표현을 단일 규칙으로 처리합니다.
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    # 공백 제거와 소문자 변환을 먼저 해 두면 Oracle 숫자/문자 혼합 입력도 같은 규칙으로 비교할 수 있습니다.
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "y", "yes"}

def _normalize_adjusted_returns_df(df: pd.DataFrame) -> pd.DataFrame:
    """adjusted_expected_returns 표준 컬럼 형태로 DataFrame을 정규화합니다."""
    # 필수 컬럼 누락을 초기에 차단해 이후 로직 오류를 방지합니다.
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

    # 원본 DataFrame을 그대로 보존하기 위해 복사본에서만 컬럼명과 타입을 정리합니다.
    normalized_df = df.copy()
    # DB/JSON 출처마다 공백이 섞일 수 있으므로 컬럼명을 문자열 + trim 기준으로 맞춥니다.
    normalized_df.columns = [str(c).strip() for c in normalized_df.columns]
    # 필수 컬럼이 비어 있으면 이후 계산식이 조용히 깨지므로 초기에 강하게 실패시킵니다.
    missing_cols = [c for c in required_cols if c not in normalized_df.columns]
    if missing_cols:
        raise ValueError(f"adjusted returns 필수 컬럼 누락: {missing_cols}")

    # 티커 키 정규화(공백 제거 + 대문자)로 DB 키를 일치시킵니다.
    normalized_df["Ticker"] = normalized_df["Ticker"].astype(str).str.strip().str.upper()
    normalized_df["Gate_Passed"] = normalized_df["Gate_Passed"].apply(_parse_bool)
    normalized_df["Adjustment_Applied"] = normalized_df["Adjustment_Applied"].apply(_parse_bool)
    return normalized_df

def _build_all_returns_map(df: pd.DataFrame) -> dict[str, float]:
    """전체 티커의 Adjusted_E_Total 맵을 생성합니다."""
    # optimize 단계에서 매 호출마다 파일 재로드하지 않도록 캐시 맵을 만듭니다.
    all_returns = {}
    for _, row in df.iterrows():
        # 티커는 캐시 키와 공분산 인덱스 키를 맞추기 위해 항상 대문자로 통일합니다.
        ticker = str(row["Ticker"]).strip().upper()
        # 기대수익률은 숫자 변환이 실패하면 버리고, 유효한 값만 맵에 적재합니다.
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
            # Vol_3M은 표준편차이므로 목적함수에서 쓰는 분산으로 맞추기 위해 제곱합니다.
            variance_map[ticker] = float(vol) ** 2
        else:
            # 최소한의 계산 지속을 위한 보수적 fallback 분산입니다.
            variance_map[ticker] = 0.01
    return variance_map

def _build_risk_snapshot_by_level(df: pd.DataFrame) -> dict[int, dict]:
    """RISK_LEVEL_PORTFOLIO_SNAPSHOT 조회 결과를 레벨별 캐시 구조로 변환합니다."""
    if df is None or df.empty:
        return {}

    # 프론트엔드 카드에 바로 넣을 라벨과 색상을 레벨별로 미리 고정합니다.
    vol_labels = {4: "Very Low", 3: "Low", 2: "Medium", 1: "High"}
    color_map = {4: "#10b981", 3: "#3b82f6", 2: "#8b5cf6", 1: "#ef4444"}

    # 최종 결과는 level -> summary/holdings 구조의 중첩 딕셔너리입니다.
    result: dict[int, dict] = {}
    # 요약 행(rank 0)과 보유 종목 행(rank > 0)을 같은 순서 규칙으로 읽기 위해 정렬합니다.
    ordered = df.sort_values(["Risk_Level", "Rank_No"], ascending=[False, True])

    for _, row in ordered.iterrows():
        try:
            risk_level = int(row["Risk_Level"])
            rank_no = int(row["Rank_No"])
        except Exception:
            continue

        # 같은 risk_level 안에서 summary 1개와 holdings 여러 개를 누적할 버킷입니다.
        bucket = result.setdefault(risk_level, {"summary": None, "holdings": []})

        if rank_no == 0:
            # rank 0은 레벨 전체 요약 행이므로 포트폴리오 메트릭만 저장하고 다음 행으로 넘어갑니다.
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

        # rank > 0 행은 실제 추천 종목 카드에 들어갈 보유 종목 정보입니다.
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
        # DB 조회 순서와 상관없이 rank 오름차순으로 카드가 고정되도록 다시 정렬합니다.
        result[risk_level]["holdings"] = sorted(
            result[risk_level]["holdings"],
            key=lambda item: int(item.get("rank", 9999)),
        )

    return result

def _snapshot_records_to_dataframe(records, datetime_columns: list[str] | None = None) -> pd.DataFrame:
    """스냅샷 레코드 리스트를 DataFrame으로 바꾸고 날짜 컬럼을 datetime으로 정규화합니다."""
    df = pd.DataFrame(records or [])
    for column in datetime_columns or []:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df

def _load_cache_from_demo_snapshot() -> bool:
    """데모 스냅샷 JSON을 읽어 실제 DB 캐시와 같은 구조로 메모리에 적재합니다."""
    try:
        snapshot = load_demo_snapshot()
    except FileNotFoundError as e:
        _cache["last_stage"] = "snapshot_missing"
        _cache["last_error"] = str(e)
        print(f"  [SYNC][SNAPSHOT][MISS] {e}")
        return False
    except Exception as e:
        _cache["last_stage"] = "snapshot_load"
        _cache["last_error"] = str(e)
        print(f"  [SYNC][SNAPSHOT][ERROR] {e}")
        return False

    # adjusted_expected_returns가 추천/최적화 계산의 주원천이므로 가장 먼저 DataFrame으로 정규화합니다.
    source_df = _snapshot_records_to_dataframe(
        snapshot.get("adjusted_expected_returns"),
        datetime_columns=["Updated_At"],
    )
    if source_df.empty:
        _cache["last_stage"] = "snapshot_empty"
        _cache["last_error"] = "adjusted_expected_returns is empty in demo snapshot"
        print("  [SYNC][SNAPSHOT][ERROR] adjusted_expected_returns is empty")
        return False

    # source_df는 원본 전체 종목 풀, filtered는 추천 후보로 줄인 하위 집합입니다.
    source_df = _normalize_adjusted_returns_df(source_df)
    filtered = _filter_candidates(source_df)

    # 공분산 payload는 티커 순서와 행렬이 반드시 함께 움직여야 하므로 둘을 동시에 검증합니다.
    covariance_payload = snapshot.get("ewma_covariance") or {}
    covariance_tickers = [
        str(ticker).strip().upper()
        for ticker in covariance_payload.get("ticker_list", [])
        if str(ticker).strip()
    ]
    covariance_matrix_raw = covariance_payload.get("matrix") or []
    covariance_matrix = np.array(covariance_matrix_raw, dtype=float) if covariance_matrix_raw else np.array([])

    # 위험 단계별 포트폴리오 스냅샷은 요약/보유 종목 구조로 다시 묶어 캐시에 저장합니다.
    risk_snapshot_df = _snapshot_records_to_dataframe(
        snapshot.get("risk_level_portfolio_snapshot"),
        datetime_columns=["Updated_At"],
    )
    risk_snapshot_by_level = _build_risk_snapshot_by_level(risk_snapshot_df)

    # 필터링 결과와 원본 전체 결과를 동시에 보관해 추천/순위 계산에서 각각 재사용합니다.
    _cache["adj_returns_df"] = filtered
    _cache["full_returns_df"] = source_df
    if covariance_tickers and covariance_matrix.size > 0 and covariance_matrix.shape[0] == covariance_matrix.shape[1]:
        # 일간 공분산을 60거래일 기준으로 확대해 나머지 계산식과 스케일을 일치시킵니다.
        covariance_matrix_scaled = covariance_matrix * HORIZON_DAYS
        _cache["cov_ticker_list"] = covariance_tickers
        _cache["cov_matrix"] = covariance_matrix_scaled
        _cache["ticker_to_cov_idx"] = {ticker: idx for idx, ticker in enumerate(covariance_tickers)}
        _cache["variance_map"] = {
            ticker: float(covariance_matrix_scaled[idx, idx])
            for idx, ticker in enumerate(covariance_tickers)
        }
    else:
        _cache["cov_ticker_list"] = None
        _cache["cov_matrix"] = None
        # 행렬이 비어 있으면 Vol_3M 기반 분산 맵으로 최소한의 최적화 입력을 유지합니다.
        _cache["ticker_to_cov_idx"] = None
        _cache["variance_map"] = _build_variance_map_from_df(source_df)
    _cache["all_returns_map"] = _build_all_returns_map(source_df)
    _cache["snapshot_source"] = "demo_snapshot_json"
    _cache["last_stage"] = "snapshot"
    _cache["last_error"] = None
    _cache["risk_snapshot_by_level"] = risk_snapshot_by_level
    _cache["risk_snapshot_metrics_count"] = int(
        sum(1 for data in risk_snapshot_by_level.values() if data.get("summary") is not None)
    )
    _cache["risk_snapshot_holdings_count"] = int(
        sum(len(data.get("holdings", [])) for data in risk_snapshot_by_level.values())
    )
    _cache["risk_snapshot_stage"] = "ok" if risk_snapshot_by_level else "empty"
    _cache["risk_snapshot_error"] = None if risk_snapshot_by_level else "snapshot risk data is empty"

    print(
        "  [SYNC][SNAPSHOT][OK] "
        f"source={len(source_df)}, filtered={len(filtered)}, "
        f"risk_levels={sorted(risk_snapshot_by_level.keys(), reverse=True)}"
    )
    return True

def _filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """추천 후보 필터(Gate 통과 + 양수 수익률)를 적용합니다."""
    # 기존 서비스 규칙을 그대로 유지합니다.
    # 게이트 통과 여부는 문자열/숫자 혼합일 수 있으므로 불리언 정규화를 다시 적용합니다.
    gate_mask = df["Gate_Passed"].apply(_parse_bool)
    # 원본 기대수익률과 조정 기대수익률이 모두 양수인 종목만 추천 풀에 남깁니다.
    e_total = pd.to_numeric(df["E_Total_3M"], errors="coerce")
    adjusted = pd.to_numeric(df["Adjusted_E_Total"], errors="coerce")
    filtered_df = df[(gate_mask) & (e_total > 0) & (adjusted > 0)].copy()
    filtered_df["Ticker"] = filtered_df["Ticker"].astype(str).str.strip().str.upper()
    filtered_df["Gate_Passed"] = filtered_df["Gate_Passed"].apply(_parse_bool)
    filtered_df["Adjustment_Applied"] = filtered_df["Adjustment_Applied"].apply(_parse_bool)
    return filtered_df

def get_cache_sync_status() -> dict:
    """캐시 동기화 상태를 확인하기 위한 진단 정보를 반환합니다."""
    # 문제가 발생했을 때 바로 원인 단계를 확인할 수 있게 합니다.
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
    # 재가동 시 즉시 현재 상태를 반환합니다.
    load_cache()
    return get_cache_sync_status()

def load_cache():
    """
    서버 시작 시 1회 호출되어 전체 데이터를 메모리에 캐싱합니다.
    ADJUSTED_EXPECTED_RETURNS 스냅샷을 DB에서 직접 조회해 사용합니다.
    """
    print("\n[Cache] 실제 데이터 캐싱 시작...")

    # 이전 캐시 상태를 먼저 초기화합니다.
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

    if should_force_demo_snapshot():
        if _load_cache_from_demo_snapshot():
            print("[Cache] forced demo snapshot mode enabled.\n")
            return
        raise RuntimeError("forced demo snapshot mode is enabled, but demo snapshot could not be loaded")

    # source_df는 실제 계산에 쓰이는 전체 스냅샷 원본, source_name은 어떤 출처를 탔는지 남기는 라벨입니다.
    source_df = pd.DataFrame()
    source_name = "DB"
    # 공분산 관련 값은 DB 조회가 성공하면 실제 티커 목록/행렬로 교체됩니다.
    ticker_list = []
    cov_matrix = np.array([])

    # DB 스냅샷과 EWMA 공분산을 직접 조회합니다.
    db = StockDBManager()
    db_connected = False
    try:
        db.connect(ensure_tables=False)
        db_connected = True

        try:
            # 조정 기대수익률 스냅샷은 추천/최적화/전체 종목 조회의 공통 입력입니다.
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
            # 공분산은 MVO와 포트폴리오 위험 계산에서만 쓰므로 별도 단계로 나눠 오류를 기록합니다.
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
        if _load_cache_from_demo_snapshot():
            print("[Cache] demo snapshot fallback loaded.\n")
            return
        raise RuntimeError(
            f"adjusted expected returns DB load failed (stage={stage}, error={error})"
        )

    # 전체 풀과 추천 후보 풀을 분리 저장해, 추천과 전체 300종목 화면이 같은 원천을 공유하도록 맞춥니다.
    filtered = _filter_candidates(source_df)
    _cache["adj_returns_df"] = filtered
    _cache["full_returns_df"] = source_df
    _cache["all_returns_map"] = _build_all_returns_map(source_df)
    _cache["snapshot_source"] = source_name

    print(f"  [SYNC][SOURCE] {source_name} 사용, 후보 {len(source_df)} -> {len(filtered)}")

    if len(ticker_list) > 0 and cov_matrix.size > 0:
        # 일간 공분산을 60거래일 기준으로 스케일링해 기존 계산식을 유지합니다.
        cov_matrix_scaled = cov_matrix * HORIZON_DAYS
        _cache["cov_ticker_list"] = ticker_list
        _cache["cov_matrix"] = cov_matrix_scaled
        _cache["ticker_to_cov_idx"] = {t: i for i, t in enumerate(ticker_list)}

        # 대각 원소만 모아 ticker -> 분산 맵으로 캐싱하면 추천 점수와 fallback 경로에서 재사용하기 쉽습니다.
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

        # 위험 단계 스냅샷은 추천 종목 API에서 최우선 사용되므로 캐시 적재 직전에 최신화합니다.
        snapshot_status = sync_risk_level_portfolio_snapshot(top_n=10, raise_on_error=False)
        _cache["risk_snapshot_stage"] = snapshot_status.get("stage")
        _cache["risk_snapshot_error"] = snapshot_status.get("error")

        snapshot_db = StockDBManager()
        snapshot_connected = False
        try:
            snapshot_db.connect(ensure_tables=False)
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
        # 이미 DB에서 레벨별 추천을 계산해 둔 경우, 서비스 응답은 그 결과를 그대로 우선 노출합니다.
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

            # 위험도는 절대값이 아니라 전체 시장 분포 안에서의 상대 위치로 다시 계산합니다.
            holding["risk_score"] = normalize_risk_score(ewma_std_60d, sigma_min, sigma_max)

            try:
                expected_log_pct = float(holding.get("expectedReturn3M", 0.0) or 0.0)
            except (TypeError, ValueError):
                expected_log_pct = 0.0
            # 스냅샷에 단순수익률 표시값이 없더라도 프론트엔드 카드가 깨지지 않도록 즉석 변환합니다.
            holding.setdefault(
                "expectedReturn3M_simple",
                round(float((np.exp(expected_log_pct / 100.0) - 1.0) * 100.0), 2),
            )
            holding.setdefault("historical_returns", {"1M": 0, "3M": 0, "6M": 0, "12M": 0})

        return holdings

    # 스냅샷이 비어 있으면 기존 유틸리티 점수 계산 로직으로 fallback 합니다.
    lam = LAMBDA_MAP.get(risk_level, get_lambda_by_level(2))
    df = _cache["adj_returns_df"]
    variance_map = _cache["variance_map"]

    if df is None or variance_map is None:
        raise RuntimeError("캐시가 초기화되지 않았습니다. load_cache()를 먼저 호출하세요.")

    # 1. 대상 종목 데이터 구성
    stocks = []
    for _, row in df.iterrows():
        # ticker는 이후 이름 매핑/공분산 조회/정렬 결과 키로 재사용되는 기준 키입니다.
        ticker = str(row["Ticker"]).strip().upper()
        # e_ri는 조정 후 기대 로그수익률이며, 순위 계산의 보상 항으로 바로 사용됩니다.
        e_ri = float(row["Adjusted_E_Total"])
        # vol_3m은 화면 설명용 참고 지표이고, 실제 목적함수 패널티에는 쓰지 않습니다.
        vol_3m = float(row.get("Vol_3M", 0.0))
        # ewma_variance_60d는 위험 패널티와 후속 sigma 계산에 재사용할 핵심 분산 값입니다.
        ewma_variance_60d = variance_map.get(ticker, 0.01)
        
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

    # pool은 위험 수준 컷오프를 지난 뒤 실제 목적함수 경쟁에 들어가는 후보 집합입니다.
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
    # result는 프론트엔드 카드에서 바로 소비할 최종 응답 포맷입니다.
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
    # 사용자가 고른 종목 중 실제 기대수익률이 캐시에 존재하는 종목만 최적화 입력으로 남깁니다.
    valid_tickers = [t for t in selected_tickers if t in all_returns]

    # 기대수익률 벡터
    mu = np.array([all_returns[t] for t in valid_tickers])

    # 공분산 sub-matrix 추출
    if cov_matrix is not None and ticker_to_idx is not None:
        # DB 공분산 사용 가능 → sub-matrix 추출
        # 공분산 행렬에 없는 티커는 제외하고, 행렬 인덱스와 동일한 순서로 다시 정렬합니다.
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

