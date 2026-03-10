"""
이 파일은 위험 단계별 추천 포트폴리오 스냅샷을 새로 계산해 저장합니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    _BOOT_PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _BOOT_PROJECT_ROOT is not None and str(_BOOT_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_BOOT_PROJECT_ROOT))

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from DB.stock_db_manager import StockDBManager

# investment-mbti-back 모듈(optimizer/risk profile) 재사용을 위한 경로 추가
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _PROJECT_ROOT / "investment-mbti-back"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from portfolio_optimizer import optimize_portfolio
from risk_profile import get_lambda_by_level

HORIZON_DAYS = 60
LAMBDA_MAP = {i: get_lambda_by_level(i) for i in range(1, 5)}
RISK_LEVEL_LABEL_MAP = {
    4: "Very Low Risk",
    3: "Low Risk",
    2: "Medium Risk",
    1: "High Risk",
}
TICKER_NAME_MAP = {
    "NVDA": "엔비디아",
    "AAPL": "애플",
    "MSFT": "마이크로소프트",
    "AMZN": "아마존",
    "GOOGL": "알파벳 A",
    "GOOG": "알파벳 C",
    "META": "메타",
    "AVGO": "브로드컴",
    "TSLA": "테슬라",
    "BRK-B": "버크셔 해서웨이",
    "BRK-A": "버크셔 해서웨이 A",
    "TSM": "TSMC",
}

def _parse_bool(value) -> bool:
    """bool를 읽기 쉬운 형태로 해석합니다."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "y", "yes"}

def _normalize_adjusted_returns_df(df: pd.DataFrame) -> pd.DataFrame:
    """보정된 수익률 df 값을 서로 비교하기 쉽게 정규화합니다."""
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

    normalized = df.copy()
    normalized.columns = [str(c).strip() for c in normalized.columns]
    missing = [c for c in required_cols if c not in normalized.columns]
    if missing:
        raise ValueError(f"adjusted returns 필수 컬럼 누락: {missing}")

    normalized["Ticker"] = normalized["Ticker"].astype(str).str.strip().str.upper()
    normalized["Gate_Passed"] = normalized["Gate_Passed"].apply(_parse_bool)
    normalized["Adjustment_Applied"] = normalized["Adjustment_Applied"].apply(_parse_bool)
    return normalized

def _build_all_returns_map(df: pd.DataFrame) -> dict[str, float]:
    """전체 수익률 map 결과를 여러 데이터를 바탕으로 조합해 만듭니다."""
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        value = pd.to_numeric(row.get("Adjusted_E_Total"), errors="coerce")
        if ticker and pd.notna(value):
            out[ticker] = float(value)
    return out

def _build_variance_map_from_df(df: pd.DataFrame) -> dict[str, float]:
    """variance map from df 결과를 여러 데이터를 바탕으로 조합해 만듭니다."""
    variance_map: dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        vol = pd.to_numeric(row.get("Vol_3M"), errors="coerce")
        if pd.notna(vol):
            variance_map[ticker] = float(vol) ** 2
        else:
            variance_map[ticker] = 0.01
    return variance_map

def _filter_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """candidates만 남기도록 걸러냅니다."""
    gate_mask = df["Gate_Passed"].apply(_parse_bool)
    e_total = pd.to_numeric(df["E_Total_3M"], errors="coerce")
    adjusted = pd.to_numeric(df["Adjusted_E_Total"], errors="coerce")
    filtered = df[(gate_mask) & (e_total > 0) & (adjusted > 0)].copy()
    filtered["Ticker"] = filtered["Ticker"].astype(str).str.strip().str.upper()
    filtered["Gate_Passed"] = filtered["Gate_Passed"].apply(_parse_bool)
    filtered["Adjustment_Applied"] = filtered["Adjustment_Applied"].apply(_parse_bool)
    return filtered

def _load_context(db: StockDBManager):
    """context 데이터를 메모리로 불러옵니다."""
    source_df = db.fetch_adjusted_expected_returns()
    if source_df.empty:
        raise RuntimeError("ADJUSTED_EXPECTED_RETURNS가 비어 있습니다.")

    source_df = _normalize_adjusted_returns_df(source_df)
    filtered_df = _filter_candidates(source_df)
    all_returns_map = _build_all_returns_map(source_df)

    ticker_list: list[str] = []
    cov_matrix = np.array([])
    try:
        ticker_list, cov_matrix = db.fetch_ewma_covariance()
    except Exception:
        ticker_list, cov_matrix = [], np.array([])

    if len(ticker_list) > 0 and cov_matrix.size > 0:
        cov_matrix_scaled = cov_matrix * HORIZON_DAYS
        ticker_to_cov_idx = {ticker: idx for idx, ticker in enumerate(ticker_list)}
        variance_map = {
            ticker: float(cov_matrix_scaled[i, i]) for i, ticker in enumerate(ticker_list)
        }
    else:
        cov_matrix_scaled = None
        ticker_to_cov_idx = None
        variance_map = _build_variance_map_from_df(source_df)

    return {
        "source_df": source_df,
        "filtered_df": filtered_df,
        "all_returns_map": all_returns_map,
        "cov_matrix": cov_matrix_scaled,
        "ticker_to_cov_idx": ticker_to_cov_idx,
        "variance_map": variance_map,
    }

def _get_recommended_stocks(
    risk_level: int,
    top_n: int,
    filtered_df: pd.DataFrame,
    variance_map: dict[str, float],
) -> list[dict]:
    """recommended 종목 정보를 조회해 반환합니다."""
    lam = LAMBDA_MAP.get(risk_level, get_lambda_by_level(2))

    stocks: list[dict] = []
    for _, row in filtered_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        e_ri = float(row["Adjusted_E_Total"])
        ewma_variance_60d = float(variance_map.get(ticker, 0.01))
        ewma_std_60d = float(np.sqrt(max(ewma_variance_60d, 0.0)))
        # 개별 종목 유틸리티: E(R_i) - 0.5 * λ * Var_i(EWMA 60일 분산)
        # 표준편차(σ) 대신 분산(Var) 항을 직접 패널티로 사용합니다.
        score = e_ri - 0.5 * lam * ewma_variance_60d
        stocks.append(
            {
                "ticker": ticker,
                "name": TICKER_NAME_MAP.get(ticker, ticker),
                "score": score,
                "sigma_ewma_60d": ewma_std_60d,
            }
        )

    stocks.sort(key=lambda x: x["sigma_ewma_60d"])

    total_count = len(stocks)
    if risk_level == 4:
        pool_size = max(int(total_count * 0.25), top_n)
    elif risk_level == 3:
        pool_size = max(int(total_count * 0.50), top_n)
    elif risk_level == 2:
        pool_size = max(int(total_count * 0.75), top_n)
    else:
        pool_size = total_count

    pool = stocks[:pool_size]
    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool[:top_n]

def _get_mvo_inputs(
    selected_tickers: list[str],
    all_returns_map: dict[str, float],
    cov_matrix: np.ndarray | None,
    ticker_to_cov_idx: dict[str, int] | None,
    variance_map: dict[str, float],
):
    """mvo inputs 정보를 조회해 반환합니다."""
    valid_tickers = [t for t in selected_tickers if t in all_returns_map]
    mu = np.array([all_returns_map[t] for t in valid_tickers], dtype=float)

    if cov_matrix is not None and ticker_to_cov_idx is not None:
        idx_list = []
        final_tickers = []
        for ticker in valid_tickers:
            if ticker in ticker_to_cov_idx:
                idx_list.append(ticker_to_cov_idx[ticker])
                final_tickers.append(ticker)

        if not final_tickers:
            return np.array([]), np.array([]), []

        idx_arr = np.array(idx_list)
        cov_sub = cov_matrix[np.ix_(idx_arr, idx_arr)]
        mu = np.array([all_returns_map[t] for t in final_tickers], dtype=float)
        return mu, cov_sub, final_tickers

    n = len(valid_tickers)
    if n == 0:
        return np.array([]), np.array([]), []

    cov_sub = np.zeros((n, n), dtype=float)
    for i, ticker in enumerate(valid_tickers):
        cov_sub[i, i] = float(variance_map.get(ticker, 0.01))
    return mu, cov_sub, valid_tickers

def _build_snapshot_rows(context: dict, top_n: int = 10) -> tuple[list[dict], list[dict]]:
    """스냅샷 rows 결과를 여러 데이터를 바탕으로 조합해 만듭니다."""
    filtered_df = context["filtered_df"]
    all_returns_map = context["all_returns_map"]
    cov_matrix = context["cov_matrix"]
    ticker_to_cov_idx = context["ticker_to_cov_idx"]
    variance_map = context["variance_map"]

    metrics_rows: list[dict] = []
    holdings_rows: list[dict] = []

    for risk_level in [4, 3, 2, 1]:
        lam = float(LAMBDA_MAP.get(risk_level, get_lambda_by_level(risk_level)))
        recommended = _get_recommended_stocks(
            risk_level=risk_level,
            top_n=top_n,
            filtered_df=filtered_df,
            variance_map=variance_map,
        )
        tickers = [item["ticker"] for item in recommended]

        if not tickers:
            metrics_rows.append(
                {
                    "risk_level": risk_level,
                    "risk_label": RISK_LEVEL_LABEL_MAP.get(risk_level, f"Level {risk_level}"),
                    "lambda_value": lam,
                    "expected_return_3m": 0.0,
                    "portfolio_std_60d": 0.0,
                    "holdings_count": 0,
                }
            )
            continue

        mu, cov_sub, valid_tickers = _get_mvo_inputs(
            selected_tickers=tickers,
            all_returns_map=all_returns_map,
            cov_matrix=cov_matrix,
            ticker_to_cov_idx=ticker_to_cov_idx,
            variance_map=variance_map,
        )
        if len(valid_tickers) == 0:
            metrics_rows.append(
                {
                    "risk_level": risk_level,
                    "risk_label": RISK_LEVEL_LABEL_MAP.get(risk_level, f"Level {risk_level}"),
                    "lambda_value": lam,
                    "expected_return_3m": 0.0,
                    "portfolio_std_60d": 0.0,
                    "holdings_count": 0,
                }
            )
            continue

        weights = np.array(optimize_portfolio(mu, cov_sub, lam), dtype=float)
        if len(weights) != len(valid_tickers):
            raise RuntimeError(
                f"weights length mismatch (risk_level={risk_level}, "
                f"weights={len(weights)}, tickers={len(valid_tickers)})"
            )

        portfolio_return_3m = float(np.dot(weights, mu)) * 100
        portfolio_var_60d = float(np.dot(weights, np.dot(cov_sub, weights)))
        portfolio_sigma_60d = float(np.sqrt(max(portfolio_var_60d, 0.0)))

        weighted_pairs = [
            (ticker, float(weight))
            for ticker, weight in zip(valid_tickers, weights)
            if float(weight) > 1e-8
        ]
        weighted_pairs.sort(key=lambda item: item[1], reverse=True)

        metrics_rows.append(
            {
                "risk_level": risk_level,
                "risk_label": RISK_LEVEL_LABEL_MAP.get(risk_level, f"Level {risk_level}"),
                "lambda_value": lam,
                "expected_return_3m": round(portfolio_return_3m, 2),
                "portfolio_std_60d": round(portfolio_sigma_60d, 4),
                "holdings_count": len(weighted_pairs),
            }
        )

        recommended_map = {item["ticker"]: item for item in recommended}
        return_map = {ticker: float(value) for ticker, value in zip(valid_tickers, mu)}

        for rank_no, (ticker, weight) in enumerate(weighted_pairs, 1):
            rec = recommended_map.get(ticker, {})
            holdings_rows.append(
                {
                    "risk_level": risk_level,
                    "rank_no": rank_no,
                    "ticker": ticker,
                    "stock_name": str(rec.get("name", TICKER_NAME_MAP.get(ticker, ticker))),
                    "weight_pct": round(float(weight) * 100, 2),
                    "expected_return_3m": round(return_map.get(ticker, 0.0) * 100, 2),
                    "sigma_ewma_60d": (
                        float(rec.get("sigma_ewma_60d"))
                        if rec.get("sigma_ewma_60d") is not None
                        else None
                    ),
                }
            )

    return metrics_rows, holdings_rows

def sync_risk_level_portfolio_snapshot(top_n: int = 10, raise_on_error: bool = False) -> dict:
    """Compute + persist RISK_LEVEL_PORTFOLIO_SNAPSHOT and return status."""
    status = {
        "stage": "error",
        "error": None,
        "source_count": 0,
        "candidate_count": 0,
        "metrics_count": 0,
        "holdings_count": 0,
    }

    db = StockDBManager()
    db_connected = False

    try:
        db.connect()
        db_connected = True

        context = _load_context(db)
        status["source_count"] = int(len(context["source_df"]))
        status["candidate_count"] = int(len(context["filtered_df"]))

        metrics_rows, holdings_rows = _build_snapshot_rows(context=context, top_n=top_n)

        db.replace_risk_level_portfolio_snapshot(metrics_rows, holdings_rows)
        status["stage"] = "ok"
        status["metrics_count"] = int(len(metrics_rows))
        status["holdings_count"] = int(len(holdings_rows))
        print(
            "[RISK_SNAPSHOT][OK] "
            f"source={status['source_count']}, candidates={status['candidate_count']}, "
            f"metrics={status['metrics_count']}, holdings={status['holdings_count']}"
        )
    except Exception as e:
        status["error"] = str(e)
        print(f"[RISK_SNAPSHOT][ERROR] {e}")
        if raise_on_error:
            raise
    finally:
        if db_connected:
            db.close()

    return status

def main():
    """메인 관련 처리를 담당하는 함수입니다."""
    parser = argparse.ArgumentParser(
        description="RISK_LEVEL_PORTFOLIO_SNAPSHOT 계산/적재"
    )
    parser.add_argument("--top-n", type=int, default=10, help="리스크 레벨별 최대 종목 수")
    args = parser.parse_args()

    status = sync_risk_level_portfolio_snapshot(top_n=args.top_n, raise_on_error=False)
    if status["stage"] != "ok":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
