"""
Regime Shift Overlay — 시장 국면 변동에 따른 예측 수익률 조정 모듈.

파이프라인 산출물(final_expected_returns.csv)의 예측 수익률을
최근 3개월 실현 수익률과 비교하여, 괴리 크기에 비례하는
Graduated Momentum Tracking Overlay를 적용합니다.

Execution:
    python3 -m Classification.multi_ticker.adjust_regime
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# 프로젝트 루트 경로 설정
if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

from common import np, pd
from DB import StockDBManager
from Classification.model_config import MULTI_TICKER_ARTIFACT_DIR

# ──────────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────────
# 벤치마크 연평균 기대수익률 (S&P 500 장기 평균)
E_RM_ANNUAL = 0.105
# 예측 Horizon (거래일 기준)
HORIZON_DAYS = 60
# 벤치마크 3개월 기대수익률
E_RM_3M = E_RM_ANNUAL * (HORIZON_DAYS / 252)
# 비례 조정 최대 강도 도달 기준 (괴리가 이 값 이상이면 weight=1.0)
ADJUSTMENT_THRESHOLD = 0.10


def _load_final_csv() -> pd.DataFrame:
    """파이프라인 산출물 CSV 로드."""
    csv_path = MULTI_TICKER_ARTIFACT_DIR / "final_expected_returns.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"파이프라인 산출물 CSV를 찾을 수 없습니다: {csv_path}\n"
            "run_all_mapping.py가 먼저 완료되어야 합니다."
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"[LOAD] final_expected_returns.csv: {len(df)}행 로드 완료")
    return df


def _load_realized_returns() -> tuple[pd.Series, pd.Series]:
    """
    DB에서 최근 60 거래일 로그 수익률을 로드하여
    종목별 3개월 실현 누적 수익률과 일별 변동성을 반환합니다.
    """
    db = StockDBManager()
    db.connect()
    try:
        # 종목별 일별 로그 수익률
        log_returns = db.fetch_log_returns()
        # S&P 500 일별 로그 수익률
        sp500_df = db.fetch_sp500_data()
    finally:
        db.close()

    log_returns.index = pd.to_datetime(log_returns.index)
    sp500_df.index = pd.to_datetime(sp500_df.index)

    # 최근 60 거래일만 사용
    recent_stock = log_returns.tail(HORIZON_DAYS)

    # 종목별 3개월 누적 로그 수익률 (절대 수익률)
    realized_3m = recent_stock.sum()

    # 종목별 3개월 변동성 (일별 표준편차 → 3개월 환산)
    daily_vol = recent_stock.std()
    vol_3m = daily_vol * math.sqrt(HORIZON_DAYS)

    print(f"[LOAD] 최근 {HORIZON_DAYS} 거래일 실현 수익률 로드 완료 ({len(realized_3m)} 종목)")
    return realized_3m, vol_3m


def run_regime_adjustment() -> pd.DataFrame:
    """
    Graduated Momentum Tracking Overlay를 적용하여
    조정된 예측 수익률 CSV를 생성합니다.
    """
    # 1. 입력 데이터 로드
    final_df = _load_final_csv()
    realized_3m, vol_3m = _load_realized_returns()

    results = []

    for _, row in final_df.iterrows():
        ticker = str(row["Ticker"]).strip().upper()
        gate_passed = bool(row.get("Gate_Passed", False))
        original_e_ret = float(row.get("Expected_Return_3M", 0.0))
        return_type = str(row.get("Return_Type", "Unknown"))

        # ── Step 2: 단위 통일 (모두 절대 수익률로 변환) ────────────────
        if return_type == "Grinold-Kahn":
            # 초과 수익률(Alpha) → 절대 수익률로 변환
            e_total = original_e_ret + E_RM_3M
        else:
            # CAPM은 이미 절대 수익률, Error 등도 그대로 사용
            e_total = original_e_ret

        # ── Step 3: 실현 수익률 및 변동성 조회 ──────────────────────
        if ticker in realized_3m.index:
            realized = float(realized_3m[ticker])
            ticker_vol = float(vol_3m[ticker])
        else:
            # DB에 없는 종목은 조정 불가 → 원래 값 유지
            realized = 0.0
            ticker_vol = 0.0

        # ── Step 4: Graduated Momentum Tracking Overlay ─────────
        gap = abs(e_total - realized)
        # 비례 가중치: gap이 작으면 약한 조정, 클수록 강한 조정
        adj_weight = min(1.0, gap / ADJUSTMENT_THRESHOLD)

        # 실현 수익률의 방향(부호)으로 변동성 부호 결정
        sign = 1.0 if realized >= 0 else -1.0

        # 조정값 계산
        adjustment = adj_weight * sign * ticker_vol
        adjusted_e_total = e_total + adjustment

        # 조정 적용 여부 판별
        adjustment_applied = gap > 0.001  # 사실상 모든 종목에 비례 적용

        results.append({
            "Ticker": ticker,
            "Gate_Passed": gate_passed,
            "Return_Type": return_type,
            "Original_E_Ret": round(original_e_ret, 6),
            "E_Total_3M": round(e_total, 6),
            "Realized_3M": round(realized, 6),
            "Gap": round(gap, 6),
            "Adj_Weight": round(adj_weight, 4),
            "Vol_3M": round(ticker_vol, 6),
            "Adjustment": round(adjustment, 6),
            "Adjusted_E_Total": round(adjusted_e_total, 6),
            "Adjustment_Applied": adjustment_applied,
        })

    # 5. 결과 DataFrame 생성 및 저장
    result_df = pd.DataFrame(results)
    output_path = MULTI_TICKER_ARTIFACT_DIR / "adjusted_expected_returns.csv"
    result_df.to_csv(output_path, index=False)

    # 요약 통계 출력
    adjusted_count = result_df["Adjustment_Applied"].sum()
    positive_count = (result_df["Adjusted_E_Total"] > 0).sum()
    negative_count = (result_df["Adjusted_E_Total"] <= 0).sum()

    print("\n" + "=" * 70)
    print("📊 Regime Shift Overlay — 조정 완료")
    print("=" * 70)
    print(f"  총 종목 수         : {len(result_df)}")
    print(f"  조정 적용 종목     : {adjusted_count}")
    print(f"  양수(+) 수익률     : {positive_count}")
    print(f"  음수(-) 수익률     : {negative_count}")
    print(f"  저장 경로          : {output_path}")
    print("=" * 70)

    # 상위/하위 5개 종목 출력
    sorted_df = result_df.sort_values("Adjusted_E_Total", ascending=False)
    print("\n📈 상위 5개 종목 (조정 후):")
    for _, r in sorted_df.head(5).iterrows():
        print(f"  {r['Ticker']:6s} | 예측: {r['E_Total_3M']*100:+6.2f}% "
              f"| 실현: {r['Realized_3M']*100:+6.2f}% "
              f"| 조정후: {r['Adjusted_E_Total']*100:+6.2f}%")

    print("\n📉 하위 5개 종목 (조정 후):")
    for _, r in sorted_df.tail(5).iterrows():
        print(f"  {r['Ticker']:6s} | 예측: {r['E_Total_3M']*100:+6.2f}% "
              f"| 실현: {r['Realized_3M']*100:+6.2f}% "
              f"| 조정후: {r['Adjusted_E_Total']*100:+6.2f}%")

    return result_df


if __name__ == "__main__":
    run_regime_adjustment()
