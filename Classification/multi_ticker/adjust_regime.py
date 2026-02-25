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
# 벤치마크 3개월 로그 기대수익률 (완벽한 덧셈 호환용)
E_RM_3M_LOG = math.log(1 + E_RM_3M)

# Tanh 비선형 조정을 위한 상수 (가중치 최대 허용치 2.0배, 기준 스케일 20% 괴리율)
# 하위 구간 가중치 유지보존 계수: 1.5
MAX_ADJUSTMENT_WEIGHT = 2.0
GAP_SCALE_FACTOR = 0.20
MAINTAIN_CURVE_FACTOR = 1.5


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

    # 최근 60 거래일의 단순 데이터 (실현 수익률 계산용)
    recent_stock = log_returns.tail(HORIZON_DAYS)

    # 종목별 최근 3개월 누적 로그 수익률 (절대 수익률)
    realized_3m = recent_stock.sum()

    # ────────────────────────────────────────────────────────
    # 사용자 제안 반영: "60일 롤링 누적수익률" 자체의 변동성 산출
    # 1. 과거 전체 데이터에 대해 매일매일 60일 누적 로그수익률을 계산
    rolling_60d_returns = log_returns.rolling(window=HORIZON_DAYS).sum()
    
    # 2. 최근 1년(252 거래일) 동안, 그 60일 누적수익률이 얼마나 변동했는지 표준편차 직접 산출
    vol_3m = rolling_60d_returns.tail(252).std()
    # ────────────────────────────────────────────────────────

    print(f"[LOAD] 최근 {HORIZON_DAYS} 거래일 실현 수익률 및 롤링 변동성 로드 완료 ({len(realized_3m)} 종목)")
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

        # ── Step 2: 단위 통일 (순수 로그 수익률(단위) 통일) ────────────────
        if return_type == "Grinold-Kahn":
            # 로그 초과 수익률(Log Alpha) + 벤치마크 로그 기대수익률
            e_total = original_e_ret + E_RM_3M_LOG
        else:
            # CAPM은 이미 capm.py에서 np.log(1 + simple)을 거친 온전한 로그 수익률
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
        
        # Tanh (쌍곡탄젠트) 비선형 가중치 산출
        # - 작은 Gap (ex: 2~10%): 기존 3.0 함수와 동일한 가중치 궤적 유지 (MAINTAIN_CURVE_FACTOR 보정)
        # - 중간 Gap (ex: 20%): 자연스럽게 S커브를 그리며 감속
        # - 극단적 Gap (ex: 30%+): 최대 MAX_ADJUSTMENT_WEIGHT(2.0) 배수로 부드럽게 한도 수렴
        adj_weight = MAX_ADJUSTMENT_WEIGHT * math.tanh(MAINTAIN_CURVE_FACTOR * (gap / GAP_SCALE_FACTOR) ** 2)

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
