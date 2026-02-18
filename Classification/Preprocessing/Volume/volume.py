"""
거래량 분석 모듈 (Self-contained)

생성되는 DataFrame:
  df_volume_features : Volume_Ratio, OBV_ROC_20 (정상성 확보)

★ DB 적재 없음
"""
import sys
from pathlib import Path

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

from common import pd, np
from DB import StockDBManager


def calculate_volume_analysis(ticker="AAPL", db=None):
    """
    티커 거래량 분석 (Volume Ratio + OBV 변화율)
    - Volume_Ratio = 당일 거래량 / 직전 20일 평균 거래량
    - OBV_ROC_20   = OBV의 20일간 변화율 (정상성 확보)
    """
    print(f"[{ticker}] Oracle DB에서 데이터 로드 중...")

    should_close = False
    if db is None:
        db = StockDBManager()
        db.connect()
        should_close = True

    try:
        data = db.fetch_ticker_data(ticker)
    finally:
        if should_close:
            db.close()

    if data.empty:
        print("DB에서 데이터를 가져오지 못했습니다. DB/update_stock_data.py를 먼저 실행하세요.")
        return None

    close = data["Close"]
    volume = data["Volume"]

    # Volume Ratio 계산
    # 직전 20일 평균 거래량 (shift(1)로 당일 제외 → 아직 장이 진행중인 거래량을 포함하지 않기 위함)
    avg_20 = volume.rolling(window=20).mean().shift(1)
    volume_ratio = volume / avg_20

    # OBV 계산
    price_change = np.sign(close.diff())
    obv = (price_change * volume).cumsum()

    # OBV 변화율 (20일) — 비정상성(Non-stationarity) 해결
    # OBV는 누적합이라 시간에 따라 무한히 커져 ML 모델에 부적합
    # 대신 20일간 OBV 변화율을 사용하여 정상성(Stationarity) 확보
    obv_roc_20 = obv.pct_change(20)

    result = pd.DataFrame({
        "Close": close.values,
        "Volume": volume.values,
        "Avg_Volume_20d": avg_20.values,
        "Volume_Ratio": volume_ratio.values,
        "OBV": obv.values,
        "OBV_ROC_20": obv_roc_20.values,
    }, index=close.index)

    # NaN 제거 (rolling window + OBV ROC로 인한 초기 데이터 불완전)
    result = result.dropna()

    print(f"\n총 {len(result)}개 거래일")

    # ML용 피처 DataFrame (Volume_Ratio + OBV_ROC_20만 추출)
    df_volume_features = result[["Volume_Ratio", "OBV_ROC_20"]].copy()

    print(f"\n{'=' * 60}")
    print(f"[결과] {ticker} 거래량 피처 (df_volume_features)")
    print(f"{'=' * 60}")
    print(f"  기간: {df_volume_features.index[0].date()} ~ {df_volume_features.index[-1].date()}")
    print(f"  건수: {len(df_volume_features)}")
    print(df_volume_features.head())
    print("  ...")
    print(df_volume_features.tail())

    # 거래량 기본 통계
    print(f"\n평균 거래량: {result['Volume'].mean():,.0f}")
    print(f"최대 거래량: {result['Volume'].max():,.0f} ({result['Volume'].idxmax().strftime('%Y-%m-%d')})")
    print(f"최소 거래량: {result['Volume'].min():,.0f} ({result['Volume'].idxmin().strftime('%Y-%m-%d')})")

    return df_volume_features


# Backward compatibility alias
def calculate_aapl_volume_analysis(db=None):
    return calculate_volume_analysis(ticker="AAPL", db=db)


if __name__ == "__main__":
    calculate_volume_analysis()
