from common import yf, pd, datetime, np
import matplotlib.pyplot as plt

def calculate_aapl_volume_analysis():
    """
    AAPL 거래량 분석 (Volume Ratio + OBV)
    - Volume_Ratio = 당일 거래량 / 직전 20일 평균 거래량
    - OBV: 상승일 +거래량, 하락일 -거래량 누적
    """
    ticker = "AAPL"
    start_date = "2016-01-01"
    end_date = "2024-09-30"

    print(f"[{ticker}] {start_date} ~ {end_date} 거래량 분석 데이터 계산 중...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print("데이터를 가져오지 못했습니다.")
        return

    # 멀티 컬럼 처리
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
        volume = data["Volume"].iloc[:, 0]
    else:
        close = data["Close"]
        volume = data["Volume"]

    # Volume Ratio 계산
    # 직전 20일 평균 거래량 (shift(1)로 당일 제외)
    avg_20 = volume.rolling(window=20).mean().shift(1)
    volume_ratio = volume / avg_20

    # OBV 계산
    price_change = np.sign(close.diff())
    obv = (price_change * volume).cumsum()
#오늘 종가 어제종가 비교
#부호(가격 변화량)
#cumsum=누적합
    result = pd.DataFrame({
        "Close": close.values,
        "Volume": volume.values,
        "Avg_Volume_20d": avg_20.values,
        "Volume_Ratio": volume_ratio.values,
        "OBV": obv.values
    }, index=close.index)
    result.index.name = "Date"

    # 20거래일 이후부터 유효 (NaN 제거)
    result = result.dropna()

    pd.set_option("display.max_rows", None)
 #                                      제한은없다
    print(f"\n총 {len(result)}개 거래일\n")
    print(result)

    # 거래량 기본 통계
    print(f"\n평균 거래량: {result['Volume'].mean():,.0f}")
    print(f"최대 거래량: {result['Volume'].max():,.0f} ({result['Volume'].idxmax().strftime('%Y-%m-%d')})")
    print(f"최소 거래량: {result['Volume'].min():,.0f} ({result['Volume'].idxmin().strftime('%Y-%m-%d')})")

    result.to_csv("aapl_volume_analysis.csv")
    print("\naapl_volume_analysis.csv 파일로 저장 완료!")

    # Volume Ratio 히스토그램
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].hist(result["Volume_Ratio"], bins=80, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=1.0, color="red", linestyle="--", label="Ratio = 1.0")
    axes[0].set_title("AAPL Volume Ratio Distribution (2016 ~ 2024-09)")
    axes[0].set_xlabel("Volume Ratio")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    axes[1].plot(result.index, result["OBV"], color="steelblue", linewidth=0.7)
    axes[1].set_title("AAPL OBV (On-Balance Volume)")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("OBV")

    plt.tight_layout()
    # 그래프끼리 겹치지않게
    plt.savefig("aapl_volume_analysis.png", dpi=150)
    # 이미지파일로 저장하는 메서드
    print("aapl_volume_analysis.png 차트 저장 완료!")
    plt.show()

if __name__ == "__main__":
    calculate_aapl_volume_analysis()
