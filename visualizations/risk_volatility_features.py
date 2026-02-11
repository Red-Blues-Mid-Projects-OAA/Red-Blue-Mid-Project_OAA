from pathlib import Path
import sys

# 프로젝트 루트를 import path에 추가해 어디서 실행해도 동작하게 처리
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import pd, np, yf
import matplotlib.pyplot as plt


def fetch_aapl_and_sp500(start_date="2018-01-01", end_date=None):
    tickers = ["AAPL", "^GSPC"]
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        raise ValueError("가격 데이터 다운로드에 실패했습니다.")

    # '^GSPC'를 분석용 컬럼명으로 변경
    df = df.rename(columns={"^GSPC": "SP500"})
    return df.dropna()


def calculate_features(price_df, lambda_val=0.94, annualize=True):
    log_returns = np.log(price_df / price_df.shift(1)).dropna()

    alpha = 1 - lambda_val

    # EWMA 변동성 (AAPL)
    ewma_var_aapl = log_returns["AAPL"].ewm(alpha=alpha, adjust=False).var()
    ewma_vol_aapl = np.sqrt(ewma_var_aapl)
    if annualize:
        ewma_vol_aapl = ewma_vol_aapl * np.sqrt(252)

    # Vol Trend: 20일 평균 - 60일 평균
    vol_ma20 = ewma_vol_aapl.rolling(20).mean()
    vol_ma60 = ewma_vol_aapl.rolling(60).mean()
    vol_trend = vol_ma20 - vol_ma60

    # EWMA Correlation (AAPL vs SP500)
    ewma_cov = log_returns["AAPL"].ewm(alpha=alpha, adjust=False).cov(log_returns["SP500"])
    ewma_var_sp500 = log_returns["SP500"].ewm(alpha=alpha, adjust=False).var()
    ewma_corr = ewma_cov / np.sqrt(ewma_var_aapl * ewma_var_sp500)

    features = pd.DataFrame(
        {
            "EWMA_VOL_AAPL": ewma_vol_aapl,
            "VOL_MA20": vol_ma20,
            "VOL_MA60": vol_ma60,
            "VOL_TREND": vol_trend,
            "EWMA_CORR_SP500": ewma_corr,
        }
    ).dropna()

    return features


def build_feature_dataframe(features):
    """
    모델 입력용으로 핵심 3개 feature만 분리한 DataFrame 생성
    """
    feature_df = features[
        ["EWMA_VOL_AAPL", "VOL_TREND", "EWMA_CORR_SP500"]
    ].copy()
    feature_df.columns = ["EWMA_VOL", "VOL_TREND", "EWMA_CORR"]
    return feature_df


def plot_features(features, output_path="visualizations/risk_volatility_features_aapl.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1) EWMA 변동성
    axes[0].plot(features.index, features["EWMA_VOL_AAPL"], label="AAPL EWMA Volatility", color="#1f77b4")
    axes[0].set_title("AAPL EWMA Volatility")
    axes[0].set_ylabel("Volatility")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    # 2) Vol Trend
    axes[1].plot(features.index, features["VOL_MA20"], label="Vol MA 20", color="#ff7f0e")
    axes[1].plot(features.index, features["VOL_MA60"], label="Vol MA 60", color="#2ca02c")
    axes[1].plot(features.index, features["VOL_TREND"], label="Vol Trend (20-60)", color="#d62728", linewidth=1.2)
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.7)
    axes[1].set_title("Volatility Trend: Expansion vs Contraction")
    axes[1].set_ylabel("Vol Level / Trend")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    # 3) EWMA Correlation
    axes[2].plot(features.index, features["EWMA_CORR_SP500"], label="EWMA Corr(AAPL, SP500)", color="#9467bd")
    axes[2].axhline(0, color="black", linewidth=1, alpha=0.7)
    axes[2].set_title("EWMA Correlation with S&P 500")
    axes[2].set_ylabel("Correlation")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()

    backend = plt.get_backend().lower()
    if "agg" in backend:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n시각화 저장 완료: {output_path}")
    else:
        plt.show()


def main():
    price_df = fetch_aapl_and_sp500(start_date="2018-01-01")
    features = calculate_features(price_df, lambda_val=0.94, annualize=True)
    feature_df = build_feature_dataframe(features)

    print("최근 Feature 샘플:")
    print(features.tail(10))
    print("\n3개 핵심 Feature DataFrame 샘플:")
    print(feature_df.tail(10))

    latest = feature_df.iloc[-1]
    state = "위험 확산중" if latest["VOL_TREND"] > 0 else "위험 축소중"
    print("\n최신 시점 요약")
    print(f"- EWMA 변동성: {latest['EWMA_VOL']:.4f}")
    print(f"- Vol Trend(20-60): {latest['VOL_TREND']:.4f} -> {state}")
    print(f"- EWMA Corr(AAPL, SP500): {latest['EWMA_CORR']:.4f}")

    plot_features(features)


if __name__ == "__main__":
    main()
