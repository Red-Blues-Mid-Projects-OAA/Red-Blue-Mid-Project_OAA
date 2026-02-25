if __package__ in (None, ""):
    import sys
    from pathlib import Path
    
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
        
    import numpy as np

    from DB.stock_db_manager import StockDBManager
else:
    from common import np
    from DB import StockDBManager


def calculate_and_save_log_returns():
    """
    STOCK_DATA에서 로그 수익률을 계산해 LOG_RETURNS에 저장합니다.
    """
    print("로그 수익률 계산 프로세스 시작...")

    db_manager = StockDBManager()
    db_manager.connect()

    try:
        prices_df = db_manager.fetch_prices()
        if prices_df.empty:
            print("주가 데이터가 없어 계산을 중단합니다.")
            return

        print(f"주가 데이터 로드 완료: {prices_df.shape}")

        # 전 종목 공통 dropna를 하지 않고 티커별 결측을 유지해야 상장일 차이로 인한 데이터 손실을 막을 수 있습니다.
        log_returns_df = np.log(prices_df / prices_df.shift(1))

        print("\n[DB 정리] 기존 LOG_RETURNS 데이터를 초기화합니다...")
        db_manager.truncate_log_returns()

        row_count = int(log_returns_df.count().sum())
        print(f"로그 수익률 데이터 {row_count}건 저장 시작...")
        db_manager.insert_log_returns(log_returns_df)
        db_manager.reorganize_log_returns()
    except Exception as e:
        print(f"계산 중 오류 발생: {e}")
    finally:
        db_manager.close()


if __name__ == "__main__":
    calculate_and_save_log_returns()
