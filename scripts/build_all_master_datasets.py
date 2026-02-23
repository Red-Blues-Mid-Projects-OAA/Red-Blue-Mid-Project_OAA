import sys
import os
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from DB import TICKERS
from Classification.Preprocessing.build_master_dataset import build_master_dataset

def main():
    print("=" * 80)
    print(f"🔥 본격적인 300개 Ticker Master Dataset 생성 파이프라인 시작 (총 {len(TICKERS)}개 종목) 🔥")
    print("=" * 80)
    
    success_count = 0
    fail_count = 0
    failed_tickers = []
    
    for i, ticker in enumerate(TICKERS):
        print(f"\n[{i+1}/{len(TICKERS)}] 🚀 '{ticker}' Master Dataset 생성 중...")
        try:
            # auto_update=False: 원천 데이터(STOCK_DATA)는 DB/update_stock_data.py 에서 
            # 한 번에 300개를 가져왔다고 가정하고 여기서는 계산만 수행
            # 만약 개별적으로 최신화가 필요하다면 True로 유지. 
            # 속도 향상을 위해 DB에 이미 다운받혀 있다고 가정하고 False로 진행하는 것이 300개 확장시 유리.
            build_master_dataset(
                ticker=ticker,
                benchmark="SP500",
                auto_update=False, 
                persist_total_features_on_update=True,
                feature_source_mode="compute" # 무조건 새로 계산
            )
            success_count += 1
            print(f"✅ '{ticker}' 처리 성공!")
            
        except Exception as e:
            fail_count += 1
            failed_tickers.append((ticker, str(e)))
            print(f"❌ '{ticker}' 처리 실패: {e}")
            
        # SQLite 부하 방지 및 시스템 안정성을 위한 짧은 대기
        time.sleep(0.1)
            
    print("\n" + "=" * 80)
    print(f"🎉 모든 Master Dataset 생성 프로세스 종료 🎉")
    print(f"  - 성공: {success_count}개")
    print(f"  - 실패: {fail_count}개")
    if failed_tickers:
        print("  - 실패 사유 요약:")
        for t, msg in failed_tickers:
            print(f"    * {t}: {msg}")
    print("=" * 80)

if __name__ == "__main__":
    main()
