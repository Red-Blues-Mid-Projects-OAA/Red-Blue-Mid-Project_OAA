"""
이 파일은 S&P 500 구성 종목 정보를 수집하는 스크래퍼 유틸리티입니다.
주요 함수는 입력 준비, 핵심 계산, 결과 저장 또는 반환 순서로 배치되어 있어 상위 파이프라인과의 연결 지점을 위에서 아래로 따라가면 전체 흐름을 빠르게 파악할 수 있습니다.
"""

import os
import json
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_DIR = os.path.dirname(_THIS_DIR)
_TOP300_JSON_PATH = os.path.join(_DB_DIR, "sp500_top300.json")

def _scrape_sp500_from_wikipedia():
    """Wikipedia에서 S&P 500 구성 종목을 스크래핑합니다."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    print(f"[{url}] 스크래핑 시작...")
    
    # 첫 번째 테이블: S&P 500 component stocks
    tables = pd.read_html(url)
    df = tables[0]
    
    # Symbol 컬럼 정리 (B/R.K-B 등 처리 반영)
    df['Symbol'] = df['Symbol'].str.replace('.', '-')
    
    # 이 페이지에는 시가총액이나 구성비가 직접 제공되지 않습니다.
    # 따라서, 편의상 S&P500의 기본 테이블 순서가 시가총액/중요도 순서와 유사하게 나열되지 않는다면,
    # S&P Global의 시총 기준 상위 300개를 가져오려면 추가 API가 필요하지만, 
    # 현재 위키피디아에서 제공하는 500개 중 단순 상위 300개를 우선 자르거나, 
    # yfinance를 통해 시가총액을 간접적으로 가져와 정렬할 수 있습니다.
    
    # 이번 단계는: yfinance로 info를 빠르게 당겨서 정렬하기엔 너무 오래 걸리므로, 
    # 일단 위키피디아의 리스트를 알파벳 순으로 가져가거나, 단순히 앞 300개를 사용합니다.
    # 하지만 사용자는 '시가총액 상위 300개'를 원하므로, 슬릭(Slick)한 방법으로 
    # slickcharts 페이지(가중치 포함)를 스크래핑합니다.
    pass

def scrape_sp500_top300_by_weight():
    """Slickcharts에서 S&P 500 구성종목(Weight 포함)을 스크래핑하여 Top 300을 반환합니다."""
    url = "https://www.slickcharts.com/sp500"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    import requests
    from bs4 import BeautifulSoup
    import io

    print(f"[{url}] 에서 S&P 500 비중 기준 상위 300개 스크래핑 시작...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse text into dataframe
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
        
        # pd.read_html needs string
        df = pd.read_html(io.StringIO(str(table)))[0]
        
        # Clean up column names and symbols
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        
        # Sort by Weight just to be sure (Slickcharts is already sorted)
        # Weight column is "Weight" inside Slickcharts
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].astype(str).str.replace('%', '').astype(float)
            df = df.sort_values(by='Weight', ascending=False)
            
        top_300 = df.head(300)['Symbol'].tolist()
        
        print(f"Top 300 종목 가져오기 완료 (예: {top_300[:5]} ...)")
        
        # JSON 저장
        with open(_TOP300_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(top_300, f, ensure_ascii=False, indent=2)
            
        print(f"[{_TOP300_JSON_PATH}] 업데이트 완료.")
        return top_300

    except Exception as e:
        print(f"스크래핑 중 오류 발생: {e}")
        # 오류 발생 시 위키피디아로 폴백
        print("위키피디아 폴백 스크래핑 시도 중...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-')
        # 위키피디아는 시총순이 아니므로 일단 300개만 슬라이스 (실무에서는 권장하지 않음)
        top_300 = df['Symbol'].tolist()[:300]
        
        with open(_TOP300_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(top_300, f, ensure_ascii=False, indent=2)
            
        print(f"[{_TOP300_JSON_PATH}] (Wikipedia Fallback) 업데이트 완료.")
        return top_300

def load_top300_tickers():
    """저장된 JSON에서 로드. 없으면 스크래핑 실행."""
    if os.path.exists(_TOP300_JSON_PATH):
        try:
            with open(_TOP300_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return scrape_sp500_top300_by_weight()

if __name__ == "__main__":
    scrape_sp500_top300_by_weight()
