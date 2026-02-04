import yfinance as yf

def verify_yf():
    print("Testing yfinance library...")
    try:
        # Fetching data for Apple (AAPL) as a test
        ticker = "AAPL"
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        print(f"Successfully connected to Yahoo Finance!")
        print(f"Ticker: {ticker}")
        print(f"Long Name: {info.get('longName', 'N/A')}")
        print(f"Current Price: {info.get('currentPrice', 'N/A')} {info.get('currency', 'USD')}")
        
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    verify_yf()
