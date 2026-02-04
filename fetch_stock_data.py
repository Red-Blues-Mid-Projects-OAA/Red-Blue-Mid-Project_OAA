import yfinance as yf
import pandas as pd
from datetime import datetime

# Define tickers
# Note: Yahoo Finance uses 'BRK-A' for Berkshire Hathaway Class A
tickers = ['NVDA', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'META', 'TSM', 'TSLA', 'AVGO', 'BRK-A']
#adfjasdfa
# Define date range
start_date = "2015-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"Fetching data for: {', '.join(tickers)}")
print(f"Date range: {start_date} to {end_date}")
#dhvjhv
# asdlasdg

try:
    # Fetch data
    # group_by='ticker' ensures we get a MultiIndex if we fetch multiple tickers, 
    # but strictly 'Close' simplifies the structure usually.
    # auto_adjust=False ensures we get the raw Close (or Adj Close if we requested it explicitly). 
    # We will just download everything and select 'Close'.
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Check if data is empty
    if data.empty:
        print("No data fetched. Please check your internet connection or ticker symbols.")
    else:
        # Display first few rows
        print("\nFirst 5 rows of fetched data:")
        print(data.head())
        
        # Save to CSV
        output_file = "top_10_stocks_2015_to_present.csv"
        data.to_csv(output_file)
        print(f"\nSuccessfully saved data to {output_file}")
        
except Exception as e:
    print(f"An error occurred: {e}")
