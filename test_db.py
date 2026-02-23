import time
from DB.stock_db_manager import StockDBManager

db = StockDBManager()
db.connect()

print("Fetching log returns...")
start = time.time()
df = db.fetch_log_returns()
print(f"Done in {time.time() - start:.2f}s, shape: {df.shape}")

print("Fetching sp500...")
start = time.time()
df2 = db.fetch_sp500_data()
print(f"Done in {time.time() - start:.2f}s, shape: {df2.shape}")

db.close()
