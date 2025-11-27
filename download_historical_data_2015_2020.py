"""
Download historical price data for 2015-2020 using Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# 30 tickers (same as current data)
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# Date range
START_DATE = "2015-01-01"
END_DATE = "2020-12-31"

print("="*100)
print("DOWNLOADING HISTORICAL DATA (2015-2020)")
print("="*100)

# Download data
print(f"\nDownloading {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")

data = yf.download(
    tickers=TICKERS,
    start=START_DATE,
    end=END_DATE,
    progress=True,
    group_by='ticker'
)

print("\nProcessing data...")

# Extract Adjusted Close prices
price_data = {}
for ticker in TICKERS:
    try:
        if ticker in data.columns.levels[0]:
            price_data[ticker] = data[(ticker, 'Adj Close')]
        else:
            print(f"Warning: {ticker} not found in downloaded data")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Create DataFrame
df = pd.DataFrame(price_data)
df.index.name = 'date'
df = df.reset_index()

# Save to CSV
output_path = Path("/home/ubuntu/quant-ensemble-strategy/data/price_data_2015_2020.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\n✅ Data saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Tickers: {len(df.columns) - 1}")

print("\n" + "="*100)
print("DOWNLOAD COMPLETE")
print("="*100)
