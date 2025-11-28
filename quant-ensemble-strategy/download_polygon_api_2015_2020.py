"""
Download historical price data for 2015-2020 using Polygon REST API
"""

import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

# Polygon API key
API_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"

# 30 tickers
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# Date range
START_DATE = "2015-01-01"
END_DATE = "2020-12-31"

print("="*100)
print("DOWNLOADING POLYGON DATA (2015-2020)")
print("="*100)

# Download data for each ticker
all_data = {}

for i, ticker in enumerate(TICKERS, 1):
    print(f"\n[{i}/{len(TICKERS)}] Downloading {ticker}...")
    
    # Polygon aggregates endpoint
    # https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{START_DATE}/{END_DATE}"
    
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "OK" and "results" in data:
            results = data["results"]
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Convert timestamp (milliseconds) to date
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Use close price (adjusted)
            df = df.set_index('date')['c']  # 'c' is close price
            df.name = ticker
            
            all_data[ticker] = df
            
            print(f"  ✅ {len(df)} days downloaded")
        else:
            print(f"  ⚠️ No data: {data.get('status', 'Unknown')}")
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Rate limiting (5 requests per second for free tier)
    time.sleep(0.25)

# Create DataFrame
if all_data:
    df = pd.DataFrame(all_data)
    df.index.name = 'date'
    df = df.sort_index()
    df = df.reset_index()
    
    # Save to CSV
    output_path = Path("/home/ubuntu/quant-ensemble-strategy/data/price_data_2015_2020_polygon.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"✅ Data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {len(df.columns) - 1}")
    print(f"Missing tickers: {set(TICKERS) - set(df.columns)}")
else:
    print("\n❌ No data downloaded")

print("\n" + "="*100)
print("DOWNLOAD COMPLETE")
print("="*100)
