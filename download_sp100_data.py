"""
Download SP100 price data for 2021-2024 using Polygon API
"""

import requests
import pandas as pd
from pathlib import Path
import time

# Polygon API key
API_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"

# SP100 tickers (top 100 S&P 500 companies by market cap)
SP100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V",
    "UNH", "XOM", "JPM", "JNJ", "WMT", "MA", "PG", "AVGO", "HD", "CVX",
    "MRK", "COST", "ABBV", "PEP", "KO", "ADBE", "CRM", "NFLX", "ACN", "TMO",
    "MCD", "CSCO", "ABT", "LIN", "DHR", "NKE", "TXN", "PM", "DIS", "VZ",
    "ORCL", "INTC", "CMCSA", "AMD", "WFC", "NEE", "COP", "UNP", "QCOM", "IBM",
    "RTX", "UPS", "LOW", "HON", "INTU", "AMGN", "MS", "CAT", "BA", "GE",
    "SPGI", "ELV", "SBUX", "PLD", "BKNG", "BLK", "AXP", "GILD", "DE", "MDT",
    "TJX", "ADP", "MDLZ", "ADI", "VRTX", "SYK", "ISRG", "REGN", "MMC", "PGR",
    "C", "CI", "CB", "LRCX", "ZTS", "SCHW", "SO", "DUK", "NOC", "BMY",
    "BSX", "ETN", "CME", "MU", "PNC", "EOG", "FI", "AMAT", "SLB", "USB"
]

# Date range
START_DATE = "2021-06-30"
END_DATE = "2024-12-31"

print("="*100)
print(f"DOWNLOADING SP100 DATA ({len(SP100_TICKERS)} tickers)")
print("="*100)

# Download data for each ticker
all_data = {}

for i, ticker in enumerate(SP100_TICKERS, 1):
    print(f"\n[{i}/{len(SP100_TICKERS)}] {ticker}...", end=" ")
    
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
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date')['c']  # 'c' is close price
            df.name = ticker
            
            all_data[ticker] = df
            
            print(f"✅ {len(df)} days")
        else:
            print(f"⚠️ No data")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Rate limiting
    time.sleep(0.15)

# Create DataFrame
if all_data:
    df = pd.DataFrame(all_data)
    df.index.name = 'date'
    df = df.sort_index()
    df = df.reset_index()
    
    # Save to CSV
    output_path = Path("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp100_2021_2024.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"✅ Data saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Tickers: {len(df.columns) - 1}")
    print(f"Missing: {set(SP100_TICKERS) - set(df.columns)}")
else:
    print("\n❌ No data downloaded")

print("\n" + "="*100)
print("DOWNLOAD COMPLETE")
print("="*100)
