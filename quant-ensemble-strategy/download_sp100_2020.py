"""
Download SP100 data for 2020 from Polygon API
"""

import requests
import pandas as pd
from pathlib import Path
import time

# API Configuration
API_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"
BASE_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

# SP100 tickers (same as before)
SP100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "XOM", "V", "PG", "JPM", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "COST", "AVGO", "KO", "ADBE", "WMT", "MCD", "CSCO", "ACN", "TMO", "LIN",
    "ABT", "NFLX", "DHR", "NKE", "VZ", "CRM", "TXN", "NEE", "ORCL", "PM",
    "WFC", "DIS", "UPS", "BMY", "RTX", "AMGN", "HON", "LOW", "QCOM", "UNP",
    "MS", "COP", "SPGI", "BA", "INTU", "SBUX", "GE", "CAT", "AMD", "PLD",
    "AMAT", "BLK", "DE", "MDT", "LMT", "GILD", "ADP", "ADI", "BKNG", "TJX",
    "ISRG", "CI", "MMC", "VRTX", "SYK", "C", "ZTS", "REGN", "PGR", "MO",
    "CB", "DUK", "SO", "BDX", "EOG", "TGT", "ITW", "USB", "SCHW", "PNC",
    "AON", "BSX", "CME", "GS", "MU", "SLB", "NOC", "MMM", "FI", "ICE"
]

def download_2020_data():
    """Download 2020 data for SP100"""
    
    print("="*80)
    print("DOWNLOADING SP100 2020 DATA")
    print("="*80)
    
    # Date range for 2020
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    all_data = []
    
    # Download each ticker
    for i, ticker in enumerate(SP100_TICKERS, 1):
        print(f"[{i}/{len(SP100_TICKERS)}] Downloading {ticker}...", end=" ")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": API_KEY
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if "results" in data and data["results"]:
                    df = pd.DataFrame(data["results"])
                    df["ticker"] = ticker
                    df["date"] = pd.to_datetime(df["t"], unit="ms")
                    df = df[["date", "ticker", "c"]]  # close price
                    df.columns = ["date", "ticker", "price"]
                    
                    all_data.append(df)
                    print(f"✓ {len(df)} days")
                else:
                    print(f"✗ No data")
            else:
                print(f"✗ Error {response.status_code}")
            
            # Rate limiting
            time.sleep(0.15)
            
        except Exception as e:
            print(f"✗ {str(e)}")
    
    # Combine all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        # Pivot to wide format
        pivot = combined.pivot(index="date", columns="ticker", values="price")
        pivot = pivot.sort_index()
        
        # Save
        output_path = Path("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp100_2020.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(output_path)
        
        print("\n" + "="*80)
        print("DOWNLOAD COMPLETE")
        print("="*80)
        print(f"Date range: {pivot.index.min()} to {pivot.index.max()}")
        print(f"Days: {len(pivot)}")
        print(f"Tickers: {len(pivot.columns)}")
        print(f"Saved to: {output_path}")
        
        return pivot
    
    return None

if __name__ == "__main__":
    download_2020_data()
