"""
Download SP100 price data for 2021-2024
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_loader_polygon import Polygon, Ticker


def get_sp100_tickers() -> list[str]:
    """Get SP100 tickers"""
    return [
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


def main():
    print("="*100)
    print("DOWNLOADING SP100 PRICE DATA (2021-2024)")
    print("="*100)
    
    # Get tickers
    tickers = get_sp100_tickers()
    print(f"Found {len(tickers)} tickers in SP100")
    
    # Initialize Polygon client
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY not set")
    
    poly = Polygon(api_key)
    
    # Download data
    data = poly.get_daily_prices(
        tickers=tickers,
        start_date="2021-01-01",
        end_date="2024-12-31",
    )
    
    # Save to CSV
    out_path = BASE_DIR / "data" / "sp100_2021_2024.csv"
    data.to_csv(out_path)
    
    print(f"\n✅ Data saved to: {out_path}")
    
    print("\n" + "="*100)
    print("DOWNLOAD COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
