"""
Polygon.io data loader
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import requests


class Polygon:
    """Polygon.io API client"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    def get_daily_prices(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get daily prices for a list of tickers"""
        all_data = []
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Downloading {ticker}...", end=" ")
            try:
                data = self._get_daily_prices_for_ticker(ticker, start_date, end_date)
                if not data.empty:
                    all_data.append(data)
                    print(f"✓ {len(data)} days")
                else:
                    print("✗ No data")
            except Exception as e:
                print(f"✗ {str(e)}")
            time.sleep(0.15)  # Rate limiting

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)
        pivot = combined.pivot(index="date", columns="ticker", values="price")
        pivot = pivot.sort_index()
        return pivot

    def _get_daily_prices_for_ticker(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get daily prices for a single ticker"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df[["date", "ticker", "c"]]  # close price
        df.columns = ["date", "ticker", "price"]
        return df


class Ticker:
    """Represents a stock ticker"""

    def __init__(self, symbol: str):
        self.symbol = symbol

    def __repr__(self) -> str:
        return f"Ticker(symbol=\"{self.symbol}\")"
