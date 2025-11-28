#!/usr/bin/env python3
# coding: utf-8

"""
PIT-safe version of download_and_prepare_data()
Removes all rows where calendardate > date (future data leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import from run_all_tests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_tests import Polygon, SF1, SP100_TICKERS, POLYGON_API_KEY, SHARADAR_API_KEY, BASE_DIR

def download_and_prepare_data_pit_safe():
    """
    PIT-safe data preparation
    1. merge_asof for PIT matching (date <= datekey)
    2. Remove rows where calendardate > date (future leak)
    3. Per-ticker ffill (maintain values until next report)
    4. Feature engineering
    5. Fill currentratio NaN with median
    6. Drop rows with missing core columns
    """
    print("\n" + "="*100)
    print("STEP 1: DOWNLOADING AND PREPARING DATA (PIT-safe)")
    print("="*100)

    prices_path = BASE_DIR / "data" / "sp100_prices_raw.csv"
    sf1_path    = BASE_DIR / "data" / "sp100_sf1_raw.csv"
    merged_path = BASE_DIR / "data" / "sp100_merged_data.csv"
    spy_path    = BASE_DIR / "data" / "spy_prices.csv"

    # Skip if already merged
    if merged_path.exists():
        print(f"✓ Loading previously merged data from {merged_path}")
        data = pd.read_csv(merged_path, parse_dates=["date"])
        print(f"  Loaded {len(data)} rows with {data['ticker'].nunique()} tickers")
        
        # SPY data loading
        if spy_path.exists():
            print(f"✓ Loading previously downloaded SPY prices from {spy_path}")
            spy_df = pd.read_csv(spy_path, parse_dates=['date'], index_col='date')
            spy_df.index = spy_df.index.tz_localize(None)
            spy_series = spy_df['close']
        else:
            print("\nDownloading SPY prices for MarketConditionGuard...")
            poly = Polygon(POLYGON_API_KEY)
            spy_series = poly.get_spy_prices("2014-01-01", "2024-12-31")
            spy_df = pd.DataFrame({'close': spy_series})
            spy_df.to_csv(spy_path)
            print(f"✓ Saved SPY prices to {spy_path}")
        
        return data, spy_series

    # -----------------------------
    # 1) Price data loading or download
    # -----------------------------
    if prices_path.exists():
        print(f"✓ Loading prices from {prices_path}")
        prices_df = pd.read_csv(prices_path, parse_dates=["date"])
    else:
        print("\nDownloading daily prices from Polygon...")
        poly = Polygon(POLYGON_API_KEY)
        prices_df = poly.get_daily_prices(SP100_TICKERS, "2014-01-01", "2024-12-31")
        prices_df.to_csv(prices_path, index=False)

    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    prices_df = prices_df.sort_values(["ticker", "date"])

    # Filter required columns
    if "close" not in prices_df.columns:
        raise ValueError("prices_df must contain 'close' column")

    # -----------------------------
    # 2) SF1 fundamental data loading or download
    # -----------------------------
    if sf1_path.exists():
        print(f"✓ Loading SF1 data from {sf1_path}")
        sf1_df = pd.read_csv(sf1_path, parse_dates=["datekey", "reportperiod", "calendardate"])
    else:
        print("\nDownloading SF1 fundamental data from Sharadar...")
        sf1 = SF1(SHARADAR_API_KEY)
        sf1_df = sf1.get_sf1_data(SP100_TICKERS, "2014-01-01", "2024-12-31")
        sf1_df.to_csv(sf1_path, index=False)

    # SF1 cleanup
    for col in ["datekey", "calendardate"]:
        sf1_df[col] = pd.to_datetime(sf1_df[col]).dt.tz_localize(None)
    sf1_df = sf1_df.sort_values(["ticker", "datekey"])

    # Keep only needed columns
    needed_cols = [
        "ticker", "datekey", "calendardate",
        "pe", "pb", "ps", "evebitda",
        "roe", "ebitdamargin", "de", "currentratio",
    ]
    missing = [c for c in needed_cols if c not in sf1_df.columns]
    if missing:
        raise ValueError(f"Missing SF1 columns: {missing}")
    sf1_df = sf1_df[needed_cols]

    print("\nPIT merge via merge_asof (date <= datekey, then calendardate <= date)...")

    # -----------------------------
    # 3) PIT merge_asof
    # -----------------------------
    prices_sorted = prices_df.sort_values(["ticker", "date"])
    sf1_sorted    = sf1_df.sort_values(["ticker", "datekey"])

    # Per-ticker merge_asof
    merged_list = []
    for tkr, px_tkr in prices_sorted.groupby("ticker"):
        sf1_tkr = sf1_sorted[sf1_sorted["ticker"] == tkr]
        if sf1_tkr.empty:
            # No fundamental data for this ticker
            px_tkr_merged = px_tkr.copy()
            for col in needed_cols:
                if col not in ["ticker", "datekey", "calendardate"]:
                    px_tkr_merged[col] = np.nan
            merged_list.append(px_tkr_merged)
            continue

        px_tkr = px_tkr.sort_values("date")
        sf1_tkr = sf1_tkr.sort_values("datekey")

        # merge_asof: use SF1 data from previous or same datekey
        m = pd.merge_asof(
            px_tkr,
            sf1_tkr,
            left_on="date",
            right_on="datekey",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_list.append(m)

    data = pd.concat(merged_list, ignore_index=True)
    print(f"  After merge_asof: {len(data)} rows")
    
    # Handle ticker column name conflicts
    if 'ticker_x' in data.columns:
        data = data.rename(columns={'ticker_x': 'ticker'})
        if 'ticker_y' in data.columns:
            data = data.drop(columns=['ticker_y'])
    
    # Remove duplicate ticker columns if any
    if 'ticker' in data.columns and data.columns.tolist().count('ticker') > 1:
        # Keep first ticker column, drop duplicates
        cols = data.columns.tolist()
        ticker_indices = [i for i, col in enumerate(cols) if col == 'ticker']
        if len(ticker_indices) > 1:
            data = data.iloc[:, [i for i in range(len(cols)) if i not in ticker_indices[1:]]]

    # -----------------------------
    # 4) 🔧 PIT LEAK REMOVAL: Remove rows where calendardate > date
    # -----------------------------
    before_leak = len(data)
    # Conservative: remove rows where calendardate > date
    leak_mask = data["calendardate"] > data["date"]
    leak_count = leak_mask.sum()
    if leak_count > 0:
        print(f"  🔧 Removing {leak_count} rows where calendardate > date (future leak)")
        data = data[~leak_mask]
    print(f"  After PIT leak filter: {len(data)} rows (removed {before_leak - len(data)})")

    # -----------------------------
    # 5) Per-ticker ffill (maintain values until next report)
    # -----------------------------
    data = data.sort_values(["ticker", "date"])
    data = data.groupby("ticker", group_keys=False).apply(lambda x: x.ffill())
    print(f"  After per-ticker ffill: {len(data)} rows")

    # -----------------------------
    # 6) Feature Engineering
    # -----------------------------
    print("\nFeature Engineering (momentum, volatility, value_proxy)...")

    data["momentum_60d"] = data.groupby("ticker")["close"].pct_change(60)

    data["volatility_30d"] = data.groupby("ticker")["close"].transform(
        lambda x: x.pct_change().rolling(30).std()
    )

    # Simple value proxy: pe
    data["value_proxy"] = data["pe"]

    print(f"  Before dropna: {len(data)} rows")
    for col in [
        "momentum_60d", "volatility_30d", "value_proxy",
        "pe", "pb", "ps", "evebitda", "roe", "ebitdamargin", "de", "currentratio"
    ]:
        na_cnt = data[col].isna().sum()
        print(f"    {col}: {na_cnt} NaNs")

    # Fill currentratio NaN with median (stabilize Quality score)
    if "currentratio" in data.columns:
        na_count = data["currentratio"].isna().sum()
        if na_count > 0:
            median_cr = data["currentratio"].median()
            print(f"  Filling {na_count} NaNs in currentratio with median={median_cr:.4f}")
            data["currentratio"] = data["currentratio"].fillna(median_cr)

    # Keep only rows with all core columns
    data = data.dropna(subset=[
        "momentum_60d", "volatility_30d", "value_proxy",
        "pe", "pb", "ps", "evebitda", "roe", "ebitdamargin", "de", "currentratio",
    ])
    print(f"  After dropna: {len(data)} rows with {data['ticker'].nunique()} tickers")

    # Final save
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(merged_path, index=False)
    print(f"\n✓ Saved PIT-safe merged data to {merged_path}")

    # SPY data loading
    if spy_path.exists():
        print(f"✓ Loading previously downloaded SPY prices from {spy_path}")
        spy_df = pd.read_csv(spy_path, parse_dates=['date'], index_col='date')
        spy_df.index = spy_df.index.tz_localize(None)
        spy_series = spy_df['close']
    else:
        print("\nDownloading SPY prices for MarketConditionGuard...")
        poly = Polygon(POLYGON_API_KEY)
        spy_series = poly.get_spy_prices("2014-01-01", "2024-12-31")
        spy_df = pd.DataFrame({'close': spy_series})
        spy_df.to_csv(spy_path)
        print(f"✓ Saved SPY prices to {spy_path}")

    return data, spy_series

if __name__ == "__main__":
    data, spy_series = download_and_prepare_data_pit_safe()
    print(f"\n✓ PIT-safe data preparation complete!")
    print(f"  Total rows: {len(data)}")
    print(f"  Tickers: {data['ticker'].nunique()}")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
