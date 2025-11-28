"""
Generate factors for SP100 2020-2024
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main():
    print("="*100)
    print("GENERATING FACTORS FOR SP100 (2020-2024)")
    print("="*100)

    # Load price data
    price_data = pd.read_csv(BASE_DIR / "data" / "sp100_2020_2024.csv", index_col=0, parse_dates=True)

    # Calculate factors
    factors = pd.DataFrame()
    for ticker in price_data.columns:
        df = pd.DataFrame(price_data[ticker])
        df.columns = ["price"]
        df["ticker"] = ticker
        df["momentum_60d"] = df["price"].pct_change(60)
        df["volatility_30d"] = df["price"].pct_change().rolling(30).std()
        df["value_proxy"] = 1 / df["price"]  # Simple value proxy
        factors = pd.concat([factors, df])

    factors = factors.reset_index().set_index(["date", "ticker"])
    factors = factors.drop(columns=["price"])
    factors = factors.dropna()

    # Save to CSV
    out_path = BASE_DIR / "data" / "factors_sp100_2020_2024.csv"
    factors.to_csv(out_path)

    print(f"\n✅ Factors saved to: {out_path}")

    print("\n" + "="*100)
    print("FACTOR GENERATION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
