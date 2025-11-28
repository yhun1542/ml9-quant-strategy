"""
Merge 2020 and 2021-2024 price data
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main():
    print("="*100)
    print("MERGING PRICE DATA")
    print("="*100)

    # Load data
    df_2020 = pd.read_csv(BASE_DIR / "data" / "sp100_2020.csv", index_col=0, parse_dates=True)
    df_2021_2024 = pd.read_csv(BASE_DIR / "data" / "sp100_2021_2024.csv", index_col=0, parse_dates=True)

    # Merge
    df = pd.concat([df_2020, df_2021_2024])
    df = df.sort_index()

    # Save to CSV
    out_path = BASE_DIR / "data" / "sp100_2020_2024.csv"
    df.to_csv(out_path)

    print(f"\n✅ Merged data saved to: {out_path}")

    print("\n" + "="*100)
    print("MERGE COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
