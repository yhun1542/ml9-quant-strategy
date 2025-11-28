"""
Extract ML9 daily returns for 2020-2024 (full version)
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backtest_ml9_sp100_2020_2024 import run_backtest


def main():
    print("="*100)
    print("EXTRACTING ML9 DAILY RETURNS (2020-2024)")
    print("="*100)
    
    # Run backtest to get results
    results = run_backtest()
    
    # Extract daily returns
    daily_returns = results["daily_returns"]
    
    # Save to JSON
    out_data = {
        "daily_returns": {
            "index": daily_returns.index.strftime("%Y-%m-%d").tolist(),
            "values": daily_returns.values.tolist(),
        }
    }
    
    out_path = BASE_DIR / "results" / "ml9_daily_returns_2020_2024.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print(f"\n✅ Daily returns saved to: {out_path}")
    
    print("\n" + "="*100)
    print("EXTRACTION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
