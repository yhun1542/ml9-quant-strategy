"""
ML9 SP100 2020-2024 3-Window Backtest
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from engines.ml_xgboost_v9_ranking import MLXGBoostV9Ranking as ML9Engine


def load_data() -> pd.DataFrame:
    """Load and merge 2020-2024 data"""
    df_2020 = pd.read_csv(BASE_DIR / "data" / "sp100_2020.csv", index_col=0, parse_dates=True)
    df_2021_2024 = pd.read_csv(BASE_DIR / "data" / "sp100_2021_2024.csv", index_col=0, parse_dates=True)
    df = pd.concat([df_2020, df_2021_2024])
    df = df.sort_index()
    return df


def run_backtest() -> dict:
    """Run the 3-window backtest"""
    df = load_data()
    
    windows = [
        {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
        {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-06-30"},
        {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    ]
    
    all_returns = []
    
    for i, w in enumerate(windows):
        print(f"\n--- Window {i+1} ---")
        print(f"Train: ... - {w['train_end']}")
        print(f"Test: {w['test_start']} - {w['test_end']}")
        
        price_data = df
        factor_data = pd.read_csv(BASE_DIR / "data" / "factors_sp100_2020_2024.csv", index_col=["date", "ticker"], parse_dates=True)
        engine = ML9Engine(price_data, factor_data)
        
        result = engine.run_walkforward_backtest()
        
        all_returns.append(result[0]["daily_returns"])
        
    # Combine returns
    daily_returns = pd.concat(all_returns).sort_index()
    daily_returns = daily_returns[~daily_returns.index.duplicated(keep='first')]    
    # Calculate metrics
    metrics = calculate_metrics(daily_returns)
    
    return {
        "daily_returns": daily_returns,
        "metrics": metrics,
    }


def calculate_metrics(ret_daily: pd.Series) -> dict:
    """Calculate performance metrics"""
    ret_daily = ret_daily.dropna()
    if ret_daily.empty:
        return {}

    days = (ret_daily.index[-1] - ret_daily.index[0]).days
    years = days / 365.25
    if years <= 0:
        return {}

    cum = (1.0 + ret_daily).prod()
    ann_ret = cum ** (1 / years) - 1.0

    vol_d = ret_daily.std(ddof=0)
    ann_vol = vol_d * np.sqrt(252.0)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    wealth = (1 + ret_daily).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_dd = dd.min()

    win_rate = (ret_daily > 0).sum() / len(ret_daily)

    return {
        "sharpe": float(sharpe),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "n_days": len(ret_daily),
    }


if __name__ == "__main__":
    results = run_backtest()
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
    print(results["metrics"])
