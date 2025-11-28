"""
SP100 2020-2024: Apply optimal Guard parameters
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from modules.market_guard_ml9 import ML9MarketConditionGuard


def load_returns_from_json(path: Path, key: str = "daily_returns") -> pd.Series:
    """Load daily returns from JSON"""
    with open(path, "r") as f:
        data = json.load(f)
    dr = data[key]
    idx = pd.to_datetime(dr["index"])
    vals = dr["values"]
    s = pd.Series(vals, index=idx).sort_index()
    s.name = path.stem
    return s


def load_spx() -> pd.Series:
    """Load SPX price series"""
    df = pd.read_csv(BASE_DIR / "data" / "spx_close.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    s = df.iloc[:, 0]
    s = s.sort_index()
    s.name = "SPX"
    return s


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


def main():
    print("="*100)
    print("SP100 2020-2024: OPTIMAL GUARD APPLICATION")
    print("="*100)
    
    # Load data
    print("\nLoading data...")
    ret_ml9_raw = load_returns_from_json(BASE_DIR / "results" / "ml9_daily_returns_2020_2024.json")
    spx = load_spx()
    
    print(f"ML9 (raw): {len(ret_ml9_raw)} days ({ret_ml9_raw.index[0].date()} ~ {ret_ml9_raw.index[-1].date()})")
    print(f"SPX: {len(spx)} days")
    
    # Apply optimal Guard
    print("\nApplying optimal Guard...")
    optimal_config = {
        "enabled": True,
        "return_lower": -0.03,
        "return_upper": 0.0,
        "scale_factor": 0.3,
    }
    
    guard = ML9MarketConditionGuard(optimal_config)
    guard.initialize(spx)
    
    ml9_scales = guard.get_scale_series(ret_ml9_raw.index)
    ret_ml9_guard = ret_ml9_raw * ml9_scales
    ret_ml9_guard.name = "ml9_optimal_guard"
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_no_guard = calculate_metrics(ret_ml9_raw)
    metrics_guard = calculate_metrics(ret_ml9_guard)
    
    # Print comparison
    print("\n" + "="*100)
    print("RESULTS COMPARISON")
    print("="*100)
    
    print(f"\n{'Metric':<20} {'No Guard':>15} {'Optimal Guard':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<20} {metrics_no_guard['sharpe']:>15.2f} {metrics_guard['sharpe']:>15.2f} {metrics_guard['sharpe']-metrics_no_guard['sharpe']:>+15.2f}")
    print(f"{'Annual Return':<20} {metrics_no_guard['annual_return']*100:>14.2f}% {metrics_guard['annual_return']*100:>14.2f}% {(metrics_guard['annual_return']-metrics_no_guard['annual_return'])*100:>+14.2f}%")
    print(f"{'Annual Vol':<20} {metrics_no_guard['annual_vol']*100:>14.2f}% {metrics_guard['annual_vol']*100:>14.2f}% {(metrics_guard['annual_vol']-metrics_no_guard['annual_vol'])*100:>+14.2f}%")
    print(f"{'Max DD':<20} {metrics_no_guard['max_dd']*100:>14.2f}% {metrics_guard['max_dd']*100:>14.2f}% {(metrics_guard['max_dd']-metrics_no_guard['max_dd'])*100:>+14.2f}%")
    print(f"{'Win Rate':<20} {metrics_no_guard['win_rate']*100:>14.2f}% {metrics_guard['win_rate']*100:>14.2f}% {(metrics_guard['win_rate']-metrics_no_guard['win_rate'])*100:>+14.2f}%")
    
    # Save results
    out_data = {
        "no_guard": metrics_no_guard,
        "optimal_guard": metrics_guard,
        "guard_config": optimal_config,
    }
    
    out_path = BASE_DIR / "results" / "ml9_optimal_guard_sp100_2020_2024.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")
    
    print("\n" + "="*100)
    print("OPTIMAL GUARD TEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
