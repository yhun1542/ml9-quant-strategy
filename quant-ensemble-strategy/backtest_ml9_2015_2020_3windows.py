"""
ML9 Engine Backtest for 2015-2020 with 3 Walk-Forward Windows
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
from pathlib import Path
from engines.ml_xgboost_v9_ranking import MLXGBoostV9Ranking
from utils.factors import compute_all_factors
import json

# Paths
BASE_DIR = Path("/home/ubuntu/quant-ensemble-strategy")
PRICE_DATA_PATH = BASE_DIR / "data" / "price_data_2015_2020_polygon.csv"
OUTPUT_PATH = BASE_DIR / "results" / "ml9_2015_2020_3windows_results.json"

print("="*100)
print("ML9 ENGINE BACKTEST (2015-2020, 3 Windows)")
print("="*100)

# Load price data
print("\nLoading price data...")
prices = pd.read_csv(PRICE_DATA_PATH)
prices["date"] = pd.to_datetime(prices["date"])
prices = prices.set_index("date")
prices.index = prices.index.tz_localize(None)

print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")

# Compute factors
print("\nComputing factors...")
factors = compute_all_factors(prices)
print(f"Factors computed: {len(factors)} days")

# Initialize ML9 engine
print("\nInitializing ML9 engine...")
engine = MLXGBoostV9Ranking(prices, factors)

# Define 3 custom windows
windows = [
    ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),  # Window 1
    ("2015-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),  # Window 2
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),  # Window 3
]

print("\nWalk-Forward Windows:")
for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
    print(f"  Window {i}: Train {tr_start} ~ {tr_end}, Test {te_start} ~ {te_end}")

# Run backtest for each window
all_returns = []
window_metrics = []

for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
    print(f"\n[Window {i}/{len(windows)}]")
    
    tr_start_ts = pd.Timestamp(tr_start)
    tr_end_ts = pd.Timestamp(tr_end)
    te_start_ts = pd.Timestamp(te_start)
    te_end_ts = pd.Timestamp(te_end)
    
    # Run backtest for this window
    daily_ret = engine._backtest_period(tr_start_ts, tr_end_ts, te_start_ts, te_end_ts)
    
    if len(daily_ret) > 0:
        all_returns.append(daily_ret)
        
        # Calculate window metrics
        ret_values = daily_ret.values
        sharpe = (ret_values.mean() * 252) / (ret_values.std() * np.sqrt(252)) if ret_values.std() > 0 else 0
        
        window_metrics.append({
            "window": i,
            "period": f"{te_start} ~ {te_end}",
            "days": len(daily_ret),
            "sharpe": float(sharpe),
            "annual_return": float((1 + ret_values.mean()) ** 252 - 1),
            "total_return": float((1 + ret_values).prod() - 1)
        })
        
        print(f"  Window {i} Sharpe: {sharpe:.2f}")

# Combine all returns
if all_returns:
    combined_returns = pd.concat(all_returns).sort_index()
    
    print(f"\n✅ Backtest complete: {len(combined_returns)} days of returns")
    
    # Calculate overall metrics
    def calculate_metrics(returns):
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = (1 + returns).prod() - 1
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = std_ret * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "days": len(returns),
            "windows": window_metrics
        }
    
    metrics = calculate_metrics(combined_returns)
    
    print("\n" + "="*100)
    print("OVERALL RESULTS (3 Test Years)")
    print("="*100)
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {metrics['annual_vol']*100:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Days: {metrics['days']}")
    
    print("\n" + "="*100)
    print("WINDOW-BY-WINDOW RESULTS")
    print("="*100)
    for wm in window_metrics:
        print(f"Window {wm['window']} ({wm['period']}): Sharpe {wm['sharpe']:.2f}, Return {wm['total_return']*100:.1f}%")
    
    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")
    
else:
    print("\n❌ No valid windows")

print("\n" + "="*100)
print("TEST COMPLETE")
print("="*100)
