"""
ML9 Engine Backtest for 2015-2020 (Period Extension Test)

목표: ML9이 2015-2020 기간에도 양수 Sharpe를 유지하는지 확인
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
OUTPUT_PATH = BASE_DIR / "results" / "ml9_2015_2020_results.json"

print("="*100)
print("ML9 ENGINE BACKTEST (2015-2020)")
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

# Run backtest
print("\nRunning backtest...")
print("This may take 5-10 minutes...")

result = engine.run_walkforward_backtest()

# Parse daily_returns from list of dicts
if isinstance(result['daily_returns'], list) and len(result['daily_returns']) > 0 and isinstance(result['daily_returns'][0], dict):
    dates = [pd.to_datetime(item['date']) for item in result['daily_returns']]
    returns = [item['ret'] for item in result['daily_returns']]
    daily_returns = pd.Series(returns, index=dates)
else:
    daily_returns = pd.Series(result['daily_returns'])

print(f"\n✅ Backtest complete: {len(daily_returns)} days of returns")

# Calculate metrics
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
    }

metrics = calculate_metrics(daily_returns)

print("\n" + "="*100)
print("RESULTS")
print("="*100)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Annual Return: {metrics['annual_return']*100:.2f}%")
print(f"Annual Volatility: {metrics['annual_vol']*100:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
print(f"Total Return: {metrics['total_return']*100:.2f}%")
print(f"Days: {metrics['days']}")

# Save results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Results saved to: {OUTPUT_PATH}")

print("\n" + "="*100)
print("INTERPRETATION")
print("="*100)

if metrics['sharpe_ratio'] >= 1.5:
    print("✅ EXCELLENT: Sharpe >= 1.5 (similar to 2021-2024)")
elif metrics['sharpe_ratio'] >= 1.0:
    print("✅ GOOD: Sharpe >= 1.0 (still strong)")
elif metrics['sharpe_ratio'] >= 0.5:
    print("⚠️ MODERATE: Sharpe >= 0.5 (positive but weaker)")
elif metrics['sharpe_ratio'] > 0:
    print("⚠️ WEAK: Sharpe > 0 (barely positive)")
else:
    print("❌ FAIL: Sharpe <= 0 (negative)")

print("\n" + "="*100)
print("TEST COMPLETE")
print("="*100)
