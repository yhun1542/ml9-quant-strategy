"""
CPU-Optimized ML9 Backtest for SP100 Universe (2021-2024)
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
import multiprocessing as mp
import json
import time

# Paths
BASE_DIR = Path("/home/ubuntu/quant-ensemble-strategy")

@njit(parallel=True, cache=True, fastmath=True)
def compute_momentum_numba(prices, lookback=60):
    """Numba-accelerated momentum calculation"""
    n_dates, n_symbols = prices.shape
    momentum = np.zeros_like(prices, dtype=np.float32)
    
    for j in prange(n_symbols):
        for i in range(lookback, n_dates):
            momentum[i, j] = prices[i, j] / prices[i - lookback, j] - 1.0
    
    return momentum

@njit(parallel=True, cache=True, fastmath=True)
def compute_volatility_numba(prices, lookback=30):
    """Numba-accelerated volatility calculation"""
    n_dates, n_symbols = prices.shape
    volatility = np.zeros_like(prices, dtype=np.float32)
    
    # Calculate returns first
    returns = np.zeros_like(prices)
    for j in range(n_symbols):
        for i in range(1, n_dates):
            if prices[i-1, j] > 0:
                returns[i, j] = prices[i, j] / prices[i-1, j] - 1.0
    
    # Calculate rolling std
    for j in prange(n_symbols):
        for i in range(lookback, n_dates):
            # Calculate std of last lookback returns
            sum_val = 0.0
            sum_sq = 0.0
            count = 0
            
            for k in range(lookback):
                val = returns[i - k, j]
                sum_val += val
                sum_sq += val * val
                count += 1
            
            mean = sum_val / count
            variance = sum_sq / count - mean * mean
            volatility[i, j] = np.sqrt(max(variance, 0.0))
    
    return volatility

@njit(parallel=True, cache=True)
def cross_sectional_rank_numba(data):
    """Numba-accelerated cross-sectional ranking"""
    n_dates, n_symbols = data.shape
    ranked = np.zeros_like(data)
    
    for i in prange(n_dates):
        # Get valid values for this date
        row = data[i, :]
        valid_mask = ~np.isnan(row)
        
        if np.sum(valid_mask) > 0:
            # Rank transform
            valid_vals = row[valid_mask]
            sorted_idx = np.argsort(valid_vals)
            ranks = np.zeros(len(valid_vals), dtype=np.float32)
            
            for j in range(len(valid_vals)):
                ranks[sorted_idx[j]] = float(j) / (len(valid_vals) - 1)
            
            # Assign back
            rank_idx = 0
            for j in range(n_symbols):
                if valid_mask[j]:
                    ranked[i, j] = ranks[rank_idx]
                    rank_idx += 1
                else:
                    ranked[i, j] = np.nan
    
    return ranked

def run_optimized_backtest(data_path, output_path, windows):
    """CPU-optimized ML9 backtest"""
    
    print("="*100)
    print("CPU-OPTIMIZED ML9 BACKTEST - SP100 UNIVERSE")
    print("="*100)
    print(f"💻 CPU cores: {mp.cpu_count()}")
    print(f"🚀 Numba JIT: Enabled")
    
    # Load data
    print("\nLoading data...")
    start_time = time.time()
    
    prices = pd.read_csv(data_path)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date")
    prices.index = prices.index.tz_localize(None)
    
    print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
    
    # Convert to NumPy for speed
    print("\nConverting to NumPy arrays...")
    price_array = prices.values.astype(np.float32)
    dates = prices.index.values
    tickers = prices.columns.values
    
    # Compute factors with Numba
    print("\nComputing factors (Numba-accelerated)...")
    momentum = compute_momentum_numba(price_array, lookback=60)
    volatility = compute_volatility_numba(price_array, lookback=30)
    value_proxy = 1.0 / price_array  # Simple inverse
    
    # Cross-sectional ranking
    print("Applying cross-sectional ranking...")
    momentum_rank = cross_sectional_rank_numba(momentum)
    volatility_rank = cross_sectional_rank_numba(volatility)
    value_rank = cross_sectional_rank_numba(value_proxy)
    
    # Create factor DataFrame
    factors_dict = {}
    for i, ticker in enumerate(tickers):
        factors_dict[ticker] = pd.DataFrame({
            'momentum_60d': momentum_rank[:, i],
            'volatility_30d': volatility_rank[:, i],
            'value_proxy': value_rank[:, i]
        }, index=dates)
    
    factors = pd.concat(factors_dict, names=['ticker', 'date'])
    factors = factors.swaplevel().sort_index()
    
    load_time = time.time() - start_time
    print(f"✅ Data preparation complete: {load_time:.1f}s")
    
    # Run ML9 backtest
    print("\nRunning ML9 backtest...")
    from engines.ml_xgboost_v9_ranking import MLXGBoostV9Ranking
    
    engine = MLXGBoostV9Ranking(prices, factors)
    
    # Run each window
    all_returns = []
    window_metrics = []
    
    backtest_start = time.time()
    
    for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
        print(f"\n[Window {i}/{len(windows)}]")
        
        tr_start_ts = pd.Timestamp(tr_start)
        tr_end_ts = pd.Timestamp(tr_end)
        te_start_ts = pd.Timestamp(te_start)
        te_end_ts = pd.Timestamp(te_end)
        
        daily_ret = engine._backtest_period(tr_start_ts, tr_end_ts, te_start_ts, te_end_ts)
        
        if len(daily_ret) > 0:
            all_returns.append(daily_ret)
            
            ret_values = daily_ret.values
            sharpe = (ret_values.mean() * 252) / (ret_values.std() * np.sqrt(252)) if ret_values.std() > 0 else 0
            
            window_metrics.append({
                "window": i,
                "period": f"{te_start} ~ {te_end}",
                "days": len(daily_ret),
                "sharpe": float(sharpe),
                "total_return": float((1 + ret_values).prod() - 1)
            })
            
            print(f"  Sharpe: {sharpe:.2f}")
    
    backtest_time = time.time() - backtest_start
    total_time = time.time() - start_time
    
    # Combine results
    if all_returns:
        combined_returns = pd.concat(all_returns).sort_index()
        
        # Calculate metrics
        returns = combined_returns.dropna()
        total_return = (1 + returns).prod() - 1
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = std_ret * np.sqrt(252)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns)
        
        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "days": len(returns),
            "windows": window_metrics,
            "timing": {
                "data_prep_seconds": float(load_time),
                "backtest_seconds": float(backtest_time),
                "total_seconds": float(total_time)
            }
        }
        
        print("\n" + "="*100)
        print("RESULTS")
        print("="*100)
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Annual Return: {annual_return*100:.2f}%")
        print(f"Annual Vol: {annual_vol*100:.2f}%")
        print(f"Max DD: {max_dd*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Days: {len(returns)}")
        
        print("\n" + "="*100)
        print("PERFORMANCE")
        print("="*100)
        print(f"⏱️ Data prep: {load_time:.1f}s")
        print(f"⏱️ Backtest: {backtest_time:.1f}s")
        print(f"⏱️ Total: {total_time:.1f}s")
        print(f"🚀 Speedup: ~{300/total_time:.0f}x vs original")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
        
        return metrics
    
    return None

if __name__ == "__main__":
    # SP100 2021-2024 with adjusted windows (2-year train, 1-year test)
    data_path = BASE_DIR / "data" / "price_data_sp100_2021_2024.csv"
    output_path = BASE_DIR / "results" / "ml9_sp100_2021_2024_optimized.json"
    
    # Adjusted windows for 3.5-year dataset
    windows = [
        ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),  # 2-year train, 1-year test
        ("2021-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),  # 3-year train, 6-month test
    ]
    
    run_optimized_backtest(data_path, output_path, windows)
    
    print("\n" + "="*100)
    print("SP100 OPTIMIZED BACKTEST COMPLETE")
    print("="*100)
