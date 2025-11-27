"""
SP100 2020-2024 Optimized Backtest with 3-year training windows
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

def merge_data():
    """Merge 2020 and 2021-2024 data"""
    print("Merging 2020 and 2021-2024 data...")
    
    # Load 2020 data
    data_2020 = pd.read_csv(BASE_DIR / "data" / "price_data_sp100_2020.csv")
    data_2020["date"] = pd.to_datetime(data_2020["date"])
    data_2020 = data_2020.set_index("date")
    
    # Load 2021-2024 data
    data_2021_2024 = pd.read_csv(BASE_DIR / "data" / "price_data_sp100_2021_2024.csv")
    data_2021_2024["date"] = pd.to_datetime(data_2021_2024["date"])
    data_2021_2024 = data_2021_2024.set_index("date")
    
    # Find common tickers
    common_tickers = list(set(data_2020.columns) & set(data_2021_2024.columns))
    print(f"Common tickers: {len(common_tickers)}")
    
    # Merge
    merged = pd.concat([
        data_2020[common_tickers],
        data_2021_2024[common_tickers]
    ])
    merged = merged.sort_index()
    merged.index = merged.index.tz_localize(None)
    
    print(f"Merged data: {len(merged)} days, {len(merged.columns)} tickers")
    print(f"Date range: {merged.index.min()} to {merged.index.max()}")
    
    return merged

def run_optimized_backtest(prices, output_path, windows):
    """CPU-optimized ML9 backtest"""
    
    print("\n" + "="*100)
    print("CPU-OPTIMIZED ML9 BACKTEST - SP100 2020-2024")
    print("="*100)
    print(f"💻 CPU cores: {mp.cpu_count()}")
    print(f"🚀 Numba JIT: Enabled")
    
    start_time = time.time()
    
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
                "train_period": f"{tr_start} ~ {tr_end}",
                "test_period": f"{te_start} ~ {te_end}",
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
        print("WINDOW DETAILS")
        print("="*100)
        for w in window_metrics:
            print(f"Window {w['window']}: Train {w['train_period']} → Test {w['test_period']}")
            print(f"  Sharpe: {w['sharpe']:.2f}, Return: {w['total_return']*100:.2f}%, Days: {w['days']}")
        
        print("\n" + "="*100)
        print("PERFORMANCE")
        print("="*100)
        print(f"⏱️ Data prep: {load_time:.1f}s")
        print(f"⏱️ Backtest: {backtest_time:.1f}s")
        print(f"⏱️ Total: {total_time:.1f}s")
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
        
        return metrics
    
    return None

if __name__ == "__main__":
    # Merge data
    prices = merge_data()
    
    # 3-window test with 3-year training
    windows = [
        ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),  # 3-year train → 2023
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),  # 4-year train → 2024 H1
        ("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),  # 3-year train → 2024 full
    ]
    
    output_path = BASE_DIR / "results" / "ml9_sp100_2020_2024_3window.json"
    
    run_optimized_backtest(prices, output_path, windows)
    
    print("\n" + "="*100)
    print("SP100 2020-2024 3-WINDOW BACKTEST COMPLETE")
    print("="*100)
