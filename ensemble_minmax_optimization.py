#!/usr/bin/env python3
# coding: utf-8

"""
ML9(Guard) + QV Ensemble Min-Max Optimization
Objective: Maximize the minimum Sharpe across rolling windows
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TRADING_DAYS = 252

def calculate_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio"""
    if returns.empty or returns.std() == 0:
        return 0.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(TRADING_DAYS)
    return sharpe

def calculate_metrics(returns: pd.Series) -> dict:
    """Calculate performance metrics"""
    if returns.empty or returns.std() == 0:
        return {
            'sharpe': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
        }
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(TRADING_DAYS)
    annual_return = mean_ret * TRADING_DAYS
    annual_vol = std_ret * np.sqrt(TRADING_DAYS)
    
    cum_ret = (1.0 + returns).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
    }

def create_ensemble(ml9_returns: pd.Series, qv_returns: pd.Series, 
                   w_ml9: float, w_qv: float) -> pd.Series:
    """Create ensemble returns with given weights"""
    # Normalize indices to remove timezone and time component
    ml9_returns = ml9_returns.copy()
    qv_returns = qv_returns.copy()
    ml9_returns.index = pd.to_datetime(ml9_returns.index).normalize()
    qv_returns.index = pd.to_datetime(qv_returns.index).normalize()
    
    # Align indices
    common_idx = ml9_returns.index.intersection(qv_returns.index)
    ml9_aligned = ml9_returns.loc[common_idx]
    qv_aligned = qv_returns.loc[common_idx]
    
    ensemble = w_ml9 * ml9_aligned + w_qv * qv_aligned
    return ensemble

def calculate_rolling_window_sharpe(returns: pd.Series, n_windows: int = 3) -> List[float]:
    """Calculate Sharpe for each rolling window"""
    window_size = len(returns) // n_windows
    sharpes = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        if i == n_windows - 1:
            # Last window takes all remaining data
            window_returns = returns.iloc[start_idx:]
        else:
            end_idx = (i + 1) * window_size
            window_returns = returns.iloc[start_idx:end_idx]
        
        sharpe = calculate_sharpe(window_returns)
        sharpes.append(sharpe)
    
    return sharpes

def optimize_ensemble(ml9_returns: pd.Series, qv_returns: pd.Series, 
                     n_windows: int = 3) -> Tuple[float, float, dict]:
    """
    Optimize ensemble weights using min-max objective
    Returns: (best_w_ml9, best_w_qv, results_dict)
    """
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MIN-MAX OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Objective: Maximize min(Sharpe) across {n_windows} windows")
    print(f"ML9(Guard) returns: {len(ml9_returns)} days")
    print(f"QV returns: {len(qv_returns)} days")
    
    # Grid search
    weights_ml9 = np.arange(0.0, 1.1, 0.1)
    results = []
    
    print(f"\nTesting {len(weights_ml9)} weight combinations...")
    
    for w_ml9 in weights_ml9:
        w_qv = 1.0 - w_ml9
        
        # Create ensemble
        ensemble_returns = create_ensemble(ml9_returns, qv_returns, w_ml9, w_qv)
        
        # Calculate window Sharpes
        window_sharpes = calculate_rolling_window_sharpe(ensemble_returns, n_windows)
        min_sharpe = min(window_sharpes)
        mean_sharpe = np.mean(window_sharpes)
        
        # Overall metrics
        overall_metrics = calculate_metrics(ensemble_returns)
        
        results.append({
            'w_ml9': w_ml9,
            'w_qv': w_qv,
            'min_sharpe': min_sharpe,
            'mean_sharpe': mean_sharpe,
            'window_sharpes': window_sharpes,
            'overall_sharpe': overall_metrics['sharpe'],
            'overall_return': overall_metrics['annual_return'],
            'overall_volatility': overall_metrics['annual_volatility'],
            'overall_mdd': overall_metrics['max_drawdown'],
            'overall_win_rate': overall_metrics['win_rate'],
        })
        
        print(f"  w_ml9={w_ml9:.1f}, w_qv={w_qv:.1f}: "
              f"min_sharpe={min_sharpe:.3f}, mean_sharpe={mean_sharpe:.3f}, "
              f"overall_sharpe={overall_metrics['sharpe']:.3f}")
    
    # Find best weights
    best_result = max(results, key=lambda x: x['min_sharpe'])
    best_w_ml9 = best_result['w_ml9']
    best_w_qv = best_result['w_qv']
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION RESULTS")
    print(f"{'='*80}")
    print(f"Best weights: ML9(Guard)={best_w_ml9:.1f}, QV={best_w_qv:.1f}")
    print(f"Min Sharpe: {best_result['min_sharpe']:.3f}")
    print(f"Mean Sharpe: {best_result['mean_sharpe']:.3f}")
    print(f"Overall Sharpe: {best_result['overall_sharpe']:.3f}")
    print(f"Window Sharpes: {[f'{s:.3f}' for s in best_result['window_sharpes']]}")
    print(f"\nOverall Performance:")
    print(f"  Annual Return: {best_result['overall_return']*100:.2f}%")
    print(f"  Annual Volatility: {best_result['overall_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {best_result['overall_mdd']*100:.2f}%")
    print(f"  Win Rate: {best_result['overall_win_rate']*100:.2f}%")
    
    return best_w_ml9, best_w_qv, results

def plot_optimization_results(results: List[dict], output_path: Path):
    """Plot optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    weights_ml9 = [r['w_ml9'] for r in results]
    min_sharpes = [r['min_sharpe'] for r in results]
    mean_sharpes = [r['mean_sharpe'] for r in results]
    overall_sharpes = [r['overall_sharpe'] for r in results]
    overall_returns = [r['overall_return'] * 100 for r in results]
    overall_mdds = [r['overall_mdd'] * 100 for r in results]
    
    # Plot 1: Sharpe Ratios
    axes[0, 0].plot(weights_ml9, min_sharpes, 'o-', label='Min Sharpe', linewidth=2)
    axes[0, 0].plot(weights_ml9, mean_sharpes, 's-', label='Mean Sharpe', linewidth=2)
    axes[0, 0].plot(weights_ml9, overall_sharpes, '^-', label='Overall Sharpe', linewidth=2)
    axes[0, 0].set_xlabel('ML9(Guard) 가중치')
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('앙상블 가중치별 Sharpe Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Annual Return
    axes[0, 1].plot(weights_ml9, overall_returns, 'o-', color='green', linewidth=2)
    axes[0, 1].set_xlabel('ML9(Guard) 가중치')
    axes[0, 1].set_ylabel('연간 수익률 (%)')
    axes[0, 1].set_title('앙상블 가중치별 연간 수익률')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Max Drawdown
    axes[1, 0].plot(weights_ml9, overall_mdds, 'o-', color='red', linewidth=2)
    axes[1, 0].set_xlabel('ML9(Guard) 가중치')
    axes[1, 0].set_ylabel('최대 낙폭 (%)')
    axes[1, 0].set_title('앙상블 가중치별 최대 낙폭')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Window Sharpes for best result
    best_result = max(results, key=lambda x: x['min_sharpe'])
    window_sharpes = best_result['window_sharpes']
    axes[1, 1].bar(range(len(window_sharpes)), window_sharpes, color='steelblue')
    axes[1, 1].axhline(y=best_result['min_sharpe'], color='red', linestyle='--', 
                       label=f'Min Sharpe: {best_result["min_sharpe"]:.3f}')
    axes[1, 1].set_xlabel('윈도우')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].set_title(f'최적 가중치 (ML9={best_result["w_ml9"]:.1f}) 윈도우별 Sharpe')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Optimization plot saved to {output_path}")

def main():
    print(f"\n{'='*80}")
    print(f"ML9(Guard) + QV ENSEMBLE OPTIMIZATION")
    print(f"{'='*80}")
    
    # Load returns
    ml9_guard_returns = pd.read_csv(RESULTS_DIR / "ml9_guard_returns.csv", 
                                    index_col=0, parse_dates=True).iloc[:, 0]
    qv_returns = pd.read_csv(RESULTS_DIR / "qv_returns.csv", 
                            index_col=0, parse_dates=True).iloc[:, 0]
    
    print(f"Loaded returns:")
    print(f"  ML9(Guard): {len(ml9_guard_returns)} days ({ml9_guard_returns.index.min().date()} to {ml9_guard_returns.index.max().date()})")
    print(f"  QV: {len(qv_returns)} days ({qv_returns.index.min().date()} to {qv_returns.index.max().date()})")
    
    # Optimize
    best_w_ml9, best_w_qv, results = optimize_ensemble(ml9_guard_returns, qv_returns, n_windows=3)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "ensemble_optimization_results.csv", index=False)
    print(f"\n✓ Results saved to {RESULTS_DIR / 'ensemble_optimization_results.csv'}")
    
    # Plot
    plot_optimization_results(results, RESULTS_DIR / "ensemble_optimization_plot.png")
    
    # Save best ensemble returns
    best_ensemble_returns = create_ensemble(ml9_guard_returns, qv_returns, best_w_ml9, best_w_qv)
    best_ensemble_returns.to_csv(RESULTS_DIR / "ensemble_best_returns.csv")
    
    # Save best metrics
    best_result = max(results, key=lambda x: x['min_sharpe'])
    best_metrics = {
        'w_ml9': best_w_ml9,
        'w_qv': best_w_qv,
        'min_sharpe': best_result['min_sharpe'],
        'mean_sharpe': best_result['mean_sharpe'],
        'window_sharpes': best_result['window_sharpes'],
        'overall_sharpe': best_result['overall_sharpe'],
        'overall_annual_return': best_result['overall_return'],
        'overall_annual_volatility': best_result['overall_volatility'],
        'overall_max_drawdown': best_result['overall_mdd'],
        'overall_win_rate': best_result['overall_win_rate'],
    }
    
    with open(RESULTS_DIR / "ensemble_best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)
    
    print(f"\n✓ Best ensemble metrics saved to {RESULTS_DIR / 'ensemble_best_metrics.json'}")
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
