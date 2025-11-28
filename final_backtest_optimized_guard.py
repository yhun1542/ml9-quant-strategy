#!/usr/bin/env python3
# coding: utf-8

"""
Final Backtest with Optimized Guard Parameters
Return Range: -3.0% ~ 0.0%
Scale Factor: 0.30
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.market_guard_ml9 import ML9MarketConditionGuard
from turbo_cpu_backtest import TurboCPUBacktest

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

def main():
    print("\n" + "="*80)
    print("FINAL BACKTEST WITH OPTIMIZED GUARD PARAMETERS")
    print("="*80)
    
    # 최적 파라미터 로딩
    with open(RESULTS_DIR / "guard_best_params.json", 'r') as f:
        best_params = json.load(f)
    
    print(f"\nOptimized Guard Parameters:")
    print(f"  Return Range: {best_params['return_lower']*100:.1f}% ~ {best_params['return_upper']*100:.1f}%")
    print(f"  Scale Factor: {best_params['scale_factor']:.2f}")
    
    # 데이터 로딩
    print(f"\nLoading data...")
    ml9_returns = pd.read_csv(RESULTS_DIR / "ml9_returns.csv", 
                              index_col=0, parse_dates=True).iloc[:, 0]
    spy_prices = pd.read_csv(DATA_DIR / "spy_prices.csv", 
                            index_col='date', parse_dates=True)
    spy_prices.index = pd.to_datetime(spy_prices.index).normalize()
    spy_series = spy_prices['close']
    
    # Guard 초기화
    print(f"\nInitializing Guard...")
    guard_config = {
        "enabled": True,
        "spx_symbol": "SPY",
        "return_lower": best_params['return_lower'],
        "return_upper": best_params['return_upper'],
        "scale_factor": best_params['scale_factor'],
        "vol_window": 20,
        "use_vol_filter": False,
    }
    
    guard = ML9MarketConditionGuard(guard_config)
    guard.initialize(spy_series)
    
    # Guard 적용
    print(f"\nApplying Guard to ML9 returns...")
    ml9_returns_normalized = ml9_returns.copy()
    ml9_returns_normalized.index = pd.to_datetime(ml9_returns_normalized.index).normalize()
    
    spy_returns = spy_series.pct_change()
    common_idx = ml9_returns_normalized.index.intersection(spy_returns.index)
    
    ml9_aligned = ml9_returns_normalized.loc[common_idx].sort_index()
    spy_aligned = spy_returns.loc[common_idx].sort_index()
    
    # NumPy 배열로 변환
    ml9_arr = ml9_aligned.values
    spy_arr = spy_aligned.values
    
    # Guard 조건
    guard_condition = (spy_arr > best_params['return_lower']) & (spy_arr <= best_params['return_upper'])
    guarded_returns_arr = np.where(guard_condition, ml9_arr * best_params['scale_factor'], ml9_arr)
    guarded_returns = pd.Series(guarded_returns_arr, index=common_idx)
    
    # Guard 활성화 통계
    guard_active_days = guard_condition.sum()
    guard_active_rate = guard_active_days / len(guard_condition) * 100
    print(f"  Guard active: {guard_active_days}/{len(guard_condition)} days ({guard_active_rate:.2f}%)")
    
    # 메트릭 계산
    print(f"\nCalculating metrics...")
    turbo = TurboCPUBacktest()
    
    # Original ML9 metrics
    ml9_metrics = turbo.compute_metrics_fast(ml9_aligned.values)
    
    # Optimized Guard metrics
    guard_metrics = turbo.compute_metrics_fast(guarded_returns.values)
    
    # 결과 출력
    print(f"\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\nML9 (No Guard):")
    print(f"  Sharpe Ratio: {ml9_metrics['sharpe']:.3f}")
    print(f"  Annual Return: {ml9_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {ml9_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {ml9_metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {ml9_metrics['win_rate']*100:.2f}%")
    
    print(f"\nML9 (Optimized Guard):")
    print(f"  Sharpe Ratio: {guard_metrics['sharpe']:.3f} ({(guard_metrics['sharpe']/ml9_metrics['sharpe']-1)*100:+.1f}%)")
    print(f"  Annual Return: {guard_metrics['annual_return']*100:.2f}% ({(guard_metrics['annual_return']-ml9_metrics['annual_return'])*100:+.2f}%)")
    print(f"  Annual Volatility: {guard_metrics['annual_volatility']*100:.2f}% ({(guard_metrics['annual_volatility']/ml9_metrics['annual_volatility']-1)*100:+.1f}%)")
    print(f"  Max Drawdown: {guard_metrics['max_drawdown']*100:.2f}% ({(guard_metrics['max_drawdown']-ml9_metrics['max_drawdown'])*100:+.2f}%)")
    print(f"  Win Rate: {guard_metrics['win_rate']*100:.2f}% ({(guard_metrics['win_rate']-ml9_metrics['win_rate'])*100:+.2f}%)")
    
    # 윈도우별 Sharpe
    print(f"\nWindow Sharpes (Optimized Guard):")
    for i, sharpe in enumerate(best_params['window_sharpes'], 1):
        print(f"  Window {i}: {sharpe:.3f}")
    print(f"  Min Sharpe: {best_params['min_sharpe']:.3f}")
    
    # 결과 저장
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    guarded_returns.to_csv(RESULTS_DIR / "ml9_optimized_guard_returns.csv")
    print(f"✓ Returns saved to {RESULTS_DIR / 'ml9_optimized_guard_returns.csv'}")
    
    final_metrics = {
        'ml9_no_guard': ml9_metrics,
        'ml9_optimized_guard': guard_metrics,
        'guard_config': guard_config,
        'guard_active_days': int(guard_active_days),
        'guard_active_rate': float(guard_active_rate),
        'window_sharpes': best_params['window_sharpes'],
        'min_sharpe': best_params['min_sharpe'],
    }
    
    with open(RESULTS_DIR / "ml9_optimized_guard_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=4)
    print(f"✓ Metrics saved to {RESULTS_DIR / 'ml9_optimized_guard_metrics.json'}")
    
    print(f"\n" + "="*80)
    print("FINAL BACKTEST COMPLETE ✓")
    print("="*80)
    
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"  Target Sharpe: 2.0+")
    print(f"  Achieved Sharpe: {guard_metrics['sharpe']:.3f}")
    print(f"  Achievement Rate: {guard_metrics['sharpe']/2.0*100:.1f}%")
    print(f"  Status: {'✅ EXCEEDED!' if guard_metrics['sharpe'] >= 2.0 else '❌ NOT MET'}")

if __name__ == "__main__":
    main()
