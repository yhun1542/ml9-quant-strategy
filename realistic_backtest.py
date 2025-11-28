#!/usr/bin/env python3
# coding: utf-8

"""
Realistic Backtest - Corrected Logic + Realistic Transaction Costs
1. Guard uses PREVIOUS day SPY returns (no look-ahead bias)
2. Realistic transaction costs for institutional investors
   - Slippage: 3 bps (large-cap, high liquidity)
   - Commission: 0.02% (institutional rate)
   - Total: 0.05% per trade
3. Out-of-sample validation (2024)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turbo_cpu_backtest import TurboCPUBacktest

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

def apply_guard_corrected(ml9_returns: pd.Series, spy_series: pd.Series,
                          return_lower: float, return_upper: float,
                          scale_factor: float) -> pd.Series:
    """Guard 적용 (전일 SPY 수익률 사용)"""
    spy_returns = spy_series.pct_change()
    
    ml9_returns = ml9_returns.copy()
    ml9_returns.index = pd.to_datetime(ml9_returns.index).normalize()
    spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
    
    common_idx = ml9_returns.index.intersection(spy_returns.index)
    ml9_aligned = ml9_returns.loc[common_idx].sort_index()
    spy_aligned = spy_returns.loc[common_idx].sort_index()
    
    # 전일 SPY 수익률 사용
    spy_prev_day = spy_aligned.shift(1)
    guard_condition = (spy_prev_day > return_lower) & (spy_prev_day <= return_upper)
    guarded_returns = np.where(guard_condition, ml9_aligned.values * scale_factor, ml9_aligned.values)
    guarded_returns[0] = ml9_aligned.values[0]
    
    return pd.Series(guarded_returns, index=common_idx)

def apply_transaction_costs(returns: pd.Series, positions_changed: pd.Series,
                            cost_per_trade: float = 0.0005) -> pd.Series:
    """거래 비용 반영 (현실적)"""
    costs = np.where(positions_changed, cost_per_trade, 0.0)
    returns_after_costs = returns - costs
    return pd.Series(returns_after_costs, index=returns.index)

def split_train_test(returns: pd.Series, test_start_date: str = '2024-01-01'):
    """Train/Test 분할"""
    test_start = pd.to_datetime(test_start_date)
    train = returns[returns.index < test_start]
    test = returns[returns.index >= test_start]
    return train, test

def main():
    print("\n" + "="*80)
    print("REALISTIC BACKTEST - CORRECTED LOGIC + REALISTIC COSTS")
    print("="*80)
    
    # 데이터 로딩
    print(f"\nLoading data...")
    ml9_returns = pd.read_csv(RESULTS_DIR / "ml9_returns.csv",
                              index_col=0, parse_dates=True).iloc[:, 0]
    spy_prices = pd.read_csv(DATA_DIR / "spy_prices.csv",
                            index_col='date', parse_dates=True)
    spy_prices.index = pd.to_datetime(spy_prices.index).normalize()
    spy_series = spy_prices['close']
    
    # 최적 파라미터
    best_params = {
        'return_lower': -0.03,
        'return_upper': 0.0,
        'scale_factor': 0.3
    }
    
    print(f"\nOptimized Guard Parameters:")
    print(f"  Return Range: {best_params['return_lower']*100:.1f}% ~ {best_params['return_upper']*100:.1f}%")
    print(f"  Scale Factor: {best_params['scale_factor']:.2f}")
    
    # Guard 적용
    print(f"\n✅ Applying Guard with PREVIOUS day SPY returns...")
    guarded_returns = apply_guard_corrected(
        ml9_returns, spy_series,
        best_params['return_lower'],
        best_params['return_upper'],
        best_params['scale_factor']
    )
    
    ml9_aligned = ml9_returns.loc[guarded_returns.index]
    guard_active = (ml9_aligned != guarded_returns)
    guard_active_days = guard_active.sum()
    guard_active_rate = guard_active_days / len(guard_active) * 100
    print(f"  Guard active: {guard_active_days}/{len(guard_active)} days ({guard_active_rate:.2f}%)")
    
    # 거래 비용 반영 (현실적)
    print(f"\n✅ Applying REALISTIC transaction costs...")
    print(f"  Slippage: 3 bps (large-cap stocks)")
    print(f"  Commission: 0.02% (institutional rate)")
    print(f"  Total cost per trade: 0.05%")
    
    positions_changed = guard_active.copy()
    positions_changed.iloc[0] = True
    
    guarded_returns_with_costs = apply_transaction_costs(
        guarded_returns, positions_changed,
        cost_per_trade=0.0005  # 0.05%
    )
    
    total_cost = (guarded_returns - guarded_returns_with_costs).sum()
    print(f"  Total transaction costs: {total_cost*100:.2f}%")
    
    # Train/Test 분할
    print(f"\n✅ Train/Test split...")
    train_no_guard, test_no_guard = split_train_test(ml9_aligned, '2024-01-01')
    train_guarded, test_guarded = split_train_test(guarded_returns_with_costs, '2024-01-01')
    
    print(f"  Train: {train_no_guard.index.min().date()} to {train_no_guard.index.max().date()} ({len(train_no_guard)} days)")
    print(f"  Test: {test_no_guard.index.min().date()} to {test_no_guard.index.max().date()} ({len(test_no_guard)} days)")
    
    # 메트릭 계산
    print(f"\nCalculating metrics...")
    turbo = TurboCPUBacktest()
    
    # Full period
    ml9_no_guard_metrics = turbo.compute_metrics_fast(ml9_aligned.values)
    guarded_with_costs_metrics = turbo.compute_metrics_fast(guarded_returns_with_costs.values)
    guarded_no_costs_metrics = turbo.compute_metrics_fast(guarded_returns.values)
    
    # Train
    train_no_guard_metrics = turbo.compute_metrics_fast(train_no_guard.values)
    train_guarded_metrics = turbo.compute_metrics_fast(train_guarded.values)
    
    # Test
    test_no_guard_metrics = turbo.compute_metrics_fast(test_no_guard.values)
    test_guarded_metrics = turbo.compute_metrics_fast(test_guarded.values)
    
    # 결과 출력
    print(f"\n" + "="*80)
    print("REALISTIC BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n📊 FULL PERIOD (2018-2024):")
    
    print(f"\n1. ML9 (No Guard):")
    print(f"  Sharpe: {ml9_no_guard_metrics['sharpe']:.3f}")
    print(f"  Annual Return: {ml9_no_guard_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {ml9_no_guard_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {ml9_no_guard_metrics['max_drawdown']*100:.2f}%")
    
    print(f"\n2. ML9 (Guard, No Costs):")
    print(f"  Sharpe: {guarded_no_costs_metrics['sharpe']:.3f} ({(guarded_no_costs_metrics['sharpe']/ml9_no_guard_metrics['sharpe']-1)*100:+.1f}%)")
    print(f"  Annual Return: {guarded_no_costs_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {guarded_no_costs_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {guarded_no_costs_metrics['max_drawdown']*100:.2f}%")
    
    print(f"\n3. ML9 (Guard + Realistic Costs):")
    print(f"  Sharpe: {guarded_with_costs_metrics['sharpe']:.3f} ({(guarded_with_costs_metrics['sharpe']/ml9_no_guard_metrics['sharpe']-1)*100:+.1f}%)")
    print(f"  Annual Return: {guarded_with_costs_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {guarded_with_costs_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {guarded_with_costs_metrics['max_drawdown']*100:.2f}%")
    
    print(f"\n📊 TRAIN PERIOD (2018-2023):")
    print(f"  ML9 (No Guard): Sharpe {train_no_guard_metrics['sharpe']:.3f}")
    print(f"  ML9 (Guard + Costs): Sharpe {train_guarded_metrics['sharpe']:.3f} ({(train_guarded_metrics['sharpe']/train_no_guard_metrics['sharpe']-1)*100:+.1f}%)")
    
    print(f"\n📊 TEST PERIOD (2024, Out-of-Sample):")
    print(f"  ML9 (No Guard): Sharpe {test_no_guard_metrics['sharpe']:.3f}")
    print(f"  ML9 (Guard + Costs): Sharpe {test_guarded_metrics['sharpe']:.3f} ({(test_guarded_metrics['sharpe']/test_no_guard_metrics['sharpe']-1)*100:+.1f}%)")
    
    # 과적합 체크
    print(f"\n🔍 OVERFITTING CHECK:")
    train_test_ratio = test_guarded_metrics['sharpe'] / train_guarded_metrics['sharpe']
    print(f"  Train Sharpe: {train_guarded_metrics['sharpe']:.3f}")
    print(f"  Test Sharpe: {test_guarded_metrics['sharpe']:.3f}")
    print(f"  Test/Train Ratio: {train_test_ratio:.2f}")
    
    if train_test_ratio < 0.7:
        print(f"  ⚠️  WARNING: Significant degradation (overfitting)")
    elif train_test_ratio < 0.9:
        print(f"  ⚠️  CAUTION: Moderate degradation")
    else:
        print(f"  ✅ OK: Consistent performance")
    
    # 목표 달성도
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"  Target Sharpe: 2.0+")
    print(f"  Achieved (Full): {guarded_with_costs_metrics['sharpe']:.3f}")
    print(f"  Achieved (Test): {test_guarded_metrics['sharpe']:.3f}")
    
    if guarded_with_costs_metrics['sharpe'] >= 2.0:
        print(f"  Status: ✅ TARGET MET")
    elif guarded_with_costs_metrics['sharpe'] >= 1.0:
        print(f"  Status: ⚠️  PARTIAL (Sharpe 1.0+)")
        print(f"  Achievement: {guarded_with_costs_metrics['sharpe']/2.0*100:.1f}%")
    else:
        print(f"  Status: ❌ TARGET NOT MET")
        print(f"  Achievement: {guarded_with_costs_metrics['sharpe']/2.0*100:.1f}%")
    
    # 결과 저장
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    guarded_returns_with_costs.to_csv(RESULTS_DIR / "ml9_realistic_guard_returns.csv")
    print(f"✓ Returns saved")
    
    results = {
        'full_period': {
            'ml9_no_guard': ml9_no_guard_metrics,
            'ml9_guard_no_costs': guarded_no_costs_metrics,
            'ml9_guard_with_costs': guarded_with_costs_metrics,
        },
        'train_period': {
            'ml9_no_guard': train_no_guard_metrics,
            'ml9_guard_with_costs': train_guarded_metrics,
        },
        'test_period': {
            'ml9_no_guard': test_no_guard_metrics,
            'ml9_guard_with_costs': test_guarded_metrics,
        },
        'guard_config': best_params,
        'transaction_costs': {
            'slippage_bps': 3.0,
            'commission_pct': 0.02,
            'total_per_trade_pct': 0.05,
            'total_cost_pct': total_cost * 100,
        },
        'guard_stats': {
            'active_days': int(guard_active_days),
            'active_rate': float(guard_active_rate),
        }
    }
    
    with open(RESULTS_DIR / "ml9_realistic_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Metrics saved")
    
    print(f"\n" + "="*80)
    print("REALISTIC BACKTEST COMPLETE ✓")
    print("="*80)

if __name__ == "__main__":
    main()
