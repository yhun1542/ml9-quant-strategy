#!/usr/bin/env python3
# coding: utf-8

"""
Corrected Backtest - No Look-Ahead Bias + Transaction Costs
1. Guard uses PREVIOUS day SPY returns
2. Transaction costs included (slippage + commission)
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
    """
    룩어헤드 바이어스 수정: 전일 SPY 수익률 사용
    
    Args:
        ml9_returns: ML9 수익률 시리즈
        spy_series: SPY 가격 시리즈
        return_lower: SPX 수익률 하한
        return_upper: SPX 수익률 상한
        scale_factor: 포지션 축소 비율
    
    Returns:
        Guard 적용된 수익률 시리즈
    """
    # SPY 수익률 계산
    spy_returns = spy_series.pct_change()
    
    # 인덱스 정규화
    ml9_returns = ml9_returns.copy()
    ml9_returns.index = pd.to_datetime(ml9_returns.index).normalize()
    spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
    
    # 공통 인덱스
    common_idx = ml9_returns.index.intersection(spy_returns.index)
    
    ml9_aligned = ml9_returns.loc[common_idx].sort_index()
    spy_aligned = spy_returns.loc[common_idx].sort_index()
    
    # 🔧 FIX: 전일 SPY 수익률 사용 (shift(1))
    spy_prev_day = spy_aligned.shift(1)
    
    # Guard 조건 (전일 SPY 수익률 기반)
    guard_condition = (spy_prev_day > return_lower) & (spy_prev_day <= return_upper)
    
    # Scale factor 적용
    guarded_returns = np.where(guard_condition, ml9_aligned.values * scale_factor, ml9_aligned.values)
    
    # 첫 날은 Guard 없음 (전일 데이터 없음)
    guarded_returns[0] = ml9_aligned.values[0]
    
    result = pd.Series(guarded_returns, index=common_idx)
    
    return result

def apply_transaction_costs(returns: pd.Series, positions_changed: pd.Series,
                            slippage_bps: float = 8.5, commission_pct: float = 0.001) -> pd.Series:
    """
    거래 비용 반영
    
    Args:
        returns: 수익률 시리즈
        positions_changed: 포지션 변경 여부 (True/False)
        slippage_bps: 슬리피지 (basis points)
        commission_pct: 수수료 (%)
    
    Returns:
        거래 비용 반영된 수익률 시리즈
    """
    # 슬리피지 + 수수료
    transaction_cost = (slippage_bps / 10000) + commission_pct
    
    # 포지션 변경 시 거래 비용 차감
    costs = np.where(positions_changed, transaction_cost, 0.0)
    
    # 수익률에서 비용 차감
    returns_after_costs = returns - costs
    
    return pd.Series(returns_after_costs, index=returns.index)

def split_train_test(returns: pd.Series, test_start_date: str = '2024-01-01'):
    """
    Train/Test 분할
    
    Args:
        returns: 수익률 시리즈
        test_start_date: 테스트 시작 날짜
    
    Returns:
        (train_returns, test_returns)
    """
    test_start = pd.to_datetime(test_start_date)
    
    train = returns[returns.index < test_start]
    test = returns[returns.index >= test_start]
    
    return train, test

def main():
    print("\n" + "="*80)
    print("CORRECTED BACKTEST - NO LOOK-AHEAD BIAS + TRANSACTION COSTS")
    print("="*80)
    
    # 데이터 로딩
    print(f"\nLoading data...")
    ml9_returns = pd.read_csv(RESULTS_DIR / "ml9_returns.csv",
                              index_col=0, parse_dates=True).iloc[:, 0]
    spy_prices = pd.read_csv(DATA_DIR / "spy_prices.csv",
                            index_col='date', parse_dates=True)
    spy_prices.index = pd.to_datetime(spy_prices.index).normalize()
    spy_series = spy_prices['close']
    
    # 최적 파라미터 (그리드 서치 결과)
    best_params = {
        'return_lower': -0.03,
        'return_upper': 0.0,
        'scale_factor': 0.3
    }
    
    print(f"\nOptimized Guard Parameters:")
    print(f"  Return Range: {best_params['return_lower']*100:.1f}% ~ {best_params['return_upper']*100:.1f}%")
    print(f"  Scale Factor: {best_params['scale_factor']:.2f}")
    
    # 🔧 FIX 1: Guard 적용 (전일 SPY 수익률 사용)
    print(f"\n🔧 FIX 1: Applying Guard with PREVIOUS day SPY returns...")
    guarded_returns = apply_guard_corrected(
        ml9_returns, spy_series,
        best_params['return_lower'],
        best_params['return_upper'],
        best_params['scale_factor']
    )
    
    # Guard 활성화 통계
    ml9_aligned = ml9_returns.loc[guarded_returns.index]
    guard_active = (ml9_aligned != guarded_returns)
    guard_active_days = guard_active.sum()
    guard_active_rate = guard_active_days / len(guard_active) * 100
    print(f"  Guard active: {guard_active_days}/{len(guard_active)} days ({guard_active_rate:.2f}%)")
    
    # 🔧 FIX 2: 거래 비용 반영
    print(f"\n🔧 FIX 2: Applying transaction costs...")
    print(f"  Slippage: 8.5 bps")
    print(f"  Commission: 0.1%")
    print(f"  Total cost per trade: ~0.185%")
    
    # 포지션 변경 감지
    positions_changed = guard_active.copy()
    positions_changed[0] = True  # 첫 날은 포지션 진입
    
    # 거래 비용 적용
    guarded_returns_with_costs = apply_transaction_costs(
        guarded_returns, positions_changed,
        slippage_bps=8.5, commission_pct=0.001
    )
    
    # 거래 비용 영향
    total_cost = (guarded_returns - guarded_returns_with_costs).sum()
    print(f"  Total transaction costs: {total_cost*100:.2f}%")
    
    # 🔧 FIX 3: Train/Test 분할 (Out-of-Sample 검증)
    print(f"\n🔧 FIX 3: Train/Test split for out-of-sample validation...")
    
    train_no_guard, test_no_guard = split_train_test(ml9_aligned, '2024-01-01')
    train_guarded, test_guarded = split_train_test(guarded_returns_with_costs, '2024-01-01')
    
    print(f"  Train period: {train_no_guard.index.min().date()} to {train_no_guard.index.max().date()} ({len(train_no_guard)} days)")
    print(f"  Test period: {test_no_guard.index.min().date()} to {test_no_guard.index.max().date()} ({len(test_no_guard)} days)")
    
    # 메트릭 계산
    print(f"\nCalculating metrics...")
    turbo = TurboCPUBacktest()
    
    # Full period
    ml9_no_guard_metrics = turbo.compute_metrics_fast(ml9_aligned.values)
    guarded_with_costs_metrics = turbo.compute_metrics_fast(guarded_returns_with_costs.values)
    
    # Train period
    train_no_guard_metrics = turbo.compute_metrics_fast(train_no_guard.values)
    train_guarded_metrics = turbo.compute_metrics_fast(train_guarded.values)
    
    # Test period (Out-of-Sample)
    test_no_guard_metrics = turbo.compute_metrics_fast(test_no_guard.values)
    test_guarded_metrics = turbo.compute_metrics_fast(test_guarded.values)
    
    # 결과 출력
    print(f"\n" + "="*80)
    print("CORRECTED BACKTEST RESULTS")
    print("="*80)
    
    print(f"\n📊 FULL PERIOD (2018-2024):")
    print(f"\nML9 (No Guard):")
    print(f"  Sharpe Ratio: {ml9_no_guard_metrics['sharpe']:.3f}")
    print(f"  Annual Return: {ml9_no_guard_metrics['annual_return']*100:.2f}%")
    print(f"  Annual Volatility: {ml9_no_guard_metrics['annual_volatility']*100:.2f}%")
    print(f"  Max Drawdown: {ml9_no_guard_metrics['max_drawdown']*100:.2f}%")
    
    print(f"\nML9 (Optimized Guard + Transaction Costs):")
    print(f"  Sharpe Ratio: {guarded_with_costs_metrics['sharpe']:.3f} ({(guarded_with_costs_metrics['sharpe']/ml9_no_guard_metrics['sharpe']-1)*100:+.1f}%)")
    print(f"  Annual Return: {guarded_with_costs_metrics['annual_return']*100:.2f}% ({(guarded_with_costs_metrics['annual_return']-ml9_no_guard_metrics['annual_return'])*100:+.2f}%)")
    print(f"  Annual Volatility: {guarded_with_costs_metrics['annual_volatility']*100:.2f}% ({(guarded_with_costs_metrics['annual_volatility']/ml9_no_guard_metrics['annual_volatility']-1)*100:+.1f}%)")
    print(f"  Max Drawdown: {guarded_with_costs_metrics['max_drawdown']*100:.2f}% ({(guarded_with_costs_metrics['max_drawdown']-ml9_no_guard_metrics['max_drawdown'])*100:+.2f}%)")
    
    print(f"\n📊 TRAIN PERIOD (2018-2023, In-Sample):")
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
        print(f"  ⚠️  WARNING: Significant performance degradation in test period (overfitting)")
    elif train_test_ratio < 0.9:
        print(f"  ⚠️  CAUTION: Moderate performance degradation in test period")
    else:
        print(f"  ✅ OK: Test performance consistent with train period")
    
    # 목표 달성도
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"  Target Sharpe: 2.0+")
    print(f"  Achieved Sharpe (Full): {guarded_with_costs_metrics['sharpe']:.3f}")
    print(f"  Achieved Sharpe (Test): {test_guarded_metrics['sharpe']:.3f}")
    
    if guarded_with_costs_metrics['sharpe'] >= 2.0:
        print(f"  Status: ✅ TARGET MET (Full Period)")
    else:
        print(f"  Status: ❌ TARGET NOT MET")
        print(f"  Achievement Rate: {guarded_with_costs_metrics['sharpe']/2.0*100:.1f}%")
    
    # 결과 저장
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    guarded_returns_with_costs.to_csv(RESULTS_DIR / "ml9_corrected_guard_returns.csv")
    print(f"✓ Returns saved to {RESULTS_DIR / 'ml9_corrected_guard_returns.csv'}")
    
    results = {
        'full_period': {
            'ml9_no_guard': ml9_no_guard_metrics,
            'ml9_guarded_with_costs': guarded_with_costs_metrics,
        },
        'train_period': {
            'ml9_no_guard': train_no_guard_metrics,
            'ml9_guarded_with_costs': train_guarded_metrics,
        },
        'test_period': {
            'ml9_no_guard': test_no_guard_metrics,
            'ml9_guarded_with_costs': test_guarded_metrics,
        },
        'guard_config': best_params,
        'transaction_costs': {
            'slippage_bps': 8.5,
            'commission_pct': 0.1,
            'total_cost_pct': total_cost * 100,
        },
        'guard_stats': {
            'active_days': int(guard_active_days),
            'active_rate': float(guard_active_rate),
        }
    }
    
    with open(RESULTS_DIR / "ml9_corrected_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
    print(f"✓ Metrics saved to {RESULTS_DIR / 'ml9_corrected_metrics.json'}")
    
    print(f"\n" + "="*80)
    print("CORRECTED BACKTEST COMPLETE ✓")
    print("="*80)

if __name__ == "__main__":
    main()
