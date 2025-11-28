#!/usr/bin/env python3
# coding: utf-8

"""
Look-Ahead Bias Check
Verify that no future information is used in backtesting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

def check_lookahead_bias():
    """룩어헤드 바이어스 체크"""
    
    print("\n" + "="*80)
    print("LOOK-AHEAD BIAS VERIFICATION")
    print("="*80)
    
    issues = []
    
    # 1. Guard 적용 시점 체크
    print("\n1. Checking Guard application timing...")
    
    ml9_returns = pd.read_csv(RESULTS_DIR / "ml9_returns.csv", 
                              index_col=0, parse_dates=True).iloc[:, 0]
    ml9_optimized = pd.read_csv(RESULTS_DIR / "ml9_optimized_guard_returns.csv",
                                index_col=0, parse_dates=True).iloc[:, 0]
    spy_prices = pd.read_csv(DATA_DIR / "spy_prices.csv",
                            index_col='date', parse_dates=True)
    
    # 인덱스 정규화
    ml9_returns.index = pd.to_datetime(ml9_returns.index).normalize()
    ml9_optimized.index = pd.to_datetime(ml9_optimized.index).normalize()
    spy_prices.index = pd.to_datetime(spy_prices.index).normalize()
    
    # SPY 수익률 계산
    spy_returns = spy_prices['close'].pct_change()
    
    # 공통 인덱스
    common_idx = ml9_returns.index.intersection(ml9_optimized.index).intersection(spy_returns.index)
    
    ml9_aligned = ml9_returns.loc[common_idx].sort_index()
    ml9_opt_aligned = ml9_optimized.loc[common_idx].sort_index()
    spy_aligned = spy_returns.loc[common_idx].sort_index()
    
    # Guard 적용 날짜 찾기
    guard_applied = (ml9_aligned != ml9_opt_aligned)
    guard_dates = guard_applied[guard_applied].index
    
    print(f"   Guard applied on {len(guard_dates)} days")
    
    # 각 Guard 적용 날짜에서 SPY 수익률 체크
    lookahead_issues = 0
    for date in guard_dates[:10]:  # 처음 10개만 체크
        spy_ret = spy_aligned.loc[date]
        
        # Guard 조건: -3% < SPY <= 0%
        if not (-0.03 < spy_ret <= 0.0):
            print(f"   ⚠️  {date.date()}: SPY return {spy_ret*100:.2f}% (outside Guard range)")
            lookahead_issues += 1
    
    if lookahead_issues > 0:
        issues.append(f"Guard applied on {lookahead_issues} days outside expected range")
        print(f"   ❌ ISSUE: Guard timing may be incorrect")
    else:
        print(f"   ✅ OK: Guard applied correctly based on same-day SPY returns")
    
    # 2. PIT 데이터 체크
    print("\n2. Checking Point-in-Time (PIT) data...")
    
    try:
        merged_data = pd.read_csv(DATA_DIR / "sp100_merged_data.csv",
                                 parse_dates=['date'])
        
        # calendardate vs timestamp 체크
        if 'calendardate' in merged_data.columns:
            # calendardate가 date보다 미래인지 체크
            merged_data['calendardate'] = pd.to_datetime(merged_data['calendardate'])
            future_data = merged_data[merged_data['calendardate'] > merged_data['date']]
            
            if len(future_data) > 0:
                issues.append(f"Found {len(future_data)} rows with future fundamental data")
                print(f"   ❌ ISSUE: {len(future_data)} rows use future fundamental data")
                print(f"   Example: {future_data[['date', 'calendardate']].head()}")
            else:
                print(f"   ✅ OK: All fundamental data is from the past (PIT compliant)")
        else:
            print(f"   ⚠️  WARNING: Cannot verify PIT - 'calendardate' column not found")
            issues.append("Cannot verify PIT compliance")
    
    except FileNotFoundError:
        print(f"   ⚠️  WARNING: Merged data file not found")
        issues.append("Cannot verify PIT compliance - file not found")
    
    # 3. 그리드 서치 데이터 누수 체크
    print("\n3. Checking grid search data leakage...")
    
    # 그리드 서치는 전체 기간(2018-2024) 데이터를 사용
    # 이는 in-sample 최적화이므로 데이터 누수 가능성 있음
    
    print(f"   ⚠️  WARNING: Grid search used full period (2018-2024)")
    print(f"   This is IN-SAMPLE optimization - may overfit")
    print(f"   Recommendation: Use Walk-Forward or Out-of-Sample validation")
    issues.append("Grid search used in-sample data (2018-2024) - overfitting risk")
    
    # 4. 리밸런싱 시점 체크
    print("\n4. Checking rebalancing timing...")
    
    # ML9 리밸런싱은 매일 발생
    # Guard는 당일 SPY 수익률 기반으로 적용
    # 문제: 당일 SPY 수익률을 알고 포지션 조정하는 것은 룩어헤드!
    
    print(f"   ❌ CRITICAL ISSUE: Guard uses SAME-DAY SPY returns")
    print(f"   This is LOOK-AHEAD BIAS!")
    print(f"   ")
    print(f"   Current logic:")
    print(f"     1. Calculate SPY return on day T")
    print(f"     2. If SPY return in [-3%, 0%], reduce position on day T")
    print(f"     3. Realize ML9 return on day T with reduced position")
    print(f"   ")
    print(f"   Problem: We don't know SPY return until end of day T")
    print(f"   ")
    print(f"   Correct logic should be:")
    print(f"     1. Calculate SPY return on day T-1 (previous day)")
    print(f"     2. If SPY return in [-3%, 0%], reduce position for day T")
    print(f"     3. Realize ML9 return on day T with reduced position")
    
    issues.append("CRITICAL: Guard uses same-day SPY returns (look-ahead bias)")
    
    # 결과 요약
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    if len(issues) == 0:
        print("\n✅ NO ISSUES FOUND")
        print("Backtest appears to be free of look-ahead bias")
    else:
        print(f"\n❌ FOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n⚠️  RECOMMENDATION:")
        print("   1. Fix Guard to use PREVIOUS day SPY returns")
        print("   2. Use Walk-Forward or Out-of-Sample validation")
        print("   3. Re-run backtest with corrected logic")
    
    return issues

if __name__ == "__main__":
    issues = check_lookahead_bias()
    
    if len(issues) > 0:
        print("\n" + "="*80)
        print("⚠️  BACKTEST RESULTS MAY BE OVERSTATED")
        print("="*80)
        sys.exit(1)
    else:
        sys.exit(0)
