# Quantitative Strategy Analysis Report

## Overview
This report presents the backtesting results for the ML9 (with/without Guard) and QV quantitative trading strategies from 2015 to 2024 on the SP100 universe.

## ML9 Engine Results (No Guard)

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.80 |
| Annualized Return | 14.84% |
| Annualized Volatility | 18.63% |
| Max Drawdown | -28.37% |
| Win Rate | 49.17% |
| Number of Trades | 785 |

## ML9 Engine Results (With Guard)

| Metric | Value | Change |
|---|---|---|
| Sharpe Ratio | 0.91 | +14.4% |
| Annualized Return | 14.32% | -0.52% |
| Annualized Volatility | 15.71% | -2.92% |
| Max Drawdown | -24.53% | -13.5% |
| Win Rate | 49.17% | +0.00% |
| Number of Trades | 785 | +0 |

**Guard Configuration:**
- SPX Return Range: -2.0% to 0.0%
- Scale Factor: 0.5 (50% position reduction)
- Volatility Filter: Disabled

## QV Engine Results

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.81 |
| Annualized Return | 13.42% |
| Annualized Volatility | 16.59% |
| Max Drawdown | -31.11% |
| Win Rate | 54.25% |
| Number of Trades | 2516 |

