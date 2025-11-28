# Quantitative Strategy Analysis Report

## Overview
This report presents the backtesting results for the ML9 (with/without Guard) and QV quantitative trading strategies from 2015 to 2024 on the SP100 universe.

## ML9 Engine Results (No Guard)

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.96 |
| Annualized Return | 17.43% |
| Annualized Volatility | 18.24% |
| Max Drawdown | -25.82% |
| Win Rate | 51.34% |
| Number of Trades | 785 |

## ML9 Engine Results (With Guard)

| Metric | Value | Change |
|---|---|---|
| Sharpe Ratio | 1.11 | +16.6% |
| Annualized Return | 17.18% | -0.25% |
| Annualized Volatility | 15.42% | -2.81% |
| Max Drawdown | -22.20% | -14.0% |
| Win Rate | 51.34% | +0.00% |
| Number of Trades | 785 | +0 |

**Guard Configuration:**
- SPX Return Range: -2.0% to 0.0%
- Scale Factor: 0.5 (50% position reduction)
- Volatility Filter: Disabled

## QV Engine Results

| Metric | Value |
|---|---|
| Sharpe Ratio | 0.77 |
| Annualized Return | 12.12% |
| Annualized Volatility | 15.82% |
| Max Drawdown | -36.63% |
| Win Rate | 53.18% |
| Number of Trades | 2516 |

