# Stage 2: LowVol Engine v1 - Final Report

**Date**: 2024-11-28  
**Status**: ✅ Complete  
**Goal**: Develop and backtest low-volatility/defensive engine, analyze correlation with ML9+Guard

---

## 📊 Executive Summary

**LowVol v1** is a **moderate candidate** for ensemble integration with ML9+Guard.

**Key Findings**:
- ✅ Moderate correlation (0.53) provides some diversification benefit
- ⚠️ Lower Sharpe (0.34 vs 0.52) limits ensemble improvement
- ✅ Lower volatility (11.05% vs 15.23%) provides risk reduction
- ✅ Better drawdown (-13.74% vs -16.53%)

**Recommendation**: Include in ensemble but with **lower weight** (20-30%) compared to ML9+Guard

---

## 1. LowVol v1 Performance

### Full Period (2015-2024)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **0.6688** |
| Annual Return | 9.68% |
| Annual Volatility | 14.48% |
| Max Drawdown | -32.13% |
| Win Rate | 54.99% |
| Trading Days | 2,433 |
| Rebalance Frequency | Monthly (116 rebalances) |

### Common Period with ML9 (2018-2024)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **0.3436** |
| Annual Return | 3.80% |
| Annual Volatility | 11.05% |
| Max Drawdown | -13.74% |
| Win Rate | 55.10% |
| Trading Days | 755 |

---

## 2. Comparison: LowVol v1 vs ML9+Guard (2018-2024)

| Metric | LowVol v1 | ML9+Guard | Difference |
|--------|-----------|-----------|------------|
| **Sharpe Ratio** | 0.3436 | **0.5165** | -0.1728 ❌ |
| Annual Return | 3.80% | **7.87%** | -4.07% ❌ |
| Annual Volatility | **11.05%** | 15.23% | -4.18% ✅ |
| Max Drawdown | **-13.74%** | -16.53% | +2.78% ✅ |
| Win Rate | **55.10%** | 51.92% | +3.18% ✅ |

**Correlation**: **0.5322** (Moderate - some diversification benefit)

---

## 3. Ensemble Analysis

### 50/50 Ensemble (LowVol + ML9+Guard)

| Metric | Value | vs ML9+Guard |
|--------|-------|--------------|
| **Sharpe Ratio** | **0.5120** | **-0.86%** ⚠️ |
| Annual Return | 5.91% | -24.91% ❌ |
| Annual Volatility | **11.55%** | **+24.20%** ✅ |
| Max Drawdown | **-14.86%** | **-10.11%** ✅ |

**Key Insights**:
- ⚠️ Sharpe slightly worse (-0.86%) due to lower LowVol returns
- ✅ Volatility significantly reduced (+24.20%)
- ✅ Max drawdown improved (-10.11%)
- **Trade-off**: Lower returns for lower risk

---

## 4. LowVol Engine Configuration

```python
LowVolConfig(
    top_quantile=0.3,         # Top 30% low-risk stocks
    long_gross=1.0,           # 100% long exposure
    short_gross=0.0,          # No short (long-only)
    long_only=True,
    use_inverse_vol=True,     # Inverse volatility weighting
    vol_lookback=63,          # 63-day volatility
    beta_lookback=252,        # 252-day beta
    beta_use=True,            # Include beta in risk score
    downside_vol_weight=0.5,  # Downside volatility weight
    beta_weight=0.5,          # Beta weight
)
```

**Risk Score Formula**:
```
risk_score = z(vol_63d) + 0.5 * z(downside_vol_63d) + 0.5 * z(|beta|)
```

**Selection**: Low risk_score (top 30%) → Long positions

---

## 5. Assessment

### Strengths

✅ **Lower volatility** (11.05% vs 15.23%)  
✅ **Better drawdown** (-13.74% vs -16.53%)  
✅ **Higher win rate** (55.10% vs 51.92%)  
✅ **Moderate correlation** (0.53) provides diversification  
✅ **Defensive characteristics** suitable for risk management

### Weaknesses

❌ **Lower Sharpe** (0.34 vs 0.52)  
❌ **Lower returns** (3.80% vs 7.87%)  
❌ **50/50 ensemble doesn't improve Sharpe** (-0.86%)

### Final Rating

**⚠️ MODERATE CANDIDATE for ensemble**

**Reason**: Low-moderate correlation (0.53) provides some diversification benefit, but lower Sharpe limits overall ensemble improvement.

---

## 6. Recommendations

### For Ensemble Integration

1. **Use lower weight** (20-30%) compared to ML9+Guard (70-80%)
2. **Increase weight during high-volatility regimes** (VIX > 25)
3. **Consider dynamic weighting** based on market conditions

### For LowVol Improvement

1. **Optimize parameters**:
   - `top_quantile`: Test 0.2, 0.25, 0.3, 0.35
   - `vol_lookback`: Test 42, 63, 126
   - `beta_weight`: Test 0.3, 0.5, 0.7

2. **Add quality factors**:
   - ROE, ROA, Debt/Equity
   - Combine low-vol + quality

3. **Consider long-short**:
   - `short_gross=0.5` (short high-risk stocks)
   - May improve Sharpe to 0.8-1.0

---

## 7. Next Steps (Stage 3)

1. **3-Engine Ensemble** (ML9+Guard + QV + LowVol)
   - Static min-max ensemble
   - Optimize weights for Sharpe ≥ 2.0

2. **Regime-Based Weighting**
   - Bull/bear/high-vol regimes
   - Dynamic weight adjustment

3. **Out-of-Sample Gatekeeper**
   - Sharpe ≥ 2.0 validation
   - Multiple OOS windows

---

## 8. Files Generated

### Code
- `engines/factor_lowvol_v1.py` - LowVol engine module
- `analysis/backtest_lowvol_v1.py` - Backtest script

### Results
- `results/lowvol_v1_returns.csv` - Daily returns (2,433 days)
- `results/lowvol_v1_metrics.json` - Performance metrics

---

## 9. Conclusion

**LowVol v1** successfully provides **defensive characteristics** with lower volatility and better drawdown, but its lower Sharpe (0.34 vs 0.52) limits its standalone value. 

**Best use case**: **Risk management layer** in ensemble, especially during high-volatility periods.

**Status**: ✅ Ready for Stage 3 (3-engine ensemble optimization)

---

**Report Date**: 2024-11-28  
**Author**: Manus AI  
**Project**: Quant Ensemble Strategy (v2.0+)
