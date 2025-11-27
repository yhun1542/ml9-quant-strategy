"""
ML9 + QV Ensemble Backtest with Market Condition Guard
Compare performance with/without guard
"""

from __future__ import annotations
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# Add modules to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from modules.market_guard_ml9 import ML9MarketConditionGuard


def load_returns_from_json(path: Path, key: str = "daily_returns") -> pd.Series:
    """Load daily returns from JSON"""
    with open(path, "r") as f:
        data = json.load(f)
    dr = data[key]
    idx = pd.to_datetime(dr["index"])
    vals = dr["values"]
    s = pd.Series(vals, index=idx).sort_index()
    s.name = path.stem
    return s


def load_spx() -> pd.Series:
    """Load SPX price series"""
    df = pd.read_csv(BASE_DIR / "data" / "spx_close.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    s = df.iloc[:, 0]
    s = s.sort_index()
    s.name = "SPX"
    return s


def calculate_metrics(ret_daily: pd.Series) -> dict:
    """Calculate performance metrics"""
    ret_daily = ret_daily.dropna()
    if ret_daily.empty:
        return {
            "sharpe": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.0,
            "n_days": 0,
        }

    days = (ret_daily.index[-1] - ret_daily.index[0]).days
    years = days / 365.25
    if years <= 0:
        years = 1.0

    cum = (1.0 + ret_daily).prod()
    ann_ret = cum ** (1 / years) - 1.0

    vol_d = ret_daily.std(ddof=0)
    ann_vol = vol_d * np.sqrt(252.0)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max DD
    wealth = (1 + ret_daily).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_dd = dd.min()

    # Win rate
    win_rate = (ret_daily > 0).sum() / len(ret_daily) if len(ret_daily) > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "n_days": len(ret_daily),
    }


def backtest_ensemble(
    ret_ml9: pd.Series,
    ret_qv: pd.Series,
    w_ml9: float = 0.6,
    w_qv: float = 0.4,
    guard: Optional[ML9MarketConditionGuard] = None,
) -> tuple[pd.Series, dict]:
    """
    Backtest ML9 + QV ensemble
    If guard is provided, apply ML9 scale factor
    """
    # Align dates
    df = pd.concat(
        [ret_ml9.rename("ml9"), ret_qv.rename("qv")],
        axis=1,
    ).dropna()

    if guard is not None:
        # Get ML9 scale factors
        ml9_scales = guard.get_scale_series(df.index)
        df["ml9_scaled"] = df["ml9"] * ml9_scales
    else:
        df["ml9_scaled"] = df["ml9"]

    # Ensemble returns
    ret_ensemble = w_ml9 * df["ml9_scaled"] + w_qv * df["qv"]
    ret_ensemble.name = "ensemble"

    # Calculate metrics
    metrics = calculate_metrics(ret_ensemble)

    return ret_ensemble, metrics


def main():
    print("="*100)
    print("ML9 + QV ENSEMBLE BACKTEST WITH MARKET CONDITION GUARD")
    print("="*100)
    
    # 1) Load data
    print("\nLoading data...")
    ret_ml9 = load_returns_from_json(BASE_DIR / "results" / "ml9_daily_returns_2020_2024.json")
    ret_qv = load_returns_from_json(BASE_DIR / "results" / "qv_daily_returns.json")
    spx = load_spx()
    
    print(f"ML9: {len(ret_ml9)} days")
    print(f"QV: {len(ret_qv)} days")
    print(f"SPX: {len(spx)} days")
    
    # 2) Initialize guard
    print("\nInitializing Market Condition Guard...")
    guard_config = {
        "enabled": True,
        "spx_symbol": "SPY",
        "return_lower": -0.02,
        "return_upper": 0.0,
        "scale_factor": 0.5,
        "vol_window": 20,
        "use_vol_filter": False,
    }
    
    guard = ML9MarketConditionGuard(guard_config)
    guard.initialize(spx)
    
    # 3) Backtest without guard
    print("\n" + "="*100)
    print("BACKTEST WITHOUT GUARD")
    print("="*100)
    
    ret_no_guard, metrics_no_guard = backtest_ensemble(
        ret_ml9, ret_qv, w_ml9=0.6, w_qv=0.4, guard=None
    )
    
    print(f"\nSharpe: {metrics_no_guard['sharpe']:.2f}")
    print(f"Annual Return: {metrics_no_guard['annual_return']*100:.2f}%")
    print(f"Annual Vol: {metrics_no_guard['annual_vol']*100:.2f}%")
    print(f"Max DD: {metrics_no_guard['max_dd']*100:.2f}%")
    print(f"Win Rate: {metrics_no_guard['win_rate']*100:.2f}%")
    
    # 4) Backtest with guard
    print("\n" + "="*100)
    print("BACKTEST WITH GUARD (ML9 50% in SPX -2%~0%)")
    print("="*100)
    
    ret_with_guard, metrics_with_guard = backtest_ensemble(
        ret_ml9, ret_qv, w_ml9=0.6, w_qv=0.4, guard=guard
    )
    
    print(f"\nSharpe: {metrics_with_guard['sharpe']:.2f}")
    print(f"Annual Return: {metrics_with_guard['annual_return']*100:.2f}%")
    print(f"Annual Vol: {metrics_with_guard['annual_vol']*100:.2f}%")
    print(f"Max DD: {metrics_with_guard['max_dd']*100:.2f}%")
    print(f"Win Rate: {metrics_with_guard['win_rate']*100:.2f}%")
    
    # 5) Comparison
    print("\n" + "="*100)
    print("COMPARISON")
    print("="*100)
    
    print(f"\nSharpe: {metrics_no_guard['sharpe']:.2f} → {metrics_with_guard['sharpe']:.2f} "
          f"({(metrics_with_guard['sharpe'] - metrics_no_guard['sharpe']):.2f})")
    print(f"Annual Return: {metrics_no_guard['annual_return']*100:.2f}% → {metrics_with_guard['annual_return']*100:.2f}% "
          f"({(metrics_with_guard['annual_return'] - metrics_no_guard['annual_return'])*100:.2f}%)")
    print(f"Annual Vol: {metrics_no_guard['annual_vol']*100:.2f}% → {metrics_with_guard['annual_vol']*100:.2f}% "
          f"({(metrics_with_guard['annual_vol'] - metrics_no_guard['annual_vol'])*100:.2f}%)")
    print(f"Max DD: {metrics_no_guard['max_dd']*100:.2f}% → {metrics_with_guard['max_dd']*100:.2f}% "
          f"({(metrics_with_guard['max_dd'] - metrics_no_guard['max_dd'])*100:.2f}%)")
    
    # 6) Save results
    results = {
        "no_guard": {
            "metrics": metrics_no_guard,
            "daily_returns": {
                "index": ret_no_guard.index.strftime("%Y-%m-%d").tolist(),
                "values": ret_no_guard.tolist(),
            },
        },
        "with_guard": {
            "metrics": metrics_with_guard,
            "daily_returns": {
                "index": ret_with_guard.index.strftime("%Y-%m-%d").tolist(),
                "values": ret_with_guard.tolist(),
            },
        },
        "guard_config": guard_config,
    }
    
    out_path = BASE_DIR / "results" / "ml9_guard_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
