"""
ML9 (Guard) + QV v2.1 Min-Max Ensemble Optimization
Find (w_ml9, w_qv) that maximizes min Sharpe across windows
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from modules.market_guard_ml9 import ML9MarketConditionGuard


@dataclass
class TestWindow:
    name: str
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


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


def calculate_sharpe(ret_daily: pd.Series) -> float:
    """Calculate Sharpe ratio (annualized, rf=0)"""
    ret_daily = ret_daily.dropna()
    if ret_daily.empty:
        return 0.0

    days = (ret_daily.index[-1] - ret_daily.index[0]).days
    years = days / 365.25
    if years <= 0:
        return 0.0

    cum = (1.0 + ret_daily).prod()
    ann_ret = cum ** (1 / years) - 1.0

    vol_d = ret_daily.std(ddof=0)
    ann_vol = vol_d * np.sqrt(252.0)

    if ann_vol <= 0:
        return 0.0

    return float(ann_ret / ann_vol)


def grid_search_minmax_sharpe(
    ret_ml9: pd.Series,
    ret_qv: pd.Series,
    windows: List[TestWindow],
    step: float = 0.1,
) -> List[Dict]:
    """
    Grid search for (w_ml9, w_qv) that maximizes min Sharpe across windows
    """
    results: List[Dict] = []

    # Align dates
    df = pd.concat(
        [ret_ml9.rename("ml9"), ret_qv.rename("qv")],
        axis=1,
    ).dropna()

    # Pre-slice windows
    window_slices: Dict[str, pd.DataFrame] = {}
    for w in windows:
        df_w = df.loc[w.start : w.end]
        if df_w.empty:
            print(f"[WARN] Window {w.name} ({w.start}~{w.end}) has no data")
        window_slices[w.name] = df_w

    ws = np.arange(0.0, 1.0 + 1e-9, step)

    for w_ml9 in ws:
        w_qv = 1.0 - w_ml9
        if w_qv < -1e-9:
            continue

        window_sharpes: Dict[str, float] = {}
        min_sharpe = float("inf")

        for w in windows:
            df_w = window_slices[w.name]
            if df_w.empty:
                window_sharpes[w.name] = 0.0
                min_sharpe = min(min_sharpe, 0.0)
                continue

            ret_combo = w_ml9 * df_w["ml9"] + w_qv * df_w["qv"]
            sh = calculate_sharpe(ret_combo)
            window_sharpes[w.name] = sh
            if sh < min_sharpe:
                min_sharpe = sh

        results.append(
            {
                "w_ml9": float(w_ml9),
                "w_qv": float(w_qv),
                "min_sharpe": float(min_sharpe),
                "window_sharpes": window_sharpes,
            }
        )

    results_sorted = sorted(results, key=lambda r: r["min_sharpe"], reverse=True)
    return results_sorted


def main():
    print("="*100)
    print("ML9 (GUARD) + QV MIN-MAX ENSEMBLE OPTIMIZATION")
    print("="*100)
    
    # 1) Load data
    print("\nLoading data...")
    ret_ml9_raw = load_returns_from_json(BASE_DIR / "results" / "ml9_daily_returns_2020_2024.json")
    ret_qv = load_returns_from_json(BASE_DIR / "results" / "qv_daily_returns.json")
    spx = load_spx()
    
    print(f"ML9 (raw): {len(ret_ml9_raw)} days")
    print(f"QV: {len(ret_qv)} days")
    print(f"SPX: {len(spx)} days")
    
    # 2) Apply Guard to ML9
    print("\nApplying Market Condition Guard to ML9...")
    guard_config = {
        "enabled": True,
        "return_lower": -0.02,
        "return_upper": 0.0,
        "scale_factor": 0.5,
    }
    guard = ML9MarketConditionGuard(guard_config)
    guard.initialize(spx)
    
    # Apply Guard
    ml9_scales = guard.get_scale_series(ret_ml9_raw.index)
    ret_ml9_guard = ret_ml9_raw * ml9_scales
    ret_ml9_guard.name = "ml9_guard"
    
    print(f"ML9 (Guard applied): {len(ret_ml9_guard)} days")
    
    # 3) Define test windows
    print("\nDefining test windows...")
    windows = [
        TestWindow("2023", "2023-01-01", "2023-12-31"),
        TestWindow("2024_H1", "2024-01-01", "2024-06-30"),
        TestWindow("2024", "2024-01-01", "2024-12-31"),
    ]
    
    for w in windows:
        print(f"  {w.name}: {w.start} ~ {w.end}")
    
    # 4) Grid search
    print("\nRunning min-max Sharpe grid search...")
    results_sorted = grid_search_minmax_sharpe(ret_ml9_guard, ret_qv, windows, step=0.05)
    
    # 5) Print top 10
    print("\n" + "="*100)
    print("TOP 10 COMBINATIONS (by min Sharpe)")
    print("="*100)
    
    print(f"\n{'Rank':<6} {'w_ml9':>8} {'w_qv':>8} {'min_Sharpe':>12} {'Sharpe_2023':>12} {'Sharpe_2024H1':>14} {'Sharpe_2024':>12}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted[:10]):
        ws = r["window_sharpes"]
        print(f"{i+1:<6} {r['w_ml9']:>8.2f} {r['w_qv']:>8.2f} {r['min_sharpe']:>12.2f} "
              f"{ws.get('2023', 0.0):>12.2f} {ws.get('2024_H1', 0.0):>14.2f} {ws.get('2024', 0.0):>12.2f}")
    
    # 6) Check if Sharpe 2.0+ is achievable
    print("\n" + "="*100)
    print("SHARPE 2.0+ ANALYSIS")
    print("="*100)
    
    sharpe_2_plus = [r for r in results_sorted if r["min_sharpe"] >= 2.0]
    
    if sharpe_2_plus:
        print(f"\n✅ Found {len(sharpe_2_plus)} combinations with min Sharpe >= 2.0!")
        print(f"\nBest combination:")
        best = sharpe_2_plus[0]
        print(f"  w_ml9 = {best['w_ml9']:.2f}, w_qv = {best['w_qv']:.2f}")
        print(f"  min Sharpe = {best['min_sharpe']:.2f}")
        print(f"  Window Sharpes: {best['window_sharpes']}")
    else:
        print(f"\n❌ No combination achieves min Sharpe >= 2.0")
        print(f"\nBest achievable min Sharpe: {results_sorted[0]['min_sharpe']:.2f}")
        print(f"  w_ml9 = {results_sorted[0]['w_ml9']:.2f}, w_qv = {results_sorted[0]['w_qv']:.2f}")
        print(f"  Window Sharpes: {results_sorted[0]['window_sharpes']}")
    
    # 7) Save results
    out_path = BASE_DIR / "results" / "ensemble_ml9guard_qv_minmax.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\n✅ Results saved to: {out_path}")
    
    print("\n" + "="*100)
    print("MIN-MAX ENSEMBLE OPTIMIZATION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
