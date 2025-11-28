# analysis/dynamic_ensemble_regime_v1.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.regime_v1 import compute_spx_regime, RegimeConfigV1

BASE_DIR = Path(__file__).resolve().parents[1]

# ===== 경로 설정 =====
RET_ML9_GUARD_PATH = BASE_DIR / "results" / "ml9_guard_returns.csv"
RET_QV_PATH        = BASE_DIR / "results" / "qv_returns.csv"
RET_LOWVOL_PATH    = BASE_DIR / "results" / "lowvol_v1_returns.csv"
SPY_PRICES_PATH    = BASE_DIR / "data" / "spy_prices.csv"


@dataclass
class RegimeWeights:
    bull: Dict[str, float]
    bear: Dict[str, float]
    high_vol: Dict[str, float]
    neutral: Dict[str, float]


def load_returns_csv(path: Path) -> pd.Series:
    """Load returns from CSV file (index_col=0)"""
    df = pd.read_csv(path, index_col=0)
    s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in s.index])
    s = s.sort_index()
    return s


def load_spy_close(path: Path) -> pd.Series:
    """Load SPY close prices"""
    df = pd.read_csv(path)
    if 't' in df.columns:
        df['t'] = pd.to_datetime(df['t'])
        df = df.set_index('t')
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    
    if 'c' in df.columns:
        s = df['c']
    else:
        s = df.iloc[:, 0]
    
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in s.index])
    s = s.sort_index()
    return s


def calculate_sharpe(ret_daily: pd.Series) -> float:
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


def calculate_metrics(ret_daily: pd.Series) -> Dict:
    ret_daily = ret_daily.dropna()
    if ret_daily.empty:
        return {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "maxdd": 0.0,
            "winrate": 0.0,
        }

    days = (ret_daily.index[-1] - ret_daily.index[0]).days
    years = days / 365.25
    if years <= 0:
        return {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "maxdd": 0.0,
            "winrate": 0.0,
        }

    cum = (1.0 + ret_daily).prod()
    ann_ret = cum ** (1 / years) - 1.0

    vol_d = ret_daily.std(ddof=0)
    ann_vol = vol_d * np.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    wealth = (1.0 + ret_daily).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    maxdd = dd.min()

    winrate = (ret_daily > 0).mean()

    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "maxdd": float(maxdd),
        "winrate": float(winrate),
    }


def build_dynamic_ensemble(
    ret_ml9: pd.Series,
    ret_qv: pd.Series,
    ret_lv: pd.Series,
    spy_close: pd.Series,
    regime_weights: RegimeWeights,
) -> tuple[pd.Series, pd.Series]:
    """
    일별 레짐에 따라 엔진 가중치를 달리하여 최종 앙상블 수익률을 생성.
    
    Returns:
        (ret_series, regime_series): 앙상블 수익률 및 레짐 시리즈
    """
    # align index
    df = pd.concat(
        [
            ret_ml9.rename("ml9"),
            ret_qv.rename("qv"),
            ret_lv.rename("lv"),
        ],
        axis=1,
    ).dropna()

    spy_close = spy_close.reindex(df.index, method="ffill")

    # 레짐 계산
    regime_series = compute_spx_regime(spy_close, RegimeConfigV1())

    # 최종 수익률
    rets = []

    for date, row in df.iterrows():
        regime = regime_series.loc[date] if date in regime_series.index else "NEUTRAL"

        if regime == "BULL":
            w = regime_weights.bull
        elif regime == "BEAR":
            w = regime_weights.bear
        elif regime == "HIGH_VOL":
            w = regime_weights.high_vol
        else:
            w = regime_weights.neutral

        # normalize just in case
        total_w = w.get("ml9", 0.0) + w.get("qv", 0.0) + w.get("lv", 0.0)
        if total_w <= 0:
            rets.append((date, 0.0))
            continue

        w_ml9 = w.get("ml9", 0.0) / total_w
        w_qv  = w.get("qv", 0.0) / total_w
        w_lv  = w.get("lv", 0.0) / total_w

        r = (
            w_ml9 * row["ml9"] +
            w_qv  * row["qv"]  +
            w_lv  * row["lv"]
        )
        rets.append((date, r))

    if not rets:
        return pd.Series(dtype=float), regime_series

    ret_series = pd.Series(
        data=[r for (_, r) in rets],
        index=[d for (d, _) in rets],
        name="ret_dyn_ensemble",
    ).sort_index()

    return ret_series, regime_series


def main():
    print("=" * 80)
    print("DYNAMIC REGIME-BASED ENSEMBLE V1")
    print("(ML9+Guard, QV, LowVol)")
    print("=" * 80)

    # 1) 수익률 로딩
    print("\nLoading returns...")
    ret_ml9 = load_returns_csv(RET_ML9_GUARD_PATH)
    ret_qv  = load_returns_csv(RET_QV_PATH)
    ret_lv  = load_returns_csv(RET_LOWVOL_PATH)
    
    print(f"  ML9+Guard: {len(ret_ml9)} days ({ret_ml9.index[0]} to {ret_ml9.index[-1]})")
    print(f"  QV: {len(ret_qv)} days ({ret_qv.index[0]} to {ret_qv.index[-1]})")
    print(f"  LowVol: {len(ret_lv)} days ({ret_lv.index[0]} to {ret_lv.index[-1]})")

    # 2) SPY 로딩
    print("\nLoading SPY prices...")
    spy_close = load_spy_close(SPY_PRICES_PATH)
    print(f"  SPY: {len(spy_close)} days ({spy_close.index[0]} to {spy_close.index[-1]})")

    # 공통 구간으로 제한 (ex: 2018-01-01~2024-12-31)
    start_date = "2018-01-01"
    end_date   = "2024-12-31"

    ret_ml9 = ret_ml9.loc[start_date:end_date]
    ret_qv  = ret_qv.loc[start_date:end_date]
    ret_lv  = ret_lv.loc[start_date:end_date]
    spy_close = spy_close.loc[start_date:end_date]

    # 3) 레짐별 weight 설정 (v1 하드코딩)
    regime_weights = RegimeWeights(
        bull={"ml9": 0.7, "qv": 0.2, "lv": 0.1},
        bear={"ml9": 0.3, "qv": 0.3, "lv": 0.4},
        high_vol={"ml9": 0.4, "qv": 0.2, "lv": 0.4},
        neutral={"ml9": 0.6, "qv": 0.2, "lv": 0.2},
    )

    print("\nRegime weights:")
    print(f"  BULL:     ML9={regime_weights.bull['ml9']:.1f}, QV={regime_weights.bull['qv']:.1f}, LV={regime_weights.bull['lv']:.1f}")
    print(f"  BEAR:     ML9={regime_weights.bear['ml9']:.1f}, QV={regime_weights.bear['qv']:.1f}, LV={regime_weights.bear['lv']:.1f}")
    print(f"  HIGH_VOL: ML9={regime_weights.high_vol['ml9']:.1f}, QV={regime_weights.high_vol['qv']:.1f}, LV={regime_weights.high_vol['lv']:.1f}")
    print(f"  NEUTRAL:  ML9={regime_weights.neutral['ml9']:.1f}, QV={regime_weights.neutral['qv']:.1f}, LV={regime_weights.neutral['lv']:.1f}")

    # 4) 동적 앙상블 수익률 생성
    print("\nBuilding dynamic ensemble...")
    ret_dyn, regime_series = build_dynamic_ensemble(
        ret_ml9,
        ret_qv,
        ret_lv,
        spy_close,
        regime_weights,
    )
    
    # Regime distribution
    regime_counts = regime_series.value_counts()
    print("\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    # 5) 전체 성과
    metrics_full = calculate_metrics(ret_dyn)
    print("\n" + "=" * 80)
    print("DYNAMIC ENSEMBLE FULL-PERIOD METRICS (2018-2024)")
    print("=" * 80)
    print(f"  Sharpe Ratio:      {metrics_full['sharpe']:.4f}")
    print(f"  Annual Return:     {metrics_full['ann_return']*100:.2f}%")
    print(f"  Annual Volatility: {metrics_full['ann_vol']*100:.2f}%")
    print(f"  Max Drawdown:      {metrics_full['maxdd']*100:.2f}%")
    print(f"  Win Rate:          {metrics_full['winrate']*100:.2f}%")

    # 6) 윈도우별 Sharpe (2018, 2021, 2024 등)
    windows = [
        ("2018", "2018-01-01", "2018-12-31"),
        ("2021", "2021-01-01", "2021-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
    ]

    window_sharpes: Dict[str, float] = {}
    print("\n" + "=" * 80)
    print("DYNAMIC ENSEMBLE WINDOW SHARPES")
    print("=" * 80)
    for name, start, end in windows:
        r_win = ret_dyn.loc[start:end]
        sh = calculate_sharpe(r_win)
        window_sharpes[name] = sh
        print(f"  {name}: {sh:.4f}")
    
    min_sharpe = min(window_sharpes.values())
    print(f"\n  Min Sharpe: {min_sharpe:.4f}")

    # 7) Stage 3 비교
    print("\n" + "=" * 80)
    print("COMPARISON WITH STAGE 3 (STATIC ENSEMBLE)")
    print("=" * 80)
    print("  Stage 3 Best (ML9 100%):")
    print("    Min Sharpe: 0.4688")
    print("    2018: 0.4688, 2021: 2.5395, 2024: 1.4181")
    print(f"\n  Stage 4 Dynamic:")
    print(f"    Min Sharpe: {min_sharpe:.4f}")
    print(f"    2018: {window_sharpes['2018']:.4f}, 2021: {window_sharpes['2021']:.4f}, 2024: {window_sharpes['2024']:.4f}")
    
    improvement = (min_sharpe - 0.4688) / 0.4688 * 100
    print(f"\n  Min Sharpe Improvement: {improvement:+.1f}%")

    # 8) Sharpe 2.0+ 평가
    print("\n" + "=" * 80)
    print("SHARPE 2.0+ ASSESSMENT")
    print("=" * 80)
    if min_sharpe >= 2.0:
        print("  ✅ Sharpe 2.0+ ACHIEVED!")
    elif min_sharpe >= 1.0:
        print(f"  ⚠️  Sharpe 2.0+ NOT achieved (current: {min_sharpe:.4f})")
        print(f"  Need {(2.0 - min_sharpe):.2f} more to reach 2.0")
        print("  Recommendation: VIX-based Guard OR additional engines")
    else:
        print(f"  ❌ Sharpe 2.0+ NOT achievable (current: {min_sharpe:.4f})")
        print(f"  Need {(2.0 / min_sharpe):.1f}x improvement")
        print("  Recommendation: Structural changes required")

    # 9) 결과 저장
    results = {
        "full_metrics": metrics_full,
        "window_sharpes": window_sharpes,
        "min_sharpe": float(min_sharpe),
        "regime_weights": {
            "bull": regime_weights.bull,
            "bear": regime_weights.bear,
            "high_vol": regime_weights.high_vol,
            "neutral": regime_weights.neutral,
        },
        "regime_distribution": {k: int(v) for k, v in regime_counts.items()},
    }

    out_path = BASE_DIR / "results" / "dynamic_ensemble_regime_v1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Dynamic Ensemble] Saved results to {out_path}")
    print("=" * 80)
    print("✅ STAGE 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
