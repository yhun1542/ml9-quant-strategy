# analysis/ensemble_3engine_minmax_v1.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# ===== 경로 설정 (필요시 수정) =====
RET_ML9_GUARD_PATH = BASE_DIR / "results" / "ml9_guard_returns.csv"
RET_QV_PATH        = BASE_DIR / "results" / "qv_returns.csv"
RET_LOWVOL_PATH    = BASE_DIR / "results" / "lowvol_v1_returns.csv"


@dataclass
class TestWindow:
    name: str
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


def load_returns_from_csv(path: Path) -> pd.Series:
    """CSV 파일에서 수익률 시계열 로딩"""
    if not path.exists():
        raise FileNotFoundError(f"Return file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # 타임존 제거
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # 날짜만 남기기
    df.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in df.index])
    
    # 첫 번째 컬럼을 수익률로 사용
    s = df.iloc[:, 0]
    s = s.sort_index()
    s.name = path.stem
    return s


def load_lowvol_returns(path: Path) -> pd.Series:
    """
    lowvol_v1_returns.csv:
      - 예: columns: ['date', 'daily_return'] or index=date, col='daily_return'
    이 함수에서 최대한 유연하게 처리.
    """
    if not path.exists():
        raise FileNotFoundError(f"LowVol return file not found: {path}")

    df = pd.read_csv(path)
    # date 컬럼 찾기
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        # 첫 컬럼이 date일 가능성
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])

    # 타임존 제거
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # 날짜만 남기기
    df.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in df.index])

    # 수익률 컬럼 찾기
    if "daily_return" in df.columns:
        s = df["daily_return"]
    elif "ret" in df.columns:
        s = df["ret"]
    elif "ret_lowvol" in df.columns:
        s = df["ret_lowvol"]
    else:
        # 첫 번째 수익률 컬럼 사용
        s = df.iloc[:, 0]

    s = s.sort_index()
    s.name = "lowvol_v1"
    return s


def calculate_sharpe(ret_daily: pd.Series) -> float:
    """Sharpe Ratio 계산"""
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
    """전체 메트릭 계산"""
    ret_daily = ret_daily.dropna()
    if ret_daily.empty:
        return {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "maxdd": 0.0,
        }

    days = (ret_daily.index[-1] - ret_daily.index[0]).days
    years = days / 365.25
    if years <= 0:
        return {
            "sharpe": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "maxdd": 0.0,
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

    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "maxdd": float(maxdd),
    }


def grid_search_minmax_sharpe_3engines(
    ret_ml9: pd.Series,
    ret_qv: pd.Series,
    ret_lv: pd.Series,
    windows: List[TestWindow],
    step: float = 0.1,
) -> List[Dict]:
    """
    3-엔진 앙상블 (ML9+Guard, QV, LowVol)에 대해:
      - w_ml9, w_qv, w_lv ≥ 0
      - w_ml9 + w_qv + w_lv = 1
    를 그리드로 탐색하고,
      - 각 윈도우 Sharpe
      - min_sharpe (해당 조합의 최악 윈도우 Sharpe)
    를 계산한 뒤, min_sharpe 기준 내림차순 정렬 리스트 반환.
    """
    results: List[Dict] = []

    df = pd.concat(
        [
            ret_ml9.rename("ml9"),
            ret_qv.rename("qv"),
            ret_lv.rename("lv"),
        ],
        axis=1,
    ).dropna()

    print(f"\nCombined data: {len(df)} days")
    print(f"  Period: {df.index[0]} to {df.index[-1]}")

    # 윈도우별 슬라이스
    window_slices: Dict[str, pd.DataFrame] = {}
    for w in windows:
        df_w = df.loc[w.start : w.end]
        if df_w.empty:
            print(f"[WARN] Window {w.name} ({w.start}~{w.end}) has no data")
        else:
            print(f"  Window {w.name}: {len(df_w)} days ({df_w.index[0]} to {df_w.index[-1]})")
        window_slices[w.name] = df_w

    ws = np.arange(0.0, 1.0 + 1e-9, step)
    total_combinations = 0

    print(f"\nGrid search: step={step}")
    print(f"  Total weight points: {len(ws)}")

    for w_ml9 in ws:
        for w_qv in ws:
            w_lv = 1.0 - w_ml9 - w_qv
            if w_lv < -1e-9:
                continue
            if w_lv < 0:
                # 음수는 제외
                continue

            total_combinations += 1

            window_sharpes: Dict[str, float] = {}
            min_sharpe = float("inf")

            for w in windows:
                df_w = window_slices[w.name]
                if df_w.empty:
                    window_sharpes[w.name] = 0.0
                    min_sharpe = min(min_sharpe, 0.0)
                    continue

                ret_combo = (
                    w_ml9 * df_w["ml9"] +
                    w_qv  * df_w["qv"] +
                    w_lv  * df_w["lv"]
                )
                sh = calculate_sharpe(ret_combo)
                window_sharpes[w.name] = sh
                if sh < min_sharpe:
                    min_sharpe = sh

            results.append(
                {
                    "w_ml9": float(w_ml9),
                    "w_qv": float(w_qv),
                    "w_lv": float(w_lv),
                    "min_sharpe": float(min_sharpe),
                    "window_sharpes": window_sharpes,
                }
            )

    print(f"  Total combinations tested: {total_combinations}")

    results_sorted = sorted(results, key=lambda r: r["min_sharpe"], reverse=True)
    return results_sorted


def main():
    print("=" * 80)
    print("3-ENGINE ENSEMBLE MIN-MAX SEARCH")
    print("(ML9+Guard, QV, LowVol)")
    print("=" * 80)

    # 1) 수익률 로딩
    print("\nLoading returns...")
    ret_ml9 = load_returns_from_csv(RET_ML9_GUARD_PATH)
    print(f"  ML9+Guard: {len(ret_ml9)} days ({ret_ml9.index[0]} to {ret_ml9.index[-1]})")
    
    ret_qv  = load_returns_from_csv(RET_QV_PATH)
    print(f"  QV: {len(ret_qv)} days ({ret_qv.index[0]} to {ret_qv.index[-1]})")
    
    ret_lv  = load_lowvol_returns(RET_LOWVOL_PATH)
    print(f"  LowVol: {len(ret_lv)} days ({ret_lv.index[0]} to {ret_lv.index[-1]})")

    # 날짜 align
    start_date = "2018-01-01"   # 필요 시 2015-01-01로 확장 가능
    end_date   = "2024-12-31"

    ret_ml9 = ret_ml9.loc[start_date:end_date]
    ret_qv  = ret_qv.loc[start_date:end_date]
    ret_lv  = ret_lv.loc[start_date:end_date]

    print(f"\nAligned period: {start_date} to {end_date}")
    print(f"  ML9+Guard: {len(ret_ml9)} days")
    print(f"  QV: {len(ret_qv)} days")
    print(f"  LowVol: {len(ret_lv)} days")

    # 2) 테스트 윈도우 정의 (실제 데이터에 맞게 수정)
    # ML9 데이터는 2018-2019, 2021-2022, 2024만 존재
    windows = [
        TestWindow("2018",      "2018-01-01", "2018-12-31"),
        TestWindow("2021",      "2021-01-01", "2021-12-31"),
        TestWindow("2024",      "2024-01-01", "2024-12-31"),
    ]

    print(f"\nTest windows:")
    for w in windows:
        print(f"  {w.name}: {w.start} to {w.end}")

    # 3) 그리드 서치
    print("\n" + "=" * 80)
    print("GRID SEARCH")
    print("=" * 80)
    
    results_sorted = grid_search_minmax_sharpe_3engines(
        ret_ml9, ret_qv, ret_lv, windows, step=0.1
    )

    # 4) 상위 20개 출력
    print("\n" + "=" * 80)
    print("TOP 20 COMBINATIONS BY MIN_SHARPE")
    print("=" * 80)
    print(f"{'Rank':<6} {'w_ML9':<8} {'w_QV':<8} {'w_LV':<8} {'MinSharpe':<12} {'Window Sharpes'}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted[:20], 1):
        window_str = ", ".join([f"{k}={v:.2f}" for k, v in r['window_sharpes'].items()])
        print(
            f"{i:<6} {r['w_ml9']:<8.2f} {r['w_qv']:<8.2f} {r['w_lv']:<8.2f} "
            f"{r['min_sharpe']:<12.4f} {window_str}"
        )

    # 최고 min_sharpe 확인
    best = results_sorted[0]
    print("\n" + "=" * 80)
    print("BEST COMBINATION")
    print("=" * 80)
    print(f"  w_ML9:      {best['w_ml9']:.2f}")
    print(f"  w_QV:       {best['w_qv']:.2f}")
    print(f"  w_LowVol:   {best['w_lv']:.2f}")
    print(f"  Min Sharpe: {best['min_sharpe']:.4f}")
    print(f"\n  Window Sharpes:")
    for k, v in best['window_sharpes'].items():
        print(f"    {k}: {v:.4f}")

    # Sharpe 2.0+ 달성 여부
    print("\n" + "=" * 80)
    print("SHARPE 2.0+ ASSESSMENT")
    print("=" * 80)
    
    sharpe_2_combos = [r for r in results_sorted if r['min_sharpe'] >= 2.0]
    
    if sharpe_2_combos:
        print(f"✅ Found {len(sharpe_2_combos)} combinations with min_sharpe ≥ 2.0")
        print(f"\n  Best combination:")
        best_2 = sharpe_2_combos[0]
        print(f"    w_ML9={best_2['w_ml9']:.2f}, w_QV={best_2['w_qv']:.2f}, w_LV={best_2['w_lv']:.2f}")
        print(f"    Min Sharpe: {best_2['min_sharpe']:.4f}")
    else:
        print(f"❌ No combinations with min_sharpe ≥ 2.0 found")
        print(f"   Maximum achievable min_sharpe: {best['min_sharpe']:.4f}")
        print(f"\n   Conclusion: Sharpe 2.0+ is NOT achievable with static 3-engine ensemble")
        print(f"   Need: Dynamic weighting OR additional engines OR cost optimization")

    # 전체 기간 메트릭 (최적 조합)
    print("\n" + "=" * 80)
    print("FULL PERIOD METRICS (Best Combination)")
    print("=" * 80)
    
    df_full = pd.concat([ret_ml9, ret_qv, ret_lv], axis=1, keys=["ml9", "qv", "lv"]).dropna()
    ret_best = (
        best['w_ml9'] * df_full["ml9"] +
        best['w_qv'] * df_full["qv"] +
        best['w_lv'] * df_full["lv"]
    )
    metrics_best = calculate_metrics(ret_best)
    
    print(f"  Sharpe Ratio:      {metrics_best['sharpe']:.4f}")
    print(f"  Annual Return:     {metrics_best['ann_return']*100:.2f}%")
    print(f"  Annual Volatility: {metrics_best['ann_vol']*100:.2f}%")
    print(f"  Max Drawdown:      {metrics_best['maxdd']*100:.2f}%")

    # 5) JSON 저장
    out_path = BASE_DIR / "results" / "ensemble_3engine_minmax_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 상위 100개만 저장 (파일 크기 제한)
    with open(out_path, "w") as f:
        json.dump(
            {
                "best_combination": best,
                "sharpe_2_0_achievable": len(sharpe_2_combos) > 0,
                "max_min_sharpe": best['min_sharpe'],
                "full_period_metrics": metrics_best,
                "top_100_combinations": results_sorted[:100],
            },
            f,
            indent=2,
        )

    print(f"\n[3-Engine MinMax] Saved results to {out_path}")
    print("\n" + "=" * 80)
    print("✅ STAGE 3 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
