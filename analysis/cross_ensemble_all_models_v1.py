#!/usr/bin/env python3
"""
ML9-Guard-v2와 다른 모델들의 크로스 앙상블 테스트
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass
class TestWindow:
    name: str
    start: str
    end: str

def load_returns_from_csv(path: Path) -> pd.Series:
    """CSV 파일에서 수익률 로딩"""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # 타임존 제거 및 날짜만 추출
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()  # 시간 부분 제거
    
    returns = df.iloc[:, 0]
    returns.name = path.stem.replace('_returns', '')
    return returns

def load_returns_from_json(path: Path, key: str = "daily_returns") -> pd.Series:
    """JSON 파일에서 수익률 로딩"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    if key in data:
        dr = data[key]
        if isinstance(dr, dict) and 'index' in dr and 'values' in dr:
            idx = pd.to_datetime(dr['index'])
            vals = dr['values']
        elif isinstance(dr, list):
            # 리스트 형식인 경우
            vals = dr
            # 인덱스는 별도로 찾아야 함
            return None
        else:
            return None
    else:
        return None
    
    s = pd.Series(vals, index=idx).sort_index()
    
    # 타임존 제거
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    
    s.name = path.stem
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
    """전체 성과 지표 계산"""
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

def grid_search_two_models(
    ret_a: pd.Series,
    ret_b: pd.Series,
    name_a: str,
    name_b: str,
    windows: List[TestWindow],
    step: float = 0.1,
) -> List[Dict]:
    """
    두 모델의 크로스 앙상블 그리드 서치
    """
    results: List[Dict] = []
    
    # 공통 날짜로 정렬
    df = pd.concat([ret_a.rename('a'), ret_b.rename('b')], axis=1).dropna()
    
    if df.empty:
        print(f"[ERROR] No common dates between {name_a} and {name_b}")
        return results
    
    print(f"  Common period: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days)")
    
    # 윈도우별 슬라이스
    window_slices: Dict[str, pd.DataFrame] = {}
    for w in windows:
        df_w = df.loc[w.start : w.end]
        if df_w.empty:
            print(f"  [WARN] Window {w.name} ({w.start}~{w.end}) has no data")
        window_slices[w.name] = df_w
    
    ws = np.arange(0.0, 1.0 + 1e-9, step)
    
    for w_a in ws:
        w_b = 1.0 - w_a
        if w_b < -1e-9:
            continue
        
        window_sharpes: Dict[str, float] = {}
        min_sharpe = float("inf")
        
        for w in windows:
            df_w = window_slices[w.name]
            if df_w.empty:
                window_sharpes[w.name] = 0.0
                min_sharpe = min(min_sharpe, 0.0)
                continue
            
            ret_combo = w_a * df_w['a'] + w_b * df_w['b']
            sh = calculate_sharpe(ret_combo)
            window_sharpes[w.name] = sh
            if sh < min_sharpe:
                min_sharpe = sh
        
        # 전체 기간 성과
        ret_full = w_a * df['a'] + w_b * df['b']
        metrics_full = calculate_metrics(ret_full)
        
        results.append(
            {
                f"w_{name_a}": float(w_a),
                f"w_{name_b}": float(w_b),
                "min_sharpe": float(min_sharpe),
                "window_sharpes": window_sharpes,
                "full_metrics": metrics_full,
            }
        )
    
    results_sorted = sorted(results, key=lambda r: r["min_sharpe"], reverse=True)
    return results_sorted

def main():
    print("=== ML9-Guard-v2 크로스 앙상블 테스트 ===\n")
    
    # ML9-Guard-v2 (테스트 모델)
    ml9_guard_path = BASE_DIR / "results" / "ml9_guard_returns.csv"
    ret_ml9_guard = load_returns_from_csv(ml9_guard_path)
    print(f"✓ ML9-Guard-v2 loaded: {len(ret_ml9_guard)} days ({ret_ml9_guard.index[0].date()} ~ {ret_ml9_guard.index[-1].date()})")
    
    # 비교 모델들
    comparison_models = [
        ("ARES7-Best", "/home/ubuntu/ares7-ensemble/results/ares7_best_ensemble_results.json", "json"),
        ("QV", BASE_DIR / "results" / "qv_returns.csv", "csv"),
        ("LowVol-v1", BASE_DIR / "results" / "lowvol_v1_returns.csv", "csv"),
    ]
    
    # 테스트 윈도우
    windows = [
        TestWindow("2018", "2018-01-01", "2018-12-31"),
        TestWindow("2021", "2021-01-01", "2021-12-31"),
        TestWindow("2024", "2024-01-01", "2024-12-31"),
    ]
    
    all_results = {}
    
    for model_name, model_path, file_type in comparison_models:
        print(f"\n{'='*60}")
        print(f"크로스 앙상블: ML9-Guard-v2 vs {model_name}")
        print(f"{'='*60}")
        
        # 모델 로딩
        if file_type == "csv":
            ret_model = load_returns_from_csv(Path(model_path))
        elif file_type == "json":
            # ARES7 JSON 파일 처리
            with open(model_path, 'r') as f:
                data = json.load(f)
            
            if 'daily_returns' in data:
                dr = data['daily_returns']
                if isinstance(dr, list) and len(dr) > 0 and 'date' in dr[0] and 'ret' in dr[0]:
                    # ARES7 형식: [{'date': ..., 'ret': ...}, ...]
                    dates = [item['date'] for item in dr]
                    rets = [item['ret'] for item in dr]
                    idx = pd.to_datetime(dates)
                    ret_model = pd.Series(rets, index=idx).sort_index()
                    ret_model.name = model_name
                elif isinstance(dr, dict) and 'dates' in dr and 'returns' in dr:
                    idx = pd.to_datetime(dr['dates'])
                    vals = dr['returns']
                    ret_model = pd.Series(vals, index=idx).sort_index()
                    ret_model.name = model_name
                else:
                    print(f"  [ERROR] Unsupported JSON format for {model_name}")
                    print(f"  daily_returns type: {type(dr)}")
                    if isinstance(dr, list) and len(dr) > 0:
                        print(f"  First item: {dr[0]}")
                    continue
            else:
                print(f"  [ERROR] No daily_returns in {model_name}")
                continue
        
        if ret_model is None or ret_model.empty:
            print(f"  [SKIP] Failed to load {model_name}")
            continue
        
        print(f"✓ {model_name} loaded: {len(ret_model)} days ({ret_model.index[0].date()} ~ {ret_model.index[-1].date()})")
        
        # 그리드 서치
        results = grid_search_two_models(
            ret_ml9_guard, ret_model,
            "ML9_Guard_v2", model_name,
            windows, step=0.1
        )
        
        if not results:
            print(f"  [SKIP] No results for {model_name}")
            continue
        
        # 상위 5개 출력
        print(f"\n  Top 5 combinations:")
        for i, r in enumerate(results[:5], 1):
            fm = r['full_metrics']
            print(
                f"    {i}. w_ML9={r['w_ML9_Guard_v2']:.1f}, w_{model_name}={r[f'w_{model_name}']:.1f}, "
                f"min_sharpe={r['min_sharpe']:.2f}, full_sharpe={fm['sharpe']:.2f}"
            )
        
        all_results[model_name] = results
    
    # 전체 결과 저장
    output_path = BASE_DIR / "results" / "cross_ensemble_all_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n✅ 전체 결과 저장: {output_path}")
    
    # 최고 조합 출력
    print(f"\n\n🏆 최고 성과 조합:")
    for model_name, results in all_results.items():
        if results:
            best = results[0]
            fm = best['full_metrics']
            print(f"\n  ML9-Guard-v2 vs {model_name}:")
            print(f"    가중치: ML9={best['w_ML9_Guard_v2']:.1f}, {model_name}={best[f'w_{model_name}']:.1f}")
            print(f"    Min Sharpe: {best['min_sharpe']:.3f}")
            print(f"    Full Sharpe: {fm['sharpe']:.3f}")
            print(f"    Return: {fm['ann_return']*100:.2f}%, MDD: {fm['maxdd']*100:.2f}%")

if __name__ == "__main__":
    main()
