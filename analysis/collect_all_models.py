#!/usr/bin/env python3
"""
전체 GitHub 리포지토리에서 백테스트 결과 수집 및 분석
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelResult:
    repo: str
    model_name: str
    file_path: str
    sharpe: float
    ann_return: float
    ann_vol: float
    maxdd: float
    days: int
    start_date: str
    end_date: str
    returns_file: Optional[str] = None

def calc_metrics(returns: pd.Series) -> Optional[Dict]:
    """수익률 시리즈로부터 성과 지표 계산"""
    returns = returns.dropna()
    if len(returns) == 0:
        return None
    
    days = (returns.index[-1] - returns.index[0]).days
    years = days / 365.25
    if years <= 0:
        return None
    
    cum = (1 + returns).prod()
    ann_ret = cum ** (1/years) - 1
    
    vol_d = returns.std()
    ann_vol = vol_d * np.sqrt(252)
    
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    maxdd = dd.min()
    
    return {
        'sharpe': float(sharpe),
        'ann_return': float(ann_ret),
        'ann_vol': float(ann_vol),
        'maxdd': float(maxdd),
        'days': len(returns),
        'start_date': returns.index[0].strftime('%Y-%m-%d'),
        'end_date': returns.index[-1].strftime('%Y-%m-%d')
    }

def scan_repo(repo_path: Path, repo_name: str) -> List[ModelResult]:
    """리포지토리에서 모든 백테스트 결과 수집"""
    results = []
    
    # results 디렉토리 확인
    results_dir = repo_path / "results"
    if not results_dir.exists():
        print(f"  [SKIP] {repo_name}: No results directory")
        return results
    
    # CSV 파일 스캔
    return_files = list(results_dir.glob("*_returns.csv")) + list(results_dir.glob("*returns*.csv"))
    
    for f in return_files:
        try:
            df = pd.read_csv(f, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 첫 번째 컬럼을 수익률로 사용
            returns = df.iloc[:, 0]
            
            metrics = calc_metrics(returns)
            if metrics:
                model_name = f.stem.replace('_returns', '').replace('returns_', '')
                
                result = ModelResult(
                    repo=repo_name,
                    model_name=model_name,
                    file_path=str(f.relative_to(repo_path)),
                    returns_file=str(f),
                    **metrics
                )
                results.append(result)
                print(f"  ✓ {model_name}: Sharpe {metrics['sharpe']:.3f}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
    
    return results

def main():
    print("=== GitHub 전체 리포지토리 모델 수집 ===\n")
    
    # 리포지토리 목록
    repos = [
        ("quant-ensemble-strategy", "/home/ubuntu/quant-ensemble-strategy"),
        ("ares7-ensemble", "/home/ubuntu/ares7-ensemble"),
        ("phoenix-pinnacle-ml", "/home/ubuntu/phoenix-pinnacle-ml"),
        ("ARES-X-V110", "/home/ubuntu/ARES-X-V110"),
    ]
    
    all_models = []
    
    for repo_name, repo_path_str in repos:
        repo_path = Path(repo_path_str)
        if not repo_path.exists():
            print(f"[SKIP] {repo_name}: Repository not found")
            continue
        
        print(f"\n📁 Scanning {repo_name}...")
        models = scan_repo(repo_path, repo_name)
        all_models.extend(models)
    
    # Sharpe 기준 정렬
    all_models_sorted = sorted(all_models, key=lambda m: m.sharpe, reverse=True)
    
    print(f"\n\n=== 전체 모델 성과 순위 (총 {len(all_models_sorted)}개) ===\n")
    print(f"{'Rank':<6} {'Repo':<30} {'Model':<40} {'Sharpe':<8} {'Return':<10} {'Vol':<10} {'MDD':<10} {'Period'}")
    print("=" * 140)
    
    for i, m in enumerate(all_models_sorted[:20], 1):
        print(f"{i:<6} {m.repo:<30} {m.model_name:<40} {m.sharpe:<8.3f} {m.ann_return*100:<10.2f}% {m.ann_vol*100:<10.2f}% {m.maxdd*100:<10.2f}% {m.start_date} ~ {m.end_date}")
    
    # JSON 저장
    output_path = Path("/home/ubuntu/quant-ensemble-strategy/results/all_models_collected.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump([asdict(m) for m in all_models_sorted], f, indent=2)
    
    print(f"\n✅ 결과 저장: {output_path}")
    print(f"   총 {len(all_models_sorted)}개 모델 수집 완료")
    
    # 상위 5개 모델 정보
    print(f"\n\n🏆 상위 5개 모델 (ML9-Guard-v2와 크로스 앙상블 후보):")
    for i, m in enumerate(all_models_sorted[:5], 1):
        print(f"\n{i}. {m.model_name} ({m.repo})")
        print(f"   Sharpe: {m.sharpe:.3f}, Return: {m.ann_return*100:.2f}%, MDD: {m.maxdd*100:.2f}%")
        print(f"   Period: {m.start_date} ~ {m.end_date} ({m.days}일)")
        print(f"   File: {m.returns_file}")

if __name__ == "__main__":
    main()
