# analysis/backtest_lowvol_v1.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import sys

import json
import numpy as np
import pandas as pd

# engines 모듈 import를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.factor_lowvol_v1 import FactorLowVolEngineV1, LowVolConfig

BASE_DIR = Path(__file__).resolve().parents[1]


def load_sp100_prices() -> pd.DataFrame:
    """SP100 가격 데이터 로딩"""
    path = BASE_DIR / "data" / "sp100_prices_raw.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["date", "ticker"])
    pivot = df.pivot(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()
    return pivot


def load_spx_close() -> pd.Series:
    """SPX/SPY 종가 데이터 로딩"""
    # SPY 데이터 사용 (이미 존재)
    path = BASE_DIR / "data" / "spy_prices.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    
    # 타임존 제거
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # 날짜만 남기기 (시간 제거)
    df.index = pd.to_datetime(df.index.date)
    
    s = df["close"]
    s.name = "SPX"
    return s


def get_monthly_rebalance_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    각 달의 첫 번째 거래일을 리밸 날짜로 사용.
    """
    df = pd.DataFrame(index=index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    first_days = df.groupby(["year", "month"]).apply(lambda x: x.index[0])
    return list(first_days.sort_values())


def portfolio_returns_from_weights(
    prices: pd.DataFrame,
    weights_by_date: Dict[pd.Timestamp, pd.Series],
    rebalance_dates: List[pd.Timestamp],
) -> pd.Series:
    """
    간단한 리밸 (no execution smoothing):
    - 리밸일 종가 기준 weight 적용
    - 다음날부터 다음 리밸 직전일까지 유지
    """
    prices = prices.sort_index()
    dates = prices.index

    rebalance_set = set(rebalance_dates)
    current_weights: pd.Series | None = None
    daily_returns = []

    for i in range(len(dates) - 1):
        date = dates[i]
        next_date = dates[i + 1]

        if date in rebalance_set and date in weights_by_date:
            current_weights = weights_by_date[date]

        if current_weights is None or current_weights.empty:
            continue

        # 수익률 계산
        daily_ret = 0.0
        for tkr, w in current_weights.items():
            if tkr not in prices.columns:
                continue
            px_now = prices.loc[date, tkr]
            px_next = prices.loc[next_date, tkr]
            if not np.isfinite(px_now) or px_now <= 0 or not np.isfinite(px_next):
                continue
            r = px_next / px_now - 1.0
            daily_ret += w * r

        daily_returns.append((next_date, daily_ret))

    if not daily_returns:
        return pd.Series(dtype=float)

    ret_series = pd.Series(
        data=[r for (_, r) in daily_returns],
        index=[d for (d, _) in daily_returns],
        name="ret_lowvol_v1",
    ).sort_index()

    return ret_series


def calculate_metrics(ret_daily: pd.Series) -> Dict:
    """백테스트 메트릭 계산"""
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


def load_returns_from_json(path: Path) -> pd.Series:
    """JSON 파일에서 수익률 시계열 로딩"""
    with open(path, "r") as f:
        data = json.load(f)
    
    if "daily_returns" in data:
        idx = pd.to_datetime(data["daily_returns"]["index"])
        vals = data["daily_returns"]["values"]
        return pd.Series(vals, index=idx)
    else:
        # 다른 형식 지원
        return pd.Series(dtype=float)


def main():
    print("=== Backtest LowVol v1 (SP100, 2015-2024) ===\n")

    # 데이터 로딩
    print("Loading data...")
    prices = load_sp100_prices()
    spx_close = load_spx_close()

    # 전체 기간 제한 (2015-01-01~2024-12-31)
    prices = prices.loc["2015-01-01":"2024-12-31"]
    spx_close = spx_close.reindex(prices.index, method="ffill")
    
    print(f"  Prices: {prices.shape[0]} days, {prices.shape[1]} tickers")
    print(f"  SPX: {len(spx_close)} days")
    print(f"  Period: {prices.index[0]} to {prices.index[-1]}\n")

    # 리밸런싱 날짜 (월간)
    rebalance_dates = get_monthly_rebalance_dates(prices.index)
    print(f"Rebalance dates: {len(rebalance_dates)} (monthly)\n")

    # LowVol 엔진 설정
    cfg = LowVolConfig(
        top_quantile=0.3,
        long_gross=1.0,
        short_gross=0.0,     # v0는 롱온리부터
        long_only=True,
        use_inverse_vol=True,
        vol_lookback=63,
        beta_lookback=252,
        beta_use=True,
        downside_vol_weight=0.5,
        beta_weight=0.5,
    )
    
    print("LowVol Engine Config:")
    print(f"  top_quantile: {cfg.top_quantile}")
    print(f"  long_gross: {cfg.long_gross}, short_gross: {cfg.short_gross}")
    print(f"  long_only: {cfg.long_only}")
    print(f"  use_inverse_vol: {cfg.use_inverse_vol}")
    print(f"  vol_lookback: {cfg.vol_lookback}, beta_lookback: {cfg.beta_lookback}")
    print(f"  beta_use: {cfg.beta_use}")
    print(f"  downside_vol_weight: {cfg.downside_vol_weight}, beta_weight: {cfg.beta_weight}\n")

    # 백테스트 실행
    print("Building portfolio...")
    engine = FactorLowVolEngineV1(cfg)
    weights_by_date = engine.build_portfolio(prices, spx_close, rebalance_dates)
    print(f"  Generated {len(weights_by_date)} rebalance weights\n")

    print("Calculating returns...")
    ret_lowvol = portfolio_returns_from_weights(prices, weights_by_date, rebalance_dates)
    print(f"  Generated {len(ret_lowvol)} daily returns\n")

    # 메트릭 계산
    metrics = calculate_metrics(ret_lowvol)

    print("=" * 60)
    print("LowVol v1 Performance Metrics:")
    print("=" * 60)
    print(f"  Sharpe Ratio:      {metrics['sharpe']:.4f}")
    print(f"  Annual Return:     {metrics['ann_return']*100:.2f}%")
    print(f"  Annual Volatility: {metrics['ann_vol']*100:.2f}%")
    print(f"  Max Drawdown:      {metrics['maxdd']*100:.2f}%")
    print(f"  Win Rate:          {metrics['winrate']*100:.2f}%")
    print("=" * 60)

    # ML9+Guard와 상관계수 계산
    print("\nCorrelation with ML9+Guard:")
    ml9_path = BASE_DIR / "results" / "ml9_guard_returns.csv"
    if ml9_path.exists():
        ml9_df = pd.read_csv(ml9_path, index_col=0)
        ml9_df.index = pd.to_datetime(ml9_df.index)
        if ml9_df.index.tz is not None:
            ml9_df.index = ml9_df.index.tz_localize(None)
        # 날짜만 남기기 (시간 제거)
        ml9_df.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in ml9_df.index])
        
        ml9_series = ml9_df.iloc[:, 0]  # 첫 번째 컬럼 사용
        ml9_series.name = "ML9"
        
        # LowVol 수익률도 날짜 정규화
        ret_lowvol_normalized = ret_lowvol.copy()
        ret_lowvol_normalized.index = pd.DatetimeIndex([pd.Timestamp(d.date()) for d in ret_lowvol.index])
        
        # 공통 날짜만 사용
        df_corr = pd.concat([ret_lowvol_normalized, ml9_series], axis=1, keys=["LowVol", "ML9"]).dropna()
        
        if len(df_corr) > 0:
            corr = df_corr.corr().iloc[0, 1]
            print(f"  Correlation(LowVol v1, ML9+Guard): {corr:.4f}")
            print(f"  Common days: {len(df_corr)}")
        else:
            print("  No common dates found")
    else:
        print(f"  ML9 returns file not found: {ml9_path}")

    # 결과 저장
    print("\nSaving results...")
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_returns = out_dir / "lowvol_v1_returns.csv"
    out_metrics = out_dir / "lowvol_v1_metrics.json"

    # 수익률 저장 (CSV)
    ret_df = pd.DataFrame({
        "date": ret_lowvol.index,
        "daily_return": ret_lowvol.values,
    })
    ret_df.to_csv(out_returns, index=False)

    # 메트릭 저장 (JSON)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Returns saved to: {out_returns}")
    print(f"  Metrics saved to: {out_metrics}")
    print("\n✅ LowVol v1 backtest complete!")


if __name__ == "__main__":
    main()
