#!/usr/bin/env python
# coding: utf-8

"""
Unified script for the complete quantitative trading strategy backtest and optimization.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
import requests
import nasdaqdatalink
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import MarketConditionGuard
sys.path.insert(0, str(Path(__file__).resolve().parent))
from modules.market_guard_ml9 import ML9MarketConditionGuard

# --- Setup ---
BASE_DIR = Path(__file__).resolve().parent
(BASE_DIR / "data").mkdir(exist_ok=True)
(BASE_DIR / "results").mkdir(exist_ok=True)

# --- Constants ---
TRADING_DAYS = 252
SP100_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "XOM", "V", "PG", "JPM", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "COST", "AVGO", "KO", "ADBE", "WMT", "MCD", "CSCO", "ACN", "TMO", "LIN",
    "ABT", "NFLX", "DHR", "NKE", "VZ", "CRM", "TXN", "NEE", "ORCL", "PM",
    "WFC", "DIS", "UPS", "BMY", "RTX", "AMGN", "HON", "LOW", "QCOM", "UNP",
    "MS", "COP", "SPGI", "BA", "INTU", "SBUX", "GE", "CAT", "AMD", "PLD",
    "AMAT", "BLK", "DE", "MDT", "LMT", "GILD", "ADP", "ADI", "BKNG", "TJX",
    "ISRG", "CI", "MMC", "VRTX", "SYK", "C", "ZTS", "REGN", "PGR", "MO",
    "CB", "DUK", "SO", "BDX", "EOG", "TGT", "ITW", "USB", "SCHW", "PNC",
    "AON", "BSX", "CME", "GS", "MU", "SLB", "NOC", "MMM", "FI", "ICE"
]

# --- API Keys ---
POLYGON_API_KEY = "w7KprL4_lK7uutSH0dYGARkucXHOFXCN"
SHARADAR_API_KEY = "H6zH4Q2CDr9uTFk9koqJ"

#<editor-fold desc="Data Loaders">
class Polygon:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    def get_daily_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        all_data = []
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] Downloading Price: {ticker}... ", end="")
            try:
                data = self._get_prices_for_ticker(ticker, start_date, end_date)
                if not data.empty:
                    all_data.append(data)
                    print(f"✓ {len(data)} days")
                else:
                    print("✗ No data")
            except Exception as e:
                print(f"✗ Error: {e}")
            time.sleep(0.5)
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def _get_prices_for_ticker(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key}
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        data = res.json()
        if "results" not in data or not data["results"]:
            return pd.DataFrame()
        df = pd.DataFrame(data["results"])
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        return df[["date", "ticker", "c"]].rename(columns={"c": "close"})

    def get_spy_prices(self, start_date: str, end_date: str) -> pd.Series:
        """Download SPY prices for MarketConditionGuard"""
        print(f"Downloading SPY prices for Guard... ", end="")
        try:
            data = self._get_prices_for_ticker("SPY", start_date, end_date)
            if not data.empty:
                data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
                spy_series = data.set_index("date")["close"]
                print(f"✓ {len(spy_series)} days")
                return spy_series
            else:
                print("✗ No data")
                return pd.Series(dtype=float)
        except Exception as e:
            print(f"✗ Error: {e}")
            return pd.Series(dtype=float)

class SF1:
    def __init__(self, api_key: str):
        self.api_key = api_key
        nasdaqdatalink.ApiConfig.api_key = api_key

    def get_sf1_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        indicators = ["pe", "pb", "ps", "evebitda", "roe", "ebitdamargin", "de", "currentratio"]
        try:
            df = nasdaqdatalink.get_table(
                "SHARADAR/SF1", ticker=tickers, dimension="ART",
                calendardate={"gte": start_date, "lte": end_date},
                qopts={"columns": ["ticker", "calendardate", "datekey", "reportperiod"] + indicators},
                paginate=True)
        except Exception as e:
            raise ValueError(f"SF1 API error: {e}")
        if df.empty:
            return pd.DataFrame()
        for col in ["datekey", "reportperiod", "calendardate"]:
            df[col] = pd.to_datetime(df[col])
        return df.sort_values(["ticker", "datekey", "reportperiod"])
#</editor-fold>

#<editor-fold desc="Performance Metrics">
@dataclass
class PerformanceMetrics:
    sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    win_rate: float
    num_trades: int

def calculate_metrics(returns: pd.Series) -> PerformanceMetrics:
    returns = returns.fillna(0.0)
    if returns.empty or returns.std() == 0:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(TRADING_DAYS) if std_ret > 0 else 0
    annual_return = mean_ret * TRADING_DAYS
    annual_vol = std_ret * np.sqrt(TRADING_DAYS)
    cum_ret = (1.0 + returns).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    return PerformanceMetrics(sharpe, annual_return, annual_vol, max_dd, win_rate, len(returns))
#</editor-fold>

#<editor-fold desc="Trading Engines">
class ML9Engine:
    def __init__(self, prices: pd.DataFrame, factors: pd.DataFrame, top_quantile: float = 0.2, prediction_horizon: int = 10, guard: Optional[ML9MarketConditionGuard] = None):
        prices["date"] = pd.to_datetime(prices["date"])
        self.prices_pivot = prices.pivot(index='date', columns='ticker', values='close')
        self.factors = factors.copy()
        self.factors["date"] = pd.to_datetime(self.factors["date"])
        self.factors.set_index(['date', 'ticker'], inplace=True)
        # MultiIndex 정렬 (슬라이싱을 위해 필수)
        self.factors = self.factors.sort_index()
        self.factors["value_proxy_inv"] = 1.0 / self.factors["value_proxy"].replace(0, np.nan)
        self._apply_cross_sectional_ranking()
        self.top_quantile = top_quantile
        self.prediction_horizon = prediction_horizon
        self.xgb_params = {
            "objective": "multi:softprob", "num_class": 3, "max_depth": 5,
            "learning_rate": 0.05, "n_estimators": 200, "subsample": 0.7,
            "colsample_bytree": 0.7, "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42,
        }
        self.guard = guard  # MarketConditionGuard instance

    def _apply_cross_sectional_ranking(self):
        all_ranks = []
        for col in ["momentum_60d", "value_proxy_inv", "volatility_30d"]:
            s = self.factors[col].unstack()
            z_scores = s.apply(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0, axis=1)
            ranks = z_scores.stack().rename(f"{col}_rank")
            all_ranks.append(ranks)
        self.factors = self.factors.join(all_ranks)

    def _prepare_ml_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"  Preparing ML dataset from {start_date.date()} to {end_date.date()}")
        data_slice = self.factors.loc[start_date:end_date].copy()
        fwd_returns = self.prices_pivot.pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        data_slice['fwd_return'] = fwd_returns.stack().rename('fwd_return')

        def quantile_target(x):
            q_low, q_high = x.quantile(self.top_quantile), x.quantile(1 - self.top_quantile)
            if pd.isna(q_low) or pd.isna(q_high) or q_low == q_high:
                return pd.Series(1, index=x.index)
            return x.apply(lambda r: 0 if r <= q_low else (2 if r >= q_high else 1))

        data_slice["target"] = data_slice.groupby(level="date")["fwd_return"].transform(quantile_target)
        feature_cols = ["momentum_60d_rank", "value_proxy_inv_rank", "volatility_30d_rank"]
        final_data = data_slice.dropna(subset=feature_cols + ['fwd_return', "target"])
        X = final_data[feature_cols]
        y = final_data["target"]
        print(f"    Dataset size: {len(X)} samples")
        return X, y

    def run_walk_forward_backtest(self, start_date: str, end_date: str, train_period_months: int = 36, test_period_months: int = 12):
        print("\n--- Running ML9 Walk-Forward Backtest ---")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        all_returns = []
        current_date = start
        while current_date < end:
            train_end = current_date + pd.DateOffset(months=train_period_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_period_months)
            if test_end > end:
                test_end = end
            if test_start >= test_end:
                break
            print(f"\n  Training: {current_date.date()} - {train_end.date()} | Testing: {test_start.date()} - {test_end.date()}")
            X_train, y_train = self._prepare_ml_dataset(current_date, train_end)
            if X_train.empty or len(X_train) < 100:
                print("    Skipping period due to insufficient training data.")
                current_date = test_start
                continue
            scaler = StandardScaler()
            model = xgb.XGBClassifier(**self.xgb_params)
            model.fit(scaler.fit_transform(X_train), y_train)
            period_returns = self._run_test_period(test_start, test_end, model, scaler)
            all_returns.append(period_returns)
            current_date = test_start
        if not all_returns:
            return pd.Series(dtype=float), PerformanceMetrics(0,0,0,0,0,0)
        final_returns = pd.concat(all_returns)
        metrics = calculate_metrics(final_returns)
        return final_returns, metrics

    def _run_test_period(self, start_date, end_date, model, scaler):
        # 리밸런싱 날짜 생성 (월말)
        rebal_dates = self.prices_pivot.loc[start_date:end_date].resample("ME").first().index
        # 리밸런싱 날짜를 실제 거래일로 매핑
        rebal_dates_actual = []
        for rebal_date in rebal_dates:
            # 해당 월말에 가장 가까운 거래일 찾기
            available_dates = self.factors.index.get_level_values('date').unique()
            closest_date = min(available_dates, key=lambda x: abs((x - rebal_date).total_seconds()))
            if abs((closest_date - rebal_date).days) <= 3:  # 3일 이내면 매칭
                rebal_dates_actual.append(closest_date)
        
        daily_returns = pd.Series(index=pd.date_range(start_date, end_date, freq="B"), dtype=float)
        portfolio = {}
        
        for date in daily_returns.index:
            # 리밸런싱 날짜인지 확인
            if any(abs((date - rd).days) == 0 for rd in rebal_dates_actual):
                # 가장 가까운 데이터 날짜 찾기
                available_dates = self.factors.index.get_level_values('date').unique()
                closest_date = min(available_dates, key=lambda x: abs((x - date).total_seconds()))
                
                if closest_date in self.factors.index.get_level_values('date'):
                    factors_at_date = self.factors.loc[pd.IndexSlice[closest_date, :], :]
                    if not factors_at_date.empty:
                        feature_cols = ["momentum_60d_rank", "value_proxy_inv_rank", "volatility_30d_rank"]
                        X_test = factors_at_date[feature_cols].dropna()
                        if not X_test.empty:
                            probas = model.predict_proba(scaler.transform(X_test))[:, 2]
                            predictions = pd.Series(probas, index=X_test.index)
                            long_tickers = predictions.nlargest(int(len(predictions) * self.top_quantile)).index.get_level_values("ticker")
                            
                            # Apply MarketConditionGuard scale factor
                            guard_scale = 1.0
                            if self.guard is not None:
                                guard_scale = self.guard.get_ml9_scale(date)
                            
                            base_weight = 1.0 / len(long_tickers) if len(long_tickers) > 0 else 0.0
                            portfolio = {ticker: base_weight * guard_scale for ticker in long_tickers} if len(long_tickers) > 0 else {}
            
            if not portfolio:
                daily_returns[date] = 0.0
                continue
            
            # 수익률 계산
            day_ret = 0.0
            # 가장 가까운 가격 날짜 찾기
            price_dates = self.prices_pivot.index
            closest_today = min(price_dates, key=lambda x: abs((x - date).total_seconds()))
            
            # 어제 날짜는 오늘보다 이전 날짜 중 가장 가까운 것
            prev_dates = [d for d in price_dates if d < closest_today]
            if prev_dates:
                closest_yesterday = max(prev_dates)
                
                for ticker, weight in portfolio.items():
                    if ticker in self.prices_pivot.columns:
                        px_today = self.prices_pivot.loc[closest_today, ticker]
                        px_yst = self.prices_pivot.loc[closest_yesterday, ticker]
                        if pd.notna(px_today) and pd.notna(px_yst) and px_yst != 0:
                            day_ret += weight * (px_today / px_yst - 1)
            
            daily_returns[date] = day_ret
        return daily_returns

class QVEngine:
    def __init__(self, top_quantile: float = 0.3, long_only: bool = True, use_inverse_vol: bool = True, vol_lookback: int = 63):
        self.top_quantile = top_quantile
        self.long_only = long_only
        self.use_inverse_vol = use_inverse_vol
        self.vol_lookback = vol_lookback

    def build_signals(self, fund_daily: pd.DataFrame) -> pd.Series:
        # date 컬럼이 이미 인덱스에 있는지 확인
        if 'date' in fund_daily.columns and 'ticker' in fund_daily.columns:
            fund_daily_indexed = fund_daily.set_index(['date', 'ticker'])
        elif isinstance(fund_daily.index, pd.MultiIndex):
            fund_daily_indexed = fund_daily
        else:
            raise ValueError("fund_daily must have 'date' and 'ticker' columns or MultiIndex")
        
        value = self._compute_value_score(fund_daily_indexed)
        quality = self._compute_quality_score(fund_daily_indexed)
        qv_raw = 0.5 * quality + 0.5 * value
        return self._xsec_zscore(qv_raw).rename("qv_score")

    def _compute_value_score(self, fund_daily: pd.DataFrame) -> pd.Series:
        pe = fund_daily["pe"]
        pb = fund_daily["pb"]
        ps = fund_daily["ps"]
        evebitda = fund_daily["evebitda"]
        z_pe = self._xsec_zscore(-pe)
        z_pb = self._xsec_zscore(-pb)
        z_ps = self._xsec_zscore(-ps)
        z_ev = self._xsec_zscore(-evebitda)
        value_raw = 0.25 * z_pe + 0.25 * z_pb + 0.25 * z_ps + 0.25 * z_ev
        return self._xsec_zscore(value_raw)

    def _compute_quality_score(self, fund_daily: pd.DataFrame) -> pd.Series:
        roe = fund_daily["roe"]
        op_mgn = fund_daily["ebitdamargin"]
        d2e = fund_daily["de"]
        curr_ratio = fund_daily["currentratio"]
        z_roe = self._xsec_zscore(roe)
        z_mgn = self._xsec_zscore(op_mgn)
        z_lev = self._xsec_zscore(-d2e)
        z_liq = self._xsec_zscore(curr_ratio)
        quality_raw = 0.35 * z_roe + 0.25 * z_mgn + 0.25 * z_lev + 0.15 * z_liq
        return self._xsec_zscore(quality_raw)

    def _xsec_zscore(self, s: pd.Series, winsor_pct: float = 0.01) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        def _winsor_zscore(x: pd.Series) -> pd.Series:
            lower, upper = x.quantile(winsor_pct), x.quantile(1 - winsor_pct)
            x_clipped = x.clip(lower, upper)
            mean, std = x_clipped.mean(), x_clipped.std(ddof=0)
            return (x_clipped - mean) / std if std > 0 else x_clipped * 0.0
        out = s.groupby(level=0).apply(_winsor_zscore)
        return out.fillna(0.0)

    def run_backtest(self, prices: pd.DataFrame, fund_daily: pd.DataFrame, start_date: str, end_date: str) -> Tuple[pd.Series, PerformanceMetrics]:
        print("\n--- Running QV Engine Backtest ---")
        prices["date"] = pd.to_datetime(prices["date"])
        prices_pivot = prices.pivot(index='date', columns='ticker', values='close')
        
        # 리밸런싱 날짜 생성 (월말)
        rebal_dates = prices_pivot.loc[start_date:end_date].resample("ME").first().index
        
        # QV 신호 계산
        qv_signals = self.build_signals(fund_daily)
        
        # 리밸런싱 날짜를 실제 거래일로 매핑
        # qv_signals의 인덱스가 MultiIndex인 경우 level=0을 사용
        signal_dates = qv_signals.index.get_level_values(0).unique()
        rebal_dates_actual = []
        for rebal_date in rebal_dates:
            closest_date = min(signal_dates, key=lambda x: abs((x - rebal_date).total_seconds()))
            if abs((closest_date - rebal_date).days) <= 3:
                rebal_dates_actual.append(closest_date)
        
        all_returns = pd.Series(index=prices_pivot.loc[start_date:end_date].index, dtype=float).fillna(0.0)
        portfolio = {}
        
        for date in all_returns.index:
            # 리밸런싱 날짜인지 확인
            if any(abs((date - rd).days) == 0 for rd in rebal_dates_actual):
                # 가장 가까운 신호 날짜 찾기
                closest_signal_date = min(signal_dates, key=lambda x: abs((x - date).total_seconds()))
                
                if closest_signal_date in qv_signals.index.get_level_values(0):
                    signals_at_date = qv_signals.loc[pd.IndexSlice[closest_signal_date, :]]
                    if not signals_at_date.empty:
                        long_tickers = signals_at_date.nlargest(int(len(signals_at_date) * self.top_quantile)).index.get_level_values("ticker")
                        
                        if self.use_inverse_vol:
                            # 가장 가까운 가격 날짜 찾기
                            price_dates = prices_pivot.index
                            closest_price_date = min(price_dates, key=lambda x: abs((x - date).total_seconds()))
                            
                            if closest_price_date in prices_pivot.index:
                                vol = prices_pivot.pct_change().rolling(self.vol_lookback).std().loc[closest_price_date]
                                if not vol.loc[long_tickers].empty:
                                    inv_vol = 1 / vol.loc[long_tickers].replace(0, np.nan).dropna()
                                    if not inv_vol.empty:
                                        weights = inv_vol / inv_vol.sum()
                                        portfolio = weights.to_dict()
                        else:
                            portfolio = {ticker: 1.0 / len(long_tickers) for ticker in long_tickers} if len(long_tickers) > 0 else {}
            
            if not portfolio:
                all_returns[date] = 0.0
                continue
            
            # 수익률 계산
            day_ret = 0.0
            price_dates = prices_pivot.index
            closest_today = min(price_dates, key=lambda x: abs((x - date).total_seconds()))
            
            prev_dates = [d for d in price_dates if d < closest_today]
            if prev_dates:
                closest_yesterday = max(prev_dates)
                
                today_prices = prices_pivot.loc[closest_today]
                yesterday_prices = prices_pivot.loc[closest_yesterday]
                
                for ticker, weight in portfolio.items():
                    if ticker in today_prices.index and ticker in yesterday_prices.index and pd.notna(yesterday_prices[ticker]) and yesterday_prices[ticker] != 0:
                        day_ret += weight * (today_prices[ticker] / yesterday_prices[ticker] - 1)
            
            all_returns[date] = day_ret
        
        metrics = calculate_metrics(all_returns)
        return all_returns, metrics
#</editor-fold>

#<editor-fold desc="Main Workflow Functions">
def download_and_prepare_data():
    print("\n" + "="*100)
    print("STEP 1: DOWNLOADING AND PREPARING DATA")
    print("="*100)
    prices_path = BASE_DIR / "data" / "sp100_prices_raw.csv"
    sf1_path = BASE_DIR / "data" / "sp100_sf1_raw.csv"
    merged_path = BASE_DIR / "data" / "sp100_merged_data.csv"
    spy_path = BASE_DIR / "data" / "spy_prices.csv"

    # 이미 머지된 데이터가 있으면 그대로 사용
    if merged_path.exists():
        print(f"✓ Loading previously merged data from {merged_path}")
        data = pd.read_csv(merged_path, parse_dates=['date'])
        print(f"  Loaded {len(data)} rows with {data['ticker'].nunique()} tickers")
        # sanity check: 펀더멘털 NaN 비율
        print("  Fundamental non-null counts after reload:")
        for col in ["pe", "pb", "ps", "evebitda", "roe", "ebitdamargin", "de", "currentratio"]:
            print(f"    {col}: {data[col].notna().sum()} non-null")
        
        # SPY 데이터 로딩
        if spy_path.exists():
            print(f"✓ Loading previously downloaded SPY prices from {spy_path}")
            spy_df = pd.read_csv(spy_path, parse_dates=['date'], index_col='date')
            spy_df.index = spy_df.index.tz_localize(None)  # Remove timezone
            spy_series = spy_df['close']
        else:
            print("\nDownloading SPY prices for MarketConditionGuard...")
            poly = Polygon(POLYGON_API_KEY)
            spy_series = poly.get_spy_prices("2014-01-01", "2024-12-31")
            spy_df = pd.DataFrame({'close': spy_series})
            spy_df.to_csv(spy_path)
            print(f"✓ Saved SPY prices to {spy_path}")
        
        return data, spy_series

    # --- 1) 가격 데이터 로딩 or 다운로드 ---
    if prices_path.exists():
        print(f"✓ Loading previously downloaded prices from {prices_path}")
        prices_df = pd.read_csv(prices_path, parse_dates=['date'])
    else:
        print("\nDownloading daily prices from Polygon...")
        poly = Polygon(POLYGON_API_KEY)
        prices_df = poly.get_daily_prices(SP100_TICKERS, "2014-01-01", "2024-12-31")
        prices_df.to_csv(prices_path, index=False)

    # 날짜/티커 정렬 (타임존 제거)
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    prices_df = prices_df.sort_values(["ticker", "date"])

    # --- 2) SF1 펀더멘털 데이터 로딩 or 다운로드 ---
    if sf1_path.exists():
        print(f"✓ Loading previously downloaded fundamental data from {sf1_path}")
        sf1_df = pd.read_csv(sf1_path, parse_dates=['datekey', 'reportperiod', 'calendardate'])
    else:
        print("\nDownloading fundamental data from Sharadar...")
        sf1 = SF1(SHARADAR_API_KEY)
        sf1_df = sf1.get_sf1_data(SP100_TICKERS, "2014-01-01", "2024-12-31")
        sf1_df.to_csv(sf1_path, index=False)

    # SF1 정렬 (타임존 제거)
    sf1_df["datekey"] = pd.to_datetime(sf1_df["datekey"]).dt.tz_localize(None)
    sf1_df = sf1_df.sort_values(["ticker", "datekey"])

    print("\nPreparing and merging data (PIT using merge_asof)...")

    # 3) merge_asof로 PIT 매칭
    # merge_asof는 by 파라미터 사용 시 각 그룹별로 정렬이 필요하므로 ticker별로 처리
    all_merged = []
    tickers = sorted(prices_df['ticker'].unique())
    
    for i, ticker in enumerate(tickers, 1):
        if i % 20 == 0:
            print(f"  Processing ticker {i}/{len(tickers)}...")
        
        p_tick = prices_df[prices_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        s_tick = sf1_df[sf1_df['ticker'] == ticker].sort_values('datekey').reset_index(drop=True)
        
        if s_tick.empty:
            # SF1 데이터가 없는 ticker는 건너뛀
            continue
        
        merged = pd.merge_asof(
            p_tick,
            s_tick,
            left_on="date",
            right_on="datekey",
            direction="backward",
            allow_exact_matches=True,
        )
        all_merged.append(merged)
    
    data = pd.concat(all_merged, ignore_index=True)
    
    # ticker 컬럼 이름 충돌 해결 (ticker_x는 prices에서, ticker_y는 sf1에서)
    if 'ticker_x' in data.columns:
        data = data.rename(columns={'ticker_x': 'ticker'})
        if 'ticker_y' in data.columns:
            data = data.drop(columns=['ticker_y'])

    print(f"  After merge_asof: {len(data)} rows")
    # 초기 펀더멘털 non-null 체크
    print("  Fundamental non-null counts right after merge:")
    for col in ["pe", "pb", "ps", "evebitda", "roe", "ebitdamargin", "de", "currentratio"]:
        if col in data.columns:
            print(f"    {col}: {data[col].notna().sum()} non-null")
        else:
            print(f"    {col}: MISSING COLUMN!")

    # 4) ticker별로 forward fill (발표 후 다음 발표 전까지 같은 값 유지)
    data = data.sort_values(['ticker', 'date'])
    data = data.groupby('ticker', group_keys=False).apply(lambda x: x.ffill())
    print(f"  After per-ticker ffill: {len(data)} rows")

    print("\nFeature Engineering...")
    # 5) 팩터 계산 (가격 기반)
    data['momentum_60d'] = data.groupby('ticker')['close'].pct_change(60)
    data['volatility_30d'] = data.groupby('ticker')['close'].transform(
        lambda x: x.pct_change().rolling(30).std()
    )

    # value_proxy는 SF1의 pe 그대로 활용 (혹은 다른 값으로 바꿀 수 있음)
    data['value_proxy'] = data['pe']

    # NaN 로그
    print(f"  Before dropna: {len(data)} rows")
    for col in ["momentum_60d", "volatility_30d", "value_proxy", "pe", "pb", "ps",
                "evebitda", "roe", "ebitdamargin", "de", "currentratio"]:
        print(f"    {col}: {data[col].isna().sum()} NaNs")

    # 6) NaN 처리 전략
    # currentratio는 결측이 많으니, 드랍 전에 간단히 median으로 채우는 옵션 (원하면 사용)
    if "currentratio" in data.columns:
        na_count = data["currentratio"].isna().sum()
        if na_count > 0:
            median_cr = data["currentratio"].median()
            print(f"  Filling {na_count} NaNs in currentratio with median={median_cr:.4f}")
            data["currentratio"] = data["currentratio"].fillna(median_cr)

    # 핵심 피처/펀더멘털이 다 있는 행만 사용 (너무 빡셌으면 일부 완화 가능)
    data = data.dropna(subset=[
        "momentum_60d", "volatility_30d", "value_proxy", "pe", "pb", "ps",
        "evebitda", "roe", "ebitdamargin", "de", "currentratio"
    ])
    print(f"  After dropna: {len(data)} rows with {data['ticker'].nunique()} tickers")

    # 최종 저장
    data.to_csv(merged_path, index=False)
    print(f"\n✓ Saved merged and prepared data to {merged_path}")
    
    # --    # SPY 데이터 로딩 (MarketConditionGuard용)
    if spy_path.exists():
        print(f"✓ Loading previously downloaded SPY prices from {spy_path}")
        spy_df = pd.read_csv(spy_path, parse_dates=['date'], index_col='date')
        spy_df.index = spy_df.index.tz_localize(None)  # Remove timezone
        spy_series = spy_df['close']
    else:
        print("\nDownloading SPY prices for MarketConditionGuard...")
        poly = Polygon(POLYGON_API_KEY)
        spy_series = poly.get_spy_prices("2014-01-01", "2024-12-31")
        spy_df = pd.DataFrame({'close': spy_series})
        spy_df.to_csv(spy_path)
        print(f"✓ Saved SPY prices to {spy_path}")
    
    return data, spy_series

def run_backtests(data, spy_series):
    print("\n" + "="*100)
    print("STEP 2: RUNNING BACKTESTS")
    print("="*100)
    prices_for_engines = data[["date", "ticker", "close"]]
    
    # --- ML9 Engine (No Guard) ---
    print("\n[1/3] ML9 Engine (No Guard)")
    ml9_engine = ML9Engine(prices=prices_for_engines.copy(), factors=data.copy())
    ml9_returns, ml9_metrics = ml9_engine.run_walk_forward_backtest(start_date="2015-01-01", end_date="2024-12-31")
    ml9_returns.to_csv(BASE_DIR / "results" / "ml9_returns.csv")
    with open(BASE_DIR / "results" / "ml9_metrics.json", "w") as f:
        json.dump(asdict(ml9_metrics), f, indent=4)
    print("✓ ML9 Engine (No Guard) backtest complete.")
    
    # --- ML9 Engine (With Guard) ---
    print("\n[2/3] ML9 Engine (With Guard)")
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
    guard.initialize(spy_series)
    
    ml9_guard_engine = ML9Engine(prices=prices_for_engines.copy(), factors=data.copy(), guard=guard)
    ml9_guard_returns, ml9_guard_metrics = ml9_guard_engine.run_walk_forward_backtest(start_date="2015-01-01", end_date="2024-12-31")
    ml9_guard_returns.to_csv(BASE_DIR / "results" / "ml9_guard_returns.csv")
    with open(BASE_DIR / "results" / "ml9_guard_metrics.json", "w") as f:
        json.dump(asdict(ml9_guard_metrics), f, indent=4)
    print("✓ ML9 Engine (With Guard) backtest complete.")
    
    # --- QV Engine ---
    print("\n[3/3] QV Engine")
    qv_engine = QVEngine()
    qv_returns, qv_metrics = qv_engine.run_backtest(prices=prices_for_engines.copy(), fund_daily=data.copy(), start_date="2015-01-01", end_date="2024-12-31")
    qv_returns.to_csv(BASE_DIR / "results" / "qv_returns.csv")
    with open(BASE_DIR / "results" / "qv_metrics.json", "w") as f:
        json.dump(asdict(qv_metrics), f, indent=4)
    print("✓ QV Engine backtest complete.")
    
    print("\n✓ All backtests complete.")
    return ml9_metrics, ml9_guard_metrics, qv_metrics

def generate_report():
    print("\n" + "="*100)
    print("STEP 3: GENERATING REPORT")
    print("="*100)
    try:
        with open(BASE_DIR / "results" / "ml9_metrics.json", "r") as f:
            ml9_metrics = json.load(f)
        with open(BASE_DIR / "results" / "ml9_guard_metrics.json", "r") as f:
            ml9_guard_metrics = json.load(f)
        with open(BASE_DIR / "results" / "qv_metrics.json", "r") as f:
            qv_metrics = json.load(f)
        
        # Calculate Guard improvement
        sharpe_improvement = ((ml9_guard_metrics["sharpe"] - ml9_metrics["sharpe"]) / ml9_metrics["sharpe"] * 100) if ml9_metrics["sharpe"] != 0 else 0
        mdd_improvement = ((ml9_guard_metrics["max_drawdown"] - ml9_metrics["max_drawdown"]) / ml9_metrics["max_drawdown"] * 100) if ml9_metrics["max_drawdown"] != 0 else 0
        
        report = f'''# Quantitative Strategy Analysis Report

## Overview
This report presents the backtesting results for the ML9 (with/without Guard) and QV quantitative trading strategies from 2015 to 2024 on the SP100 universe.

## ML9 Engine Results (No Guard)

| Metric | Value |
|---|---|
| Sharpe Ratio | {ml9_metrics["sharpe"]:.2f} |
| Annualized Return | {ml9_metrics["annual_return"]*100:.2f}% |
| Annualized Volatility | {ml9_metrics["annual_volatility"]*100:.2f}% |
| Max Drawdown | {ml9_metrics["max_drawdown"]*100:.2f}% |
| Win Rate | {ml9_metrics["win_rate"]*100:.2f}% |
| Number of Trades | {ml9_metrics["num_trades"]} |

## ML9 Engine Results (With Guard)

| Metric | Value | Change |
|---|---|---|
| Sharpe Ratio | {ml9_guard_metrics["sharpe"]:.2f} | {sharpe_improvement:+.1f}% |
| Annualized Return | {ml9_guard_metrics["annual_return"]*100:.2f}% | {(ml9_guard_metrics["annual_return"] - ml9_metrics["annual_return"])*100:+.2f}% |
| Annualized Volatility | {ml9_guard_metrics["annual_volatility"]*100:.2f}% | {(ml9_guard_metrics["annual_volatility"] - ml9_metrics["annual_volatility"])*100:+.2f}% |
| Max Drawdown | {ml9_guard_metrics["max_drawdown"]*100:.2f}% | {mdd_improvement:+.1f}% |
| Win Rate | {ml9_guard_metrics["win_rate"]*100:.2f}% | {(ml9_guard_metrics["win_rate"] - ml9_metrics["win_rate"])*100:+.2f}% |
| Number of Trades | {ml9_guard_metrics["num_trades"]} | {ml9_guard_metrics["num_trades"] - ml9_metrics["num_trades"]:+d} |

**Guard Configuration:**
- SPX Return Range: -2.0% to 0.0%
- Scale Factor: 0.5 (50% position reduction)
- Volatility Filter: Disabled

## QV Engine Results

| Metric | Value |
|---|---|
| Sharpe Ratio | {qv_metrics["sharpe"]:.2f} |
| Annualized Return | {qv_metrics["annual_return"]*100:.2f}% |
| Annualized Volatility | {qv_metrics["annual_volatility"]*100:.2f}% |
| Max Drawdown | {qv_metrics["max_drawdown"]*100:.2f}% |
| Win Rate | {qv_metrics["win_rate"]*100:.2f}% |
| Number of Trades | {qv_metrics["num_trades"]} |

'''
        with open(BASE_DIR / "FINAL_REPORT.md", "w") as f:
            f.write(report)
        print("\n✓ Final report generated: FINAL_REPORT.md")
        print("Script finished successfully.")
    except FileNotFoundError as e:
        print(f"Error generating report: {e}. Check if backtests ran successfully.")
#</editor-fold>

def main():
    start_time = time.time()
    try:
        data, spy_series = download_and_prepare_data()
        run_backtests(data, spy_series)
        generate_report()
    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
