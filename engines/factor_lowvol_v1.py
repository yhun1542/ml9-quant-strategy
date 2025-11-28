# engines/factor_lowvol_v1.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class LowVolConfig:
    top_quantile: float = 0.3         # 저위험 상위 분위수 비중 (0.3 → 상위 30% 롱)
    long_gross: float = 1.0           # 롱 포지션 총합
    short_gross: float = 0.0          # 숏 포지션 총합 (v0에서는 보통 0.0 또는 0.5)
    long_only: bool = True            # True이면 롱온리
    use_inverse_vol: bool = True      # True이면 inverse-vol weighting
    vol_lookback: int = 63            # 변동성/다운사이드 볼 lookback (일수)
    beta_lookback: int = 252          # 베타 계산용 lookback (일수)
    beta_use: bool = True             # 베타를 risk_score에 포함할지 여부
    downside_vol_weight: float = 0.5  # downside vol 가중치
    beta_weight: float = 0.5          # |beta| 가중치 (beta_use=True일 때만)


class FactorLowVolEngineV1:
    """
    저변동/Defensive 엔진 v1.
    - 가격 데이터와 SPX 인덱스를 기반으로 변동성/베타/다운사이드 볼을 계산.
    - risk_score가 낮을수록 "안전한 종목".
    - 상위 top_quantile 저위험 종목 롱 (옵션: 고위험 종목 숏).
    """

    def __init__(self, cfg: Optional[LowVolConfig] = None):
        self.cfg = cfg or LowVolConfig()

    def _calc_vol_factors(
        self,
        prices: pd.DataFrame,
        spx_close: pd.Series,
    ) -> pd.DataFrame:
        """
        prices: DataFrame, index=date, columns=tickers
        spx_close: Series, index=date, name='SPX'

        반환: MultiIndex (date, ticker)의 risk factor DataFrame
        """
        # 정렬 및 align
        prices = prices.sort_index()
        spx_close = spx_close.sort_index()
        spx_close = spx_close.reindex(prices.index, method="ffill")

        # 일간 수익률
        ret = prices.pct_change()
        spx_ret = spx_close.pct_change()

        # 63일 변동성
        vol_63d = ret.rolling(self.cfg.vol_lookback).std()

        # 다운사이드 볼 (음수 수익률만)
        down_ret = ret.copy()
        down_ret[down_ret > 0] = 0.0
        down_vol_63d = down_ret.rolling(self.cfg.vol_lookback).std()

        # 베타 계산 (옵션)
        if self.cfg.beta_use:
            # 공분산 / 분산
            cov = (
                ret
                .rolling(self.cfg.beta_lookback)
                .cov(spx_ret)
            )
            var_spx = spx_ret.rolling(self.cfg.beta_lookback).var()
            # var_spx를 DataFrame과 align
            var_spx_df = pd.DataFrame(
                {col: var_spx for col in ret.columns},
                index=ret.index,
            )
            beta = cov / var_spx_df
        else:
            beta = pd.DataFrame(
                np.nan,
                index=ret.index,
                columns=ret.columns,
            )

        # MultiIndex (date, ticker)로 변환
        vols = vol_63d.stack().to_frame("vol_63d")
        downs = down_vol_63d.stack().to_frame("down_vol_63d")
        betas = beta.stack().to_frame("beta")

        df = vols.join(downs).join(betas)
        df.index.names = ["date", "ticker"]

        return df

    def _xsec_zscore(self, s: pd.Series) -> pd.Series:
        """
        날짜별 cross-sectional z-score (NaN/inf 방어 포함)
        """
        def _z(x: pd.Series) -> pd.Series:
            x = x.replace([np.inf, -np.inf], np.nan)
            if x.isna().all():
                return pd.Series(0.0, index=x.index)
            mean = x.mean()
            std = x.std(ddof=0)
            if std == 0 or np.isnan(std):
                return pd.Series(0.0, index=x.index)
            return (x - mean) / std

        out = s.groupby(level="date").transform(_z)
        return out.fillna(0.0)

    def build_signals(
        self,
        prices: pd.DataFrame,
        spx_close: pd.Series,
    ) -> pd.Series:
        """
        변동성/다운볼/베타 기반 risk_score 생성.
        - 낮을수록 안전.
        반환: Series[(date, ticker)] -> risk_score
        """
        factors = self._calc_vol_factors(prices, spx_close)

        z_vol  = self._xsec_zscore(factors["vol_63d"])
        z_down = self._xsec_zscore(factors["down_vol_63d"])

        if self.cfg.beta_use:
            z_beta = self._xsec_zscore(factors["beta"].abs())
        else:
            z_beta = pd.Series(0.0, index=factors.index)

        risk_raw = (
            z_vol +
            self.cfg.downside_vol_weight * z_down +
            self.cfg.beta_weight * z_beta
        )

        risk_score = self._xsec_zscore(risk_raw)
        return risk_score.rename("risk_score")

    def build_portfolio(
        self,
        prices: pd.DataFrame,
        spx_close: pd.Series,
        rebalance_dates: List[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        prices: DataFrame, index=date, columns=tickers
        spx_close: Series, index=date
        rebalance_dates: 포트 리밸 날짜 리스트

        반환: {rebalance_date: weight Series(ticker -> weight)}
        """
        risk_score = self.build_signals(prices, spx_close)
        # risk_score index: (date, ticker)

        # 일간 수익률 (inverse-vol weight용)
        ret = prices.pct_change()
        vol = ret.rolling(self.cfg.vol_lookback).std()

        weights_by_date: Dict[pd.Timestamp, pd.Series] = {}

        for d in rebalance_dates:
            if d not in risk_score.index.get_level_values("date"):
                continue
            if d not in vol.index:
                continue

            cs = risk_score.loc[d].dropna()  # index: ticker, value: risk_score
            if cs.empty:
                continue

            # risk_score 낮은 종목 = 저위험 (롱 후보)
            n = len(cs)
            n_long  = max(int(n * self.cfg.top_quantile), 1)
            n_short = max(int(n * self.cfg.top_quantile), 1)

            cs_sorted = cs.sort_values(ascending=True)  # 낮을수록 안전
            long_names  = cs_sorted.head(n_long).index
            short_names = cs_sorted.tail(n_short).index

            # Long leg
            if self.cfg.use_inverse_vol:
                vols_long = vol.loc[d, long_names]
                inv_long  = 1.0 / vols_long
                inv_long  = inv_long.replace([np.inf, -np.inf], np.nan).dropna()
                if inv_long.empty:
                    continue
                w_long_raw = inv_long
            else:
                w_long_raw = pd.Series(1.0, index=long_names)

            w_long = w_long_raw / w_long_raw.sum() * self.cfg.long_gross
            portfolio = w_long.to_dict()

            # Short leg (옵션)
            if (not self.cfg.long_only) and self.cfg.short_gross > 0:
                if self.cfg.use_inverse_vol:
                    vols_short = vol.loc[d, short_names]
                    inv_short  = 1.0 / vols_short
                    inv_short  = inv_short.replace([np.inf, -np.inf], np.nan).dropna()
                    if not inv_short.empty:
                        w_short_raw = inv_short
                    else:
                        w_short_raw = pd.Series(dtype=float)
                else:
                    w_short_raw = pd.Series(1.0, index=short_names)

                if not w_short_raw.empty:
                    w_short = -w_short_raw / w_short_raw.sum() * self.cfg.short_gross
                    for tkr, w in w_short.items():
                        portfolio[tkr] = portfolio.get(tkr, 0.0) + w

            if portfolio:
                w = pd.Series(portfolio)
                weights_by_date[d] = w

        return weights_by_date
