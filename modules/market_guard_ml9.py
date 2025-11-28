"""
ML9 Market Condition Guard
Automatically reduce ML9 position when SPX is in -2%~0% range
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketGuardConfig:
    enabled: bool = True
    spx_symbol: str = "SPY"
    return_lower: float = -0.02
    return_upper: float = 0.0
    scale_factor: float = 0.5
    vol_window: int = 20
    use_vol_filter: bool = False
    vol_percentile: float = 0.66
    lookback_days: int = 365


class ML9MarketConditionGuard:
    """
    Market condition guard for ML9 engine
    Reduces ML9 position when SPX return is in (-2%, 0%] range
    """
    
    def __init__(self, config: dict):
        self.cfg = MarketGuardConfig(
            enabled=config.get("enabled", True),
            spx_symbol=config.get("spx_symbol", "SPY"),
            return_lower=config.get("return_lower", -0.02),
            return_upper=config.get("return_upper", 0.0),
            scale_factor=config.get("scale_factor", 0.5),
            vol_window=config.get("vol_window", 20),
            use_vol_filter=config.get("use_vol_filter", False),
            vol_percentile=config.get("vol_percentile", 0.66),
            lookback_days=config.get("lookback_days", 365),
        )
        self.spx_close: Optional[pd.Series] = None
        self.spx_ret: Optional[pd.Series] = None
        self.spx_vol: Optional[pd.Series] = None

    def initialize(self, spx_close: pd.Series):
        """
        Initialize with SPX price series
        Calculate daily returns and rolling volatility
        """
        if not self.cfg.enabled:
            logger.info("[ML9Guard] disabled in config")
            return

        spx_close = spx_close.sort_index().dropna()
        # Normalize index to remove timezone and time component
        spx_close.index = pd.to_datetime(spx_close.index).normalize()
        self.spx_close = spx_close

        spx_ret = spx_close.pct_change().dropna()
        self.spx_ret = spx_ret

        spx_vol = spx_ret.rolling(self.cfg.vol_window).std()
        self.spx_vol = spx_vol

        logger.info(
            f"[ML9Guard] Initialized with {len(spx_close)} SPX data points"
        )

    def _get_spx_features(self, date: datetime) -> tuple[float, float]:
        """
        Get SPX daily return and rolling volatility for a specific date
        Returns the closest past trading day
        """
        if self.spx_ret is None or self.spx_vol is None:
            return 0.0, 0.0

        idx = self.spx_ret.index
        if len(idx) == 0:
            return 0.0, 0.0

        # Find closest past date
        date_floor = pd.Timestamp(date).normalize()
        past_idx = idx[idx <= date_floor]
        if len(past_idx) == 0:
            return 0.0, 0.0

        d_used = past_idx[-1]

        r_spx = float(self.spx_ret.loc[d_used])
        v_spx = float(self.spx_vol.loc[d_used]) if not np.isnan(self.spx_vol.loc[d_used]) else 0.0
        return r_spx, v_spx

    def get_ml9_scale(self, date: Optional[datetime] = None) -> float:
        """
        Get ML9 position scale factor (0~1) for current time
        Default: 1.0
        If SPX return is in (return_lower, return_upper], apply scale_factor
        Optional: vol filter (use_vol_filter=true) applies only in high vol regime
        """
        if not self.cfg.enabled:
            return 1.0

        if date is None:
            date = datetime.utcnow()

        r_spx, v_spx = self._get_spx_features(date)

        # Default: no effect
        scale = 1.0

        # Return condition
        if self.cfg.return_lower < r_spx <= self.cfg.return_upper:
            # Vol filter option
            if self.cfg.use_vol_filter and self.spx_vol is not None:
                # Calculate percentile threshold from recent vol distribution
                vol_vals = self.spx_vol.dropna().values
                if len(vol_vals) > 0:
                    thr = np.quantile(vol_vals, self.cfg.vol_percentile)
                    if v_spx >= thr:
                        scale = self.cfg.scale_factor
            else:
                scale = self.cfg.scale_factor

        return float(scale)
    
    def get_scale_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Get ML9 scale factors for a series of dates
        """
        scales = []
        for date in dates:
            scale = self.get_ml9_scale(date)
            scales.append(scale)
        
        return pd.Series(scales, index=dates, name="ml9_scale")
