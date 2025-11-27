"""
ETAS (Earthquake-Triggered Aftershock Sequences) Risk Manager
Predict and manage aftershock risk following market shocks
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
class ETASConfig:
    enabled: bool = True
    shock_threshold: float = -0.03  # -3% drop triggers shock
    aftershock_window: int = 5  # Days to watch for aftershocks
    decay_rate: float = 0.5  # Exponential decay of risk
    min_scale: float = 0.3  # Minimum position scale during high risk
    lookback_days: int = 365


class ETASRiskManager:
    """
    ETAS-based risk manager
    Detects market shocks and reduces positions during aftershock periods
    """
    
    def __init__(self, config: dict):
        self.cfg = ETASConfig(
            enabled=config.get("enabled", True),
            shock_threshold=config.get("shock_threshold", -0.03),
            aftershock_window=config.get("aftershock_window", 5),
            decay_rate=config.get("decay_rate", 0.5),
            min_scale=config.get("min_scale", 0.3),
            lookback_days=config.get("lookback_days", 365),
        )
        self.spx_ret: Optional[pd.Series] = None
        self.shock_dates: list = []

    def initialize(self, spx_close: pd.Series):
        """
        Initialize with SPX price series
        Detect historical shocks
        """
        if not self.cfg.enabled:
            logger.info("[ETAS] disabled in config")
            return

        spx_close = spx_close.sort_index().dropna()
        spx_ret = spx_close.pct_change().dropna()
        self.spx_ret = spx_ret

        # Detect shocks
        shocks = spx_ret[spx_ret <= self.cfg.shock_threshold]
        self.shock_dates = shocks.index.tolist()

        logger.info(
            f"[ETAS] Initialized with {len(spx_ret)} days, "
            f"detected {len(self.shock_dates)} shocks"
        )

    def _calculate_aftershock_risk(self, date: datetime) -> float:
        """
        Calculate aftershock risk for a specific date
        Risk decays exponentially from shock date
        Returns risk score (0~1, higher = more risk)
        """
        if not self.shock_dates:
            return 0.0

        date_ts = pd.Timestamp(date).normalize()
        
        # Find recent shocks within aftershock window
        risk = 0.0
        for shock_date in self.shock_dates:
            shock_ts = pd.Timestamp(shock_date).normalize()
            days_since = (date_ts - shock_ts).days
            
            if 0 <= days_since <= self.cfg.aftershock_window:
                # Exponential decay: risk = exp(-decay_rate * days)
                decay_risk = np.exp(-self.cfg.decay_rate * days_since)
                risk = max(risk, decay_risk)
        
        return float(risk)

    def get_position_scale(self, date: Optional[datetime] = None) -> float:
        """
        Get position scale factor (0~1) based on aftershock risk
        High risk → lower scale (min_scale)
        Low risk → full scale (1.0)
        """
        if not self.cfg.enabled:
            return 1.0

        if date is None:
            date = datetime.utcnow()

        risk = self._calculate_aftershock_risk(date)
        
        # Scale: 1.0 (no risk) → min_scale (max risk)
        scale = 1.0 - risk * (1.0 - self.cfg.min_scale)
        
        return float(scale)
    
    def get_scale_series(self, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Get position scale factors for a series of dates
        """
        scales = []
        for date in dates:
            scale = self.get_position_scale(date)
            scales.append(scale)
        
        return pd.Series(scales, index=dates, name="etas_scale")
