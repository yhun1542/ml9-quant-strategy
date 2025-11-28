# utils/regime_v1.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


RegimeLabel = Literal["BULL", "BEAR", "HIGH_VOL", "NEUTRAL"]


@dataclass
class RegimeConfigV1:
    vol_window: int = 20
    ret_window: int = 63
    dd_window: int = 126
    vol_high_pct: float = 0.7    # 상위 30% → high vol
    dd_bear_threshold: float = -0.15  # 6개월 기준 -15% 이하 DD → bear
    ma_long: int = 200


def compute_spx_regime(
    spx_close: pd.Series,
    cfg: RegimeConfigV1 | None = None,
) -> pd.Series:
    """
    SPX (또는 SPY) 종가 시계열에서 레짐 레이블 시리즈를 생성한다.

    Regime 정의 (v1):
      - BULL:
          - price > MA(200)
          - ret_63d > 0
          - vol_20d <= vol_low_mid
      - BEAR:
          - ret_63d < 0
          - dd_126d <= dd_bear_threshold (예: -15%)
      - HIGH_VOL:
          - vol_20d >= vol_high_quantile
          - BEAR가 아닌 구간
      - NEUTRAL: 그 외
    """
    cfg = cfg or RegimeConfigV1()

    spx_close = spx_close.sort_index()
    ret = spx_close.pct_change()

    vol_20d = ret.rolling(cfg.vol_window).std()
    ret_63d = spx_close.pct_change(cfg.ret_window)
    # Drawdown: 126일 기준
    rolling_max = spx_close.rolling(cfg.dd_window, min_periods=1).max()
    dd_126d = spx_close / rolling_max - 1.0

    ma_200 = spx_close.rolling(cfg.ma_long).mean()

    # vol high threshold
    vol_quantile = vol_20d.quantile(cfg.vol_high_pct)
    vol_high = vol_20d >= vol_quantile

    # BULL
    cond_bull = (
        (ma_200.notna()) &
        (spx_close > ma_200) &
        (ret_63d > 0) &
        (vol_20d <= vol_quantile)  # 하위 70% → low/mid vol
    )

    # BEAR (완화된 조건: OR 사용)
    cond_bear = (
        (ret_63d < -0.05) |  # 63일 수익률 -5% 이하
        (dd_126d <= cfg.dd_bear_threshold)  # OR 126일 DD -15% 이하
    )

    # HIGH_VOL (bear 아닌 high vol)
    cond_highvol = vol_high & (~cond_bear)

    regime = pd.Series(index=spx_close.index, dtype="object")
    regime.loc[:] = "NEUTRAL"
    regime[cond_bull] = "BULL"
    regime[cond_bear] = "BEAR"
    regime[cond_highvol] = "HIGH_VOL"

    return regime
