"""Position sizing (plan, Trading layer -> Sizing and hedging).

    notional = min( capital * risk_scale * edge / sigma_daily^2,
                    capital * cap_capital_frac,
                    ADV * cap_adv_frac )

With edge=None this returns the cap alone, used to cost a trade before sizing it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.config import TradingConfig


def size_positions(df: pd.DataFrame, cfg: TradingConfig, edge: pd.Series | None) -> pd.Series:
    adv = df["adv_usd"].fillna(df["adv_usd"].median()).fillna(0)
    cap = np.minimum(cfg.capital * cfg.cap_capital_frac, adv * cfg.cap_adv_frac)
    if edge is None:
        return pd.Series(cap, index=df.index)
    sigma2 = df["sigma_daily"].fillna(df["sigma_daily"].median()).fillna(0.02) ** 2
    raw = cfg.capital * cfg.risk_scale * edge.clip(lower=0) / sigma2
    return pd.Series(np.minimum(raw, cap), index=df.index)
