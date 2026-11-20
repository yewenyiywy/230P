"""Trade decision (plan, Trading layer -> Trade rule).

    side  = sign(forecast), restricted by the variant and the no-short rules
    edge  = side * forecast - estimated cost
    trade = edge > kappa * sigma_hat

Sizing depends on cost (impact grows with size) and cost on sizing, so we
size first at the cap, cost that size, then let `sizing` shrink it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.config import TradingConfig
from newsalpha.trading.costs import total_cost
from newsalpha.trading.sizing import size_positions


def decide(preds: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
    df = preds.copy()
    side = pd.Series(np.sign(df["pred"]), index=df.index)
    if cfg.variant in ("long_only", "long_only_hedged"):
        side = side.clip(lower=0)
    side = side.where(~((side < 0) & (df["price"] < cfg.no_short_below_price)), 0)
    df["side"] = side
    df["notional"] = size_positions(df, cfg, edge=None)
    df["cost_est"] = total_cost(df, side, df["notional"], cfg)
    df["edge"] = side * df["pred"] - df["cost_est"]
    df["trade"] = (side != 0) & (df["edge"] > cfg.kappa * df["sigma_hat"].fillna(np.inf))
    df["notional"] = np.where(df["trade"], size_positions(df, cfg, edge=df["edge"]), 0.0)
    return df
