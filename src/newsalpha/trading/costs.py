"""Cost model (plan, Trading layer -> Cost model). All costs are returns (fractions).

entry spread   half the NBBO spread at entry (observed at decision time)
exit spread    half the median closing spread over the prior 20 sessions (estimate)
impact         k * sigma_daily * sqrt(notional / ADV)       (square-root law)
borrow         shorts only: bps/yr * holding days / 252
hedge          sector-ETF round trip: 2 * etf half-spread, when hedged
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.config import TradingConfig


def entry_half_spread(df: pd.DataFrame) -> pd.Series:
    mid = (df["entry_bid"] + df["entry_ask"]) / 2
    return (df["entry_ask"] - df["entry_bid"]) / 2 / mid


def exit_half_spread_est(df: pd.DataFrame) -> pd.Series:
    fallback = df.get("spread_bps_entry", 10.0)
    return df["avg_exit_spread_bps"].fillna(fallback) / 2 / 1e4


def impact(df: pd.DataFrame, notional: pd.Series, cfg: TradingConfig) -> pd.Series:
    sigma = df["sigma_daily"].fillna(df["sigma_daily"].median()).fillna(0.02)
    adv = df["adv_usd"].fillna(df["adv_usd"].median())
    return cfg.impact_k * sigma * np.sqrt((notional / adv).clip(lower=0))


def borrow(df: pd.DataFrame, side: pd.Series, cfg: TradingConfig) -> pd.Series:
    days = df["horizon_days"].clip(lower=1)
    return (side < 0) * cfg.borrow_bps_per_year / 1e4 * days / 252


def hedge_cost(df: pd.DataFrame, cfg: TradingConfig) -> pd.Series:
    hedged = cfg.variant in ("long_short_hedged", "long_only_hedged")
    return pd.Series(2 * cfg.etf_half_spread_bps / 1e4 if hedged else 0.0, index=df.index)


def total_cost(
    df: pd.DataFrame, side: pd.Series, notional: pd.Series, cfg: TradingConfig
) -> pd.Series:
    return (
        entry_half_spread(df)
        + exit_half_spread_est(df)
        + impact(df, notional, cfg)
        + borrow(df, side, cfg)
        + hedge_cost(df, cfg)
    )
