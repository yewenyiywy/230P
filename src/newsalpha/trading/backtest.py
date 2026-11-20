"""Event-driven backtest for ONE (entry delay, horizon) strategy.

Each traded event is realised at executable prices:
    long   buy at entry ask, sell at exit bid
    short  sell at entry bid, buy back at exit ask
then hedged with the sector ETF (hedged variants), charged impact, borrow and
hedge costs, and booked on the exit session. Days with no trade earn zero:
not trading is a valid outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.calendar import sessions_in_range
from newsalpha.config import TradingConfig
from newsalpha.trading.costs import borrow, hedge_cost, impact
from newsalpha.trading.rules import decide


def realise(trades: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
    t = trades.copy()
    long_ret = t["exit_bid"] / t["entry_ask"] - 1
    short_ret = (t["entry_bid"] - t["exit_ask"]) / t["entry_bid"]
    t["gross_ret"] = np.where(t["side"] > 0, long_ret, short_ret)
    hedged = cfg.variant in ("long_short_hedged", "long_only_hedged")
    t["hedge_ret"] = -t["side"] * (np.exp(t["bench_ret"]) - 1) if hedged else 0.0
    t["impact_cost"] = impact(t, t["notional"], cfg)
    t["other_cost"] = borrow(t, t["side"], cfg) + hedge_cost(t, cfg)
    t["net_ret"] = t["gross_ret"] + t["hedge_ret"] - t["impact_cost"] - t["other_cost"]
    t["pnl"] = t["notional"] * t["net_ret"]
    t["exit_session"] = (
        t["exit_ts"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    return t


def backtest(
    preds: pd.DataFrame, cfg: TradingConfig, delay_min: int, horizon: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (trades, daily_returns). daily_returns covers every session in the period."""
    sub = preds[(preds["delay_min"] == delay_min) & (preds["horizon"] == horizon)]
    if sub.empty:
        return sub.assign(pnl=[]), pd.Series(dtype=float)
    decided = decide(sub, cfg)
    trades = realise(decided[decided["trade"]], cfg)
    last = sub["exit_ts"].max().tz_convert("America/New_York").tz_localize(None).normalize()
    sessions = sessions_in_range(sub["session_date"].min(), last)
    daily = trades.groupby("exit_session")["pnl"].sum().reindex(sessions, fill_value=0.0)
    return trades, daily / cfg.capital
