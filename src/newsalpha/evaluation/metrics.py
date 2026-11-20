"""Metrics. Daily return series are fractions of capital; per-trade figures in bps."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649


def sharpe(daily: pd.Series, periods: int = 252) -> float:
    sd = daily.std(ddof=1)
    return float(daily.mean() / sd * math.sqrt(periods)) if sd and sd > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    equity = daily.cumsum()
    return float((equity - equity.cummax()).min()) if len(equity) else float("nan")


def deflated_sharpe(daily: pd.Series, n_trials: int) -> float:
    """Bailey & Lopez de Prado (2014): P(true SR > best-of-N-by-luck SR), per-period units.

    The variance of SR across trials is approximated by its null sampling
    variance 1/T, which is conservative when trials are highly correlated.
    """
    x = daily.dropna().to_numpy()
    t = len(x)
    if t < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    skew, kurt = stats.skew(x), stats.kurtosis(x, fisher=False)
    if n_trials > 1:
        v = 1.0 / t
        sr0 = math.sqrt(v) * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e))
        )
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom))


def rank_ic(df: pd.DataFrame, by: str = "session_date", min_obs: int = 5) -> dict[str, float]:
    ics = []
    for _, g in df.groupby(by):
        if len(g) >= min_obs and g["pred"].nunique() > 1:
            ics.append(stats.spearmanr(g["pred"], g["target"]).statistic)
    s = pd.Series(ics, dtype=float).dropna()
    if s.empty:
        return {"mean": float("nan"), "t": float("nan"), "n_days": 0}
    t = s.mean() / s.std(ddof=1) * math.sqrt(len(s)) if len(s) > 1 and s.std(ddof=1) > 0 else np.nan
    return {"mean": float(s.mean()), "t": float(t), "n_days": int(len(s))}


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"n_trades": 0}
    r = trades["net_ret"]
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "n_trades": int(len(r)),
        "net_bps_per_trade": float(r.mean() * 1e4),
        "gross_bps_per_trade": float((trades["gross_ret"] + trades["hedge_ret"]).mean() * 1e4),
        "est_cost_bps_per_trade": float(trades["cost_est"].mean() * 1e4),
        "hit_rate": float((r > 0).mean()),
        "payoff_ratio": float(wins.mean() / -losses.mean())
        if len(wins) and len(losses)
        else np.nan,
        "net_pnl": float(trades["pnl"].sum()),
    }


def strategy_metrics(trades: pd.DataFrame, daily: pd.Series, n_trials: int) -> dict[str, float]:
    return {
        **trade_stats(trades),
        "net_sharpe": sharpe(daily),
        "deflated_sharpe": deflated_sharpe(daily, n_trials),
        "max_drawdown": max_drawdown(daily),
        "n_days": int(len(daily)),
    }
