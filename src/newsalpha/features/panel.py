"""Build the PANEL: one row per (event, entry delay d, holding horizon).

Timing (plan, alignment rules 2-6), with p0 = tau - 1 min (last quote before the news):

    features  use data stamped <= entry_ts = tau + d     (quotes)  or prior sessions only (daily stats)
    target    log mid(exit)/mid(entry) minus the same for the hedge ETF
    executable prices (entry/exit bid/ask) are carried for the backtest

Rows whose entry would be at or after the t0 close are dropped: we only enter
intraday. The no-look-ahead property is enforced by tests/test_leakage.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from newsalpha.align import QuoteBook
from newsalpha.calendar import horizon_exit, session_close
from newsalpha.config import RunConfig
from newsalpha.schemas import EXTRACTIONS, PANEL_ID_COLS, PANEL_LABEL_COLS, validate

HORIZON_DAYS = {"close_t0": 0, "t1": 1, "t3": 3, "t5": 5}
TEXT_FEATURES = [
    "direction",
    "magnitude",
    "confidence",
    "eps_surprise",
    "rev_surprise",
    "rating_change",
    "target_change",
    "guidance_change",
    "is_earnings",
    "is_analyst",
    "is_guidance",
]


def daily_stats(bars: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per symbol and session: stats from PRIOR sessions only (shifted by one)."""
    b = bars.copy()
    b["session_date"] = (
        b["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    b["dollar_vol"] = b["close"] * b["volume"]
    d = (
        b.groupby(["symbol", "session_date"])
        .agg(close=("close", "last"), volume=("volume", "sum"), dollar_vol=("dollar_vol", "sum"))
        .reset_index()
        .sort_values(["symbol", "session_date"])
    )
    g = d.groupby("symbol")
    d["ret"] = g["close"].transform(lambda s: np.log(s).diff())
    d["sigma_daily"] = g["ret"].transform(lambda s: s.rolling(window, min_periods=5).std().shift(1))
    d["adv_usd"] = g["dollar_vol"].transform(
        lambda s: s.rolling(window, min_periods=5).mean().shift(1)
    )
    d["minute_volume"] = g["volume"].transform(
        lambda s: (s.rolling(window, min_periods=5).mean() / 390).shift(1)
    )
    d["mom_5d"] = g["close"].transform(lambda s: np.log(s).diff(5).shift(1))
    return d[["symbol", "session_date", "sigma_daily", "adv_usd", "minute_volume", "mom_5d"]]


def daily_spreads(quotes: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Median closing spread (bps) over PRIOR sessions: the ex-ante exit-cost estimate."""
    q = quotes.copy()
    q["session_date"] = (
        q["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    )
    q["spread_bps"] = (q["ask"] - q["bid"]) / ((q["ask"] + q["bid"]) / 2) * 1e4
    d = q.groupby(["symbol", "session_date"])["spread_bps"].last().reset_index()
    d = d.sort_values(["symbol", "session_date"])
    d["avg_exit_spread_bps"] = d.groupby("symbol")["spread_bps"].transform(
        lambda s: s.rolling(window, min_periods=3).median().shift(1)
    )
    return d[["symbol", "session_date", "avg_exit_spread_bps"]]


def _volume_since(bars_sym: pd.DataFrame, start: pd.Series, end: pd.Series) -> np.ndarray:
    ts = bars_sym["ts"].dt.tz_convert(None).to_numpy().astype("datetime64[ns]")
    cum = np.r_[0.0, np.cumsum(bars_sym["volume"].to_numpy(dtype=float))]
    s = np.searchsorted(
        ts, start.dt.tz_convert(None).to_numpy().astype("datetime64[ns]"), side="left"
    )
    e = np.searchsorted(
        ts, end.dt.tz_convert(None).to_numpy().astype("datetime64[ns]"), side="left"
    )
    return cum[e] - cum[s]  # bars with start <= ts < end, i.e. fully finished before entry


def build_panel(
    events: pd.DataFrame,
    extractions: pd.DataFrame,
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
    cfg: RunConfig,
    daily_bars: pd.DataFrame | None = None,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`bars` are minute bars (at least around events); `daily_bars`, if given, feed the
    20-day stats instead (one bar per session is enough), which keeps real-data pulls small."""
    validate(extractions, EXTRACTIONS, "extractions")
    book = QuoteBook(quotes)
    bench = cfg.data.benchmark
    stats = daily_stats(daily_bars if daily_bars is not None else bars).merge(
        daily_spreads(quotes), on=["symbol", "session_date"], how="left"
    )
    bars_by_sym = {s: g.sort_values("ts") for s, g in bars.groupby("symbol")}

    ev = events.merge(
        extractions.drop(columns=["method"]), on="article_id", how="left", validate="many_to_one"
    )
    for t in ("earnings", "analyst", "guidance"):
        ev[f"is_{t}"] = (ev["event_type"] == t).astype(float)
    extra_cols = ["novelty_llm"] if "novelty_llm" in ev and ev["novelty_llm"].notna().any() else []
    if signals is not None and len(signals):
        ev = ev.merge(signals, on="event_id", how="left", validate="one_to_one")
        extra_cols += [c for c in signals.columns if c != "event_id"]

    rows = []
    for d in cfg.align.delays_min:
        for h in cfg.align.horizons:
            x = ev.copy()
            x["delay_min"] = d
            x["horizon"] = h
            x["entry_ts"] = x["tau"] + pd.Timedelta(minutes=d)
            x["exit_ts"] = [horizon_exit(s, h) for s in x["session_date"]]
            close_t0 = pd.to_datetime([session_close(s) for s in x["session_date"]], utc=True)
            x = x[(x["entry_ts"] < close_t0) & (x["exit_ts"] > x["entry_ts"])]
            rows.append(x)
    panel = pd.concat(rows, ignore_index=True)

    out = []
    for sym, g in panel.groupby("symbol"):
        g = g.copy()
        hedge = cfg.data.sector_etf.get(sym, bench)
        p0 = g["tau"] - pd.Timedelta(minutes=1)
        pre = p0 - pd.Timedelta(minutes=cfg.align.pre_window_min)
        q_entry = book.at(sym, g["entry_ts"])
        q_exit = book.at(sym, g["exit_ts"])
        m0, m_pre = book.mid(sym, p0), book.mid(sym, pre)
        b0, b_pre = book.mid(hedge, p0), book.mid(hedge, pre)
        b_entry, b_exit = book.mid(hedge, g["entry_ts"]), book.mid(hedge, g["exit_ts"])
        me, mx = q_entry["mid"].to_numpy(), q_exit["mid"].to_numpy()

        g["ar_react"] = np.log(me / m0) - np.log(b_entry / b0)
        g["pre_runup"] = np.log(m0 / m_pre) - np.log(b0 / b_pre)
        g["spread_bps_entry"] = (q_entry["ask"] - q_entry["bid"]).to_numpy() / me * 1e4
        g["quote_imb"] = (
            (q_entry["bid_size"] - q_entry["ask_size"])
            / (q_entry["bid_size"] + q_entry["ask_size"])
        ).to_numpy()
        g["price"] = m0
        g["entry_bid"], g["entry_ask"] = q_entry["bid"].to_numpy(), q_entry["ask"].to_numpy()
        g["exit_bid"], g["exit_ask"] = q_exit["bid"].to_numpy(), q_exit["ask"].to_numpy()
        g["bench_ret"] = np.log(b_exit / b_entry)
        g["target"] = np.log(mx / me) - g["bench_ret"]
        if sym in bars_by_sym:
            g["vol_since_tau"] = _volume_since(bars_by_sym[sym], g["tau"], g["entry_ts"])
        else:
            g["vol_since_tau"] = np.nan
        out.append(g)
    panel = pd.concat(out, ignore_index=True)

    panel = panel.merge(stats, on=["symbol", "session_date"], how="left")
    panel["vol_surge"] = panel["vol_since_tau"] / (panel["minute_volume"] * panel["delay_min"])
    panel["horizon_days"] = panel["horizon"].map(HORIZON_DAYS).astype(float)
    panel["overnight"] = panel["overnight"].astype(float)
    panel = panel[(panel["price"] >= cfg.data.min_price) & panel["target"].notna()]

    keep = (
        PANEL_ID_COLS
        + PANEL_LABEL_COLS
        + ["event_type", "adv_usd", "sigma_daily", "price", "avg_exit_spread_bps"]
        + TEXT_FEATURES
        + ["ar_react", "pre_runup", "spread_bps_entry", "quote_imb", "vol_surge", "mom_5d"]
        + ["overnight", "horizon_days"]
        + extra_cols
    )
    return panel[keep].reset_index(drop=True)
