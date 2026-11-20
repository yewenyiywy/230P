"""Data contracts: the only tables that pass between pipeline stages.

Every stage function takes and returns DataFrames with these columns. All
timestamps are timezone-aware UTC. `validate` is called at stage boundaries,
so a broken contract fails loudly where it is produced, not three stages later.

Table          Produced by                 Key
-------------  --------------------------  ------------------------
ARTICLES       data.sources.*              (article_id, symbol)
MINUTE_BARS    data.sources.*              (symbol, ts)  ts = minute start
QUOTES         data.sources.*              (symbol, ts)  ts = snapshot time
EVENTS         align.build_events          event_id
EXTRACTIONS    extraction.*                (article_id, method)
PANEL          features.panel.build_panel  (event_id, delay_min, horizon)
PREDICTIONS    models.walk_forward         (event_id, delay_min, horizon)
TRADES         trading.backtest            (event_id, delay_min, horizon)
"""

from __future__ import annotations

import pandas as pd

ARTICLES = {
    "article_id": "object",
    "symbol": "object",
    "created_at": "datetime64[ns, UTC]",
    "updated_at": "datetime64[ns, UTC]",
    "headline": "object",
    "summary": "object",
    "body": "object",
    "source": "object",
}

MINUTE_BARS = {
    "symbol": "object",
    "ts": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

QUOTES = {
    "symbol": "object",
    "ts": "datetime64[ns, UTC]",
    "bid": "float64",
    "ask": "float64",
    "bid_size": "float64",
    "ask_size": "float64",
}

EVENTS = {
    "event_id": "object",
    "article_id": "object",
    "symbol": "object",
    "created_at": "datetime64[ns, UTC]",
    "tau": "datetime64[ns, UTC]",  # event minute: first full minute after created_at
    "session_date": "datetime64[ns]",  # trading session t0 (naive date)
    "overnight": "bool",
    "body_usable": "bool",
}

EXTRACTIONS = {
    "article_id": "object",
    "method": "object",
    "event_type": "object",
    "direction": "float64",  # -1, 0, +1
    "magnitude": "float64",  # 1..5
    "confidence": "float64",  # 0..1
    "eps_surprise": "float64",
    "rev_surprise": "float64",
    "rating_change": "float64",
    "target_change": "float64",
    "guidance_change": "float64",
}

# PANEL: id/time columns + label columns; every other column is a feature.
PANEL_ID_COLS = [
    "event_id",
    "symbol",
    "session_date",
    "delay_min",
    "horizon",
    "entry_ts",
    "exit_ts",
]
PANEL_LABEL_COLS = ["target", "entry_bid", "entry_ask", "exit_bid", "exit_ask", "bench_ret"]
PANEL_META_COLS = ["event_type", "adv_usd", "sigma_daily", "price", "avg_exit_spread_bps"]


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """All columns that are neither ids, labels nor trading metadata."""
    excluded = set(PANEL_ID_COLS) | set(PANEL_LABEL_COLS) | set(PANEL_META_COLS)
    return [c for c in panel.columns if c not in excluded]


class ContractError(ValueError):
    pass


def validate(df: pd.DataFrame, contract: dict[str, str], name: str) -> pd.DataFrame:
    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    for col, dtype in contract.items():
        is_utc = isinstance(df[col].dtype, pd.DatetimeTZDtype) and str(df[col].dt.tz) == "UTC"
        if "UTC" in dtype and not is_utc:
            raise ContractError(f"{name}.{col}: expected UTC timestamps, got {df[col].dtype}")
    return df
