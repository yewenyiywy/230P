import numpy as np
import pandas as pd
import pytest

from newsalpha.config import TradingConfig
from newsalpha.trading.backtest import realise
from newsalpha.trading.rules import decide


def _rows(**over) -> pd.DataFrame:
    base = {
        "event_id": ["e1", "e2"],
        "event_type": ["earnings", "analyst"],
        "pred": [0.02, -0.02],
        "sigma_hat": [0.01, 0.01],
        "price": [50.0, 50.0],
        "entry_bid": [49.99, 49.99],
        "entry_ask": [50.01, 50.01],
        "exit_bid": [50.99, 48.99],
        "exit_ask": [51.01, 49.01],
        "bench_ret": [0.0, 0.0],
        "avg_exit_spread_bps": [4.0, 4.0],
        "spread_bps_entry": [4.0, 4.0],
        "sigma_daily": [0.02, 0.02],
        "adv_usd": [5e7, 5e7],
        "horizon_days": [1.0, 1.0],
        "exit_ts": pd.to_datetime(["2025-01-03 21:00", "2025-01-03 21:00"], utc=True),
    }
    base.update(over)
    return pd.DataFrame(base)


def test_long_short_trades_both_sides():
    d = decide(_rows(), TradingConfig(kappa=0.1))
    assert list(d["side"]) == [1, -1] and d["trade"].all()
    assert (d["cost_est"] > 0).all() and (d["notional"] > 0).all()


@pytest.mark.parametrize("variant", ["long_only", "long_only_hedged"])
def test_long_only_never_shorts(variant):
    d = decide(_rows(), TradingConfig(kappa=0.1, variant=variant))
    assert (d["side"] >= 0).all() and not d.loc[1, "trade"]


def test_no_shorts_below_price_floor():
    d = decide(_rows(price=[5.0, 5.0]), TradingConfig(kappa=0.1, no_short_below_price=10))
    assert d.loc[1, "side"] == 0


def test_high_kappa_means_no_trade():
    d = decide(_rows(), TradingConfig(kappa=50))
    assert not d["trade"].any() and (d["notional"] == 0).all()


def test_realise_uses_bid_ask_and_costs():
    cfg = TradingConfig(kappa=0.1, variant="long_only")
    t = realise(decide(_rows(), cfg).query("trade"), cfg)
    long_gross = 50.99 / 50.01 - 1  # buy at ask, sell at bid
    assert np.isclose(t.iloc[0]["gross_ret"], long_gross)
    assert t.iloc[0]["net_ret"] < long_gross  # impact charged on top


def test_short_return_sign():
    cfg = TradingConfig(kappa=0.1)
    t = realise(decide(_rows(), cfg).query("trade"), cfg)
    short = t[t["side"] < 0].iloc[0]
    assert np.isclose(short["gross_ret"], (49.99 - 49.01) / 49.99)
