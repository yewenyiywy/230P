import numpy as np
import pandas as pd

from newsalpha.evaluation.metrics import deflated_sharpe, max_drawdown, rank_ic, sharpe


def test_sharpe_and_drawdown():
    r = pd.Series([0.01, -0.02, 0.01, 0.01])
    assert np.isclose(sharpe(r), r.mean() / r.std() * np.sqrt(252))
    assert np.isclose(max_drawdown(r), -0.02)


def test_deflated_sharpe_falls_with_more_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    assert deflated_sharpe(r, 1) > deflated_sharpe(r, 10) > deflated_sharpe(r, 1000)


def test_deflated_sharpe_of_noise_is_not_significant():
    rng = np.random.default_rng(1)
    assert deflated_sharpe(pd.Series(rng.normal(0, 0.01, 500)), 50) < 0.5


def test_rank_ic_perfect():
    df = pd.DataFrame({"session_date": [1] * 6, "pred": range(6), "target": range(6)})
    assert np.isclose(rank_ic(df)["mean"], 1.0)
