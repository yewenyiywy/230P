"""Walk-forward splits with purging (plan, Evaluation -> Protocol).

Folds are calendar months. For a fold starting at T:
    test   = rows whose entry_ts is in [T, T + 1 month)
    train  = rows whose exit_ts < T - embargo   (purge: no label may overlap the test period)
The first fold is the first month at or after `first_test` with >= min_train rows of training data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    start: pd.Timestamp
    end: pd.Timestamp
    train: np.ndarray  # boolean mask over the panel
    test: np.ndarray


def walk_forward(
    panel: pd.DataFrame,
    first_test: str | pd.Timestamp,
    months: int = 1,
    min_train: int = 200,
    embargo: pd.Timedelta | None = None,
    last_test: str | pd.Timestamp | None = None,
) -> Iterator[Fold]:
    embargo = embargo if embargo is not None else pd.Timedelta(0)
    entry, exit_ = panel["entry_ts"], panel["exit_ts"]
    start = pd.Timestamp(first_test, tz="UTC")
    stop = pd.Timestamp(last_test, tz="UTC") if last_test else entry.max() + pd.Timedelta(seconds=1)
    while start < stop:
        end = min(start + pd.DateOffset(months=months), stop)
        test = ((entry >= start) & (entry < end)).to_numpy()
        train = (exit_ < start - embargo).to_numpy()
        if test.any() and train.sum() >= min_train:
            yield Fold(start, end, train, test)
        start = end
