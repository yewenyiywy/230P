"""Return models: forecast the abnormal return from entry to exit.

One pooled model covers all entry delays and horizons; `delay_min` and
`horizon_days` are features, so the model can learn decay. Each model also
reports a forecast scale sigma_hat per (delay, horizon) used by the trade rule.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from newsalpha.config import ModelConfig


class ReturnModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> ReturnModel: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class RidgeModel:
    def __init__(self, alpha: float = 10.0):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.cols: list[str] = []
        self.medians: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RidgeModel:
        self.cols = list(X.columns)
        self.medians = X.median(numeric_only=True).fillna(0)
        self.model.fit(X.fillna(self.medians).fillna(0).to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.cols].fillna(self.medians).fillna(0).to_numpy())


class LGBMModel:
    DEFAULTS = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "verbose": -1,
    }

    def __init__(self, **params):
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**{**self.DEFAULTS, **params})
        self.cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LGBMModel:
        self.cols = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.cols])


def make_model(cfg: ModelConfig) -> ReturnModel:
    if cfg.kind == "ridge":
        return RidgeModel(**cfg.params)
    if cfg.kind == "lgbm":
        return LGBMModel(**cfg.params)
    raise ValueError(f"unknown model kind {cfg.kind}")
