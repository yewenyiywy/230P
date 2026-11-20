"""Data source adapters. Pick one with `get_source(cfg)`."""

from __future__ import annotations

from newsalpha.config import DataConfig
from newsalpha.data.sources.base import MarketData


def get_source(cfg: DataConfig) -> MarketData:
    if cfg.source == "synthetic":
        from newsalpha.data.sources.synthetic import SyntheticWorld

        return SyntheticWorld.generate(cfg)
    if cfg.source == "alpaca":
        from newsalpha.data.sources.alpaca import AlpacaSource

        return AlpacaSource.from_env(cfg)
    raise ValueError(f"unknown data source {cfg.source}")
