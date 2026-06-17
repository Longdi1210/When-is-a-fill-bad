from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResearchConfig:
    seed: int = 42
    output_dir: str = "outputs"
    data_mode: str = "synthetic_validation"
    instrument: str = "BTC-USD"
    venue: str = "synthetic"
    btc_data_path: str = "data/raw/btc_usd_events.csv"
    steps: int = 5000
    fundamental_price: int = 10000
    tick_size: int = 1
    order_size: int = 5
    fill_horizon: int = 80
    markout_horizons: list[int] = field(default_factory=lambda: [10, 50, 100, 500])
    markout_horizon_for_value: int = 100
    lookback_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200, 500])
    flow_weighting: str = "uniform"
    shuffle_block_size: int = 200
    fees_ticks: float = 0.05
    queue_ahead_fraction: float = 0.45
    queue_ahead_max: float = 20.0
    queue_ahead_sensitivity: list[float] = field(default_factory=lambda: [0.25, 0.45, 0.75])
    min_spread_ticks: int = 1
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    eligible_every_n_events: int = 1
    rolling_window: int = 20
    models: dict[str, Any] = field(default_factory=lambda: {"fill": "logistic", "markout": "ridge"})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResearchConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**payload)
