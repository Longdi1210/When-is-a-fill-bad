from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FORMATION_WINDOWS_SECONDS = [1, 2, 5, 10, 30, 60]
MARKOUT_HORIZONS_SECONDS = [1, 5, 10, 30, 60, 300]
SIDES = ["buy", "sell"]


@dataclass(frozen=True)
class SideColumns:
    market: str
    cancel: str
    replenish: str
    depth: str
    imbalance: str
    sign: float


SIDE_COLUMNS = {
    "buy": SideColumns(
        market="market_sell_pressure",
        cancel="bid_cancellation_pressure",
        replenish="bid_limit_replenishment",
        depth="bid_depth_top_15",
        imbalance="imbalance_top_15",
        sign=1.0,
    ),
    "sell": SideColumns(
        market="market_buy_pressure",
        cancel="ask_cancellation_pressure",
        replenish="ask_limit_replenishment",
        depth="ask_depth_top_15",
        imbalance="imbalance_top_15",
        sign=-1.0,
    ),
}


def passive_markout_bps(future_return_bps: pd.Series, side: str) -> pd.Series:
    return SIDE_COLUMNS[side].sign * future_return_bps


def side_component_frame(data: pd.DataFrame, side: str) -> pd.DataFrame:
    cols = SIDE_COLUMNS[side]
    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(data["timestamp"], utc=True),
            "date": pd.to_datetime(data["timestamp"], utc=True).dt.date.astype(str),
            "side": side,
            "spread": data["spread"].astype(float),
            "visible_depth": data[cols.depth].astype(float),
            "book_imbalance": (cols.sign * data[cols.imbalance].astype(float)),
            "aggressive_market_pressure": data[cols.market].astype(float),
            "same_side_cancellation": data[cols.cancel].astype(float),
            "same_side_replenishment": data[cols.replenish].astype(float),
            "one_second_abs_return_bps": data["one_second_return_bps"].abs().astype(float),
        }
    )
    return result


def add_time_aggregates(frame: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    ordered = frame.sort_values("timestamp").set_index("timestamp")
    out = frame.copy()
    for window in windows:
        roll = ordered[
            [
                "aggressive_market_pressure",
                "same_side_cancellation",
                "same_side_replenishment",
                "visible_depth",
                "one_second_abs_return_bps",
            ]
        ].rolling(f"{window}s", closed="both")
        sums = roll.sum()
        depth_mean = ordered[["visible_depth"]].rolling(f"{window}s", closed="both").mean()
        out[f"market_pressure_{window}s"] = sums["aggressive_market_pressure"].to_numpy()
        out[f"cancellation_{window}s"] = sums["same_side_cancellation"].to_numpy()
        out[f"replenishment_{window}s"] = sums["same_side_replenishment"].to_numpy()
        out[f"visible_depth_{window}s"] = depth_mean["visible_depth"].to_numpy()
        out[f"recent_volatility_{window}s"] = sums["one_second_abs_return_bps"].to_numpy()
        out[f"p1_market_{window}s"] = out[f"market_pressure_{window}s"]
        out[f"p2_market_cancel_{window}s"] = out[f"market_pressure_{window}s"] + out[f"cancellation_{window}s"]
        out[f"p3_full_{window}s"] = (
            out[f"market_pressure_{window}s"] + out[f"cancellation_{window}s"] - out[f"replenishment_{window}s"]
        )
        out[f"p3_depth_norm_{window}s"] = safe_divide(out[f"p3_full_{window}s"], out[f"visible_depth_{window}s"])
    return out


def build_side_dataset(data: pd.DataFrame, side: str, windows: list[int], horizons: list[int]) -> pd.DataFrame:
    out = add_time_aggregates(side_component_frame(data, side), windows)
    for horizon in horizons:
        out[f"markout_bps_{horizon}s"] = passive_markout_bps(data[f"future_return_bps_{horizon}s"], side)
        out[f"adverse_{horizon}s"] = (out[f"markout_bps_{horizon}s"] < 0).astype(float)
    return out


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def split_masks(frame: pd.DataFrame, split_table: pd.DataFrame) -> dict[str, pd.Series]:
    dates = pd.to_datetime(frame["timestamp"], utc=True).dt.date.astype(str)
    masks = {}
    for row in split_table.itertuples(index=False):
        masks[row.split] = (dates >= str(row.start_date_utc)) & (dates <= str(row.end_date_utc))
    return masks


def train_quantile_bins(values: pd.Series, train_mask: pd.Series, n_bins: int = 10) -> np.ndarray:
    train_values = values.loc[train_mask].replace([np.inf, -np.inf], np.nan).dropna()
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(train_values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        center = float(train_values.median()) if len(train_values) else 0.0
        edges = np.array([center - 1.0, center, center + 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    return pd.Series(np.digitize(values, edges[1:-1], right=True) + 1, index=values.index)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, mean, mean
    half_width = 1.96 * float(clean.std(ddof=1)) / np.sqrt(len(clean))
    return mean, mean - half_width, mean + half_width


def rank_correlation(x: pd.Series, y: pd.Series) -> float:
    xy = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(xy) < 3 or xy["x"].nunique() < 2 or xy["y"].nunique() < 2:
        return np.nan
    return float(xy["x"].rank().corr(xy["y"].rank()))


def linear_fit_predict(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_mean = train_x.mean()
    x_std = train_x.std(ddof=0).replace(0, 1.0)
    x_train = ((train_x - x_mean) / x_std).fillna(0.0).to_numpy()
    x_test = ((test_x - x_mean) / x_std).fillna(0.0).to_numpy()
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    coef = np.linalg.lstsq(x_train, train_y.to_numpy(), rcond=None)[0]
    return x_test @ coef, coef


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y = y_true.to_numpy()
    residual = y - y_pred
    baseline = y - y.mean()
    denom = float(np.sum(baseline**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denom if denom > 0 else np.nan
    mae = float(np.mean(np.abs(residual)))
    rank = rank_correlation(pd.Series(y_pred), pd.Series(y))
    directional = float((np.sign(y_pred) == np.sign(y)).mean())
    return {"r2": r2, "mae": mae, "rank_correlation": rank, "directional_accuracy": directional}


def local_block_shuffle(frame: pd.DataFrame, columns: list[str], block: str = "5min", seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shuffled = frame.copy()
    blocks = pd.to_datetime(shuffled["timestamp"], utc=True).dt.floor(block)
    for _, index in shuffled.groupby(blocks).groups.items():
        idx = np.array(list(index))
        if len(idx) < 2:
            continue
        for column in columns:
            shuffled.loc[idx, column] = shuffled.loc[rng.permutation(idx), column].to_numpy()
    return shuffled
