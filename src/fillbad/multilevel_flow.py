from __future__ import annotations

import numpy as np
import pandas as pd


DEPTH_LEVELS = [1, 3, 5, 10, 15]
ALL_LEVELS = list(range(15))


def side_prefix(side: str) -> str:
    return "bids" if side == "buy" else "asks"


def side_sign(side: str) -> float:
    return 1.0 if side == "buy" else -1.0


def level_columns(prefix: str, field: str) -> list[str]:
    return [f"{prefix}_{field}_{i}" for i in ALL_LEVELS]


def cumulative_depth(depth_by_level: pd.DataFrame, levels: list[int] = DEPTH_LEVELS) -> pd.DataFrame:
    out = pd.DataFrame(index=depth_by_level.index)
    for level in levels:
        out[f"depth_{level}"] = depth_by_level.iloc[:, :level].sum(axis=1)
    return out


def depth_shape(depth_by_level: pd.DataFrame, distance_by_level: pd.DataFrame) -> pd.DataFrame:
    total = depth_by_level.sum(axis=1).replace(0, np.nan)
    near = depth_by_level.iloc[:, :5].sum(axis=1)
    deep = depth_by_level.iloc[:, 5:].sum(axis=1)
    level_numbers = np.arange(1, depth_by_level.shape[1] + 1)
    weighted_level = depth_by_level.mul(level_numbers, axis=1).sum(axis=1) / total
    weighted_distance = depth_by_level.mul(distance_by_level.abs().to_numpy(), axis=0).sum(axis=1) / total
    slope = depth_by_level.iloc[:, :5].mean(axis=1) - depth_by_level.iloc[:, 10:].mean(axis=1)
    return pd.DataFrame(
        {
            "near_touch_depth_share": (near / total).fillna(0.0),
            "deep_book_depth_share": (deep / total).fillna(0.0),
            "depth_weighted_level": weighted_level.fillna(0.0),
            "depth_weighted_distance": weighted_distance.fillna(0.0),
            "depth_slope_near_minus_deep": slope.fillna(0.0),
        },
        index=depth_by_level.index,
    )


def weighted_level_flow(flow_by_level: pd.DataFrame, scheme: str, cumulative_depth_by_level: pd.DataFrame | None = None) -> pd.Series:
    n = flow_by_level.shape[1]
    if scheme == "uniform":
        weights = np.ones(n)
    elif scheme == "inverse_level":
        weights = 1.0 / np.arange(1, n + 1)
    elif scheme == "inverse_cumulative_depth":
        if cumulative_depth_by_level is None:
            raise ValueError("cumulative_depth_by_level is required for inverse_cumulative_depth")
        return (flow_by_level / cumulative_depth_by_level.replace(0, np.nan)).mean(axis=1).fillna(0.0)
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    weights = weights / weights.sum()
    return flow_by_level.mul(weights, axis=1).sum(axis=1)


def potential_penetration_class(shock_notional: pd.Series, lagged_depths: pd.DataFrame) -> pd.Series:
    labels = []
    for shock, row in zip(shock_notional, lagged_depths.itertuples(index=False)):
        depths = list(row)
        if not np.isfinite(shock) or shock <= 0:
            labels.append("no_shock")
        elif shock <= depths[0]:
            labels.append("contained_touch")
        elif shock <= depths[2]:
            labels.append("threatens_near_book")
        elif shock <= depths[3]:
            labels.append("threatens_medium_depth")
        elif shock <= depths[4]:
            labels.append("threatens_deep_visible")
        else:
            labels.append("exceeds_visible_depth")
    return pd.Series(labels, index=shock_notional.index)


def side_adjusted_markout_bps(future_mid: pd.Series, base_mid: pd.Series, side: str) -> pd.Series:
    future = pd.Series(np.asarray(future_mid, dtype=float), index=base_mid.index)
    base = pd.Series(np.asarray(base_mid, dtype=float), index=base_mid.index)
    return side_sign(side) * 10000.0 * np.log(future / base)


def quote_survives(pre_quote: pd.Series, future_best: pd.Series, side: str) -> pd.Series:
    future = pd.Series(np.asarray(future_best, dtype=float), index=pre_quote.index)
    pre = pd.Series(np.asarray(pre_quote, dtype=float), index=pre_quote.index)
    if side == "buy":
        return future >= pre
    if side == "sell":
        return future <= pre
    raise ValueError(f"unknown side: {side}")
