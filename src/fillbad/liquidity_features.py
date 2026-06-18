from __future__ import annotations

import numpy as np
import pandas as pd


def depth_imbalance(bid_depth: pd.Series, ask_depth: pd.Series) -> pd.Series:
    denominator = (bid_depth + ask_depth).replace(0, np.nan)
    return ((bid_depth - ask_depth) / denominator).fillna(0.0)


def signed_market_pressure(market_buy_pressure: pd.Series, market_sell_pressure: pd.Series) -> pd.Series:
    return market_buy_pressure - market_sell_pressure


def signed_cancellation_pressure(ask_cancellation: pd.Series, bid_cancellation: pd.Series) -> pd.Series:
    return ask_cancellation - bid_cancellation


def signed_replenishment(ask_limit_replenishment: pd.Series, bid_limit_replenishment: pd.Series) -> pd.Series:
    return ask_limit_replenishment - bid_limit_replenishment


def visible_liquidity_fragility(
    market_buy_pressure: pd.Series,
    market_sell_pressure: pd.Series,
    ask_cancellation: pd.Series,
    bid_cancellation: pd.Series,
    ask_replenishment: pd.Series,
    bid_replenishment: pd.Series,
) -> pd.Series:
    upward = market_buy_pressure + ask_cancellation - ask_replenishment
    downward = market_sell_pressure + bid_cancellation - bid_replenishment
    return upward - downward


def zscore_from_train(values: pd.Series, train_mask: pd.Series) -> pd.Series:
    train_values = values.loc[train_mask]
    mean = train_values.mean()
    std = train_values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return values * 0.0
    return (values - mean) / std
