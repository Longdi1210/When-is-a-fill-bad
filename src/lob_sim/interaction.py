from __future__ import annotations

import math

import numpy as np
import pandas as pd


def passive_side_sign(side: str) -> int:
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    raise ValueError(f"Unknown passive side: {side}")


def aggressive_flow_against_passive(side: str, buy_trade_qty: float, sell_trade_qty: float) -> float:
    signed_aggressive_flow = buy_trade_qty - sell_trade_qty
    return -passive_side_sign(side) * signed_aggressive_flow


def signed_flow_persistence(
    side: str,
    buy_trade_qty: pd.Series,
    sell_trade_qty: pd.Series,
    weighting: str = "uniform",
) -> float:
    trade_qty = buy_trade_qty + sell_trade_qty
    denom = float(trade_qty.sum())
    if denom <= 0:
        return 0.0
    pressure = aggressive_flow_against_passive(side, buy_trade_qty, sell_trade_qty)
    if weighting == "exponential":
        weights = np.exp(np.linspace(-1.0, 0.0, len(trade_qty)))
        return float((pressure * weights).sum() / max((trade_qty * weights).sum(), 1e-12))
    return float(pressure.sum() / denom)


def passive_depth(events: pd.DataFrame, side: str) -> pd.Series:
    return events["bid_depth"] if side == "buy" else events["ask_depth"]


def queue_depletion_components(side: str, window_events: pd.DataFrame, current_depth: float, epsilon: float = 1e-9) -> dict:
    if window_events.empty:
        return {
            "total_depletion": 0.0,
            "trade_depletion": 0.0,
            "cancel_depletion": 0.0,
            "replenishment": 0.0,
        }
    start_depth = float(passive_depth(window_events, side).iloc[0])
    denominator = max(start_depth, epsilon)
    if side == "buy":
        trade_qty = float(window_events["sell_trade_qty"].sum())
        cancel_qty = float(window_events.loc[window_events["cancel_side"] == "buy", "cancel_qty"].sum())
    else:
        trade_qty = float(window_events["buy_trade_qty"].sum())
        cancel_qty = float(window_events.loc[window_events["cancel_side"] == "sell", "cancel_qty"].sum())
    total_depletion = (start_depth - current_depth) / denominator
    trade_depletion = trade_qty / denominator
    cancel_depletion = cancel_qty / denominator
    replenishment = max(trade_depletion + cancel_depletion - total_depletion, 0.0)
    return {
        "total_depletion": float(total_depletion),
        "trade_depletion": float(trade_depletion),
        "cancel_depletion": float(cancel_depletion),
        "replenishment": float(replenishment),
    }


def add_flow_depletion_features(
    orders: pd.DataFrame,
    events: pd.DataFrame,
    windows: list[int],
    weighting: str = "uniform",
) -> pd.DataFrame:
    events_by_step = events.set_index("step", drop=False)
    result = orders.copy()
    for window in windows:
        flow_values = []
        total_dep_values = []
        trade_dep_values = []
        cancel_dep_values = []
        replenish_values = []
        interaction_values = []
        for _, order in result.iterrows():
            step = int(order["step"])
            side = order["side"]
            history = events_by_step[(events_by_step["step"] >= step - window) & (events_by_step["step"] < step)]
            flow = signed_flow_persistence(side, history["buy_trade_qty"], history["sell_trade_qty"], weighting)
            current_depth = float(order["bid_depth"] if side == "buy" else order["ask_depth"])
            components = queue_depletion_components(side, history, current_depth)
            trade_depletion = components["trade_depletion"]
            flow_values.append(flow)
            total_dep_values.append(components["total_depletion"])
            trade_dep_values.append(trade_depletion)
            cancel_dep_values.append(components["cancel_depletion"])
            replenish_values.append(components["replenishment"])
            interaction_values.append(flow * trade_depletion)
        result[f"flow_persistence_{window}"] = flow_values
        result[f"total_depletion_{window}"] = total_dep_values
        result[f"trade_depletion_{window}"] = trade_dep_values
        result[f"cancel_depletion_{window}"] = cancel_dep_values
        result[f"replenishment_{window}"] = replenish_values
        result[f"flow_depletion_interaction_{window}"] = interaction_values
        result[f"abs_flow_persistence_{window}"] = result[f"flow_persistence_{window}"].abs()
    return result


def interaction_feature_columns(window: int) -> tuple[list[str], list[str], list[str], list[str]]:
    controls = [
        "spread",
        "queue_ahead",
        "signed_queue_imbalance",
        "recent_volatility",
        "recent_mid_move",
        "side_is_buy",
    ]
    m1 = controls + [f"flow_persistence_{window}"]
    m2 = m1 + [f"trade_depletion_{window}", f"total_depletion_{window}"]
    m3 = m2 + [f"flow_depletion_interaction_{window}"]
    return controls, m1, m2, m3

