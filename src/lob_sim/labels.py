from __future__ import annotations

import numpy as np
import pandas as pd

from .book import Side
from .config import ResearchConfig


def side_sign(side: str) -> int:
    if side == Side.BUY.value:
        return 1
    if side == Side.SELL.value:
        return -1
    raise ValueError(f"Unknown side: {side}")


def signed_markout(side: str, fill_price: float, future_mid: float, tick_size: float) -> float:
    return side_sign(side) * (future_mid - fill_price) / tick_size


def expected_posting_value(fill_probability: float, spread_ticks: float, expected_markout: float, fees_ticks: float) -> float:
    return fill_probability * (spread_ticks / 2.0 + expected_markout - fees_ticks)


def construct_passive_orders(events: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    rows: list[dict] = []
    valid = events.dropna(subset=["best_bid", "best_ask", "mid_price", "spread"]).copy()
    valid = valid[valid["spread"] >= config.min_spread_ticks]
    for _, event in valid.iloc[:: config.eligible_every_n_events].iterrows():
        for side in [Side.BUY.value, Side.SELL.value]:
            same_depth = event["bid_depth"] if side == Side.BUY.value else event["ask_depth"]
            price = event["best_bid"] if side == Side.BUY.value else event["best_ask"]
            rows.append(
                {
                    "obs_id": f"{int(event['step'])}_{side}",
                    "step": int(event["step"]),
                    "side": side,
                    "submission_price": float(price),
                    "queue_ahead": min(config.queue_ahead_max, max(0.0, float(same_depth) * config.queue_ahead_fraction)),
                    "order_size": config.order_size,
                    "mid_price": float(event["mid_price"]),
                    "spread": float(event["spread"]),
                    "spread_bps": float(event["spread_bps"]),
                    "bid_depth": float(event["bid_depth"]),
                    "ask_depth": float(event["ask_depth"]),
                    "queue_imbalance": float(event["queue_imbalance"]),
                    "microprice_deviation": float(event["microprice_deviation"]),
                    "recent_signed_trade_flow": float(event["recent_signed_trade_flow"]),
                    "recent_trade_volume": float(event["recent_trade_volume"]),
                    "recent_volatility": float(event["recent_volatility"]),
                    "recent_mid_move": float(event["recent_mid_move"]),
                }
            )
    return pd.DataFrame(rows)


def _depletion_for_order(future: pd.DataFrame, side: str, price: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == Side.BUY.value:
        trade = np.where((future["sell_trade_qty"] > 0) & (future["trade_price"] <= price), future["sell_trade_qty"], 0.0)
        cancel = np.where(
            (future["cancel_side"] == Side.BUY.value) & (future["cancel_price"] == price),
            future["cancel_qty"],
            0.0,
        )
    else:
        trade = np.where((future["buy_trade_qty"] > 0) & (future["trade_price"] >= price), future["buy_trade_qty"], 0.0)
        cancel = np.where(
            (future["cancel_side"] == Side.SELL.value) & (future["cancel_price"] == price),
            future["cancel_qty"],
            0.0,
        )
    total = trade + cancel
    return trade.astype(float), cancel.astype(float), total.astype(float)


def label_orders(orders: pd.DataFrame, events: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    events_by_step = events.set_index("step", drop=False)
    max_horizon = max(max(config.markout_horizons), config.fill_horizon)
    labeled = []
    for _, order in orders.iterrows():
        start = int(order["step"])
        future = events[(events["step"] > start) & (events["step"] <= start + config.fill_horizon)].copy()
        row = order.to_dict()
        row.update(
            {
                "filled": 0,
                "censored": 1,
                "time_to_fill": np.nan,
                "fill_step": np.nan,
                "fill_price": np.nan,
                "fill_event": "",
                "trade_depletion_share": np.nan,
                "adverse_fill": np.nan,
            }
        )
        for horizon in config.markout_horizons:
            row[f"markout_{horizon}"] = np.nan

        if not future.empty:
            trade_dep, cancel_dep, total_dep = _depletion_for_order(future, row["side"], row["submission_price"])
            cum_total = total_dep.cumsum()
            threshold = float(row["queue_ahead"]) + float(row["order_size"])
            fill_positions = np.flatnonzero(cum_total >= threshold)
            if len(fill_positions):
                fill_pos = int(fill_positions[0])
                fill_step = int(future.iloc[fill_pos]["step"])
                fill_window_trade = float(trade_dep[: fill_pos + 1].sum())
                fill_window_cancel = float(cancel_dep[: fill_pos + 1].sum())
                row["filled"] = 1
                row["censored"] = 0
                row["time_to_fill"] = fill_step - start
                row["fill_step"] = fill_step
                row["fill_price"] = row["submission_price"]
                row["fill_event"] = "trade" if trade_dep[fill_pos] > 0 else "cancel_proxy"
                denom = fill_window_trade + fill_window_cancel
                row["trade_depletion_share"] = fill_window_trade / denom if denom > 0 else np.nan
                for horizon in config.markout_horizons:
                    mark_step = fill_step + horizon
                    if mark_step in events_by_step.index:
                        mid_column = "markout_mid_price" if "markout_mid_price" in events_by_step.columns else "mid_price"
                        future_mid = float(events_by_step.loc[mark_step, mid_column])
                        row[f"markout_{horizon}"] = signed_markout(
                            row["side"],
                            float(row["fill_price"]),
                            future_mid,
                            config.tick_size,
                        )
                value_horizon = config.markout_horizon_for_value
                markout = row.get(f"markout_{value_horizon}", np.nan)
                row["adverse_fill"] = int(markout < 0) if not pd.isna(markout) else np.nan
        labeled.append(row)
    result = pd.DataFrame(labeled)
    result["side_is_buy"] = (result["side"] == Side.BUY.value).astype(int)
    result["signed_queue_imbalance"] = result["queue_imbalance"] * result["side"].map({Side.BUY.value: 1, Side.SELL.value: -1})
    result["signed_recent_trade_flow"] = result["recent_signed_trade_flow"] * result["side"].map({Side.BUY.value: 1, Side.SELL.value: -1})
    result["signed_microprice_deviation"] = result["microprice_deviation"] * result["side"].map({Side.BUY.value: 1, Side.SELL.value: -1})
    return result
