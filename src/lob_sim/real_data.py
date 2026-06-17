from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "event_id",
    "event_type",
    "side",
    "price",
    "size",
    "best_bid",
    "best_ask",
    "bid_size",
    "ask_size",
}

OPTIONAL_COLUMNS = {
    "trade_sign",
    "trade_size",
    "order_id",
}


def validate_btc_schema(events: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(f"BTC event data missing required columns: {missing}")
    result = events.copy()
    if "trade_sign" not in result:
        result["trade_sign"] = 0
    if "trade_size" not in result:
        result["trade_size"] = 0.0
    if "order_id" not in result:
        result["order_id"] = ""
    if (result["best_ask"] <= result["best_bid"]).any():
        raise ValueError("BTC event data contains non-positive spread rows.")
    return result


def load_btc_events(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"BTC event file not found: {source}. Add Coinbase BTC-USD L2/L3 events or point btc_data_path to a valid CSV."
        )
    events = pd.read_csv(source)
    return validate_btc_schema(events)


def normalize_btc_events(events: pd.DataFrame) -> pd.DataFrame:
    validated = validate_btc_schema(events)
    normalized = pd.DataFrame(
        {
            "step": range(1, len(validated) + 1),
            "timestamp": validated["timestamp"],
            "event_type": validated["event_type"],
            "order_side": validated["side"],
            "order_price": validated["price"],
            "order_qty": validated["size"],
            "best_bid": validated["best_bid"],
            "best_ask": validated["best_ask"],
            "mid_price": (validated["best_bid"] + validated["best_ask"]) / 2.0,
            "markout_mid_price": (validated["best_bid"] + validated["best_ask"]) / 2.0,
            "spread": validated["best_ask"] - validated["best_bid"],
            "bid_depth": validated["bid_size"],
            "ask_depth": validated["ask_size"],
            "buy_trade_qty": (validated["trade_sign"] > 0) * validated["trade_size"],
            "sell_trade_qty": (validated["trade_sign"] < 0) * validated["trade_size"],
            "trade_qty": validated["trade_size"].abs(),
            "trade_price": validated["price"],
            "cancel_side": "",
            "cancel_qty": 0.0,
            "cancel_price": float("nan"),
        }
    )
    normalized["queue_imbalance"] = (normalized["bid_depth"] - normalized["ask_depth"]) / (
        normalized["bid_depth"] + normalized["ask_depth"]
    ).replace(0, 1)
    normalized["microprice_deviation"] = 0.0
    normalized["recent_signed_trade_flow"] = (normalized["buy_trade_qty"] - normalized["sell_trade_qty"]).rolling(20, min_periods=1).sum()
    normalized["recent_trade_volume"] = normalized["trade_qty"].rolling(20, min_periods=1).sum()
    normalized["recent_volatility"] = normalized["mid_price"].diff().abs().rolling(20, min_periods=1).mean().fillna(0.0)
    normalized["recent_mid_move"] = normalized["mid_price"].diff(20).fillna(0.0)
    normalized["spread_bps"] = 10000 * normalized["spread"] / normalized["mid_price"]
    return normalized

