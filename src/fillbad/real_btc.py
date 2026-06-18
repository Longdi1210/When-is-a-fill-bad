from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


RAW_CSV = Path("data/raw/kaggle/coinbase_btc/BTC_1sec.csv")
PARQUET_DIR = Path("data/processed/kaggle_btc")
CANONICAL_PARQUET = Path("data/processed/kaggle_btc_canonical.parquet")
REPORT_DIR = Path("data/reports/kaggle_btc")
AUDIT_FIGURE_DIR = Path("outputs/figures/real_btc_audit")


@dataclass(frozen=True)
class ColumnInfo:
    group: str
    level: int | None
    side: str | None
    event_type: str | None
    confidence: str
    notes: str


def classify_column(name: str) -> ColumnInfo:
    if name == "system_time":
        return ColumnInfo("timestamp", None, None, None, "verified", "UTC timestamp string in the CSV.")
    if name == "Unnamed: 0":
        return ColumnInfo("derived_fields", None, None, None, "high", "CSV row index exported as a column.")
    if name in {"midpoint", "spread"}:
        return ColumnInfo("derived_fields", None, None, None, "verified", "Provided midpoint/spread field.")
    if name in {"buys", "sells"}:
        side = "buy" if name == "buys" else "sell"
        return ColumnInfo("market-order activity", None, side, "market", "medium", "Aggregate buy/sell activity over the one-second interval.")
    pattern = re.match(r"^(bids|asks)_(distance|notional|cancel_notional|limit_notional|market_notional)_(\d+)$", name)
    if pattern:
        side_raw, kind, level_raw = pattern.groups()
        side = "bid" if side_raw == "bids" else "ask"
        level = int(level_raw)
        group_map = {
            "distance": f"{side} prices",
            "notional": f"{side} sizes",
            "cancel_notional": "cancellation activity",
            "limit_notional": "limit-order activity",
            "market_notional": "market-order activity",
        }
        event_map = {
            "distance": "snapshot",
            "notional": "snapshot",
            "cancel_notional": "cancel",
            "limit_notional": "limit",
            "market_notional": "market",
        }
        notes = {
            "distance": "Relative distance from midpoint; price is inferred as midpoint * (1 + distance).",
            "notional": "Visible notional at this book level.",
            "cancel_notional": "One-second cancellation notional at this side/level.",
            "limit_notional": "One-second limit-order notional at this side/level.",
            "market_notional": "One-second market-order notional consuming this side/level.",
        }[kind]
        confidence = "high" if kind in {"distance", "notional"} else "medium"
        return ColumnInfo(group_map[kind], level, side, event_map[kind], confidence, notes)
    return ColumnInfo("unknown fields", None, None, None, "low", "No parser rule matched this column name.")


def parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def level_columns(prefix: str, field: str, levels: int = 15) -> list[str]:
    return [f"{prefix}_{field}_{i}" for i in range(levels)]


def canonical_columns(levels: int = 15) -> list[str]:
    cols = ["system_time", "midpoint", "spread", "buys", "sells"]
    for side in ("bids", "asks"):
        for field in ("distance", "notional", "cancel_notional", "limit_notional", "market_notional"):
            cols.extend(level_columns(side, field, levels))
    return cols


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_canonical_table(df: pd.DataFrame, levels: int = 15) -> pd.DataFrame:
    result = pd.DataFrame()
    result["timestamp"] = parse_timestamps(df["system_time"])
    midpoint = df["midpoint"].astype(float)
    result["mid_price"] = midpoint
    result["spread"] = df["spread"].astype(float)
    result["best_bid"] = midpoint * (1.0 + df["bids_distance_0"].astype(float))
    result["best_ask"] = midpoint * (1.0 + df["asks_distance_0"].astype(float))

    bid_depths = df[level_columns("bids", "notional", levels)].astype(float)
    ask_depths = df[level_columns("asks", "notional", levels)].astype(float)
    bid_markets = df[level_columns("bids", "market_notional", levels)].astype(float)
    ask_markets = df[level_columns("asks", "market_notional", levels)].astype(float)
    bid_cancels = df[level_columns("bids", "cancel_notional", levels)].astype(float)
    ask_cancels = df[level_columns("asks", "cancel_notional", levels)].astype(float)
    bid_limits = df[level_columns("bids", "limit_notional", levels)].astype(float)
    ask_limits = df[level_columns("asks", "limit_notional", levels)].astype(float)

    result["bid_depth_level_1"] = bid_depths.iloc[:, 0]
    result["ask_depth_level_1"] = ask_depths.iloc[:, 0]
    result["bid_depth_top_5"] = bid_depths.iloc[:, :5].sum(axis=1)
    result["ask_depth_top_5"] = ask_depths.iloc[:, :5].sum(axis=1)
    result["bid_depth_top_15"] = bid_depths.sum(axis=1)
    result["ask_depth_top_15"] = ask_depths.sum(axis=1)
    for suffix in ("level_1", "top_5", "top_15"):
        bid = result[f"bid_depth_{suffix}"]
        ask = result[f"ask_depth_{suffix}"]
        result[f"imbalance_{suffix}"] = safe_ratio(bid - ask, bid + ask).fillna(0.0)

    result["market_buy_pressure"] = ask_markets.sum(axis=1)
    result["market_sell_pressure"] = bid_markets.sum(axis=1)
    result["net_market_pressure"] = result["market_buy_pressure"] - result["market_sell_pressure"]
    result["bid_cancellation_pressure"] = bid_cancels.sum(axis=1)
    result["ask_cancellation_pressure"] = ask_cancels.sum(axis=1)
    result["net_cancellation_pressure"] = result["ask_cancellation_pressure"] - result["bid_cancellation_pressure"]
    result["bid_limit_replenishment"] = bid_limits.sum(axis=1)
    result["ask_limit_replenishment"] = ask_limits.sum(axis=1)
    result["net_replenishment"] = result["ask_limit_replenishment"] - result["bid_limit_replenishment"]
    result["visible_bid_depth_change"] = result["bid_depth_top_15"].diff().fillna(0.0)
    result["visible_ask_depth_change"] = result["ask_depth_top_15"].diff().fillna(0.0)
    result["upward_fragility_raw"] = (
        result["market_buy_pressure"] + result["ask_cancellation_pressure"] - result["ask_limit_replenishment"]
    )
    result["downward_fragility_raw"] = (
        result["market_sell_pressure"] + result["bid_cancellation_pressure"] - result["bid_limit_replenishment"]
    )
    result["signed_fragility_raw"] = result["upward_fragility_raw"] - result["downward_fragility_raw"]
    result["absolute_fragility_raw"] = result[["upward_fragility_raw", "downward_fragility_raw"]].abs().max(axis=1)
    visible_depth = result["bid_depth_top_15"] + result["ask_depth_top_15"]
    result["signed_fragility_depth_norm"] = safe_ratio(result["signed_fragility_raw"], visible_depth).fillna(0.0)
    result["one_second_return_bps"] = 10000.0 * np.log(result["mid_price"]).diff().fillna(0.0)
    return result


def add_future_return_labels(canonical: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    base = canonical.sort_values("timestamp").copy()
    lookup = base[["timestamp", "mid_price"]].rename(columns={"timestamp": "future_timestamp", "mid_price": "future_mid_price"})
    for horizon in horizons:
        target = base["timestamp"] + pd.to_timedelta(horizon, unit="s")
        merged = pd.DataFrame({"future_timestamp": target}).merge(lookup, on="future_timestamp", how="left")
        future_mid = merged["future_mid_price"]
        log_ret = np.log(future_mid / base["mid_price"])
        base[f"future_return_{horizon}s"] = log_ret
        base[f"future_return_bps_{horizon}s"] = 10000.0 * log_ret
        base[f"future_mid_change_{horizon}s"] = future_mid - base["mid_price"]
        base[f"future_direction_{horizon}s"] = np.sign(base[f"future_mid_change_{horizon}s"]).fillna(0.0)
        base[f"future_absolute_move_{horizon}s"] = base[f"future_mid_change_{horizon}s"].abs()
    return base


def chronological_split_dates(timestamps: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(timestamps, utc=True).dt.date.unique()).sort_values().reset_index(drop=True)
    n = len(dates)
    train_end = max(int(n * 0.60), 1)
    val_end = max(int(n * 0.80), train_end + 1)
    rows = [
        ("train", dates.iloc[0], dates.iloc[train_end - 1]),
        ("validation", dates.iloc[train_end], dates.iloc[val_end - 1]),
        ("test", dates.iloc[val_end], dates.iloc[-1]),
    ]
    return pd.DataFrame(rows, columns=["split", "start_date_utc", "end_date_utc"])
