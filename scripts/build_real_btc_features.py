from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.dataset as ds

from fillbad.real_btc import (
    AUDIT_FIGURE_DIR,
    CANONICAL_PARQUET,
    PARQUET_DIR,
    canonical_columns,
    add_future_return_labels,
    build_canonical_table,
    chronological_split_dates,
)


HORIZONS_SECONDS = [1, 5, 10, 30, 60, 300]


def load_needed_columns() -> pd.DataFrame:
    dataset = ds.dataset(PARQUET_DIR, format="parquet", partitioning="hive", exclude_invalid_files=True)
    table = dataset.to_table(columns=canonical_columns())
    return table.to_pandas()


def save_audit_figures(canonical: pd.DataFrame) -> None:
    AUDIT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sample = canonical.iloc[:: max(len(canonical) // 20_000, 1)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sample["timestamp"], sample["mid_price"], linewidth=0.8)
    ax.set_title("Coinbase BTC Mid-Price Overview")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Mid-price (USD)")
    fig.tight_layout()
    fig.savefig(AUDIT_FIGURE_DIR / "01_mid_price_time_series.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    canonical["spread"].clip(upper=canonical["spread"].quantile(0.99)).hist(ax=ax, bins=80)
    ax.set_title("Spread Distribution")
    ax.set_xlabel("Spread (USD, clipped at 99th percentile)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(AUDIT_FIGURE_DIR / "02_spread_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    depth = (canonical["bid_depth_top_15"] + canonical["ask_depth_top_15"]).clip(
        upper=(canonical["bid_depth_top_15"] + canonical["ask_depth_top_15"]).quantile(0.99)
    )
    depth.hist(ax=ax, bins=80)
    ax.set_title("Visible Top-15 Depth Distribution")
    ax.set_xlabel("Visible notional depth (USD, clipped at 99th percentile)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(AUDIT_FIGURE_DIR / "03_visible_depth_distribution.png", dpi=160)
    plt.close(fig)

    hourly = canonical.set_index("timestamp")[
        ["net_market_pressure", "net_cancellation_pressure", "net_replenishment"]
    ].resample("1h").sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly.plot(ax=ax, linewidth=0.8)
    ax.set_title("Hourly Order-Flow, Cancellation, and Replenishment Activity")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Net notional activity")
    fig.tight_layout()
    fig.savefig(AUDIT_FIGURE_DIR / "04_activity_by_time.png", dpi=160)
    plt.close(fig)


def main() -> None:
    start = time.perf_counter()
    raw = load_needed_columns()
    canonical = build_canonical_table(raw)
    canonical = add_future_return_labels(canonical, HORIZONS_SECONDS)
    canonical = canonical.sort_values("timestamp").reset_index(drop=True)
    CANONICAL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_parquet(CANONICAL_PARQUET, compression="zstd", index=False)
    split = chronological_split_dates(canonical["timestamp"])
    split_path = "outputs/tables/main/real_btc_splits.csv"
    pd.DataFrame(split).to_csv(split_path, index=False)
    save_audit_figures(canonical)
    report = {
        "canonical_feature_table": str(CANONICAL_PARQUET),
        "shape": [int(canonical.shape[0]), int(canonical.shape[1])],
        "start_datetime_utc": canonical["timestamp"].min().isoformat(),
        "end_datetime_utc": canonical["timestamp"].max().isoformat(),
        "horizons_seconds": HORIZONS_SECONDS,
        "split_table": split_path,
        "runtime_seconds": time.perf_counter() - start,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
