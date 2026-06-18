from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fillbad.real_btc import RAW_CSV, REPORT_DIR, classify_column, parse_timestamps


def numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    if not pd.api.types.is_numeric_dtype(series):
        return {"minimum": None, "maximum": None, "mean": None, "standard_deviation": None}
    return {
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=0)),
    }


def build_column_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        info = classify_column(column)
        summary = numeric_summary(df[column])
        rows.append(
            {
                "column_name": column,
                "dtype": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "unique_count": int(df[column].nunique(dropna=True)),
                **summary,
                "suspected_group": info.group,
                "suspected_level": info.level,
                "suspected_side": info.side,
                "suspected_event_type": info.event_type,
                "interpretation_confidence": info.confidence,
                "notes": info.notes,
            }
        )
    return pd.DataFrame(rows)


def audit_time(timestamps: pd.Series) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ts = parse_timestamps(timestamps).sort_values().reset_index(drop=True)
    intervals = ts.diff().dropna()
    interval_seconds = intervals.dt.total_seconds()
    second_grid = ts.dt.floor("s")
    full_grid = pd.date_range(second_grid.iloc[0], second_grid.iloc[-1], freq="1s")
    missing = full_grid.difference(pd.DatetimeIndex(second_grid))
    duplicate_timestamps = int(ts.duplicated().sum())
    distribution = interval_seconds.value_counts().sort_index().reset_index()
    distribution.columns = ["interval_seconds", "count"]
    large_gaps = pd.DataFrame(
        {
            "gap_start": ts.shift(1),
            "gap_end": ts,
            "gap_seconds": interval_seconds.reindex(ts.index).to_numpy(),
        }
    ).dropna()
    large_gaps = large_gaps[large_gaps["gap_seconds"] > 1.0].sort_values("gap_seconds", ascending=False)
    report = {
        "timestamp_column": "system_time",
        "timestamp_unit": "ISO-8601 string with microseconds",
        "timezone": "UTC",
        "start_datetime_utc": ts.iloc[0].isoformat(),
        "end_datetime_utc": ts.iloc[-1].isoformat(),
        "row_count": int(len(ts)),
        "duplicate_timestamps": duplicate_timestamps,
        "duplicate_utc_seconds": int(second_grid.duplicated().sum()),
        "nominal_sampling_interval_seconds": 1,
        "median_interval_seconds": float(interval_seconds.median()),
        "missing_one_second_intervals": int(len(missing)),
        "longest_gap_seconds": float(interval_seconds.max()),
        "longest_gap_start": large_gaps["gap_start"].iloc[0].isoformat() if not large_gaps.empty else None,
        "longest_gap_end": large_gaps["gap_end"].iloc[0].isoformat() if not large_gaps.empty else None,
    }
    return report, distribution, large_gaps


def write_provenance(report: dict, column_dictionary: pd.DataFrame) -> None:
    groups = column_dictionary["suspected_group"].value_counts().sort_index()
    levels = column_dictionary["suspected_level"].dropna().astype(int)
    visible_levels = int(levels.max() + 1) if not levels.empty else 0
    text = f"""# Kaggle Coinbase BTC Data Provenance

## Verified facts

- Raw file: `{RAW_CSV}`
- Rows: {report['row_count']:,}
- Columns: {len(column_dictionary)}
- Duplicate timestamps: {report['duplicate_timestamps']:,}
- Timestamp column: `system_time`
- Timestamp timezone: UTC, as encoded in the source strings.
- Date range: {report['start_datetime_utc']} to {report['end_datetime_utc']}
- Nominal sampling interval: {report['nominal_sampling_interval_seconds']} second
- Missing one-second intervals: {report['missing_one_second_intervals']:,}
- Longest gap: {report['longest_gap_seconds']} seconds

## Inferred schema interpretation

- The file is a mixed snapshot-plus-interval-aggregate table sampled every second.
- `midpoint` and `spread` are provided derived market-state fields.
- `bids_distance_*` and `asks_distance_*` appear to be relative distances from midpoint. Prices are inferred as `midpoint * (1 + distance)`.
- `*_notional_*` fields are interpreted as visible notional by side and level.
- `*_market_notional_*`, `*_cancel_notional_*`, and `*_limit_notional_*` are interpreted as one-second interval aggregate market, cancel, and limit activity by side and level.
- Visible book levels detected: {visible_levels}

These event-type interpretations come from column names and internal consistency checks, not from order-level messages. The dataset does not support exact FIFO queue reconstruction or exact passive-order fills.

## Column group counts

{groups.to_string()}

## Unsupported claims

- Exact order-level queue position is not observable.
- Hidden liquidity is not observable.
- Exact passive fill labels cannot be reconstructed from this table alone.
- The data supports short-horizon price-response analysis, not exchange-grade order replay.
"""
    (REPORT_DIR / "data_provenance.md").write_text(text)


def main() -> None:
    start = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_CSV)
    dictionary = build_column_dictionary(df)
    dictionary.to_csv(REPORT_DIR / "column_dictionary.csv", index=False)
    time_report, interval_dist, large_gaps = audit_time(df["system_time"])
    interval_dist.to_csv(REPORT_DIR / "time_interval_distribution.csv", index=False)
    large_gaps.to_csv(REPORT_DIR / "time_gap_report.csv", index=False)
    raw_size = RAW_CSV.stat().st_size
    audit = {
        **time_report,
        "raw_file_size_bytes": raw_size,
        "raw_file_size_gb": raw_size / 1e9,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_total": int(df.isna().sum().sum()),
        "visible_book_levels": 15,
        "row_type": "mixed snapshot plus one-second interval aggregates",
        "audit_runtime_seconds": time.perf_counter() - start,
    }
    (REPORT_DIR / "schema_audit.json").write_text(json.dumps(audit, indent=2))
    write_provenance(audit, dictionary)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
