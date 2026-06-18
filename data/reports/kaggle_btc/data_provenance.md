# Kaggle Coinbase BTC Data Provenance

## Verified facts

- Raw file: `data/raw/kaggle/coinbase_btc/BTC_1sec.csv`
- Rows: 1,030,728
- Columns: 156
- Duplicate timestamps: 0
- Timestamp column: `system_time`
- Timestamp timezone: UTC, as encoded in the source strings.
- Date range: 2021-04-07T11:32:42.122161+00:00 to 2021-04-19T09:54:22.386544+00:00
- Nominal sampling interval: 1 second
- Missing one-second intervals: 173
- Longest gap: 30.000617 seconds

## Inferred schema interpretation

- The file is a mixed snapshot-plus-interval-aggregate table sampled every second.
- `midpoint` and `spread` are provided derived market-state fields.
- `bids_distance_*` and `asks_distance_*` appear to be relative distances from midpoint. Prices are inferred as `midpoint * (1 + distance)`.
- `*_notional_*` fields are interpreted as visible notional by side and level.
- `*_market_notional_*`, `*_cancel_notional_*`, and `*_limit_notional_*` are interpreted as one-second interval aggregate market, cancel, and limit activity by side and level.
- Visible book levels detected: 15

These event-type interpretations come from column names and internal consistency checks, not from order-level messages. The dataset does not support exact FIFO queue reconstruction or exact passive-order fills.

## Column group counts

suspected_group
ask prices               15
ask sizes                15
bid prices               15
bid sizes                15
cancellation activity    30
derived_fields            3
limit-order activity     30
market-order activity    32
timestamp                 1

## Unsupported claims

- Exact order-level queue position is not observable.
- Hidden liquidity is not observable.
- Exact passive fill labels cannot be reconstructed from this table alone.
- The data supports short-horizon price-response analysis, not exchange-grade order replay.
