# Real BTC Validation

## 1. Why real-data validation is needed

The synthetic layer tests exact hypothetical fills and post-fill markouts under controlled replay assumptions. The real Coinbase BTC layer checks whether analogous quote-consumption conditions appear in observed market states.

## 2. Data and evidence boundary

- Dataset: `data/processed/kaggle_btc_canonical.parquet`
- Rows: 1,030,728
- Date range: 2021-04-07 11:32:42.122161+00:00 to 2021-04-19 09:54:22.386544+00:00
- Frequency: one-second sampled market-state and interval-aggregate data
- Visible levels: 15
- Exact real passive fills: not observed
- FIFO queue position: not reconstructable

## 3. Passive-side execution-pressure proxy

For a passive buy at the bid, pressure uses aggressive sell activity, bid-side cancellations, bid-side limit replenishment, and bid depth. For a passive sell at the ask, the same construction uses aggressive buy activity, ask-side cancellations, ask-side limit replenishment, and ask depth.

P1 is market pressure only. P2 adds same-side cancellation. P3 adds cancellation and subtracts same-side replenishment. The main proxy is P3 divided by visible passive-side depth.

## 4. Future markout definition

Future markout is side-adjusted mid-price response. Positive is favorable to the passive quote; negative is adverse. Buy-side markout is future mid return; sell-side markout reverses the sign.

## 5. Chronological experimental design

| split | start_date_utc | end_date_utc |
|---|---|---|
| train | 2021-04-07 | 2021-04-13 |
| validation | 2021-04-14 | 2021-04-16 |
| test | 2021-04-17 | 2021-04-19 |

Formation windows: [1, 2, 5, 10, 30, 60]

Response horizons: [1, 5, 10, 30, 60, 300]

## 6. Main pressure-markout result

The validation-selected display scale is W=10s and H=60s. On the untouched test period, the average high-minus-low P3 depth-normalized pressure contrast is -0.8872 bps. Negative values mean higher execution pressure is associated with worse passive-side future markout.

## 7. Cancellation and replenishment contribution

Mean test incremental R2, M2-M1 cancellation contribution: 0.000102.

Mean test incremental R2, M3-M2 replenishment contribution: 0.000124.

Mean test incremental R2, M4-M0 full-proxy contribution: -0.000091.

## 8. Time-scale result

The formation-window x response-horizon map is saved as `outputs/figures/real_btc_main/04_formation_response_map.png`. It should be interpreted as a proxy response map, not an exact fill map.

## 9. Null test

The local 5-minute block shuffle preserves broad local regimes while disrupting pressure/markout alignment. The mean real-minus-shuffled high-minus-low contrast is 0.1756 bps.

## 10. Daily stability

Daily stability is stored in `outputs/tables/main/real_btc_daily_stability.csv`. The table reports high-minus-low markout, rank correlation, a simple daily coefficient, and count by date and side.

## 11. Comparison with synthetic exact-fill experiment

See `outputs/tables/main/synthetic_real_comparison.csv`. The synthetic layer tests exact-fill mechanics; the real layer tests aggregated execution-pressure conditions.

## 12. Why does the proxy change sign?

At W=10s and H=60s, the mean test high-minus-low contrast changes from 1.2222 bps for P2 to -1.9727 bps for raw P3 and -0.8872 bps for depth-normalized P3. This shows that subtracting same-side replenishment and then sorting by the depth-normalized proxy materially changes which states are classified as high pressure.

The component audit explains the sign reversal. Market flow alone has a mean contrast of -0.2347 bps and cancellation alone has a favorable mean contrast of 1.7624 bps, so P2 is positive. The negative-replenishment component has a mean contrast of -2.0962 bps, which pulls raw P3 negative. The inverse-depth denominator contrast is -1.6488 bps; low displayed depth contributes to extreme pressure values, but depth normalization attenuates rather than creates the raw P3 adverse contrast.

The adverse P3 result is concentrated more on passive buys (-1.6516 bps) than passive sells (-0.1228 bps). Leave-one-day-out stability also shows day dependence: when 2021-04-18 is removed, the average P3 contrast is 0.2366 bps. The mechanism audit therefore supports a regime-dependent proxy result rather than a uniform execution-pressure law.

The detailed mechanism table is `outputs/tables/main/real_btc_mechanism_audit.csv`. It separates market flow, cancellation, replenishment, raw P2, raw P3, depth-normalized P3, and inverse-depth denominator effects by side, test day, depth tercile, volatility tercile, and spread regime.

## 13. Supported findings

- Real Coinbase BTC data supports side-adjusted post-quote markout labels.
- Market, cancellation, limit replenishment, and visible depth are available as one-second aggregate proxies.
- The repository now links exact synthetic fills with real execution-pressure validation without claiming exact real fills.
- Higher pressure states are associated with more adverse passive-side markout in selected side/window/horizon regimes.

## 14. Partially supported findings

- Cancellation and replenishment add small average incremental R2 beyond market pressure, but the effect size is modest.
- The full depth-normalized proxy provides useful ordering in selected bins, but it does not consistently outperform controls across the full model grid.
- Daily stability is mixed; the selected pressure-markout contrast changes magnitude and sign across test days and sides.

## 15. Unsupported findings

- Exact passive fill probability is not measured in the real BTC layer.
- Exact FIFO queue position is not reconstructable.
- Any cancellation/replenishment contribution must be read through the one-second aggregate data boundary.
- The local shuffled null does not support a strong claim that the full proxy captures a stable sequence effect across all scales.

## 16. Limitations

The dataset is one-second aggregated, not order-level MBO. Hidden liquidity, latency, partial fills, and queue priority are not observed. The real-data result is a mechanism validation of post-quote price response, not a deployable trading claim.

Runtime: 327.92 seconds.
