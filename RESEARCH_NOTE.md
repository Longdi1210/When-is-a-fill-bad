# Research Note

## 1. Research Question

Can market states associated with easier passive execution also produce worse execution quality?

## 2. Why Fill Likelihood And Execution Quality Differ

Fill likelihood asks whether a passive quote executes. Execution quality asks what happens after execution. A passive order can be easier to fill precisely when liquidity is being consumed or withdrawn, which may make the subsequent price response adverse.

## 3. Two-Layer Evidence Design

The project has two layers:

- synthetic controlled data: exact hypothetical fills, fill likelihood, post-fill signed markout;
- real Coinbase BTC data: passive-side execution-pressure proxy, future side-adjusted post-quote markout.

The real-data layer validates mechanism conditions. It is not an exact real-fill study.

## 4. Synthetic Exact-Fill Experiment

The synthetic experiment submits passive buy and sell orders at the best bid and best ask. It constructs fill labels, time-to-fill, censoring, and signed post-fill markouts. It remains the exact-fill validation layer.

The synthetic result supports the core distinction between fill likelihood and markout, but the richer flow-depletion interaction is weak and unstable.

## 5. Real BTC Data And Provenance

| Item | Value |
|---|---:|
| Dataset | Coinbase BTC one-second LOB/order-flow table |
| Rows | 1,030,728 |
| Date range | 2021-04-07 11:32:42 UTC to 2021-04-19 09:54:22 UTC |
| Visible depth | 15 levels |
| Activity fields | market, limit, cancel notional |
| Data type | mixed snapshot + one-second interval aggregates |

Unsupported: exact FIFO queue position, exact passive fills, hidden liquidity, and intrasecond event ordering.

## 6. Passive-Side Execution-Pressure Proxies

Each valid timestamp creates two observations: passive buy at the best bid and passive sell at the best ask.

For passive buys, pressure uses market sells, bid cancellation, bid limit replenishment, and visible bid depth. For passive sells, pressure uses market buys, ask cancellation, ask limit replenishment, and visible ask depth.

```text
P1 = market pressure / visible passive depth
P2 = (market pressure + same-side cancellation) / visible passive depth
P3 = (market pressure + same-side cancellation - same-side replenishment) / visible passive depth
```

## 7. Side-Adjusted Markout

Markout is measured in basis points. Positive is favorable for the passive trader; negative is adverse.

```text
buy markout  =  10^4 log(mid_{t+H} / mid_t)
sell markout = -10^4 log(mid_{t+H} / mid_t)
```

Future labels use timestamp-aware joins rather than row offsets.

## 8. Chronological Split

| Split | Dates |
|---|---|
| train | 2021-04-07 to 2021-04-13 |
| validation | 2021-04-14 to 2021-04-16 |
| test | 2021-04-17 to 2021-04-19 |

Formation windows: `[1, 2, 5, 10, 30, 60]` seconds.

Markout horizons: `[1, 5, 10, 30, 60, 300]` seconds.

The display configuration selected from train/validation is `W=10s, H=60s`.

## 9. Main Result

At `W=10s, H=60s` on the untouched test set:

| Proxy | High-minus-low markout |
|---|---:|
| P1 market-only | -0.2347 bps |
| P2 market + cancel | +1.2222 bps |
| P3 full depth-normalized | -0.8872 bps |

Execution pressure shows adverse ordering in selected short/intermediate-horizon configurations, but the richer proxy does not robustly beat the simpler market-only signal.

## 10. Market-Only Versus Richer Proxies

At `W=10s, H=60s`:

| Quantity | Value |
|---|---:|
| Cancellation increment R2 | +0.000160 |
| Replenishment increment R2 | -0.000268 |
| Full proxy vs controls R2 | -0.000649 |
| Market-only rank correlation | +0.089046 |
| Full-proxy rank correlation | +0.085100 |

P2 changes sign relative to P3, and P3 remains adverse. However, the richer proxy does not improve the model metrics.

## 11. Formation-Window By Response-Horizon Result

P3 test high-minus-low markout:

| H \\ W | 1 | 2 | 5 | 10 | 30 | 60 |
|---:|---:|---:|---:|---:|---:|---:|
| 1s | -0.1681 | -0.1417 | -0.0964 | -0.0803 | -0.0422 | -0.0472 |
| 5s | -0.2615 | -0.2179 | -0.1763 | -0.1718 | -0.1336 | -0.1731 |
| 10s | -0.2870 | -0.2558 | -0.2489 | -0.2659 | -0.2314 | -0.3360 |
| 30s | -0.4948 | -0.4398 | -0.4928 | -0.5880 | -0.5909 | -1.0367 |
| 60s | -0.6576 | -0.6016 | -0.6878 | -0.8872 | -1.1081 | -2.1007 |
| 300s | -0.3484 | -0.2178 | -0.2614 | +0.0503 | +1.2142 | +2.0954 |

## 12. 300-Second Dissipation And Reversal

The adverse ordering is strongest at short and intermediate horizons. It dissipates and reverses at 300 seconds for longer formation windows. This is a horizon-dependent result; it should not be labeled mean reversion without a separate test.

## 13. Daily Stability And 2021-04-18 Concentration

| Date | Side | High-minus-low markout |
|---|---|---:|
| 2021-04-17 | buy | -0.0351 |
| 2021-04-18 | buy | -5.5869 |
| 2021-04-19 | buy | +0.0834 |
| 2021-04-17 | sell | +0.7801 |
| 2021-04-18 | sell | -1.2803 |
| 2021-04-19 | sell | -0.5667 |

The aggregate buy-side result is materially influenced by 2021-04-18.

## 14. Local Shuffled-Null Result

| Null statistic | Value |
|---|---:|
| Mean real-minus-shuffle | +0.1756 bps |
| Median real-minus-shuffle | -0.0542 bps |
| Share real more adverse | 56.94% |
| Mean real | -0.2914 bps |
| Mean shuffled | -0.4669 bps |

The local null does not support a strong sequence-specific mechanism.

## 15. Supported Findings

- The real BTC pipeline is complete and reproducible.
- Side-adjusted post-quote markouts are timestamp-aware.
- Short/intermediate adverse ordering appears in selected configurations.
- Market-order pressure gives the clearest simple ordering.

## 16. Partially Supported Findings

- Full execution pressure carries information in selected regimes.
- The effect is stronger for specific days and horizons.
- Cancellation and replenishment help diagnose regimes, but not as stable performance improvements.

## 17. Unsupported Findings

- The full proxy robustly beats market-only.
- Cancellation and replenishment add stable incremental value.
- The local shuffled null confirms a local sequence mechanism.
- The effect is stable across all days and horizons.
- Real data provide exact passive fill probabilities.

## 18. Limitations

- One-second aggregation removes intrasecond ordering.
- Exact FIFO fills are unavailable.
- The sample covers one venue and a short date range.
- Hidden liquidity, fees, latency, and partial fills are not modeled.
- No live trading or profitability claim is made.

## 19. What Exact L3 Data Would Be Required Next

The next empirical step would require single-venue order-level L3/MBO data with order identifiers, trades, cancellations, queue updates, and sufficiently precise timestamps. That would allow exact passive fill analysis rather than proxy validation.
