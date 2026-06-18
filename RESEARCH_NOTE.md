# Research Note

## 1. Research Question

Can market states associated with easier passive execution also produce worse execution quality?

The project separates execution likelihood from execution quality. A passive quote can become easier to execute when visible liquidity is being consumed or withdrawn, but the subsequent price response can still be adverse to the passive trader.

## 2. Why Fill Likelihood And Fill Quality Differ

Fill likelihood asks whether a passive order executes. Fill quality asks what happens after execution. A fast fill can be valuable when it captures spread without adverse price movement, or harmful when the quote is being traded through.

## 3. Two-Layer Evidence Design

The repository has two complementary layers:

- controlled synthetic exact-fill experiment;
- real Coinbase BTC execution-pressure validation.

The synthetic layer supports exact hypothetical fills and post-fill signed markouts. The real BTC layer does not contain exact passive fills, so it uses passive-side execution pressure as an empirical proxy for quote-consumption conditions.

## 4. Synthetic Exact-Fill Experiment

The synthetic experiment submits hypothetical passive buy and sell orders at the best bid and best ask. It estimates fill likelihood and post-fill signed markout under controlled replay assumptions.

Current synthetic results show that fill-score bins can have different realized fill rates and different signed markouts. The flow-depletion interaction remains weak and is not reported as a strong discovery.

## 5. Real BTC Data And Provenance

The real dataset is `data/processed/kaggle_btc_canonical.parquet`.

| Item | Value |
|---|---:|
| Rows | 1,030,728 |
| Canonical columns | 61 |
| Date range | 2021-04-07 11:32:42 UTC to 2021-04-19 09:54:22 UTC |
| Sampling | Approximately one second |
| Visible depth | 15 levels |
| Activity fields | market, limit, cancellation notional |

The data are mixed LOB snapshots plus one-second interval aggregates.

## 6. Evidence Boundary

The real BTC layer supports pressure and markout analysis, not exact execution reconstruction.

Unsupported in the real data:

- exact FIFO queue position;
- exact order-level passive fills;
- hidden liquidity;
- exact intrasecond event ordering.

## 7. Passive-Side Execution-Pressure Proxies

Each timestamp creates two passive-side observations:

- passive buy at the best bid;
- passive sell at the best ask.

For passive buys, aggressive pressure is market sells; same-side cancellation is bid cancellation; replenishment is bid limit notional; passive depth is visible bid depth.

For passive sells, aggressive pressure is market buys; same-side cancellation is ask cancellation; replenishment is ask limit notional; passive depth is visible ask depth.

The main proxy family is:

```text
P1 = market pressure / visible passive depth
P2 = (market pressure + same-side cancellation) / visible passive depth
P3 = (market pressure + same-side cancellation - same-side replenishment) / visible passive depth
```

All signs are from the passive trader's perspective. Positive future markout is favorable; negative future markout is adverse.

## 8. Chronological Experimental Design

| Split | Dates |
|---|---|
| train | 2021-04-07 to 2021-04-13 |
| validation | 2021-04-14 to 2021-04-16 |
| test | 2021-04-17 to 2021-04-19 |

Formation windows are `[1, 2, 5, 10, 30, 60]` seconds. Future markout horizons are `[1, 5, 10, 30, 60, 300]` seconds.

Pressure quantile boundaries and the display scale are selected without using the test set. The selected display scale is `W=10s, H=60s`.

## 9. Main Nonparametric Result

On the untouched test set, the full depth-normalized P3 proxy has an average high-minus-low markout contrast of `-0.8872 bps` at `W=10s, H=60s`. Negative means higher execution pressure is associated with worse passive-side future markout.

This result is directional but conditional rather than uniform.

## 10. Incremental Role Of Cancellation

The mean test incremental R2 from adding cancellation to market pressure is:

```text
M2 - M1 = +0.000102
```

The effect is positive but small. It should be interpreted as modest incremental information, not a large economic effect.

## 11. Incremental Role Of Replenishment

The mean test incremental R2 from adding replenishment is:

```text
M3 - M2 = +0.000124
```

The P2-to-P3 sign reversal is more informative than the average R2 increment. At `W=10s, H=60s`, P2 is positive while P3 is negative because subtracting same-side replenishment changes which states enter the high-pressure bucket.

## 12. Depth-Conditioned Mechanism

The pressure effect is regime-dependent. In the mechanism audit, passive-buy P3 is most adverse on 2021-04-18 and in high-volatility states. Depth normalization attenuates the raw P3 adverse contrast, so low displayed depth contributes to extreme values but does not by itself create the full effect.

## 13. Multi-Scale Response

The formation-window by response-horizon map is stored in `outputs/figures/real_btc_main/04_formation_response_map.png`. It shows a conditional response pattern rather than one stable universal time scale.

## 14. Daily Stability

The selected P3 result is day-dependent:

- buy side, 2021-04-18: strongly adverse;
- buy side, 2021-04-17 and 2021-04-19: weak or near neutral;
- sell side: mixed.

Removing 2021-04-18 changes the average P3 contrast to `+0.2366 bps`, so the main real-data effect is not stable across all test days.

## 15. Local-Null Test

The local 5-minute shuffled null preserves broad regimes while disrupting local alignment between pressure and future markout. The mean real-minus-shuffled high-minus-low contrast is `+0.1756 bps`, which does not support a strong sequence-effect claim across all scales.

## 16. Synthetic-Real Comparison

The synthetic layer tests exact-fill logic. The real layer tests whether related pressure states appear in real one-second Coinbase BTC data.

The two layers are not numerically comparable. They are evidence complements:

- synthetic: exact hypothetical fill and post-fill markout;
- real BTC: execution-pressure proxy and post-quote side-adjusted markout.

## 17. Supported Findings

- The project cleanly separates execution likelihood from execution quality.
- Synthetic exact-fill labels and markouts are reproducible.
- Real BTC passive-side pressure proxies can be built from market, cancel, limit, and depth fields.
- Higher pressure states show more adverse passive-side markout in selected regimes.

## 18. Partially Supported Findings

- Cancellation and replenishment add small incremental information beyond market flow.
- The P3 proxy helps in selected regimes, especially passive-buy / high-volatility / 2021-04-18 states.
- The time-scale result is conditional, not a single stable characteristic scale.

## 19. Unsupported Findings

- Real BTC exact passive fill probability is not observed.
- FIFO queue reconstruction is not supported by the one-second aggregate data.
- The full composite proxy does not dominate controls across every side, day, window, and horizon.
- The local-null result does not justify a strong causal sequence claim.

## 20. Limitations

- One-second aggregation hides intrasecond ordering.
- Hidden liquidity is unavailable.
- Real passive executions are not labeled.
- Fees, rebates, latency, and partial fills are not modeled in the real layer.
- The current result is a mechanism validation, not a trading strategy.

## 21. What Exact Real-Fill Data Would Be Required Next

The next empirical requirement is single-venue L3/MBO data with order identifiers, trades, cancellations, queue updates, and timestamps precise enough to reconstruct queue position. That would allow the real-data layer to move from execution-pressure proxy validation to exact passive-fill analysis.
