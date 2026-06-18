# Portfolio Brief

## Question

Can passive-side states associated with easier execution also produce worse post-fill or post-quote price outcomes?

## Relevance

Fill likelihood and execution quality are not the same object. A passive quote may be more likely to trade when market pressure is moving through visible liquidity, and that same state may create adverse post-fill or post-quote markout.

## Research Architecture

- Controlled synthetic exact-fill experiment for passive-order replay and signed markout labels.
- Coinbase BTC one-second validation layer with 1,030,728 observations and 15 visible book levels.
- Side-adjusted buy/sell markout convention where positive is favorable to the passive trader.
- Static pressure baselines using market flow, cancellation, replenishment, and displayed depth.
- Dynamic shock analysis: pressure shock, potential depth penetration, absorption, quote survival, and recovery path.
- Chronological validation, stratified local null, focused tests, and one-command reproduction.

## Result

The static proxy hypothesis is weak: adding cancellation and replenishment to market pressure does not reliably improve the real-data baseline. The stronger result comes from the dynamic representation. On the test period, multi-level shock absorption separates adverse post-quote states much more clearly than market-only or static P3 proxies.

| Side | Static market-only | Static P3 | Dynamic shock absorption |
|---|---:|---:|---:|
| buy | -2.4431 bps | -1.7760 bps | -10.3845 bps |
| sell | -1.7864 bps | -1.5306 bps | -11.2888 bps |

The result is framed carefully: dynamic state ordering is informative, while linear predictive R2 remains small and exact real fills are not observed.

## Technical Signal

- High-frequency data engineering and Parquet-based processing.
- Market-microstructure label design.
- Side-aware sign conventions.
- Timestamp-aware future response construction.
- Mechanism testing under chronological validation.
- Null-model design and negative-result discipline.
- Compact, reproducible research packaging.

## Evidence Boundary

The real BTC dataset supports quote-pressure and visible-depth validation, not exact FIFO queue reconstruction. The project makes no claim of live trading profitability or production execution readiness.

## Next Empirical Step

Run the same labeling and validation logic on single-venue L3/MBO data with order identifiers and exact queue updates.
