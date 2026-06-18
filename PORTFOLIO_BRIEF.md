# Portfolio Brief

## Question

Can passive-side states associated with easier execution also produce worse post-fill or post-quote price outcomes?

## Research Positioning

This is a controlled execution-quality research project. The synthetic layer validates exact passive-fill labels. The real Coinbase BTC layer uses one-second quote-pressure episodes as a proxy because exact FIFO fills are not available.

## Identification Upgrade

An initial dynamic result showed large separation, but the time windows overlapped. The final version hardens the design:

```text
shock formation [t-10s, t]
early absorption observation (t, t+5s]
future outcome (t+5s, t+5s+H]
```

The absorption score excludes quote survival and future markout. This turns the project from a descriptive LOB episode study into a stricter predictive microstructure audit.

## Evidence

On 22,300 strict shock episodes, early absorption separates future passive-side outcomes at short horizons:

| Side | 10s strong - weak markout | 30s strong - weak markout | 60s strong - weak markout |
|---|---:|---:|---:|
| buy | +0.8446 bps | +1.5725 bps | +1.7922 bps |
| sell | +0.6761 bps | +1.6727 bps | +0.4369 bps |

The 10s and 30s effects have positive block-bootstrap confidence intervals on both sides. The 60s result is weaker, and the expanded 200-seed stratified null is directionally supportive but not decisive.

## Technical Signal

- High-frequency data engineering and Parquet-based processing.
- Side-adjusted markout conventions.
- Temporal leakage audit.
- Non-overlapping label design.
- Train-only thresholding and scaling.
- Block-bootstrap uncertainty.
- Stratified local null with 200 seeds.
- Clear separation of supported, partial, and unsupported claims.

## Evidence Boundary

The real BTC dataset supports quote-pressure validation, not exact FIFO queue reconstruction. The project makes no live trading, profitability, optimal-execution, or production-system claim.

## Next Empirical Step

Run the same strict labeling logic on single-venue L3/MBO data with order identifiers and exact queue updates.
