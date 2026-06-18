# Portfolio Brief

## Question

Can easier passive execution coincide with worse execution quality?

## Why It Matters

Passive orders are often ranked by execution likelihood, but a fast fill can be adverse if the quote is being consumed as the market moves through it. The project studies the gap between getting filled and getting a good fill.

## What I Built

- Controlled synthetic LOB experiment with exact hypothetical fills.
- Side-adjusted post-fill markout labels.
- Real Coinbase BTC validation using 1,030,728 one-second LOB observations.
- Passive-side execution-pressure proxies for buy and sell quotes.
- Timestamp-aware future markout labels.
- Chronological train/validation/test design.
- Nested model comparisons and pressure quantile tests.
- Depth-conditioned, daily-stability, and local-null diagnostics.

## Evidence

The synthetic layer validates exact-fill mechanics. The real BTC layer shows partial empirical support: higher passive-side execution pressure is associated with worse future markout in selected regimes, especially passive-buy states on 2021-04-18. Cancellation and replenishment add small incremental information, but the full proxy is not uniformly stable across all days and horizons.

## Research Value

The project demonstrates research question design, data provenance control, label construction, leakage-aware validation, mechanism attribution, null testing, and honest reporting of mixed evidence.

## Boundary

The real BTC data are one-second aggregates. Exact FIFO queue position and exact real passive fills are not observed.

## Reproduce

```bash
make real-btc-validation
python -m pytest tests
```
