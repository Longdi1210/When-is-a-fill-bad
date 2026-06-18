# Portfolio Brief

## Problem

Can easy passive execution coincide with poor execution quality?

## Data

- 1,030,728 real Coinbase BTC observations.
- 15 LOB levels.
- Market, limit, and cancel notional.
- 2021-04-07 to 2021-04-19 UTC.

## What I Built

- Controlled synthetic exact-fill layer.
- Real execution-pressure proxy layer.
- Side-adjusted markouts.
- Chronological evaluation.
- Multi-scale heatmap.
- Daily stability analysis.
- Local shuffled null.
- Reproducible pipeline.

## Main Finding

Market-order pressure provides the most stable simple ordering of future markout. Adding cancellation and replenishment does not robustly improve out-of-sample performance. The adverse relation is strongest at short and intermediate horizons and is concentrated in selected days.

| Question | Result |
|---|---|
| Does execution pressure order future markout? | Partly, at short/intermediate horizons |
| Does full proxy beat market-only? | No |
| Do cancellation and replenishment add stable value? | No |
| Is the effect stable by day? | No, one day materially influences the buy side |
| Does the local null confirm sequence structure? | No |
| Is the pipeline reproducible? | Yes |

## Research Judgment

The project rejects a stronger composite-signal hypothesis instead of tuning until it appears successful.

## Visible Skills

- High-frequency data engineering.
- Time-aware feature construction.
- Leakage control.
- Model comparison.
- Null testing.
- Regime analysis.
- Reproducibility.
- Mechanism interpretation.

## Evidence Boundary

Real data validates quote-consumption conditions, not exact FIFO fills.
