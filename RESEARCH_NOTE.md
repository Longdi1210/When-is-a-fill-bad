# Research Note

## 1. Research Question

Can passive-order states with higher fill likelihood produce worse post-fill execution quality?

The project separates two objects that are often conflated:

- the probability or timing of passive execution;
- the signed markout after the order is filled.

## 2. Controlled Environment And Data Provenance

The current analysis uses a controlled synthetic event stream. No real exchange data are processed.

| Item | Value |
|---|---|
| Data source | synthetic_validation |
| Exchange | none |
| Instrument | simulated |
| Events | 3500 |
| Hypothetical passive orders | 6428 |
| Fill rate | 0.5708 |

The synthetic environment is retained as a validation sandbox for replay logic, label construction, side conventions, leakage checks, and model comparison.

## 3. Passive-Order Replay

At eligible event times, the replay submits two hypothetical passive orders:

- buy order at the current best bid;
- sell order at the current best ask.

The queue-ahead estimate is a capped displayed-depth proxy. Fills are inferred when trade and cancellation depletion clear queue ahead plus the fixed order size. Cancellations can advance queue position but are not treated as fills by themselves. No fill within the horizon is censored.

The current simulator does not reconstruct exact exchange queue position.

## 4. Fill And Markout Definitions

For a buy fill:

```text
markout = (future_mid - fill_price) / tick_size
```

For a sell fill, the sign is reversed.

Positive signed markout is favorable to the passive trader. Negative signed markout is adverse.

Markout horizons:

```text
[10, 50, 100, 500]
```

## 5. Experimental Design

Features are computed at submission time. Future fills, future queue movement, and future markouts are labels only.

The split is chronological:

| Split | Start step | End step | Observations |
|---|---:|---:|---:|
| train | 1 | 2182 | 3856 |
| validation | 2183 | 2825 | 1286 |
| test | 2826 | 3500 | 1286 |

The audit checks pass for timestamp separation, duplicate observations, target leakage, markout timing, nonnegative queue ahead, and positive spread.

## 6. Baseline Results

The fill-score bins show that execution likelihood and execution quality can rank states differently.

| Fill-score bin | Realized fill rate | Signed markout conditional on fill |
|---:|---:|---:|
| 1 | 0.4651 | -10.1165 |
| 5 | 0.7829 | -0.7192 |
| 8 | 0.5039 | +0.7524 |
| 10 | 0.8140 | -1.4665 |

The strongest supported result is not that high fill probability is always toxic. It is that fill likelihood alone is an incomplete objective: states with similar or higher fill likelihood can have materially different post-fill markout.

## 7. Flow And Queue Mechanism Test

The mechanism test uses:

- local signed-flow persistence;
- passive-side trade depletion;
- passive-side cancellation depletion;
- a flow-depletion interaction.

Nested fill models:

- M0: controls only;
- M1: controls + flow persistence;
- M2: controls + flow persistence + depletion;
- M3: controls + flow persistence + depletion + interaction.

The key comparison is M3-M2.

## 8. Chronological Robustness

In the current run, the M3 interaction does not deliver stable incremental fill-prediction value.

At W=50:

| Metric | Value |
|---|---:|
| M3 fill ROC AUC | 0.5261 |
| M3-M2 AUC | -0.00087 |

Across windows, M3-M2 is mostly near zero or negative. This is a weak mechanism result.

## 9. Local Shuffled Null

The local shuffled null disrupts local trade-sign ordering while preserving block-level structure. The null diagnostic does not support a strong positive interaction claim. Some windows differ from the shuffled sequence, but the effect is not stable enough to claim a robust mechanism.

## 10. Supported Findings

- Passive-order replay is reproducible.
- Fill and markout labels are consistently side-adjusted.
- Chronological evaluation is implemented.
- Fill likelihood and post-fill execution quality are measured jointly.
- Fill-score bins show that execution likelihood and execution quality can diverge.

## 11. Unsupported Or Unstable Findings

- The flow-depletion interaction does not improve fill prediction robustly.
- No clear finite event-history scale is identified.
- The shuffled-null comparison is not a strong positive mechanism result.
- The current outputs are not empirical Bitcoin evidence.
- The prototype does not imply a trading strategy.

## 12. Limitations

- Synthetic data only.
- Queue ahead is a proxy.
- Hidden liquidity is absent.
- Latency is simplified.
- Partial-fill treatment is simplified.
- Event generation is simplified.
- No exchange-specific microstructure.
- No live execution claim.

## 13. Real-Data Extension

The next empirical step is narrow: replace the synthetic event stream with a single-venue BTC-USD L2 or L3 event stream and rerun the same labeling and evaluation pipeline.

