# Research Note

## 1. Research Question

Can passive-side states associated with easier execution also produce worse post-fill or post-quote price outcomes?

The project separates execution likelihood from execution quality. A passive quote can become easier to trade when visible liquidity is being consumed, but the economic quality of that execution depends on what happens after the quote is exposed.

## 2. Evidence Layers

The synthetic layer provides exact hypothetical passive fills, time-to-fill, censoring, and signed post-fill markout. It validates label construction and side conventions.

The real Coinbase BTC layer does not contain order IDs or exact FIFO fills. It therefore studies quote-pressure shocks and future post-quote outcomes:

```text
shock pressure -> visible-depth penetration -> early absorption -> later quote survival and markout
```

## 3. Data Provenance

| Item | Value |
|---|---:|
| Dataset | Coinbase BTC one-second LOB/order-flow table |
| Rows | 1,030,728 |
| Date range | 2021-04-07 11:32:42 UTC to 2021-04-19 09:54:22 UTC |
| Visible depth | 15 levels |
| Activity fields | market, limit, cancel notional |
| Data type | mixed snapshot + one-second interval aggregates |

Unsupported: exact FIFO queue position, exact passive fills, hidden liquidity, partial fills, and intrasecond event ordering.

## 4. Temporal Identification And Leakage Audit

The first dynamic analysis was descriptive: absorption, quote survival, and markout paths were measured over overlapping post-shock windows. That produced large separation, but the design allowed absorption variables to overlap with later outcomes.

The final analysis uses a strict three-stage design:

```text
[t-10s, t]       shock formation
(t, t+5s]        early absorption observation
(t+5s, t+H]      future outcome
```

Absorption uses only:

- limit additions during the first 5 seconds;
- cancellations during the first 5 seconds;
- top-5 depth recovery by `t+5s`;
- spread recovery by `t+5s`.

It excludes quote survival and future markout. Scaling and absorption thresholds are fit on the training period only.

The primary future markout baseline is the midpoint at `t+5s`. The pre-shock-baseline markout is retained only as a descriptive total episode path.

The machine-readable audit is saved at:

`outputs/tables/audit/temporal_identification_audit.csv`

## 5. Shock Episode Design

| Metric | Value |
|---|---:|
| Shock window | 10 seconds |
| Shock threshold | train-period 95th percentile |
| Absorption window | 5 seconds |
| Valid strict shock episodes | 22,300 |
| Train episodes | 10,700 |
| Validation episodes | 6,732 |
| Test episodes | 4,868 |
| Outcome total times after shock | 10, 30, 60, 120, 300 seconds |

The 5-second absorption window is pre-specified as the main window and is not selected on the test set.

## 6. Strict Strong-Versus-Weak Absorption Result

On the untouched test set:

| Side | Total time | Weak markout | Strong markout | Strong - weak | 95% block-bootstrap CI | Quote-survival diff |
|---|---:|---:|---:|---:|---:|---:|
| buy | 10s | -0.5022 | +0.3424 | +0.8446 | [+0.3975, +1.3342] | +7.63 pp |
| buy | 30s | -1.0854 | +0.4871 | +1.5725 | [+0.3329, +3.2433] | +3.89 pp |
| buy | 60s | -1.2677 | +0.5245 | +1.7922 | [-0.1318, +3.8304] | +2.83 pp |
| sell | 10s | -0.4214 | +0.2546 | +0.6761 | [+0.2724, +1.1462] | +6.47 pp |
| sell | 30s | -0.7154 | +0.9573 | +1.6727 | [+0.5720, +3.3205] | +4.79 pp |
| sell | 60s | -0.1629 | +0.2740 | +0.4369 | [-1.3973, +2.3350] | +0.62 pp |

The strict design supports short-horizon separation at 10s and 30s on both sides. The 60s effect is weaker, especially on the sell side.

## 7. Overlap Versus Non-Overlap Attenuation

At 60 seconds:

| Side | Design | Strong - weak markout |
|---|---|---:|
| buy | descriptive total path | +2.2641 bps |
| buy | strict future after absorption | +1.7922 bps |
| sell | descriptive total path | +1.1808 bps |
| sell | strict future after absorption | +0.4369 bps |

The strict design attenuates the descriptive separation. This is expected and is now the basis of the public claim.

## 8. Interaction-Sign Audit

The grouped strong-versus-weak paths are the primary result. Regression interaction coefficients are secondary because the interaction is conditional on standardized main effects and can be hard to read when shock intensity and absorption are correlated.

At 60 seconds:

| Side | Shock coef | Absorption coef | Shock x absorption coef | 95% CI |
|---|---:|---:|---:|---:|
| buy | +0.0776 | +0.4333 | -0.0257 | see `interaction_sign_audit.csv` |
| sell | -0.0678 | +0.0467 | +0.0075 | see `interaction_sign_audit.csv` |

The audit table documents sign convention, conditional interpretation, and grouped-effect consistency:

`outputs/tables/audit/interaction_sign_audit.csv`

## 9. Expanded Stratified Null

The null uses 200 seeds. It preserves date, side, shock-intensity bin, pre-shock depth bin, spread regime, volatility regime, and time-of-day block, while disrupting the alignment between early absorption and later outcomes.

At 60 seconds:

| Side | Real strong - weak | Null mean | Empirical p-value |
|---|---:|---:|---:|
| buy | +1.7922 bps | +0.3813 bps | 0.1045 |
| sell | +0.4369 bps | +0.1226 bps | 0.6517 |
| combined | +1.0881 bps | +0.1748 bps | 0.1542 |

The null is directionally supportive for the buy side but not decisive overall.

## 10. Concentration Diagnostics

At 60 seconds, the test-set group counts are:

| Side | State | Count | Effective non-overlap count | 2021-04-18 fraction | Mean | Median |
|---|---|---:|---:|---:|---:|---:|
| buy | weak | 722 | 595 | 57.76% | -1.2677 | -0.5022 |
| buy | strong | 899 | 752 | 52.95% | +0.5245 | +0.4329 |
| sell | weak | 848 | 710 | 56.60% | -0.1629 | -0.0184 |
| sell | strong | 674 | 575 | 48.07% | +0.2740 | +0.1402 |

The result is not produced by only a handful of observations, but 2021-04-18 contributes a large fraction of the test episodes and remains an important stress-regime limitation.

## 11. Supported Findings

- The project now has an explicit temporal leakage audit.
- Strict early absorption at `t+5s` separates later quote survival at 10s and 30s.
- Strict early absorption separates future markout at 10s and 30s on both passive sides.
- The original descriptive result attenuates but does not disappear under non-overlapping windows.
- The codebase provides reproducible tests, tables, figures, and audit outputs.

## 12. Partially Supported Findings

- The buy-side 60s result remains directionally positive but has a confidence interval crossing zero.
- The expanded null is directionally supportive for buy-side markout but not decisive.
- The static-vs-dynamic comparison remains useful, but the strict result is smaller than the earlier overlapping dynamic result.

## 13. Unsupported Findings

- The strict design does not support a strong 300s predictive claim.
- The expanded null does not establish a robust sequence-specific effect across both sides.
- Real data do not provide exact passive fill probabilities or FIFO execution labels.
- The result should not be presented as a deployable trading model.

## 14. Reproduction

```bash
make dynamic-lob
python -m pytest tests
```

Earlier layers:

```bash
make real-btc-validation
make reproduce
```

## 15. Next Empirical Step

The natural extension is single-venue L3/MBO data with order identifiers, trades, cancellations, and exact queue updates. That would allow exact passive fill reconstruction instead of the current one-second quote-pressure proxy.
