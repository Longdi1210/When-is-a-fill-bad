# Research Note

## 1. Research Question

Can passive-side states associated with easier execution also produce worse post-fill or post-quote price outcomes?

The project treats this as an execution-quality question, not as a generic return-forecasting task. Execution likelihood and execution quality are different objects: a quote can become easier to hit precisely when visible liquidity is being consumed or withdrawn.

## 2. Evidence Design

The repository has two evidence layers.

The synthetic layer provides exact hypothetical passive fills, time-to-fill, censoring, and signed post-fill markout. It validates the label design and side convention in a controlled setting.

The real Coinbase BTC layer does not contain exact FIFO fills. It therefore uses execution-pressure and shock-response proxies: large market-order pressure against the passive side, potential penetration through displayed depth, post-shock replenishment or cancellation, best-quote survival, and future side-adjusted markout.

## 3. Data Provenance

| Item | Value |
|---|---:|
| Dataset | Coinbase BTC one-second LOB/order-flow table |
| Rows | 1,030,728 |
| Date range | 2021-04-07 11:32:42 UTC to 2021-04-19 09:54:22 UTC |
| Visible depth | 15 levels |
| Activity fields | market, limit, cancel notional |
| Data type | mixed snapshot + one-second interval aggregates |

Unsupported by this dataset: exact FIFO queue position, exact passive fills, hidden liquidity, partial fills, and intrasecond event ordering.

## 4. Side-Adjusted Markout

For both synthetic fills and real post-quote outcomes, positive markout is favorable to the passive trader.

```text
buy markout  =  10^4 log(mid_{t+H} / mid_t)
sell markout = -10^4 log(mid_{t+H} / mid_t)
```

Future labels use timestamp-aware joins with tolerance checks rather than row shifts across possible time gaps.

## 5. Static Proxy Baseline

The first real-data validation used static execution-pressure proxies:

```text
P1 = market pressure / visible passive depth
P2 = (market pressure + same-side cancellation) / visible passive depth
P3 = (market pressure + same-side cancellation - same-side replenishment) / visible passive depth
```

At `W=10s, H=60s` on the chronological test set:

| Proxy | High-minus-low markout |
|---|---:|
| P1 market-only | -0.2347 bps |
| P2 market + cancel | +1.2222 bps |
| P3 full depth-normalized | -0.8872 bps |

The static result is weak. Cancellation and replenishment do not add stable out-of-sample value, and the local shuffled null is close to the real sequence. This is retained as a useful falsification: a plausible composite pressure score is not automatically a better execution-quality signal.

## 6. Dynamic Shock-Absorption Analysis

The dynamic layer separates the sequence into:

```text
shock -> potential visible-depth penetration -> absorption -> quote survival -> markout response
```

Shock episodes are selected using train-period 95th percentile thresholds of 10-second market pressure relative to lagged top-5 displayed depth. The same thresholds are then applied chronologically.

| Metric | Value |
|---|---:|
| Shock episodes | 22,372 |
| Train episodes | 10,745 |
| Validation episodes | 6,747 |
| Test episodes | 4,880 |
| Formation window | 10 seconds |
| Response horizons | 1, 2, 5, 10, 30, 60, 120, 300 seconds |

Absorption combines post-shock net replenishment, top-5 depth recovery, and best-quote survival. This is not a pre-trade alpha signal; it is a state-response diagnostic after a quote-pressure shock.

## 7. Dynamic Versus Static Result

On the untouched test period, multi-level shock absorption produces a much clearer adverse state ordering than the static baselines:

| Side | Representation | High-minus-low 60s markout | Rank correlation |
|---|---|---:|---:|
| buy | market-only static | -2.4431 bps | -0.0244 |
| buy | static P3 proxy | -1.7760 bps | -0.0148 |
| buy | top-level dynamic | -1.3574 bps | +0.0005 |
| buy | multi-level shock absorption | -10.3845 bps | -0.2416 |
| sell | market-only static | -1.7864 bps | -0.0406 |
| sell | static P3 proxy | -1.5306 bps | +0.0056 |
| sell | top-level dynamic | -0.0308 bps | -0.0298 |
| sell | multi-level shock absorption | -11.2888 bps | -0.2915 |

The important change is not model complexity. It is the representation: large pressure by itself is less informative than pressure conditioned on visible-depth penetration and subsequent absorption failure.

## 8. Absorption States

At the 60-second response horizon:

| Side | Absorption state | Count | Mean markout | Quote survival |
|---|---|---:|---:|---:|
| buy | weak absorption | 3,293 | -6.7425 bps | 0.1971 |
| buy | partial absorption | 3,822 | -0.8327 bps | 0.4611 |
| buy | strong absorption | 3,899 | +3.1792 bps | 0.6516 |
| sell | weak absorption | 4,139 | -6.3097 bps | 0.1858 |
| sell | partial absorption | 3,720 | -0.3576 bps | 0.4739 |
| sell | strong absorption | 3,499 | +3.4253 bps | 0.6424 |

Weak absorption corresponds to poor passive-side markout and low quote survival on both sides. Strong absorption corresponds to favorable markout and higher quote survival.

## 9. Local Projection Boundary

Linear projections using shock ratio, absorption score, interaction, spread, depth, and recent volatility have small R2 even when rank ordering is meaningful.

At 60 seconds:

| Side | R2 | MAE | Rank correlation |
|---|---:|---:|---:|
| buy | 0.0024 | 12.2267 bps | 0.3288 |
| sell | 0.0035 | 11.8599 bps | 0.3428 |

This is a useful boundary. The dynamic representation orders states; it does not create a high-accuracy predictive model.

## 10. Stratified Null

The local null preserves date, side, and potential penetration class while shuffling markout alignment. This disrupts the link between absorption path and future markout without globally reshuffling the market.

| Side | Representation | Real high-minus-low | Null mean high-minus-low |
|---|---|---:|---:|
| buy | multi-level shock absorption | -10.3845 bps | -1.0732 bps |
| sell | multi-level shock absorption | -11.2888 bps | -3.1358 bps |

The null comparison supports the value of the dynamic absorption state, while the short sample and small linear R2 prevent a stronger live-trading claim.

## 11. Supported Findings

- The repository implements reproducible synthetic exact-fill labels and real BTC post-quote validation.
- Static market/cancel/replenishment proxies are insufficient as a standalone mechanism.
- Dynamic decomposition into shock, penetration, absorption, and quote survival produces clearer adverse markout ordering.
- Weak absorption states have negative markout and low quote survival on both passive sides.
- The dynamic state ordering is stronger than the stratified local null average.

## 12. Unsupported Or Bounded Findings

- Exact real passive fill probability is not observed.
- The static P3 proxy does not robustly outperform market-only pressure.
- Linear models have small R2, so the result should not be presented as a high-accuracy return forecast.
- One venue and one short sample do not establish a universal market law.
- No trading profitability, optimal execution, or market-making claim is made.

## 13. Reproduction

```bash
make dynamic-lob
python -m pytest tests
```

Earlier layers:

```bash
make real-btc-validation
make reproduce
```

## 14. Next Empirical Step

The natural extension is single-venue L3/MBO data with order identifiers, trades, cancellations, and precise queue updates. That would allow exact passive fill reconstruction instead of the current one-second quote-pressure proxy.
