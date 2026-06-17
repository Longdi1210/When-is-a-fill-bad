# Research Sweep Summary

This summary aggregates 40 simulations: 4 scenarios x 10 random seeds.

## Scenario Means

| Scenario | Avg Spread | Avg Microprice Deviation | Avg OFI | Realized Volatility | Resilience Score |
|---|---:|---:|---:|---:|---:|
| calm_baseline | 1.2247 | 0.0206 | 0.0757 | 0.00000342 | 0.6642 |
| information_shock | 2.3220 | 0.1427 | 0.2468 | 0.00001650 | 0.5392 |
| resilient_market | 1.2882 | 0.0226 | 0.2550 | 0.00000547 | 0.6511 |
| fragile_liquidity | 3.0342 | 0.2742 | 0.2398 | 0.00002494 | 0.6616 |

## Interpretation

The information-shock scenario widens spreads, increases microprice deviation, and raises realized volatility. When resilience-style liquidity providers are more active, spreads and microprice deviation move back toward the calm baseline even under high informed-order-flow pressure.

This is the main career-market story: the project uses a physics-style perturbation experiment to study order-book stability, liquidity replenishment, and queue imbalance under stress.

