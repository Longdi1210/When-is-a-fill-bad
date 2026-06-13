# When Is a Fill Bad?

## Research Question

Can pre-fill limit-order-book states explain when a passive fill preserves spread capture and when it suffers post-fill adverse drift?

## Motivation

A passive limit-order fill is economically ambiguous. It may represent successful liquidity provision, but it may also indicate adverse selection if the mid-price moves against the filled order shortly after execution.

## Method

I build a simplified noisy limit-order-book simulator to study passive buy orders. The simulator tracks queue position, order imbalance, volatility, fill events, post-fill drift, and net execution value.

## Metrics

- Fill probability

- Spread capture

- Post-fill drift

- Adverse drift

- Net execution value

- Bad fill ratio

## Results

1. Queue position controls fill probability.

2. Negative pre-fill imbalance leads to worse post-fill drift for passive buy orders.

3. Volatility expands bad-fill risk, especially under adverse imbalance regimes.

4. A simple imbalance-based passive filter reduces bad-fill ratio and improves adverse-selection-adjusted execution quality, at the cost of lower fill rate.

## Limitations

This is a controlled mechanism study, rather than a production trading model. The simulator uses simplified queue dynamics, symmetric volatility shocks, no hidden liquidity, no venue-specific matching rules, no latency model, and no fee/rebate structure.

## Next Steps

- Add queue-aware fill calibration.

- Add real LOB data extension.

- Estimate conditional adverse drift from data.

- Replace rule-based filter with a data-calibrated execution filter.