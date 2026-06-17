# Portfolio Brief

## Question

Can easier passive fills also be worse fills?

## Why It Matters

Fill likelihood and execution quality are different objects. A passive order can become easier to fill because the queue ahead is being depleted, while the post-fill price move can still be adverse.

## What I Built

- Event-driven synthetic LOB experiment.
- Hypothetical passive-order replay.
- Queue-aware features.
- Fill and signed-markout labels.
- Chronological validation.
- Leakage checks.
- Nested model comparison.
- Local shuffled-null diagnostic.
- Reproducible figures and tables.

## Evidence

The pipeline separates execution likelihood from post-fill quality. In the current controlled run, fill-score bins show materially different fill rates and signed markouts. The proposed flow-depletion interaction remains unstable across event windows, so it is reported as a weak mechanism result rather than a discovery.

## Research Value

The project demonstrates problem formulation, event-driven simulation, label design, side-adjusted evaluation, chronological validation, mechanism testing, negative-result discipline, and reproducibility.

## Next Step

Run the unchanged pipeline on a single-venue BTC-USD L2 or L3 dataset.

