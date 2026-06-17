# Repository Audit

## Files And Directories

The repository contains a compact Python package under `src/lob_sim/`, runnable scripts under `scripts/`, unit tests under `tests/`, generated outputs under `outputs/`, configuration under `config/`, older thesis-oriented configuration under `configs/`, and an archive folder.

There are no notebooks and no committed raw BTC-USD data files. Current empirical-looking outputs are generated from synthetic data only.

## Reusable Modules

- `src/lob_sim/book.py`: reusable price-time-priority matching engine.
- `src/lob_sim/agents.py`: reusable synthetic agent logic for validation data.
- `src/lob_sim/data.py`: current synthetic event generator with event type, trade side, cancellation, top-of-book state, and synthetic markout-mid proxy.
- `src/lob_sim/labels.py`: current passive-order construction, fill detection, censoring, signed markout, and posting-value helpers.
- `src/lob_sim/models.py`: chronological split and baseline logistic/ridge model helpers.
- `src/lob_sim/evaluation.py`, `src/lob_sim/plots.py`, `src/lob_sim/policy.py`, `src/lob_sim/audit.py`: useful but currently tailored to the previous decile/policy analysis.

## Duplicated Or Redundant Material

- `outputs/PORTFOLIO_BRIEF.md` and `outputs/PROJECT_SUMMARY.md` are old Codex-generated thesis/portfolio summaries and are no longer aligned with the requested BTC microstructure project.
- `archive/codex_intermediate/` already contains old sweep summaries and sweep CSV files.
- `scripts/run_research_sweep.py` and `scripts/run_thesis_transfer_experiment.py` are old scenario-sweep scripts. They are not used by the active fill-toxicity analysis.
- `outputs/events.csv`, `outputs/features.csv`, `outputs/trades.csv`, and `outputs/summary.json` are legacy demo outputs.

## Current Synthetic Assumptions

- The active path uses synthetic LOB data, not BTC data.
- Price-time priority is assumed in the matching engine.
- Directional informed bursts and a synthetic efficient-mid proxy are deliberately embedded to create adverse-selection coupling.
- Top-of-book cancellation turnover is included.
- The synthetic generator emits enough fields to test replay, fill labels, signed markouts, and trade/cancel queue-depletion logic.

## Current Fill-Label Logic

- Hypothetical passive orders are posted at current best bid for buys and best ask for sells.
- Queue ahead is a capped fraction of same-side displayed depth.
- Trade depletion plus cancellation-ahead depletion advances queue position.
- Fill occurs when cumulative depletion exceeds queue ahead plus fixed order size.
- Cancellations alone are allowed to advance queue position but are not treated as economic fills.
- Non-filled observations within the fill horizon are censored.

## Current Markout Convention

- Positive signed markout is favorable to the passive trader.
- Buy fill: `(future_mid - fill_price) / tick_size`.
- Sell fill: sign is reversed.

## Current Split And Leakage Controls

- Chronological train/validation/test split is implemented by event step.
- Current audit checks disjoint order observations, chronological split boundaries, nonnegative time-to-fill, markout after fill, duplicate observations, and feature allowlist.
- Missing checks: target-column exclusion in arbitrary feature matrices, impossible queue values, zero/negative spread checks, and BTC timestamp overlap checks.

## Current Gaps Relative To Requested Project

- No BTC real-data adapter or schema validator.
- No public separation between synthetic validation and real-data analysis.
- No local signed-flow persistence feature across multiple lookback windows.
- No passive-side depletion feature over multiple lookback windows.
- No `flow persistence x passive-side trade depletion` interaction scan.
- No nested M0-M3 model comparison by lookback window.
- No formation-window by markout-horizon scale map.
- No response surfaces with shared flow/depletion axes.
- No local shuffled null preserving slow conditions while disrupting local sequence.
- README still answers the previous broader fill-probability toxicity question rather than the final BTC-USD interaction question.

## Refactor Plan

1. Archive obsolete Codex summaries and old sweep/demo outputs.
2. Keep the synthetic generator as a validation sandbox under `validation/synthetic/`.
3. Add a BTC data adapter with schema validation and a small mock fixture.
4. Add side-adjusted flow persistence, passive-side depletion, and interaction features for multiple event windows.
5. Add nested model scans for fill and markout.
6. Add local shuffled-null analysis.
7. Generate the six primary figures and required CSV tables.
8. Add focused tests for sign adjustment, persistence, depletion, interaction, split/audit, and shuffled-null determinism.
9. Rewrite README last using actual generated outputs.

