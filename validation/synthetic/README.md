# Synthetic Validation Sandbox

This directory documents the role of the synthetic pipeline.

The synthetic generator is retained only to:

- test passive-order replay and label logic;
- test side-adjusted flow/depletion features;
- verify that the analysis can detect a deliberately embedded adverse-selection mechanism;
- provide a deterministic fallback when real BTC-USD data are unavailable.

Synthetic outputs must not be presented as empirical evidence about Bitcoin.

Run from the repository root:

```bash
python3 scripts/run_main_analysis.py
```

