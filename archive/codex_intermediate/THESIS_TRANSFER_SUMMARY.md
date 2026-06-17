# Thesis-Transfer Sweep Summary

This experiment uses the PhD thesis as the primary source of the project narrative:

`Numerical Study of the Attosecond Free-Electron Laser Pulse Generation in the Soft X-ray Regime at the SwissFEL`

The goal is not to claim that FEL physics and market microstructure are the same system. The goal is to show a transferable research workflow: stochastic simulation, delay effects, selective amplification, parameter tapering, robustness across seeds, and mechanism-driven interpretation.

## Scenario Mapping

| Thesis Concept | Market-Microstructure Analogy |
|---|---|
| SASE shot-noise start | Noise-started order-flow dynamics |
| ESASE current spike | Concentrated informed-order-flow burst |
| Slicing with taper | Selective liquidity around active price slices |
| Mode-locking with chicane delay | Delay/latency control aligning repeated liquidity replenishment |
| TGU strong taper | Aggressive parameter optimisation to prevent instability under stress |

## Scenario Means

| Scenario | Avg Spread | Avg Microprice Deviation | Avg OFI | Realized Volatility | Resilience Score |
|---|---:|---:|---:|---:|---:|
| sase_noise_start | 1.1865 | 0.0168 | 0.0927 | 0.00000320 | 0.7525 |
| esase_current_spike | 2.4413 | 0.1667 | 0.2558 | 0.00001787 | 0.4539 |
| slicing_selective_resonance | 1.1381 | 0.0152 | 0.2027 | 0.00000417 | 0.7387 |
| mode_locking_delay_control | 1.1048 | 0.0136 | 0.2317 | 0.00000281 | 0.7737 |
| tgu_strong_taper | 1.0668 | 0.0088 | 0.2442 | 0.00000227 | 0.8299 |

## Interpretation

The ESASE/current-spike analogy produces the most stressed book: wider spreads, larger microprice deviation, higher order-flow imbalance, and higher volatility. The mode-locking and TGU-strong-taper analogies reduce spread and microprice deviation while improving resilience, which mirrors the thesis logic that delay control and taper optimisation can compress unstable dynamics into cleaner, shorter structures.

This is the strongest portfolio angle: the project turns the PhD thesis into an HFT-relevant simulation story about stochastic dynamics, delay, queue pressure, and parameter optimisation.

