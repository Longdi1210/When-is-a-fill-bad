# Portfolio Brief: Limit Order Book Lab

## Positioning

This project is designed as a bridge from your PhD thesis on attosecond free-electron-laser pulse generation to high-frequency quantitative research and engineering.

The thesis is the primary source of the narrative:

- Title: Numerical Study of the Attosecond Free-Electron Laser Pulse Generation in the Soft X-ray Regime at the SwissFEL.
- System: SwissFEL Athos soft X-ray FEL.
- Methods: SASE/ESASE, slicing, mode-locking, transverse gradient undulator, slippage analysis, superradiance interpretation, undulator taper optimisation.
- Workflow: stochastic simulations over random shot noise, mechanism-driven interpretation, parameter scans, time-frequency analysis, and robustness checks.

The project translates that background into market microstructure:

- SASE shot-noise amplification becomes noise-started order-flow dynamics.
- ESASE current spikes become concentrated informed-order-flow bursts.
- Slippage and magnetic-chicane delay control become latency and queue-position effects.
- Slicing and taper optimisation become selective liquidity provision around active price levels.
- Mode-locking becomes alignment of repeated liquidity replenishment under autocorrelated flow.
- Spectrum and pulse-duration analysis become microprice, spread, volatility, and resilience metrics.

## What The Project Demonstrates

- A price-time priority matching engine.
- Limit, market, and cancel order handling.
- Heterogeneous agents: noise, informed, cancellation, and resilience liquidity providers.
- Shock experiments with controlled pre/during/post windows.
- Microstructure features: spread bps, queue imbalance, microprice deviation, signed order-flow imbalance.
- Research sweep across calm, shocked, resilient, and fragile-liquidity scenarios.

## Current Empirical Result

Across 40 simulations, information shocks widen spreads and increase microprice deviation. The resilient-market scenario keeps spread and microprice deviation close to the calm baseline, showing how adaptive liquidity replenishment can stabilize the book under informed-order-flow pressure.

See `outputs/THESIS_TRANSFER_SUMMARY.md` for the thesis-aligned scenario table, and `outputs/RESEARCH_SWEEP_SUMMARY.md` for the generic market-stress table.

## Strong Interview Narrative

I built this project to show how the numerical mindset from my PhD thesis on attosecond FEL pulse generation transfers to high-frequency market microstructure. In my thesis, I studied how short pulses emerge from stochastic shot noise, slippage, delay control, mode-locking, and taper optimisation. In this project, I implement a controlled limit-order-book laboratory where order-flow shocks, latency-like delay, queue imbalance, and liquidity replenishment can be perturbed and measured reproducibly. The point is not that photons and orders are the same system; the point is that both require mechanism-based simulation, stochastic validation, and careful interpretation of time-domain and frequency-domain signals.

## Next Upgrade For HFT Roles

The highest-impact next step is to add a C++ matching-engine backend with Python bindings, then benchmark throughput and latency. That would connect your existing C++/HPC background directly to quant developer expectations.
