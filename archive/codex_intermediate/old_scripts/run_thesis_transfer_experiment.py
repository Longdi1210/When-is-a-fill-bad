from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob_sim.phd_layer import PhDProfile
from lob_sim.simulation import run_simulation


def main() -> None:
    base = PhDProfile.from_json(ROOT / "configs" / "phd_profile.json")
    scenarios = [
        (
            "sase_noise_start",
            "Noise-started order flow, analogous to SASE shot-noise amplification.",
            replace(base, informed_agent_share=0.10, resilience_agent_share=0.10, cancellation_pressure=0.06, order_flow_autocorrelation=0.20),
        ),
        (
            "esase_current_spike",
            "A concentrated informed-flow burst, analogous to ESASE current-spike enhancement.",
            replace(base, informed_agent_share=0.30, resilience_agent_share=0.10, cancellation_pressure=0.08, order_flow_autocorrelation=0.48),
        ),
        (
            "slicing_selective_resonance",
            "Liquidity is selectively supplied near the active price slice, analogous to maintaining resonance with an undulator taper.",
            replace(base, informed_agent_share=0.22, resilience_agent_share=0.22, cancellation_pressure=0.06, order_flow_autocorrelation=0.34),
        ),
        (
            "mode_locking_delay_control",
            "Stronger replenishment under autocorrelated flow, analogous to delay control aligning repeated pulses.",
            replace(base, informed_agent_share=0.24, resilience_agent_share=0.30, cancellation_pressure=0.05, order_flow_autocorrelation=0.55),
        ),
        (
            "tgu_strong_taper",
            "Aggressive resilience under stress, analogous to a strong TGU taper preventing pulse broadening.",
            replace(base, informed_agent_share=0.26, resilience_agent_share=0.38, cancellation_pressure=0.04, order_flow_autocorrelation=0.42),
        ),
    ]

    rows = []
    for scenario, thesis_analogy, profile in scenarios:
        for seed in range(12):
            result = run_simulation(profile=profile, steps=700, seed=20_000 + seed)
            summary = result["summary"]
            rows.append(
                {
                    "scenario": scenario,
                    "thesis_analogy": thesis_analogy,
                    "seed": seed,
                    "trades": summary["trades"],
                    "trade_volume": summary["trade_volume"],
                    "average_spread": summary["average_spread"],
                    "average_abs_imbalance": summary["average_abs_imbalance"],
                    "average_abs_microprice_deviation": summary["average_abs_microprice_deviation"],
                    "average_abs_order_flow_imbalance": summary["average_abs_order_flow_imbalance"],
                    "realized_volatility": summary["realized_volatility"],
                    "resilience_score": summary["resilience_score"],
                }
            )

    output = ROOT / "outputs" / "thesis_transfer_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} thesis-transfer scenario rows to {output}")


if __name__ == "__main__":
    main()

