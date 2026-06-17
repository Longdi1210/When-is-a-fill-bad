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
        ("calm_baseline", replace(base, informed_agent_share=0.08, resilience_agent_share=0.10, cancellation_pressure=0.04)),
        ("information_shock", replace(base, informed_agent_share=0.28, resilience_agent_share=0.10, cancellation_pressure=0.10)),
        ("resilient_market", replace(base, informed_agent_share=0.28, resilience_agent_share=0.26, cancellation_pressure=0.06)),
        ("fragile_liquidity", replace(base, informed_agent_share=0.28, resilience_agent_share=0.04, cancellation_pressure=0.18)),
    ]

    rows = []
    for name, profile in scenarios:
        for seed in range(10):
            result = run_simulation(profile=profile, steps=600, seed=10_000 + seed)
            summary = result["summary"]
            rows.append(
                {
                    "scenario": name,
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

    output = ROOT / "outputs" / "research_sweep.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} scenario rows to {output}")


if __name__ == "__main__":
    main()

