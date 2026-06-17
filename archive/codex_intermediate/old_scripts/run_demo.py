from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lob_sim.phd_layer import PhDProfile
from lob_sim.simulation import run_simulation, write_outputs


def main() -> None:
    profile = PhDProfile.from_json(ROOT / "configs" / "phd_profile.json")
    result = run_simulation(profile=profile, steps=500, seed=42)
    write_outputs(result, ROOT / "outputs")
    print("Wrote demo outputs to outputs/")
    print(result["summary"])


if __name__ == "__main__":
    main()

