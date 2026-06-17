from __future__ import annotations

import argparse
import json

from .phd_layer import PhDProfile
from .simulation import run_simulation, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PhD-flavored limit order book simulation.")
    parser.add_argument("--profile", default="configs/phd_profile.json", help="Path to a PhD profile JSON file.")
    parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PhDProfile.from_json(args.profile)
    result = run_simulation(profile=profile, steps=args.steps, seed=args.seed)
    write_outputs(result, args.out)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

