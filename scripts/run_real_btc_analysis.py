from __future__ import annotations

from pathlib import Path
import subprocess


COMMANDS = [
    ["python", "scripts/audit_kaggle_btc.py"],
    ["python", "scripts/convert_kaggle_btc.py"],
    ["python", "scripts/build_real_btc_features.py"],
]


def main() -> None:
    for command in COMMANDS:
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    print("Real BTC audit and feature build complete. Modeling is intentionally deferred to the next phase.")


if __name__ == "__main__":
    main()
