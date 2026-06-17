from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_main_analysis.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

