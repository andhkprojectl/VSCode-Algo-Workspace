"""
monteCarlo.py
=============
Wrapper for Monte Carlo test mode (TASKS.md T7.5).

CLI arguments: --strategy <id>  (e.g. S4003)
Runs the monteCarlo test file with folder prefix <id> under VS_4000_strategy.
E.g. `python monteCarlo.py --strategy S4003` runs
`VS_4000_strategy/VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py`.

Any extra arguments are forwarded to the strategy monteCarlo program,
e.g. `python monteCarlo.py --strategy S4003 --config path/to/config.yaml`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRATEGY_ROOT = ROOT / "VS_4000_strategy"
MODE_FOLDER = "5 OtherTest"
FILE_SUFFIX = "_monteCarlo.py"


def normalizeStrategyId(raw):
    """'s4003' / '4003' -> 'S4003'."""
    sid = str(raw).strip().upper()
    if not sid.startswith("S"):
        sid = "S" + sid
    return sid


def findStrategyProgram(strategy_id):
    """Locate <strategy_id>_monteCarlo.py under VS_4000_strategy/VS_<num>_*/."""
    sid = normalizeStrategyId(strategy_id)
    num = sid[1:]
    candidates = sorted(STRATEGY_ROOT.glob(f"VS_{num}_*"))
    if not candidates:
        available = sorted(
            p.name for p in STRATEGY_ROOT.glob("VS_*") if p.is_dir())
        raise FileNotFoundError(
            f"no strategy folder VS_{num}_* under {STRATEGY_ROOT}. "
            f"available: {available}")
    strategy_dir = candidates[0]
    mode_dir = strategy_dir / MODE_FOLDER
    exact = mode_dir / f"{sid}{FILE_SUFFIX}"
    if exact.exists():
        return exact
    matches = sorted(mode_dir.glob(f"{sid}_*.py"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"no monteCarlo program {sid}{FILE_SUFFIX} under {mode_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="MonteCarlo wrapper (TASKS.md T7.5). Extra args are "
                    "forwarded to the strategy monteCarlo program.")
    ap.add_argument("--strategy", required=True,
                    help="strategy id, e.g. S4003")
    args, extra = ap.parse_known_args()
    try:
        program = findStrategyProgram(args.strategy)
    except FileNotFoundError as e:
        print(f"monteCarlo.py: error: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"monteCarlo.py: running {program}")
    rs = subprocess.run([sys.executable, str(program), *extra], cwd=ROOT)
    sys.exit(rs.returncode)


if __name__ == "__main__":
    main()
