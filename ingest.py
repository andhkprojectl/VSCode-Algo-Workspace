"""
ingest.py
=========
Wrapper for ingest mode (TASKS.md T7.1).

Runs `VS_0006_dataFunc/VS_6003_GetMarketDataToDb/1min/ingest.py` (T6.1),
which fetches market data from IB and saves it to MariaDB. All ingest
parameters come from config.yaml (workspace root) section `Ingest`;
credentials from VS_0002_config/.env.

Usage: python ingest.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROGRAM = (ROOT / "VS_0006_dataFunc" / "VS_6003_GetMarketDataToDb"
           / "1min" / "ingest.py")


def main():
    if not PROGRAM.exists():
        print(f"ingest.py: error: ingest program not found: {PROGRAM}",
              file=sys.stderr)
        sys.exit(2)
    print(f"ingest.py: running {PROGRAM}")
    rs = subprocess.run([sys.executable, str(PROGRAM), *sys.argv[1:]],
                        cwd=ROOT)
    sys.exit(rs.returncode)


if __name__ == "__main__":
    main()
