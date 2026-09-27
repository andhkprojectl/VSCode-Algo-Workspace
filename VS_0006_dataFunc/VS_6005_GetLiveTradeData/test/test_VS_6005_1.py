"""
test_VS_6005_1.py
=================
Requirement: test_VS_6005_1.txt
Run getLiveTradeIB1MinData for the CME future ES Dec '26 (front month) from
2026-09-18 to 2026-09-22 23:59, then write the result to
C:\\Project\\ProjectLife\\VSCode Algo Workspace DataFile\\VS_0006_dataFunc\\test_VS_6005_1.csv
with columns [datetime, open, close, high, low, volume].
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                 # VS_6005_GetLiveTradeData\test
_MODULE_DIR = _HERE.parent                              # VS_6005_GetLiveTradeData
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from VS_6005_1_1min_glm import getLiveTradeIB1MinData  # noqa: E402

OUT_PATH = (r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile"
            r"\VS_0006_dataFunc\test_VS_6005_1.csv")


def main():
    df = getLiveTradeIB1MinData("ES", None, "2026-09-18 00:00", "2026-09-22 23:59")
    print(f"\nresult: {df.shape[0]} rows x {df.shape[1]} cols, "
          f"{df.index[0]} -> {df.index[-1]}")

    out = df.reset_index().rename(columns={"index": "datetime"})
    out = out[["datetime", "open", "close", "high", "low", "volume"]]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"written: {OUT_PATH} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
