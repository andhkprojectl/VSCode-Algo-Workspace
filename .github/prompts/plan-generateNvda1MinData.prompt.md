## Plan: Generate NVDA 1-Minute Data Fetch Script

Create a new Python script `NVDA_20260101_20260615_1Min.py` under `VS_0006_data/VS_6001_GetMarketDataToCsv/` to fetch 1-minute NVDA data from 2026-01-01 to 2026-06-15, based on the existing `NVDA_20250101_20260430_5Min.py`.

**Steps**

1. **Create `NVDA_20260101_20260615_1Min.py`** — Copy the structure from `NVDA_20250101_20260430_5Min.py` and modify the following parameters:
   - File: `VS_0006_data/VS_6001_GetMarketDataToCsv/NVDA_20260101_20260615_1Min.py`
   - Change `outcsvFileName` to `NVDA_20260101_20260615_1Min.csv`
   - Change `startDate` from `"20250501"` to `"20260101"`
   - Change `endDate` from `"20260430"` to `"20260615"`
   - Change `period1` from `5` (5-min bars) to `1` (1-min bars)
   - Remove the commented-out dead code block (old conn1/symbolName variables) to keep the file clean

**Relevant files**
- `c:\Project\ProjectLife\VSCode Algo Workspace\VS_0006_data\VS_6001_GetMarketDataToCsv\NVDA_20250101_20260430_5Min.py` — source template to copy from
- `c:\Project\ProjectLife\VSCode Algo Workspace\VS_0006_data\VS_6001_GetMarketDataToCsv\NVDA_20260101_20260615_1Min.py` — new file to create

**Changes summary**

| Parameter | Original (5Min) | New (1Min) |
|-----------|-----------------|------------|
| Output CSV | `NVDA_20250101_20260430_5Min.csv` | `NVDA_20260101_20260615_1Min.csv` |
| `startDate` | `"20250501"` | `"20260101"` |
| `endDate` | `"20260430"` | `"20260615"` |
| `period1` | `5` | `1` |

**Verification**
1. Run `NVDA_20260101_20260615_1Min.py` and confirm the console prints `Successfully retrieved csvExcelPath: C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcelAll`
2. Confirm `outcsvFileName` prints: `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcelAll\NVDA_20260101_20260615_1Min.csv`
3. Confirm the script fetches 1-minute NVDA data for the date range 2026-01-01 to 2026-06-15

**Decisions**
- Reuse the same `.env` loading and path resolution pattern from the existing script
- Keep the AmiBroker format conversion block at the end for consistency
- Remove commented-out dead code to keep the new file clean
