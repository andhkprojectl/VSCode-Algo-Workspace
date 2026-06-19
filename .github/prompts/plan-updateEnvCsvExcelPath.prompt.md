## Plan: Update .env and Python file to use absolute csvExcelPath

Update the workspace `.env` file to use an absolute path for `csvExcelPath`, and clean up the Python file to correctly use that absolute path directly.

**Steps**

1. **Update `.env` file** — Change `csvExcelPath` from the relative path `../../VSCode Algo Workspace DataFile/csvExcel` to the absolute path `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcelAll`
   - File: `VS_0002_config/.env`

2. **Clean up `NVDA_20250101_20260430_5Min.py`** — Remove the dead code that computes `absolute_csv_path` (lines 27-29) since it is never used, and the `csvExcelPath` will now be an absolute path so no relative-to-env-dir resolution is needed.
   - Remove lines 27-29: `env_dir = dotenv_path.parent` and `absolute_csv_path = (env_dir / csv_excel_path).resolve()`
   - The existing `outcsvFileName = Path(csv_excel_path) / "NVDA_20250101_20260430_5Min.csv"` on line 32 already works correctly with an absolute path — no change needed there.

**Relevant files**
- `c:\Project\ProjectLife\VSCode Algo Workspace\VS_0002_config\.env` — update `csvExcelPath` value
- `c:\Project\ProjectLife\VSCode Algo Workspace\VS_0006_data\VS_6001_GetMarketDataToCsv\NVDA_20250101_20260430_5Min.py` — remove dead `absolute_csv_path` code block

**Verification**
1. Run `NVDA_20250101_20260430_5Min.py` and confirm the console prints `Successfully retrieved csvExcelPath: C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcelAll`
2. Confirm `outcsvFileName` prints the correct full path: `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcelAll\NVDA_20250101_20260430_5Min.csv`

**Decisions**
- The `.env` file already exists at `VS_0002_config/.env` — no need to create a new one
- Using absolute path eliminates the need for relative path resolution logic in the Python file
