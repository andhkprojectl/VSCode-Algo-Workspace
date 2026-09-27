# Monte Carlo Test Program — S8004 (NVDA 5-min)

Generate a backtest python program.

## Requirement

- **Python file:** `VS_4004_20260204_NVDA/6 MonteCarlo/monteCarlo_S8004_v1.py`
- **Purpose:** run a Monte Carlo test using the strategy in
  `VS_4004_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py`
  (class `strategyIRB1000_V1`).

## Monte Carlo test data file

`C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcel\NVDA_20250101_20260430_5Min.csv`

- stock data is NVDA
- first row is header
- 5-minute time frame

| Column | Field | Format |
|--------|-------|--------|
| A | datetime | MM/DD/YYYY hh24:mi |
| B | date | MM/DD/YYYY |
| C | time | hh24:mi:ss |
| D | high price | |
| E | low price | |
| F | close price | |
| G | open price | |
| H | volume | |
| I | symbol name | |

## Backtest output result

Output 2 files.

### 1st file — backtest statistics file

- HTML format
- File:
  `C:\Project\ProjectLife\VSCode Algo Workspace - DataFile\backTestResult\VS_4004_20260204_NVDA\VS_4004_20260204_NVDA\S8004_GenerateFromPromptQwen3.7Max_<YYYYMMDDHH24MISS>.html`
- At least include the below values:

| Statistic |
|-----------|
| Initial Capital |
| End Capital |
| Net Profit |
| Net Profit % |
| Exposure % |
| Annual Return % |
| Total Commission Cost |
| Number of trades |
| Number of Long trades |
| Number of Short trades |
| Number of wins |
| Number of wins % |
| Total win amount |
| Number of loss |
| Number of loss % |
| Total loss amount |
| Average Profit/loss |
| Average Profit/loss % |
| Maximum trade drawdown |
| Maximum system drawdown |
| CAR/MaxDD |
| Profit Factor |
| Sharpe Ratio |
| Ulcer Index |
| K-Ratio |

- Sharpe ratio with annual risk-free rate 3%

### 2nd file — backtest trade file

- CSV format
- File:
  `C:\Project\ProjectLife\VSCode Algo Workspace - DataFile\backTestResult\VS_4004_20260204_NVDA\VS_4004_20260204_NVDA\S8004_GenerateFromPromptQwen3.7Max_<YYYYMMDDHH24MISS>.csv`
- Columns:

| Column | Field | Format |
|--------|-------|--------|
| A | datetime | MM/DD/YYYY hh24:mi |
| B | date | MM/DD/YYYY |
| C | time | hh24:mi:ss |
| D | high price | |
| E | low price | |
| F | close price | |
| G | open price | |
| H | volume | |
| I | symbol name | |
