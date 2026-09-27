# Look-Ahead Test Program — S8001_4 (NVDA 5-min)

Generate a testing python program.

## Requirement

- **Python file:** `VS_4004_20260204_NVDA/5 OtherTest/lookAheadTest_S8004_v1.py`
- **Purpose:** run a look-ahead test using the strategy in
  `VS_4004_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py`
  (class `strategyIRB1000_V1`). Determine whether this strategy has a
  look-ahead issue, i.e. changing future OHLCV values will affect previous-bar
  buy/short/sell/cover signals.

## Input data file

`C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4004_20260204_NVDA\csvExcel\NVDA_20250101_20260430_5Min_.csv`

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

## Output result of look-ahead test

All output files are saved to folder
`C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4004_20260204_NVDA\LimitTestResult`.

Output 1 file — the backtest statistics file:

- HTML format, filename `S8001_4_LookAheadTest_Qwen3.7Max_<YYYYMMDDHH24MISS>.html`
- plot chart, show which buy or short bar has look-ahead issue
- show total number of buy
- show total number of buy with look-ahead issue
- show total number of short
- show total number of short with look-ahead issue

## Logic

1. Find all buy, sell, short, cover signals from the input data file using
   strategy `S8001_4_GenerateFromPromptQwen37Max.py`.
2. Loop for each bar, starting from the last bar to the first bar. Each step
   is 1 bar, hence the sequence is last bar, last bar − 1, last bar − 2, …,
   until the 1st bar.
3. For each step, set `-1` to open, high, low, close, volume (OHLCV) of the
   current bar. After setting OHLCV to `-1`, if the current bar is not a
   buy/short/sell/cover signal and the latest bar with a buy/short/sell/cover
   signal before the current bar disappears, then this strategy has a
   look-ahead issue. That bar with a disappearing buy or short signal means
   that bar has a look-ahead issue.
4. Tune the program so it does not run too long.
5. Use multiprocessing technique to make the program run faster.
6. If more than 5 bars are found with look-ahead issues, stop the program and
   output the result.
7. Only test the last 2 months of data in the input CSV file.
8. Print the date/time before and after the program run, and print on screen
   how many hours and minutes it took to run the program.
