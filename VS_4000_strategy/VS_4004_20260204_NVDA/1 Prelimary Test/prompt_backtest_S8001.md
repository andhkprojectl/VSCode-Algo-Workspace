# prompt_backtest_S8001 — Backtest Specification

## Overview

Generate and maintain a backtest Python program at `VS_4004_20260204_NVDA/3 BackTest/backtest_S8004_v1.py` that backtests the IRB1000 V1 strategy on NVDA 5-minute data.

This document consolidates the incremental requirements from `prompt_backtest_S8001_V1.md` through `V6.md`.

---

## 1. Base Program (V1)

- **Python file**: `VS_4004_20260204_NVDA/3 BackTest/backtest_S8004_v1.py`
- **Strategy**: class `strategyIRB1000_V1` in `VS_4004_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py`
- **Data file**: `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcel\NVDA_20250101_20260430_5Min.csv`
  - Stock: NVDA, 5-minute timeframe, first row is header
  - Columns:
    - A: datetime — format `MM/DD/YYYY hh24:mi`
    - B: date — format `MM/DD/YYYY`
    - C: time — format `hh24:mi:ss`
    - D: high price
    - E: low price
    - F: close price
    - G: open price
    - H: volume
    - I: symbol name

### Output File 1 — Backtest Statistics (HTML)

- Path: `C:\Project\ProjectLife\VSCode Algo Workspace - DataFile\backTestResult\VS_4004_20260204_NVDA\VS_4004_20260204_NVDA\S8004_GenerateFromPromptQwen3.7Max_<YYYYMMDDHH24MISS>.html`
- Must include at least:
  - Initial Capital
  - End Capital
  - Net Profit
  - Net Profit %
  - Exposure %
  - Annual Return %
  - Total Commission Cost
  - Number of Trades
  - Number of Long Trades
  - Number of Short Trades
  - Number of Wins
  - Number of Wins %
  - Total Win Amount
  - Number of Losses
  - Number of Losses %
  - Total Loss Amount
  - Average Profit/Loss
  - Average Profit/Loss %
  - Maximum Trade Drawdown
  - Maximum System Drawdown
  - CAR/MaxDD
  - Profit Factor
  - Sharpe Ratio
  - Ulcer Index
  - K-Ratio
- Sharpe Ratio uses an annual risk-free rate of 3%.

### Output File 2 — Backtest Trades (CSV)

- Path: `C:\Project\ProjectLife\VSCode Algo Workspace - DataFile\backTestResult\VS_4004_20260204_NVDA\VS_4004_20260204_NVDA\S8004_GenerateFromPromptQwen3.7Max_<YYYYMMDDHH24MISS>.csv`

---

## 2. Trades CSV Format Update & Equity Chart (V2)

- Update the trades CSV columns to:
  - A: symbol name
  - B: trade type — `Long` or `Short`
  - C: buy/short entry datetime — format `DD/MM/YYYY hh24:mi`
  - D: buy/short entry price
  - E: sell/cover exit datetime — format `DD/MM/YYYY hh24:mi`
  - F: sell/cover exit price
  - G: % change
  - H: Profit
  - I: % Profit
  - J: buy/short shares
  - K: position value
  - L: cumulative profit
  - M: number of bars between entry and exit
  - N: MAE
  - O: MFE
- Records sorted in **descending** order of column C.
- Add an **equity chart** to the HTML report.

---

## 3. Explore CSV (V3)

- After the backtest, create a **3rd output file** (may require touching the strategy file if necessary):
  - `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\backTestResult\VS_4004_20260204_NVDA\S8004_GenerateFromPromptQwen3.7Max_explore_<YYYYMMDDHH24MISS>.csv`
- Write a record when `buy00` or `short00` is true in the strategy.
- Columns:
  - A: Symbol
  - B: signal type — `buy00` or `short00`
  - C: datetime of the signal — format `DD/MM/YYYY hh24:mi`
  - D–G: Open, Close, High, Low
  - H–R: strategy values — `data['buy00']`, `data['short00']`, `data['iRbBullish']`, `data['iRbBearish']`, `data['regSlopema10']`, `data['regSlopema10'].shift(1)`, `data['pRtEma20Rankper']`, `data['ema20DiffCAbsRank']`, `data['pRtEma20Rankper']`, `data['pRtEma20Rankper'].shift(1)`, `data['pRtEma20Rankper'].shift(2)`
- Records sorted in **descending** order of column C.

---

## 4. Explore CSV — Write Every Bar in a Date Range (V4)

- Update the write condition for the explore CSV:
  - Write a record for **every 5-minute bar from 04/22 to 04/30**, regardless of whether a `buy00` or `short00` signal exists.
- Columns stay the same.
- Records sorted in **descending** order of column C datetime.

---

## 5. Explore CSV — Additional Strategy Columns (V5)

- Update function `generate_explore_csv`, add these columns from the strategy:
  - `data['rt10']`
  - `data['ema20DiffCAbsRound0']`
  - `data['ema20DiffCAbsRank']`
  - `data['sumEma20DiffCAbsRank']`
  - `data['sumEma20DiffCAbsUpRankper']`
  - `data['pRtEma20Rankper']`

---

## 6. Explore CSV — Fix + Parameterized Rank Columns (V6)

- Fix the error in `generate_explore_csv` caused by the update to `S8004_GenerateFromPromptQwen37Max.py`.
- Add columns:
  - `data[f'sumEma20DiffCAbsRank_{r}']` for `r in range(1, 6)` — 5 new columns
  - `data[f'sumEma20DiffCAbsUpRank_{r}']` for `r in range(1, 6)` — 5 new columns
  - `data['pRtEma20Rankper']`
