# Plan: S4003_1.py — ES 1-Min Price-Action Strategy (Python Conversion, V1)

Plan ID: prompt_strategy_S4003_V1
Source requirement: `VS_4000_strategy/VS_4003_20260827_ES/1 Prompt/prompt_strategy_S4003_V1.txt`

---

## 1. Objective

Convert the AmiBroker strategy `backtest_4003_1 trim_4_python.afl` into a single Python file
`S4003_1.py` containing class `strategyS4003V1`, backtested on ES 1-minute bars loaded from
MariaDB, producing 3 result files (statistics HTML, trades CSV, explore CSV).

**Parity requirement (requirement #7)**: backtesting `strategyS4003V1` from **2026-07-13 to
2026-08-17** must reproduce the same **number of trades, net profit, and win rate** as the
AmiBroker report `backtest_4003_1 - Backtest Report.html` (see §2a).

---

## 2. Source Materials / References

| Item | Path |
|---|---|
| AFL source (primary logic) | `VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/amibroker/backtest_4003_1 trim_4_python.afl` |
| AmiBroker settings (secondary) | `VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/amibroker/backtest_4003_1 trim_4_python.apx` |
| AmiBroker reference report (validation) | `VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/amibroker/backtest_4003_1 - Backtest Report.html` |
| DB access pattern | `VS_0007_dbAndFile/mariaDB/IBDb.py` (mysql-connector-python, env vars) |
| DB table DDL | `VS_0007_dbAndFile/mariaDB/amibroker/createTicker1Min.sql` |
| Test programs (class must be usable as parameter) | `VS_0003_test/backtest.py`, `VS_0003_test/otherTest.py`, `VS_0003_test/monteCarloSimulation.py`, `VS_0003_test/walkForwardTest.py` |
| Prior-art dual-mode strategy class | `VS_4000_strategy/VS_4002_20260620_NVDA/2 Strategy/S8002_1_BB.py` |

AmiBroker reference results (validation targets, engine differences may cause small deviations):
Initial capital 80,000 | Net profit 2,740.05 | 74 trades (10 long, 64 short) | transaction costs 1,073 (= 14.50/trade).

### 2a. Parity targets (requirement #7 — must match exactly)

Backtest period: **2026-07-13 to 2026-08-17** (same period as the AmiBroker report).

| Metric | AmiBroker value (parity target) |
|---|---|
| Number of trades | 74 (10 long, 64 short) |
| Net profit | 2,740.05 (long 414.15 + short 2,325.90) |
| Win rate | 50.00% (37 winners / 37 losers) |

If any of the 3 metrics differs, calibrate (see §8 step 10 and §10 notes) until they match.

---

## 3. Deliverable

- **File**: `VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/S4003_1.py`
- **Class**: `strategyS4003V1` — all strategy logic inside the class (helpers may be module-level functions).
- Output folder (auto-create if missing):
  `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES\backTestResult\`
- Dependencies: `pandas`, `numpy`, `mysql-connector-python`, `backtesting` (for the Strategy-subclass mode), `matplotlib` (equity chart). No TA-Lib.

---

## 4. Data Source

- MariaDB, database `IBTradingDb`, user `ibUser1`@`localhost`, password from env `DB_PASSWORD`
  (default empty), port env `DB_PORT` (default 3306) — follow `IBDb.py` conventions.
- Table `ticker1Min` columns: `id, ticker, datetime1, open, high, low, close, volume`
  (unique key `ticker + datetime1`).
- Query: `SELECT datetime1, open, high, low, close, volume FROM ticker1Min WHERE ticker = 'ES'
  AND datetime1 BETWEEN %s AND %s ORDER BY datetime1 ASC`.
- `start_date` / `end_date` are class attributes. **Default parity window:
  `2026-07-13` to `2026-08-17`** (inclusive) — the exact period of the AmiBroker reference
  report required by requirement #7.
- Returned DataFrame: DatetimeIndex from `datetime1`, columns renamed to
  `Open, High, Low, Close, Volume` (backtesting.py convention), sorted ascending.
- Symbol name attribute: `symbol = 'ES'`.

---

## 5. Strategy Logic (converted from AFL, active branches only)

All signals are evaluated on **completed bar i**; entry happens on **bar i+1** at its Open.
Warm-up: at least 100 bars (rolling percentile) — no signals before warm-up.

### 5.1 Indicators (pure pandas)

| Name | Definition |
|---|---|
| `atr14` | TR = max(H−L, \|H−C_prev\|, \|L−C_prev\|); Wilder smoothing period 14 |
| `max3BarAtr14_1` | `atr14.rolling(3).max().shift(1)` — max ATR of previous 3 bars, excluding current |
| `maxCO` / `minCO` | max(C,O) / min(C,O) |
| `isBullish` / `isBearish` | C > O / C < O |
| `bodySize` | abs(C−O) |
| `bodySizeRankTop` | `bodySize.rolling(100, min_periods=100).quantile(0.70)` |
| `bodySizePer` | abs(C−O)/(H−L), guard H==L → 0 |
| `isLargeBody` | bodySize >= bodySizeRankTop AND bodySizePer >= 0.65 |
| `largeBullishBody` / `largeBearishBody` | isBullish/isBearish AND isLargeBody |
| `vwap50` | `Sum(C·V, 50) / Sum(V, 50)` rolling 50-bar VWAP on Close (Nz → 0) |
| `crossUpVWap50` | AFL `Cross(C, vwap50)`: C_prev <= vwap50_prev AND C > vwap50; OR (O < vwap50 AND C > vwap50) |
| `crossDownVWap50` | AFL `Cross(vwap50, C)`: C_prev >= vwap50_prev AND C < vwap50; OR (O > vwap50 AND C < vwap50) |
| `is5BarLowL` / `is5BarLowMinCO` | current bar is the lowest of last 5 bars: L == LLV(L,5) / minCO == LLV(minCO,5) |
| `regSlop3Vwap50Bar` | LinRegSlope(vwap50, 3) |
| `regSlop5vwap50Bar` | LinRegSlope(vwap50, 5) |
| `regSlop5vwap50BarL0` | regSlop5vwap50Bar < −0.01 |
| `regSlop5vwap50BarL0Sum15` | `regSlop5vwap50BarL0.rolling(5).sum()` (AFL sums over 5 despite the name) |

LinRegSlope(x, n): OLS slope of x over the last n bars (rolling, per bar).

### 5.2 Entry signals (bar i)

```
priceActionUp4Bar2 = (largeBullishBody[i-3] OR largeBullishBody[i-2])
                 AND (crossUpVWap50[i-3]   OR crossUpVWap50[i-2])
                 AND largeBearishBody[i-1] AND crossDownVWap50[i-1]
                 AND (is5BarLowL OR is5BarLowMinCO)          # current bar lowest of 5
                 AND isBullish

buy00 = priceActionUp4Bar2                                  # AFL: buy00 = buy01_3_2
```

```
priceActionDown3Bar1 = isBullish[i-2] AND isBearish[i-1]
                 AND (minCO[i-2] > minCO[i-1] OR L[i-2] > L[i-1])
                 AND (maxCO[i-2] < maxCO[i-1] OR H[i-2] < H[i-1])
                 AND isBearish AND crossDownVWap50

short02 = regSlop3Vwap50Bar[i-1] > 0 AND regSlop3Vwap50Bar < 0
     AND Sum(crossDownVWap50, 4) > 0
     AND isBearish AND isBearish[i-1]
     AND regSlop5vwap50BarL0Sum15 >= 4

short00 = priceActionDown3Bar1 OR short02
```

### 5.3 Entry execution (bar i+1)

- `Buy = buy00[i]` (signal shifted 1 bar); time filter currently `True` (keep as attribute).
- Entry price: **Open of bar i+1** (AFL loop uses `tradePrice0 = O`).
  - AFL also sets `BuyPrice = O − atr14[i]/4` / `ShortPrice = O + atr14[i]/4` (limit offset).
    Implement as option `use_limit_offset = False` by default; validate against the AmiBroker
    report and flip if it matches better.
- Reverse on signal: if in long and a Short triggers, force-exit long at Open and enter short
  same bar (and vice versa) — AFL `shortForceSell` / `buyForceCover`.

### 5.4 Exits (in-position, checked each bar)

Let `stopPt = max3BarAtr14_1` at entry bar, `profitPt = 1.5 * max3BarAtr14_1`:

| Exit | Long | Short | Exit price |
|---|---|---|---|
| Stop loss | Low <= entry − stopPt | High >= entry + stopPt | stop price |
| Profit take | High >= entry + profitPt | Low <= entry − profitPt | profit price |
| N-bar stop | 15 bars after entry (see note) | 15 bars after entry | Open |
| Force (opposite signal) | short signal while long | buy signal while short | Open |

N-bar note (preserve AFL asymmetry): long uses `sellStopNBarPeriod` initialized to 15 at entry
and shifted forward when consecutive Buy signals keep firing; short uses fixed `stopPeriod1 = 15`
with the same "skip if current bar is also an entry signal" rule. Same-bar exit allowed
(AFL `AllowSameBarExit = true`); profit/stop checks run on the entry bar too.

### 5.5 Sizing and costs

- `NumContracts = 1`; ES point value `point_value = 50` (futures).
- `initial_capital = 80000` (matches AmiBroker report).
- `commission = 14.50` per round-turn trade (derived: 1,073 / 74 from report); configurable.

---

## 6. Class Interface — usable as a parameter in all 4 test programs

Follow the dual-mode pattern of `S8002_1_BB.py`:

1. `class strategyS4003V1(Strategy)` — subclasses `backtesting.Strategy`
   - `init()`: register indicators via `self.I(...)`
   - `next()`: implement 5.2–5.4 entries/exits (incl. stop-profit/stop-loss via intrabar
     High/Low checks or `sl`/`tp` orders; N-bar exit; force-close on opposite signal)
   - works with `VS_0003_test/backtest.py` (`backtestStrategy(strategyS4003V1, ...)`) and
     `VS_0003_test/walkForwardTest.py`
2. `@classmethod generate_signals(df) -> df`
   - computes all indicators, adds `buy00`, `short00`, and `signal` column
     (1 = buy entry bar, −1 = short entry bar, 0 = hold; signal at bar T is derived from
     buy00/short00 at bar T−1 only → look-ahead safe)
   - required by `VS_0003_test/otherTest.py::lookAheadBiasTest`
3. `@classmethod run_backtest(df, init_balance, position_size) -> list[float]`
   - manual bar-by-bar loop implementing 5.2–5.5; returns per-trade P&L in dollars
     (net of commission)
   - required by `VS_0003_test/monteCarloSimulation.py`
4. `@classmethod run_full_backtest(df) -> results`
   - drives the custom engine for the 3 output files in §7 (trades list + equity curve +
     statistics), also used by `__main__`.

`__main__` smoke test: load data from MariaDB, run full backtest, write the 3 files, print summary.

---

## 7. Output Files

Folder: `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES\backTestResult\`
Timestamp `<YYYYMMDDHH24MISS>` = `datetime.now().strftime('%Y%m%d%H%M%S')`.

### 7.1 File 1 — statistics HTML: `S4003_1_<ts>.html`

Include at least:

- Initial Capital, End Capital, Net Profit, Net Profit %
- Exposure % (bars-in-market / total bars)
- Annual Return % (CAGR from equity curve)
- Total Commission Cost
- Number of trades / Long trades / Short trades
- Number of wins, wins %, Total win amount
- Number of loss, loss %, Total loss amount
- Average Profit/Loss, Average Profit/Loss %
- Maximum trade drawdown (worst single trade %), Maximum system drawdown (peak-to-trough of equity curve, %)
- CAR/MaxDD, Profit Factor (gross win / gross loss)
- **Sharpe Ratio with annual risk-free rate 3%** (per-bar returns annualized)
- Ulcer Index (sqrt of mean squared drawdown from equity peaks, 14-bar window)
- K-Ratio (OLS slope of log-equity vs. bar index / slope standard error)
- **Embedded equity chart** (matplotlib PNG, base64-embedded inline in the HTML)

### 7.2 File 2 — trades CSV: `S4003_1_<ts>.csv`

(Note: requirement txt says `S8002_1_<ts>.csv`; use `S4003_1_` — S8002 is a copy-paste artifact.)

| Col | Content |
|---|---|
| A | Symbol (ES) |
| B | Trade type: Long or Short |
| C | Entry date time, `DD/MM/YYYY HH:MM` |
| D | Entry price |
| E | Exit date time, `DD/MM/YYYY HH:MM` |
| F | Exit price |
| G | % change = (exit − entry)/entry × 100 (sign-corrected for short) |
| H | Profit in $ (net of commission; short = (entry − exit) × 50 × contracts − commission) |
| I | % Profit = Profit / (entry × 50 × contracts) × 100 |
| J | Shares (= contracts, 1) |
| K | Position value = entry × 50 × contracts |
| L | Cumulative profit |
| M | Number of bars held (entry bar → exit bar) |
| N | MAE in $ (max adverse excursion while in trade, 1 contract) |
| O | MFE in $ (max favorable excursion while in trade, 1 contract) |

Records in **descending order of column C** (newest entry first).

### 7.3 File 3 — explore CSV: `S4003_1_explore_<ts>.csv`

- One row **every 1-minute bar** in the test range, signal or not.
- Columns:
  - A: Symbol
  - B: `buyer00` / `short00` / empty — which entry signal is true on that bar
  - C: bar date time, `DD/MM/YYYY HH:MM`
  - D–G: Open, Close, High, Low
  - H: `cCrossUpBBTop` — carries S4003 `buy00` boolean (column name kept per requirement template)
  - I: `cCrossDownBBBottom` — carries S4003 `short00` boolean
- Records in **descending order of column C**.

---

## 8. Implementation Steps

1. **Skeleton + data loader**: create `S4003_1.py`, module docstring (reference this plan),
   `load_data_from_db()` following `IBDb.py` connection conventions; column normalization.
2. **Indicator engine**: pure pandas implementations of §5.1 (vectorized, no loops except
   rolling kernels); unit sanity checks (ATR, VWAP, LinRegSlope, rolling percentile).
3. **Signals**: `buy00` / `short00` per §5.2 as a `compute_signals(df)` classmethod.
4. **Trade simulation loop**: manual engine per §5.3–5.5 producing trade records
   (entry/exit/bar counts/MAE/MFE) and the bar-by-bar equity curve.
5. **Statistics + HTML**: metrics of §7.1 with 3% risk-free Sharpe; embed equity chart.
6. **CSV writers**: §7.2 and §7.3 with exact columns, datetime format, descending sort.
7. **backtesting.py wrapper**: `init()`/`next()` faithful to the same rules.
8. **`generate_signals` + `run_backtest`** classmethods for otherTest/monteCarlo.
9. **`__main__`**: end-to-end run (DB → backtest → 3 files → console summary).
10. **Parity calibration run**: run the full backtest on the default window 2026-07-13 →
    2026-08-17 and compare number of trades, net profit, win rate against §2a targets.
    If mismatched, iterate on the calibration levers in §10 (entry price mode, ATR smoothing
    variant, warm-up handling, percentile window) until all 3 metrics match.

---

## 9. Verification

1. `python S4003_1.py` runs end-to-end; 3 files created with correct names/columns/formats.
2. **Parity check (requirement #7, hard gate)**: backtest over 2026-07-13 → 2026-08-17 must
   produce **exactly**: 74 trades (10 long, 64 short), net profit 2,740.05, win rate 50.00%
   (37/74) — same as the AmiBroker report. A mismatch in any of the 3 metrics fails
   verification; investigate typical causes (ATR smoothing variant, limit-offset entry mode,
   percentile window, warm-up bars before 2026-07-13, DB bar completeness for the period).
3. `otherTest.lookAheadBiasTest(strategyS4003V1, ...)` → PASSED (no look-ahead).
4. `backtest.backtestStrategy(strategyS4003V1, <csv export of DB data>)` runs.
5. `walkForwardTest.walkTestStrategy1(strategyS4003V1, ...)` runs.
6. `monteCarloSimulation.monteCarloSimulation1(strategyS4003V1, ...)` runs on
   `run_backtest` P&L list.
7. Spot-check explore CSV: rows are 1 minute apart; buy00/short00 bars match trade entries
   shifted by 1 bar.

---

## 10. Notes / Open Items

- Requirement's File-2 prefix `S8002_1_` is treated as a typo → `S4003_1_`.
- Explore columns H/I keep the template names `cCrossUpBBTop`/`cCrossDownBBBottom` but carry
  S4003's `buy00`/`short00` (S4003 has no Bollinger Band logic).
- **Validated calibration (2026-08-27, parity achieved)**: `entry_price_mode='limit'` with
  AmiBroker price-bound clamping (long `max(O − ATR14[i-1]/4, Low)`, short
  `min(O + ATR14[i-1]/4, High)`), `atr_mode='wilder'`, full 120-day indicator warm-up loaded
  before 2026-07-13 with entries gated to the window, `LLVBars` oldest-occurrence tie-break.
  Result: 74/74 trades matched AmiBroker entry/exit prices to display precision (≤0.0007),
  net 2,740.22 vs 2,740.05 (sub-tick float residual from AB's 3-dp report rounding),
  win rate 50.00% exact, commission 1,073.00 exact.
- AFL short-side N-bar stop uses fixed 15 while long side shifts on consecutive buys —
  asymmetry preserved intentionally.
- AmiBroker `Percentile` = rolling window percentile value; `min_periods=100` (no partial windows).
- All AFL comment-only branches (return-target `buyRt*`, divergence, stoch setups, MA filters)
  are exploratory columns in AmiBroker only — **not** ported unless needed later.
