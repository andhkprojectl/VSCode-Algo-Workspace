## Plan: S4003_1.py Strategy Implementation (per prompt_strategy_S4003_V1.txt)

Implement a Python strategy file converted from AmiBroker AFL/APX, exposing a
`strategyS4003V1` class compatible with all four test engines and three
backtest output artifacts; verify against the AmiBroker reference report.
**Steps**

Phase A — AFL conversion understanding
1. Analyze source AFL [backtest_4003_1 trim_4_python.afl] and `.apx` — extract: VWAP (50-bar, month-reset variant `vwap50_2`), Stochastics (9,3 / 14,3 / 40,4 / 60,5), ATR(14), EMA(9), price-action patterns (`priceActionUp4Bar2` → `buy01_3_2`, `priceActionDown3Bar1` → `short01_4`, `priceActionDown3Bar2`/`short02`), divergence, body-size percentile ranks, and the bar-loop exit engine (stop loss `max3BarAtr14_1`, profit target `1.5*max3BarAtr14_1`, N-bar stop `stopPeriod1=15`, signal reversal force exits). *indipendent*, blocks Phase B.

Phase B — Strategy file
2. Create VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/S4003_1.py containing class `strategyS4003V1`. Use S8002_1_BB.py as structural template — helper indicator functions, `init`/`next`, `generate_signals`, `run_backtest`. *depends on 1*
3. Data loader: MariaDB `IBTradingDb.ticker1Min`, user `ibUser1@localhost`, mirroring IBDb.py connector pattern (mysql.connector, env-var-overridable host/user/password/database). Query `ticker, datetime1, open, high, low, close, volume`; parse to pandas DataFrame with DatetimeIndex. *depends on 2*
4. Class methods: `init`/`next` (backtesting.Strategy subclass) for backtest.py; `generate_signals(df)` for otherTest.py look-ahead test; `run_backtest(df, init_balance, position_size)` returning per-trade P&L list for monteCarloSimulation.py; compatible with walkForwardTest.py. *depends on 3*
5. Conversion fidelity: all indicator math with no look-ahead (Elliott wave / divergence entries use bar-relative shifted refs identical to AFL `Ref(x, -n)` semantics). *depends on 4*

Phase C — Backtest artifacts
6. Output file 1 (HTML stats) — `…DataFile\VS_4003_20260827_ES\backTestResult\S4003_1_<YYYYMMDDHH24MISS>.html` with all metrics: Initial/End Capital, Net Profit, Net Profit %, Exposure %, Annual Return %, Total Commission, Trades/Long/Short counts, Wins/Wins %, Total win amount, Loss/Loss %, Total loss amount, Avg P/L, Avg P/L %, Max trade DD, Max system DD, CAR/MaxDD, Profit Factor, Sharpe (rf=3%), Ulcer Index, K-Ratio; plus an equity chart. *depends on 5*
7. Output file 2 (trades CSV) — `S4003_1_<YYYYMMDDHH24MISS>.csv` columns A–O = symbol, Long/Short, entry datetime (DD/MM/YYYY hh24:mi), entry price, exit datetime, exit price, %Change, Profit, %Profit, Shares, Position value, Cumulative profit, Bars held, MAE, MFE; descending entry time. *depends on 5*
8. Output file 3 (explore CSV) — `S4003_1_explore_<timestamp>.csv` every 1 minute regardless of signal: symbol, buy00/short00 flag, datetime (DD/MM/YYYY hh24:mi), Open/Close/High/Low, `cCrossUpBBTop`, `cCrossDownBBBottom`; descending datetime. *depends on 5*

Phase D — Validation
9. Run backtest on 2026-07-13 → 2026-08-17 and compare Number of trades, Net profit, Win rate against backtest_4003_1 - Backtest Report.html. Tolerances to lock with user. *depends on 8*

**Relevant files**
- VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/S4003_1.py — new strategy module (create)
- VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/amibroker/* — conversion source
- VS_0007_dbAndFile/mariaDB/IBDb.py — DB connection template
- VS_0003_test/backtest.py — expects `Backtest`, `Strategy`, `init`/`next`, `_trades`
- VS_0003_test/otherTest.py — expects `strategy1.generate_signals(df)` returning `signal` column for lookAheadBiasTest; also holds monkeyTest
- VS_0003_test/monteCarloSimulation.py — expects `strategy1.run_backtest(df, init_balance, position_size)` returning P&L list
- VS_0003_test/walkForwardTest.py — standard backtesting.Strategy subclass use
- VS_4000_strategy/VS_4002_20260620_NVDA/2 Strategy/S8002_1_BB.py — structural/code template
- VS_4000_strategy/VS_4003_20260827_ES/1 Prompt/prompt_strategy_S4003_kimi_V1.md — this plan (prompt source)

**Verification**
1. Import test: `from S4003_1 import strategyS4003V1` — passes
2. generate_signals: returns DataFrame with `signal`, `cCrossUpBBTop`, `cCrossDownBBBottom`; pandas index DatetimeIndex; no look-ahead verified via otherTest.lookAheadBiasTest
3. run_backtest: returns list[float] P&L
4. Output artifact filenames and column formats match requirements exactly (col A–O mapping, DD/MM/YYYY hh24:mi format, descending order)
5. HTML contains all 24 metrics; Sharpe annualized rf=3%
6. Compare against AmiBroker HTML report for date range 2026-07-13 to 2026-08-17 on three metrics: trades, net profit, win rate

**Decisions**
- VWAP: port the 50-bar variant (`vwap50_2`) as entry condition reference (user confirmed).
- Sharpe ratio: computed from annualized returns (not daily); risk-free rate = 3%/year (user confirmed).
- Trades CSV filename uses `S4003_1_<YYYYMMDDHH24MISS>.csv` (user confirmed typo fix).
- Class exposes DB fetch helper as classmethod (reads from ticker1Min) so test engines can operate on either CSV or DB-loaded DataFrame
- Signal reversal: when Buy fires while in short position, force cover at trade price then enter long (mirrors AFL `buyForceCover`/`shortForceSell` semantics)
- Backtest entry price = current bar Open (`tradePrice0 = O`); live entry assumed Close
- Stop/profit exits emulate AFL `ApplyStop` active in backtest when `isTesting=True` and `ApplyStop` disabled — manual bar-loop logic must be replicated in Python
- `explore` CSV writes one row per 1-minute bar even when no signal, from the loaded time range in DB

All clarification items resolved — proceeding to implementation.