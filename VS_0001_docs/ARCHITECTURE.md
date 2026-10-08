# ARCHITECTURE — ES Algo Trading System

## 1. Components

| Component | File | Responsibility |
| --- | --- | --- |
| Config | `VS_4003_20260827_ES/config.yaml` (planned - TASKS.md T0.2) | Strategy-local parameters for all modes; credentials only in `VS_0002_config/.env` |
| Ingest | `VS_6000_DataSource/ibMarketData.py` | Connect, request historical bars, subscribe live bars |
| Strategy | `VS_4003_20260827_ES/2 Strategy/S4003_strategy.py` | Strategy class `strategyS4003`. Concrete strategy logic, base on S4003_7_randomforest_glm.py |
| BacktestEngine | `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py` | Replay bars from DB, apply strategy, simulate fills, output backtest result |
| WalkForward | `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py` | In-sample optimize -> out-of-sample validate, folds, output walkforward result |
| LiveTrader | `VS_4003_20260827_ES/6 liveTrade/S4003_live.py` | Callback loop: new bar -> strategy -> signal -> order |
| OrderExec | `VS_0008_liveTrade/ibAutotrade.py` | `doTrade0`: place and track IB orders for BUY/SELL/SHORT/COVER signals |

## 2. Data Model

refer to VS_0007_dbAndFile/mariaDB. schema IBTradingDb. tables ticker1Min (1-min bars) and ticker5Min (5-min bars)

## 3. Strategy Interface

- Strategy class `strategyS4003` in `S4003_strategy.py`; one strategy class per strategy folder
- Input: bars DataFrame (from MariaDB for tests, from IB live stream for live trade) plus parameters from the strategy `config.yaml`
- Output per bar: one signal of `BUY`, `SELL`, `SHORT`, `COVER`, `HOLD`
- The same strategy class is reused by backtest, walk-forward, other test and live trade

## 4. Flows

### Ingest

`ingest.py` -> VS_6003_GetMarketDataToDb -> ibMarketData ->  getAllTypesTicketDataWithTimeFromIB -> IBClient.connect -> reqHistoricalData(1 min, range) -> save to DB -> disconnect

### Backtest

`backtest.py` --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/3 BackTest/S4003_backtest.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name -> call strategy in S4003_strategy.py -> Metrics.report()

### Walk-forward

`walkforward.py` --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name, step of IIS, IIS window size (trading days), endDateIsOOS (1 = analysisEnd is last OOS day; 0 = analysisEnd is last IIS day -> extra final results row selects w1/n1 for the next trading day, OOS columns blank), walkforward test parameter range -> call strategy in S4003_strategy.py --> optimizer: for each fold:
  in-sample window (last iisDays trading days before the OOS day) -> grid/random search params -> pick best by objective
  out-of-sample window -> evaluate best params -> record
aggregate folds -> if thresholds pass -> save config/strategyS4003_wf.json


### Other Test
`lookAhead.py` --strategy 4003 --name lookAhead -> VS_4000_strategy/VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name -> call strategy in S4003_strategy.py --> output html strategy contain look ahead issue or not, detail of look ahead issue
`monteCarlo.py` --strategy 4003 --name monteCarlo -> VS_4000_strategy/VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name, number of runs (default 5000) -> call strategy in S4003_strategy.py --> output html table of percentile 99%, 95%, 90%, 75%, 50% final equity, annual return, max draw down, %max draw down, equity chart, draw down chart

### Live

`liveTrade.py` --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py -> load config.yaml -> load symbol name, start date, end date, positionsize, initCapital, tradePort: tcp/ip port connecting to IB TWS or IB gateway -> IBClient.connect -> reqHistoricalData(keepUpToDate=True) --> call strategy in S4003_strategy.py -> place order -> track fill -> output trade defail transaction to csv, P&L and statistics html


