# ARCHITECTURE — ES Algo Trading System

## 1. Components

| Component | File | Responsibility |
| --- | --- | --- |
| Ingest | `VS_6000_DataSource/ibMarketData.py` | Connect, request historical bars, subscribe live bars |
| Strategy | `VS_4003_20260827_ES/2 Strategy/S4003_strategy.py` | Test S4003 strategy. Concrete strategy logic, base on S4003_7_randomforest_glm.py |
| BacktestEngine | `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py` | Replay bars from DB, apply strategy, simulate fills, output backtest result |
| WalkForward | `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py` | In-sample optimize -> out-of-sample validate, folds, output walkforward result |
| LiveTrader | `VS_4003_20260827_ES/6 liveTrade/S4003_live.py` | Callback loop: new bar -> strategy -> signal -> order |

## 2. Data Model

refer to VS_0007_dbAndFile/mariaDB. table ticker1Min. schema IBTradingDb


## 3. Flows

### Ingest

main.py ingest -> VS_6003_GetMarketDataToDb -> ibMarketData ->  getAllTypesTicketDataWithTimeFromIB -> IBClient.connect -> reqHistoricalData(1 min, range) -> save to DB -> disconnect

### Backtest

main.py backtest --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/3 BackTest/S4003_backtest.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name -> call strategy in S4003_strategy.py -> Metrics.report()

### Walk-forward

main.py walkForwardTest --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name. step of IIS , table name, IIS window size, walkforward test parameter range,  -> call strategy in S4003_strategy.py --> optimizer: for each fold:
  in-sample window -> grid/random search params -> pick best by objective
  out-of-sample window -> evaluate best params -> record
aggregate folds -> if thresholds pass -> save config/strategyS4003V1_wf.json

### Other Test
main.py 5 otherTest --strategy 4003 --name lookAhead -> VS_4000_strategy/VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name -> call strategy in S4003_strategy.py --> output html strategy contain look ahead issue or not, detail of look ahead issue
main.py otherTest --strategy 4003 --name monteCarlo -> VS_4000_strategy/VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py -> load config.yaml -> load symbol name, start date, end date, initCapital, commission amount, table name, number of runs (default 5000) -> call strategy in S4003_strategy.py --> output html table of percentile 99%, 95%, 90%, 75%, 50% final equity, annual return, max draw down, %max draw down, equity chart, draw down chart


### Live

main.py live --strategy 4003 -> VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py -> load config.yaml -> load symbol name, start date, end date, positionsize, initCapital, tradePort: tcp/ip port connecting to IB TWS or IB gateway -> IBClient.connect -> reqHistoricalData(keepUpToDate=True) --> call strategy in S4003_strategy.py -> place order -> track fill -> output trade defail transaction to csv, P&L and statistics html


