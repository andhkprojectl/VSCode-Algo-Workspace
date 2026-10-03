# TASKS — Build Order

## Phase 1 — Strategy

- [ ] T1.1 Implement `VS_4003_20260827_ES/2 Strategy/S4003_strategy.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_7_randomforest_glm.py`,  copy strategy class used in S4003_7_randomforest_glm.py to this file, rename strategy class `strategyS4003V1` to `strategyS4003` (only store strategy class `strategyS4003`).  All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Strategy`. Do not depends on files within folder `1 Prelimary Test` anymore.

## Phase 2 — Back Test and Walkforward Test

- [ ] T2.1 Implement `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_7_randomforest_glm.py`, copy walkforward function used in `S4003_7_randomforest_glm.py` to this file, call strategy class in file `S4003_strategy.py` to perform walkforward test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `WalkForward Test`. Do not depends on files within folder `1 Prelimary Test` anymore. Walkforward test output file from `S4003_walkforward.py` same as S4003_7_randomforest_glm.py, change to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/walkForwardTestResult`. Data got from mariaDb, table ticker1Min

- [ ] T2.2 Implement `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_4_randomforest_glm.py`, copy backtest function used in `S4003_4_randomforest_glm.py` to this file, call strategy class in file `S4003_strategy.py` to perform backtest . All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Backtest Test`. Do not depends on files within folder `1 Prelimary Test` anymore. BackTest output file from `S4003_4_randomforest_glm.py` same as S4003_7_randomforest_glm.py. Data got from mariaDb, table ticker1Min



## Phase 3 — Other Test

- [ ] T3.1 Implement `VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py`: Call strategy class in file `S4003_strategy.py` to perform look ahead test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `LookAhead Test`. Do not depends on files within folder `1 Prelimary Test` anymore. Look Ahead test output file from `S4003_lookAhead.py` to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/lookAheadTestResult`, output html strategy contain look ahead issue or not, detail of look ahead issue. Data got from mariaDb, table ticker1Min

- [ ] T3.2 Implement `VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py`: Call strategy class in file `S4003_strategy.py` to perform monteCarlo test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `MonteCarlo Test`. Do not depends on files within folder `1 Prelimary Test` anymore. monteCarlo test output file from `S4003_monteCarlo.py` to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/monteCarloResult`, output html table of percentile 99%, 95%, 90%, 75%, 50% final equity, annual return, max draw down, %max draw down, equity chart, draw down chart. Data got from mariaDb, table ticker1Min

## Phase 4 — Live Trading

- [ ] T4.1 Implement `VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py`: call IB function reqHistoricalData(keepUpToDate=True), if bar change, call back function call strategy class in file `S4003_strategy.py`. If buy/short/sell/cover signal generate, call function `doTrade0` in file `VS_0008_liveTrade/ibAutotrade.py` to place order. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Live Trade`. There is a parameter tradePort: tcp/ip port connecting to IB TWS or IB gateway. Do not depends on files within folder `1 Prelimary Test` anymore. live trade output file to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/liveTrade`, output files include log file storing each buy/short/sell/cover record, debug log from doTrade0

## Phase 5 — Hardening

- [ ] T5.1 Unit tests: database, strategy signals, order mapping (pytest)
- [ ] T5.2 README with run instructions for all five modes