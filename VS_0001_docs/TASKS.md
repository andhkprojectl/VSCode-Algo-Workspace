# TASKS — Build Order

## Phase 0 — Foundation

- [x] T0.1 Create `VS_4000_strategy/VS_4003_20260827_ES/config.yaml`: local strategy configuration with one section per phase, used by later tasks: `Strategy` (T1.1), `WalkForward Test` (T2.1), `Backtest Test` (T2.2), `LookAhead Test` (T3.1), `MonteCarlo Test` (T3.2), `Live Trade` (T4.1). Parameter list per section follows PRD.md FR-7. Never store credentials in config.yaml; credentials stay in `VS_0002_config/.env`.

- [x] T0.2 Create `config.yaml` (workspace root): global strategy configuration with section `Ingest` (T6.1). Parameter list per section follows PRD.md T6.1. Never store credentials in config.yaml; credentials stay in `VS_0002_config/.env`.


## Phase 1 — Strategy

- [x] T1.1 Implement `VS_4003_20260827_ES/2 Strategy/S4003_strategy.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_7_randomforest_glm.py`,  copy strategy class used in S4003_7_randomforest_glm.py to this file, rename strategy class `strategyS4003V1` to `strategyS4003` (only store strategy class `strategyS4003`).  All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Strategy`. Do not depends on files within folder `1 Prelimary Test` anymore.

## Phase 2 — Back Test and Walkforward Test

- [] T2.1 Implement `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_7_randomforest_glm.py`, copy walkforward function used in `S4003_7_randomforest_glm.py` to this file, call strategy class in file `S4003_strategy.py` to perform walkforward test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `WalkForward Test`. Do not depends on files within folder `1 Prelimary Test` anymore. Walkforward test output file from `S4003_walkforward.py` same as S4003_7_randomforest_glm.py, change to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/walkForwardResult`. Data got from mariaDb, table ticker1Min. logic follow FR-4. In html output, show result of walkforward parameter w1 (profit take weight), n1 (max open position period) after testing IIS and going to apply to next day OOS. IIS window counts trading days (last iisDays trading days before each OOS day), not calendar days. Config `WalkForward Test` has parameter `endDateIsOOS`: 1 = analysisEnd is the last OOS day; 0 = analysisEnd is the last IIS day and the results CSV/HTML get one extra final row with blank OOS date carrying w1/n1 for the next trading day (OOS columns blank until bars exist in MariaDB)

- [x] T2.2 Implement `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py`: Base on strategy in file `VS_4003_20260827_ES\1 Prelimary Test\S4003_4_randomforest_glm.py`, copy backtest function used in `S4003_4_randomforest_glm.py` to this file, call strategy class in file `S4003_strategy.py` to perform backtest . All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Backtest Test`. Do not depends on files within folder `1 Prelimary Test` anymore. BackTest output file from `S4003_4_randomforest_glm.py` same as S4003_7_randomforest_glm.py. Data got from mariaDb, table ticker1Min



## Phase 3 — Other Test

- [x] T3.1 Implement `VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py`: Call strategy class in file `S4003_strategy.py` to perform look ahead test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `LookAhead Test`. Do not depends on files within folder `1 Prelimary Test` anymore. Look Ahead test output file from `S4003_lookAhead.py` to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/lookAheadTestResult`, output html strategy contain look ahead issue or not, detail of look ahead issue. Data got from mariaDb, table ticker1Min

- [x] T3.2 Implement `VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py`: Call strategy class in file `S4003_strategy.py` to perform monteCarlo test. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `MonteCarlo Test`. Do not depends on files within folder `1 Prelimary Test` anymore. monteCarlo test output file from `S4003_monteCarlo.py` to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/monteCarloResult`, output html table of percentile 99%, 95%, 90%, 75%, 50% final equity, annual return, max draw down, %max draw down, equity chart, draw down chart. Data got from mariaDb, table ticker1Min

## Phase 4 — Live Trading

- [x] T4.1 Implement `VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py`: call IB function reqHistoricalData(keepUpToDate=True), if bar change, call back function call logic base on `S4003_walkforward.py`, parameter w1, n1 got from config file config.yaml. If buy/short/sell/cover signal generate, call function `doTrade0` in file `VS_0008_liveTrade/ibAutotrade.py` to place order. All parameter store in config file `VS_4003_20260827_ES/config.yaml` under section called `Live Trade`. There is a parameter tradePort: tcp/ip port connecting to IB TWS or IB gateway. Do not depends on files within folder `1 Prelimary Test` anymore. live trade output file to folder `C:/Project/ProjectLife/VSCode Algo Workspace DataFile/VS_4003_20260827_ES/liveTrade`, output files include log file storing each buy/short/sell/cover record, debug log from doTrade0

## Phase 5 — Hardening

- [ ] T5.1 Unit tests: database, strategy signals, order mapping (pytest)
- [x] T5.2 README with run instructions for all modes

## Phase 6 — Ingest
- [x] T6.1 implement `VS_6003_GetMarketDataToDb/1min/ingest.py`: logic and function same as DB_NQ_20250101_20260702_1Min.py. When call function ibMarketData.getAllTypesTicketDataWithTimeFromIB, instead of hard code, move parameters symbolName, startDate, startTime, endDate, endTime, period1, tickerType, isConFuture, futureExpireDate, futureExchange to global configuration file `config.yaml` (workspace root), under section Ingest

## Phase 7 — CLI command py file

- [x] T7.1 Implement `ingest.py`: wrapper for ingest mode. When run this program, actually call `VS_6003_GetMarketDataToDb/1min/ingest.py`, see T6.1

- [x] T7.2 Implement `backtest.py`: Wrapper for backtesting mode. CLI arguments: `--strategy <id>`, actually call backtest file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `backtest.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py`

- [x] T7.3 Implement `walkforward.py`: Wrapper for walkforward mode. CLI arguments: `--strategy <id>`, actually call walkforward test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `walkforward.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py`

- [x] T7.4 Implement `lookAhead.py`: Wrapper for lookAhead mode. CLI arguments: `--strategy <id>`, actually call lookAhead test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `lookAhead.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py`

- [x] T7.5 Implement `monteCarlo.py`: Wrapper for monteCarlo mode. CLI arguments: `--strategy <id>`, actually call monteCarlo test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `monteCarlo.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py`

- [ ] T7.6 Implement `liveTrade.py`: Wrapper for liveTrade mode. CLI arguments: `--strategy <id>`, actually call liveTrade test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `liveTrade.py` with parameter --strategy S4003, actually run file `VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py`