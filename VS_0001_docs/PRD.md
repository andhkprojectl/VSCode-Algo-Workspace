# PRD — ES Futures 1-Minute Algo Trading System

## 1. Objective

Automated trading system for Multiple strategies, each strategy trades its own symbols, timeframe, broker.
Market data comes from Interactive Brokers (IB) via the TWS/Gateway API
The system supports 5 modes: data ingestion, backtesting, walk forward test, other test, and live trading.
Some files already exists in the workspace. Keep files unchanged if file name not specific in project structure

## 2. Modes

| Mode | Trigger | Data source | Data saved? |
| --- | --- | --- | --- |
| Ingest | `ingest.py` | IB `reqHistoricalData` | Yes, local DB |
| Backtest | `backtest.py` | Local DB | No |
| Walk-forward | `walkforward.py` | Local DB | No |
| lookAhead | `lookAhead.py` | Local DB | No |
| monteCarlo | `monteCarlo.py` | Local DB | No |
| Live | `liveTrade.py` | IB `reqHistoricalData` + `keepUpToDate=True` | Yes (optional log) |

## 3. Functional Requirements

### FR-1 Data Ingestion (non-live)

- Connect to IB TWS/Gateway (paper account first).
- Multiple programs, each program get its own symbol name, start date, end date, time frame (e.g. 1min, 5min, etc)
- Each program has a section in configuration file config.yaml with its own parameter
- parameter include symbol name, start date, end date, time frame, and output table name
- data save to mariaDb
- under folder VS_0006_dataFunc/VS_6003_GetMarketDataToDb
- run `ingest.py`


### FR-2 Strategy Framework

- Support multiple strategies. Each strategy has its own folder under parent folder
- parent folder is VS_4000_strategy
- strategy folder name is VS_4XXX_<YYYYMMDD>_<SYMBOL_NAME>, e.g. VS_4003_20260827_ES
- Some py programs for prelimiary test, in 1 Prelimary Test. Files in this folder can keep unchange
- strategy file put in folder "2 Strategy". strategy file name S4XXX_strategy.py, e.g. S4003_strategy.py
- strategy file has a class strategyS4XXX, e.g. strategyS4003
- Backtest, walkforward test, other test, live trade program call strategy class to trigger trading signal
- Trading signal is triggered, generate buy/sell/short/cover signal
- Parameter passed to strategy include symbol name, start date, end date, time frame, and position size
- Parameters save in strategy folder config.yaml
- testing or live data pass to strategy class for analysis 


### FR-3 Backtesting

- each strategy has its own backtest program in "3 BackTest" folder
- Backtest file name S4XXX_backtest.py, e.g. S4003_backtest.py
- run `backtest.py`
- backtest get data from folder VS_6004_GetDataFromMariaDb/VS_6004_GetDataFromMariaDb.py
- Parameter is symbol name, start date, end date, table name
- Output folder C:\Project\ProjectLife\VSCode Algo Workspace DataFile\<strategy folder name>\backTestResult
- Output file: comparison csv, parameter analysis, parameter value, backtest result (html format)
- refer to file in backTestResult\glm\8 for reference backtest result
- Backtest may be run multiple times with different periods and parameter sets.


### FR-4 Walk-Forward Test

- For each fold: optimize 2 parameters in-sample window, validate on out-of-sample window
- 2 parameters are profit take weight and profit take period
- 1st parameter is profit take weight, variable w1. profit take price = buy/short price +- w1*(atr 14 period, previous bar)
- w1 range from 1.5 to 3.0, step 0.5
- 2nd parameter is max open position period, variable n1. any open position must be closed after n1 bars.
- n1 range from 10 to 25, step 5
- Walk-forward test uses rolling window approach with overlapping in-sample (IIS) and out-of-sample (OOS) periods.
- IIS/OOS window size are in strategy config file: IIS window size: 14 trading days, OOS window size: 1 trading day
- IIS window size counts trading days (days with bars in the DB), not calendar days: the in-sample window of an OOS day is the last N trading days strictly before it
- config parameter endDateIsOOS: 1 = analysisEnd is the last OOS day (last results row has OOS date = analysisEnd); 0 = analysisEnd is the last IIS day (results get one extra final row with blank OOS date, carrying the w1/n1 selected for the next trading day; its OOS columns stay blank until that day has bars in MariaDB)
- start date, end date, step of IIS are configured in the strategy config file
- each strategy has its own backtest program in "4 Walk Forward Test" folder
- Walkforward file name S4XXX_walkforward.py, e.g. S4003_walkforward.py
- run `walkforward.py`
- walkForwardTest get data from folder VS_6004_GetDataFromMariaDb/VS_6004_GetDataFromMariaDb.py
- Parameter is symbol name, start date, end date, table name
- Output folder C:\Project\ProjectLife\VSCode Algo Workspace DataFile\<strategy folder name>\walkForwardResult
- Output file: comparison csv, parameter analysis, parameter value, walkForwardTest result (html format)
- Output: best parameter set per fold + aggregated out-of-sample performance.
- in html, please show result of walkforward parameter w1, n1 after testing IIS and going to apply to next day OOS


### FR-5 Other Test
- each strategy has its own other test program in "5 OtherTest" folder
- Other Test include 2 types: Look Ahead and MonteCarlo
- Other test file name S4XXX_lookAhead.py and S4XXX_monteCarlo.py, e.g. S4003_lookAhead.py and S4003_monteCarlo.py
- run `lookAhead.py`, `monteCarlo.py`
- otherTest get data from folder VS_6004_GetDataFromMariaDb/VS_6004_GetDataFromMariaDb.py
- Parameter is symbol name, start date, end date, table name, and test type
- Output folder C:\Project\ProjectLife\VSCode Algo Workspace DataFile\<strategy folder name>\otherTestResult
- Output file: lookAhead|monteCarlo test result in html format. 
- lookAhead output html file show whether the strategy has look ahead bias (future data leakage) by comparing predictions with actual future price movements.
- monteCarlo output html file show probability of success and max drawdown from simulated random price paths.
- refer to file in otherTestResult\glm\8 for reference other test result
- Other test may be run multiple times with different periods and parameter sets.
- lookAhead test perform test, try to find out is strategy has look ahead issue
- MonteCarlo test perform test, simulate random price paths to estimate probability of success and find max drawdown

### FR-6 Live Trading
- Data: `ib.reqHistoricalData(..., barSizeSetting='1 min', keepUpToDate=True)`.
- Maintain rolling bars in an in-memory pandas DataFrame (updated in the callback).
- On each new bar (callback trigger), trigger buy/sell/short/cover signal base on walkforward file S4003_walkforward.py, with fix w1, n1 set in config.yaml.
- On signal:
- `BUY`  -> place BUY market order (entry long)
- `SELL` -> place SELL market order (exit long)
- `SHORT`-> place SELL market order (entry short)
- `COVER`-> place BUY market order (exit short)
- Position is parameter in strategy local configure file. default 1 contract
- Order execution via IB TWS/Gateway API (function `doTrade0` in `VS_0008_liveTrade/ibAutotrade.py`)
- Maintain position tracking and risk limits
- Log trade execution details to database


### FR-7 Configuration
- each strategy folder has its own config.yaml, include below parameter
    - symbol name, start date, end date, time frame, and position size
    - commission amount (USD) per trade or per contract
    - initial capital
    - backtest, walkforward test, other test result output folder name
    - walkforward test parameters: IIS start date, IIS end date, OOS step, rolling window:true/false, parameter 1, parameter 2 for tuning parameters, endDateIsOOS (1 = analysisEnd is last OOS day, 0 = analysisEnd is last IIS day)
    - liveTrade=true/false

## 4. Acceptance Criteria

- [ ] Ingesting 1 year of ES 1-min bars completes without duplicate rows. No missing bar
- [ ] Backtest on 3M, 6M, and 1Y periods each produce a metrics report.
- [ ] Walk-forward completes all folds and exports. IIS period show best parameter for OOS testing
- [ ] Live mode (paper account) receives 1-min bars via `keepUpToDate`, triggers strategyS4003 per bar, and routes signals to IB orders.

## 5. Out of Scope

- Monitoring dashboard UI (`VS_0005_monitor` is reserved and not part of the 5 modes).
- Experimental / playground programs (`VS_9999_test_program` is reserved).
- Brokers other than Interactive Brokers; all order flow goes through IB TWS/Gateway.
- Market-data sources other than IB for the 5 modes (`yFinanceMarketData.py` is a helper tool only).
- Portfolio-level capital allocation or cross-strategy risk management; each strategy runs standalone with its own config.yaml.
- Real-money live trading before paper-account validation; live mode requires backtest, walk-forward and other tests to pass first.

