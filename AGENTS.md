# AGENTS.md — Project Context for AI Assistant

## Project Overview
Automated trading system for Multiple strategies, each strategy trades its own symbols, timeframe, broker.
Market data comes from Interactive Brokers (IB) via the TWS/Gateway API
The system supports 5 modes: data ingestion, backtesting, walk forward test, other test, and live trading.
Some files already exists in the workspace. Keep files unchanged if file name not specific in project structure

## Tech Stack
- Language: Python 3.13+
- IB connectivity: Either via IB gateway or IB TWS
- Data storage: MariaDB. Already exists MariaDB. Schema, user and table structure refer to `VS_0007_dbAndFile\mariaDB`
- Data processing: pandas, numpy
- Config: Global configuration file  and local strategy configuration file. File `config.yaml` (never hardcode credentials)
- Tests: pytest

## Project Structure
```
VSCode Algo Workspace/
├── AGENTS.md
├── VS_0001_docs
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── TASKS.md
├── VS_0002_config
│   └── config.yaml # environment variables (IB / MariaDB ),credentials), not exists yet, need later create
├── VS_0003_test
│   ├── backtest.py             # backtest functions, called from strategy files in VS_4000_strategy
│   ├── monteCarloSimulation.py # Monte Carlo functions, called from strategy files in VS_4000_strategy
│   ├── walkForwardTest.py     # walk-forward test functions, called from strategy files in VS_4000_strategy
│   └── otherTest.py            # other tests (e.g. look-ahead test), called from strategy files in VS_4000_strategy
├── VS_0004_broker
│   ├── IB                   
│   │   ├── ib_client.py # IB API client for TWS/Gateway connectivity, create later
│   └── Futu
│       └── fu_client.py         # Futu API client, create later
├── VS_0006_dataFunc            # data/feature programs that serve other programs in the workspace
│   ├── VS_6000_DataSource
│   │   └── ibMarketData.py             # get market data from IB and save to local MariaDB
│   ├── VS_6001_GetMarketDataToCsv
│   │   ├── NVDA_20250101_20260430_5Min.py      # get 5-min NVDA data from IB, save to csv
│   │   └── NVDA_20260101_20260615_1Min.py      # get 1-min NVDA data from IB, save to csv
│   ├── VS_6003_GetMarketDataToDb
│   │   ├── 1min
│   │   │   └── DB_NQ_20250101_20260702_1Min.py     # 1-min NQ data from IB -> MariaDB ticker1Min
│   │   └── 5min
│   │       ├── DB_NQ_20250101_20260702_5Min.py     # 5-min NQ data from IB -> MariaDB ticker5Min
│   │       └── DB_NVDA_20250101_20260430_5Min.py   # 5-min NVDA data from IB -> MariaDB ticker5Min
│   ├── VS_6004_GetDataFromMariaDb
│   │   └── VS_6004_GetDataFromMariaDb.py   # load_1Min_data_from_db: read 1-min bars from MariaDB
│   ├── VS_6005_GetLiveTradeData
│   │   ├── VS_6005_1_1min_glm.py           # getLiveTradeIB1MinData: consolidate MariaDB + IB live 1-min bars into df
│   │   ├── VS_6005_2_1min_glm.md           # requirement doc for VS_6005_1
│   └── VS_6006_FeatureEngineering
│       ├── F6006_FeatureScoreRanker_1_glm.py   # feature score ranker for strategies in VS_4000_strategy
│       ├── F6006_GeneralFeature_1_kimi.py      # feature engineering (kimi variant)
│       ├── F6006_GeneralFeature_2_glm.py       # feature engineering (glm variant)
│       └── F6006_GeneralFeature_2_glm.md       # requirement doc for F6006_GeneralFeature_2_glm.py
├── VS_0007_dbAndFile
│   └── mariaDB
│       ├── mariaDb.md              # doc for mariaDb.py
│       ├── mariaDb.py              # MariaDB functions (saveDfToTicker5Min etc.)
│       └── sql
│           ├── createIBDbUser.sql      # create database schema, user and privileges for IBTradingDb
│           ├── createTicker1Min.sql    # create ticker1Min table
│           └── createTicker5Min.sql    # create ticker5Min table
├── VS_4000_strategy
│   ├── VS_4002_20260620_NVDA
│   │   ├── 1 Prelimary Test
│   │   │   ├── open_router_token_cost.docx
│   │   │   ├── plan_backtest_S8002_V1.txt
│   │   │   └── plan_strategy_S8002_V1.txt
│   │   ├── 2 Strategy
│   │   │   └── S8002_1_BB.py           # Bollinger Band strategy
│   │   └── 3 BackTest                  # (empty - reserved)
│   ├── VS_4003_20260827_ES             # ES intraday strategy family
│   │   ├── 1 Prelimary Test
│   │   │   ├── prompt_strategy_S4003_V1.txt        # original requirement
│   │   │   ├── prompt_strategy_S4003_kimi_V1.md    # kimi plan doc
│   │   │   └── prompt_strategy_S4003_fm_glm_V1.md  # glm plan doc
│   │   ├── 2 Strategy
│   │   │   ├── S4003_1_glm.py                  # base engine: strategy logic + MariaDB loader (AFL parity port)
│   │   │   ├── S4003_1_kimi.py                 # kimi variant of the base engine
│   │   │   ├── S4003_2_randomforest_glm.py     # RandomForest ML gate on base signals
│   │   │   ├── S4003_2_randomforest_kimi.py    # kimi variant
│   │   │   ├── S4003_3_randomforest_glm.py     # shared helpers: param grids, walk-forward utils, exit cost
│   │   │   ├── S4003_3_randomforest_kimi.py    # kimi variant
│   │   │   ├── S4003_4_linearRegression_glm.py # OLS gate variant
│   │   │   ├── S4003_4_randomforest_glm.py     # daily stop-loss/take-profit test (stn1/sln1)
│   │   │   ├── S4003_5_randomforest_glm.py     # walk-forward helpers: simulate_window, metrics, sharpe
│   │   │   ├── S4003_6_randomforest_glm.py     # target-driven WF with OOS-day gate
│   │   │   ├── S4003_7_randomforest_glm.py     # WF with OOS-day gate + calibration window (IS 14d)
│   │   │   ├── S4003_8_randomforest_glm.py     # rolling WF: IIS 60d -> OOS 7d, step 1 day, 1-year span
│   │   │   ├── kimi variants: S4003_{4_linearRegression,4_randomforest,5_linearRegression,5_randomforest,6_randomforest}_kimi.py, S4003_{7,8}_randomforest_kimi.py
│   │   │   ├── amibroker
│   │   │   │   ├── backtest_4003_1 trim_4_python.afl   # AmiBroker AFL source of the port
│   │   │   │   ├── backtest_4003_1 trim_4_python.apx   # AmiBroker analysis project
│   │   │   │   └── backtest_4003_1 - Backtest Report.html
│   │   │   ├── run_log.txt
│   │   │   └── requirement variants: S4003_{2..8}_*.txt (glm/kimi ur, OLS/RF per program)
│   │   ├── 5 OtherTest                     # (empty - reserved)
│   │   └── 7 liveTrade
│   │       └── L4003_1_randomForest_glm.txt    # live-trade requirement for the RF model
│   └── VS_4004_20260204_NVDA           # NVDA intraday strategy family
│       ├── 1 Prelimary Test
│       │   ├── plan-nvdaFeatureAnalysis.prompt.md
│       │   ├── prompt_backtest_S8001.md            # backtest requirement
│       │   ├── prompt_lookAheadTest_S8001_V1.md    # look-ahead test requirement
│       │   ├── prompt_montecarlo_S8001_V1.md       # Monte Carlo test requirement
│       │   ├── prompt_strategy_S8001.md            # strategyIRB1000_V1 consolidated requirement (V1-V3)
│       │   └── readMe.md
│       ├── 2 Strategy
│       │   ├── S8001_1_ConvertInCubationNVDAByGemini.py
│       │   ├── S8001_2_ConvertFromGemini.py        # NVDA port of the IB autotrade program
│       │   ├── S8001_3_TestConvertPromptGemini.py
│       │   ├── S8001_4_GenerateFromPromptQwen37Max.py  # strategyIRB1000_V1 implementation
│       ├── 3 BackTest
│       │   ├── backtest_S8004_V1.py
│       ├── 4 Walk Forward Test             # (empty - reserved)
│       ├── 5 OtherTest
│       │   ├── lookAheadTest_S8004_v1.py   # look-ahead test for S8001_4
│       │   └── test_terminal.py
│       ├── 6 liveTrade                     # (empty - reserved)
│       ├── config.yaml                     # local strategy configuration
│       └── readMe.txt
├── VS_0008_liveTrade
│   ├── IB
│   │   ├── displayIBLiveInfo.py    # display IB live info (see displayLiveInfo.txt)
│   │   ├── displayLiveInfo.txt     # description for displayIBLiveInfo.py
│   │   └── autotrade
│   │       ├── VS_8101_autotrade.md
│   │       ├── VS_8101_autotrade.py        # doTrade0: IB bracket-order trading (port of autoTrade.afl doTrade00)
│   │       ├── L8001_autotrade_step.txt    # build notes
```

## Commands
- Ingest data: `python -m main ingest --period 1min --start 2025-01-01 --end 2025-06-30` 
- Backtest: `python -m main backtest --strategy 4003`
- Walk-forward: `python -m main walkforward --strategy 4003`
- Other Test (Look Ahead): `python -m main otherTest --strategy 4003 --name lookAhead`
- Other Test (MonteCarlo): `python -m main otherTest --strategy 4003 --name monteCarlo`
- Live: `python -m main live --strategy 4003`


## Code Conventions
- snake_case for folder, PascalCase for classes, camelCase for functions
- Every strategy has a individual folder under VS_4000_strategy. The strategy is an individual class within a py program
- Each strategy folder has its own config.yaml configuration file and its own readme.txt to describe the strategy
- Each strategy folder has its own backtest, walkforward test, other test folder, generate output to its own folder
- Each strategy folder has its own live trade folder, run its own strategy class for live trade
- Backtet, walkforward, other test, live trade are indvidual functions and able to run strategy class
- Signal enum: `BUY`, `SELL`, `SHORT`, `COVER`, `HOLD` (long/short both supported).
- All timestamps stored in UTC; convert to America/Chicago only for display.
- Never commit IB credentials. Use `.env` + `config.yaml`.

## Safety / Constraints
- folder VS_0005_monitor, VS_9999_test_program reserved, ignore 
- Before live trade, must pass backtest, walkforward test, other test
