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
│   ├── readme.txt          # description of the work space
│   └── TASKS.md
├── VS_0002_config
│   ├── config.yaml         # global configuration file applied to files in workspace
├── VS_0003_test/
│   ├── backtest.py         # include functions for backtesting, called from strategy file in VS_4000_strategy
│   ├── monteCarloSimulation.py # include functions for monteCarlo, called from strategy file in VS_4000_strategy
│   ├── walkforwardTest.py # include functions for walkforward testing, called from strategy file in VS_4000_strategy
│   └── otherTest.py # include functions for other test,e.g.look ahead test, called from strategy file in VS_4000_strategy
├── VS_0004_broker/
│   ├── IB/
│   │   ├── ib_client.py     # include functions related to IB broker, called from other py file
│   ├── Futu/
│   │   ├── fu_client.py     # include functions related to broker Futu, called from other py file
├── VS_0006_dataFunc/
│   ├── IB/
│   │   ├── ib_client.py     # include functions related to IB broker, called from other py file
│   ├── Futu/
│   │   ├── fu_client.py     # include functions related to broker Futu, called from other py file
├── VS_0007_dbAndFile/
│   ├── mariaDB/
│   │   ├── mariaDb.md     # mark down file for maria DB
│   │   ├── mariaDb.py     # include function operate mariaDb
│   │   ├── sql/
│   │   │   ├── createIBDbUser.sql # Create database schema, user and privileges for IBTradingDb
│   │   │   ├── createTicker1Min.sql # sql create table
│   │   │   ├── createTicker5Min.sql # sql create table
├── VS_4000_strategy/
│   ├── VS_0006_dataFunc/VS_6006_FeatureEngineering
│   ├── VS_4002_20260620_NVDA
│   ├── VS_4003_20260827_ES
│   ├── VS_4004_20260204_NVDA
│   ├── config.py            # load config.yaml
│   ├── data/
│   │   ├── ib_client.py     # IB connection, reqHistoricalData, keepUpToDate subscriptions
│   │   └── database.py      # SQLite read/write for 1-min bars
│   ├── strategies/
│   │   ├── base.py          # StrategyBase abstract class (on_bar -> Signal)
│   │   └── strategy1.py     # Strategy1 implementation
│   ├── backtest/
│   │   ├── engine.py        # event-driven backtester over local DB bars
│   │   └── metrics.py       # PnL, Sharpe, max drawdown, win rate
│   ├── walkforward/
│   │   └── optimizer.py     # in-sample optimize -> out-of-sample validate
│   └── live/
│       ├── live_trader.py   # callback-driven live loop
│       └── order_manager.py # map signals -> IB orders (BUY/SELL/SHORT/COVER)
├── VS_4000_strategy/

└── tests/
VS_0008_liveTrade

## Commands
- Install: `pip install -r requirements.txt`
- Ingest data: `python -m src.main ingest --start 2025-01-01 --end 2025-06-30`
- Backtest: `python -m src.main backtest --strategy strategy1 --period 6M`
- Walk-forward: `python -m src.main walkforward --strategy strategy1 --period 1Y`
- Live: `python -m src.main live --strategy strategy1 --params config/strategy1_wf.json`
- Tests: `pytest -v`

## Code Conventions
- snake_case for files/functions, PascalCase for classes.
- Every strategy must inherit `StrategyBase` and live in its own file under `src/strategies/`.
- Signal enum: `BUY`, `SELL`, `SHORT`, `COVER`, `HOLD` (long/short both supported).
- All timestamps stored in UTC; convert to America/Chicago only for display.
- ES contract: use a helper that always resolves the current front-month contract (rollover).
- Never commit IB credentials. Use `.env` + `config.yaml`.

## Safety / Constraints
- folder VS_0005_monitor, VS_9999_test_program reserved, ignore 
- Live orders must pass a `live_trading: true` flag AND a confirmation prompt.
- Max position size and daily loss limit enforced in `order_manager.py` before any order is sent.
- Never call `ib.disconnect()` inside bar callbacks.
