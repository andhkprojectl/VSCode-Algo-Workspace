# AGENTS.md — Project Context for AI Assistant

## Project Overview
Automated trading system for Mulitple strategy, each strategy trade its own symbols, timeframe, broker
Market data comes from Interactive Brokers (IB) via the TWS/Gateway API.
The system supports three modes: data ingestion, backtesting (local DB), and live trading.

## Tech Stack
- Language: Python 3.11+
- IB connectivity: `ib_insync` (wraps the official IB API; supports `reqHistoricalData` with `keepUpToDate=True`)
- Data storage: SQLite (via SQLAlchemy) — file: `data/es_bars.db`
- Data processing: pandas, numpy
- Config: `config.yaml` (never hardcode credentials)
- Tests: pytest

## Project Structure
```
es-algo-trading/
├── AGENTS.md
├── config.yaml
├── requirements.txt
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── TASKS.md
├── src/
│   ├── main.py              # entry point: mode = ingest | backtest | walkforward | live
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
└── tests/
```

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
- Live orders must pass a `live_trading: true` flag AND a confirmation prompt.
- Max position size and daily loss limit enforced in `order_manager.py` before any order is sent.
- Never call `ib.disconnect()` inside bar callbacks.
