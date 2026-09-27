# TASKS — Build Order

## Phase 1 — Foundation

- [ ] T1.1 Create repo, `requirements.txt` (ib_insync, pandas, sqlalchemy, pyyaml, pytest), `config.yaml`
- [ ] T1.2 Implement `src/config.py` loader + `.env` handling for IB credentials
- [ ] T1.3 Implement `src/data/database.py`: SQLite `bars` table + upsert + `load_bars(start, end)`

## Phase 2 — Data

- [ ] T2.1 Implement `src/data/ib_client.py`: connect/disconnect, front-month ES contract resolver
- [ ] T2.2 Implement `ingest` mode: reqHistoricalData for 3M/6M/1Y -> upsert to DB
- [ ] T2.3 Verify ingested data: row counts, no duplicates, UTC timestamps

## Phase 3 — Strategy & Backtest

- [ ] T3.1 Implement `src/strategies/base.py` (Signal enum, StrategyBase abstract class)
- [ ] T3.2 Implement `src/strategies/strategy1.py` with configurable parameters
- [ ] T3.3 Implement `src/backtest/engine.py` (bar replay, fill simulation, long+short)
- [ ] T3.4 Implement `src/backtest/metrics.py` and backtest CLI (`--period 3M|6M|1Y`)
- [ ] T3.5 Run backtests for 3M, 6M, 1Y; save reports under `results/`

## Phase 4 — Walk-Forward

- [ ] T4.1 Implement `src/walkforward/optimizer.py` (folds, in-sample optimize, OOS validate)
- [ ] T4.2 Define pass thresholds (OOS Sharpe > 1.0, max DD < limit) in `config.yaml`
- [ ] T4.3 Run walk-forward for 3M, 6M, 1Y horizons; export `config/strategy1_wf.json` if passed

## Phase 5 — Live Trading

- [ ] T5.1 Implement `src/live/order_manager.py`: Signal -> BUY/SELL/SHORT/COVER IB orders, position + risk checks
- [ ] T5.2 Implement `src/live/live_trader.py`: `keepUpToDate=True` subscription, rolling DataFrame, per-bar strategy call
- [ ] T5.3 Paper-account live test: verify bars arrive, signals fire, orders fill
- [ ] T5.4 Add optional live bar logging to DB; add graceful shutdown (cancel subscriptions, flatten flag)

## Phase 6 — Hardening

- [ ] T6.1 Unit tests: database, strategy signals, order mapping (pytest)
- [ ] T6.2 README with run instructions for all four modes