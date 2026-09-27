# ARCHITECTURE — ES Algo Trading System

## 1. Components

| Component | File | Responsibility |
| --- | --- | --- |
| IBClient | `src/data/ib_client.py` | Connect, request historical bars, subscribe live bars (`keepUpToDate=True`) |
| Database | `src/data/database.py` | SQLite upsert/read of 1-min bars |
| StrategyBase | `src/strategies/base.py` | Abstract `on_bar(df) -> Signal`, shared param loading |
| Strategy1 | `src/strategies/strategy1.py` | Concrete strategy logic |
| BacktestEngine | `src/backtest/engine.py` | Replay bars from DB, apply strategy, simulate fills |
| Metrics | `src/backtest/metrics.py` | PnL, Sharpe, max drawdown, win rate |
| WalkForward | `src/walkforward/optimizer.py` | In-sample optimize -> out-of-sample validate, folds |
| LiveTrader | `src/live/live_trader.py` | Callback loop: new bar -> strategy -> signal -> order |
| OrderManager | `src/live/order_manager.py` | Signal -> IB order mapping, position/risk checks |

## 2. Data Model

Table `bars`:

```javascript
symbol      TEXT   -- e.g. 'ES'
contract_id TEXT   -- resolved front-month contract local symbol
timestamp   TEXT   -- UTC, ISO-8601
open        REAL
high        REAL
low         REAL
close       REAL
volume      INTEGER
PRIMARY KEY (contract_id, timestamp)
```

## 3. Flows

### Ingest

```javascript
main.py ingest -> IBClient.connect -> reqHistoricalData(1 min, range)
-> for each bar -> Database.upsert -> disconnect
```

### Backtest

```javascript
main.py backtest --period 6M -> Database.load_bars(start, end)
-> BacktestEngine.replay(df, Strategy1(params)) -> Metrics.report()
```

### Walk-forward

```javascript
optimizer: for each fold:
  in-sample window -> grid/random search params -> pick best by objective
  out-of-sample window -> evaluate best params -> record
aggregate folds -> if thresholds pass -> save config/strategy1_wf.json
```

### Live

```javascript
IBClient.connect -> reqHistoricalData(keepUpToDate=True)
-> on bar callback: append to rolling DataFrame (keep last N bars)
-> Strategy1.on_bar(df) -> Signal
-> if Signal != HOLD: OrderManager.execute(signal)
  position checks -> build IB MarketOrder -> ib.placeOrder -> track fill
```

## 4. Key Design Decisions

- `ib_insync` for async IB I/O; callbacks run in the event loop — keep strategy work lightweight.
- One contract resolution helper ensures backtest and live use consistent ES front-month symbols.
- Strategy params identical in backtest, walk-forward, and live (single source of truth: JSON file).
- Paper account enforced via config flag `account_type: paper` until explicitly switched.