# PRD — ES Futures 1-Minute Algo Trading System

## 1. Objective

Build a Python system that trades CME ES futures on a 1-minute timeframe using
Interactive Brokers for data and order execution, with local data storage,
backtesting, walk-forward validation, and a callback-driven live trading loop.

## 2. Modes

| Mode | Trigger | Data source | Data saved? |
| --- | --- | --- | --- |
| Ingest | `main.py ingest` | IB `reqHistoricalData` | Yes, local DB |
| Backtest | `main.py backtest` | Local DB | No |
| Walk-forward | `main.py walkforward` | Local DB | No |
| Live | `main.py live` | IB `reqHistoricalData` + `keepUpToDate=True` | Yes (optional log) |

## 3. Functional Requirements

### FR-1 Data Ingestion (non-live)

- Connect to IB TWS/Gateway (paper account first).
- Request 1-minute historical bars for the current ES front-month contract.
- Ingest over configurable periods: 3 months, 6 months, 1 year.
- Save bars to local SQLite DB (`bars` table) with upsert semantics (no duplicates).

### FR-2 Strategy Framework

- Abstract `StrategyBase` class with `on_bar(df) -> Signal`.
- `Strategy1` implemented in its own class/file: `src/strategies/strategy1.py`.
- Strategy parameters loaded from JSON/YAML so backtest and live share the same definition.

### FR-3 Backtesting

- Event-driven backtester reading bars from the local DB.
- Configurable test periods: 3M / 6M / 1Y (via CLI or config).
- Outputs: trade log, equity curve, metrics (net PnL, Sharpe, max drawdown, win rate).
- Backtest may be run multiple times with different periods and parameter sets.

### FR-4 Walk-Forward Test

- For each fold: optimize parameters on in-sample window, validate on out-of-sample window.
- Walk-forward must also support multiple runs: 3M / 6M / 1Y total horizons.
- Output: best parameter set per fold + aggregated out-of-sample performance.
- If performance passes predefined thresholds (e.g., OOS Sharpe > 1.0, max DD < limit), export final parameters to `config/strategy1_wf.json`.

### FR-5 Live Trading

- Data: `ib.reqHistoricalData(..., barSizeSetting='1 min', keepUpToDate=True)`.
- Maintain rolling bars in an in-memory pandas DataFrame (updated in the callback).
- On each new bar (callback trigger), run `Strategy1.on_bar(df)` using the walk-forward parameters.
- On signal:
- `BUY`  -> place BUY market order (entry long)
- `SELL` -> place SELL market order (exit long)
- `SHORT`-> place SELL market order (entry short)
- `COVER`-> place BUY market order (exit short)
- Position tracking to avoid duplicate entries; respect max position size.

## 4. Acceptance Criteria

- [ ] Ingesting 6 months of ES 1-min bars completes without duplicate rows.
- [ ] Backtest on 3M, 6M, and 1Y periods each produce a metrics report.
- [ ] Walk-forward completes all folds and exports `strategy1_wf.json` only when thresholds pass.
- [ ] Live mode (paper account) receives 1-min bars via `keepUpToDate`, triggers Strategy1 per bar, and routes signals to IB orders.
- [ ] All modes run from `src/main.py` with CLI arguments.

## 5. Out of Scope

- Multi-strategy portfolio, optimization of multiple instruments, GUI.