# Plan: S8002_1_BB.py — Bollinger Band Cross Strategy (NVDA 5-min)

**TL;DR** — Create `VS_8000_Strategy/VS_8002_20260620_NVDA/2 Strategy/S8002_1_BB.py` with class `strategyS8002BBV1` implementing a BB-cross strategy on NVDA 5-min bars. The class subclasses `backtesting.Strategy` (for `backtest.py`/`walkForwardTest.py`/`monkeyTest`) and additionally exposes `generate_signals(df)` and `run_backtest(df, init_balance, position_size)` methods (for `lookAheadBiasTest`/`monteCarloSimulation`). Live trade (item 4) is excluded per your instruction.

---

## Confirmed decisions

- **Fill timing**: Order placed when current bar `buy00` true; `backtesting.py` fills at NEXT bar Open (matches "previous bar buy00" semantics).
- **Stop order**: Backtest uses `sl=stop_price` (market-when-touched). The 0.06 limit offset is live-only and excluded from backtest.
- **Signal methods**: `generate_signals(df)` returns `signal` col (1/-1/0); `run_backtest(...)` simulates full exits (5-bar, ATR stop, 15:55 force close) and returns per-trade P&L list.
- **Live trade**: IGNORED per user (item 4 TBD).

---

## Steps

### Phase 1 — Class skeleton & dual-mode support

1. Create `S8002_1_BB.py` with imports (`pandas`, `numpy`, `backtesting.Strategy`).
2. Define `strategyS8002BBV1(Strategy)` with class attributes: `bb_period=20`, `bb_std=2`, `atr_period=7`, `atr_mult=2.5`, `hold_bars=5`, `position_size=110`, `commission=0.0`, `slippage=0.03`, `is_live=False`, `entry_start='09:30'`, `entry_end='14:55'`, `force_close_time='15:55'`.

### Phase 2 — Indicator computation

3. `init()`: compute `bbTop`, `bbBottom` (rolling mean/std on Close, period 20, std 2), `atr7` (pure pandas TR rolling mean). Register via `self.I()`. Init `self._entry_bar = None`.
4. `staticmethod _compute_indicators(df)`: pure-pandas version (no `backtesting.py` dependency) returning df with `bbTop`, `bbBottom`, `atr7`, `cCrossUpBBTop`, `cCrossDownBBBottom`:
   - `cCrossUpBBTop = (prev Close <= bbTop) & (cur Close > bbTop)`
   - `cCrossDownBBBottom = (prev Close >= bbBottom) & (cur Close < bbBottom)`

### Phase 3 — backtesting.py `next()` logic

5. `next()`:
   - **Buy**: if no position AND prev-bar `cCrossUpBBTop` true AND time in [09:30, 14:55] → `self.buy(size=position_size, sl=Close[-1] - atr7[-2]*2.5)`. Record entry bar.
   - **Short**: symmetric → `self.sell(size=position_size, sl=Close[-1] + atr7[-2]*2.5)`.
   - **5-bar exit**: if bars since entry ≥ 5 → `self.position.close()`.
   - **Force close at 15:55**: if time == 15:55 and in position → close.
   - `backtesting.py` fills at next bar Open automatically.

### Phase 4 — `generate_signals(df)` for lookAheadBiasTest

6. Parse datetime index, call `_compute_indicators`, compute `buy00`/`short00`, shift by 1 (signal triggers on prev bar), apply time filter, return df with `signal` col (1=buy, -1=short, 0=hold). Deterministic, no look-ahead.

### Phase 5 — `run_backtest(df, init_balance, position_size)` for monteCarloSimulation

7. Manual loop over df (pattern from `S8001_2_ConvertFromGemini.py`):
   - Track `isInLong`/`isInShort`, `entry_bar_index`, `entry_price`, `stop_price`.
   - Entry: when prev-bar signal triggers, `entry_price = Open[current]` (backtest mode).
   - Exits: 5-bar exit (at Open), ATR stop (Low ≤ stop for long / High ≥ stop for short), 15:55 force close.
   - Return list of per-trade P&L = `(exit_price - entry_price) * position_size` (long) / negative for short.

### Phase 6 — Verification

8. Run `backtest.py` with `strategyS8002BBV1` against NVDA 5-min CSV.
9. Run `otherTest.py` (`lookAheadBiasTest` + `monkeyTest`).
10. Run `monteCarloSimulation.py`.
11. Run `walkForwardTest.py`.

---

## Relevant files

- `VS_8000_Strategy/VS_8002_20260620_NVDA/2 Strategy/S8002_1_BB.py` — **NEW**, the strategy class
- `VS_0003_test/backtest.py` — uses `Backtest(df, strategy1, ...)`; expects `Strategy` subclass with `init`/`next`, `commission` attr
- `VS_0003_test/otherTest.py` — `lookAheadBiasTest` needs `generate_signals(df)`→`signal` col; `monkeyTest` needs `Strategy` subclass
- `VS_0003_test/monteCarloSimulation.py` — needs `run_backtest(df, init_balance, position_size)`→P&L list
- `VS_0003_test/walkForwardTest.py` — uses `Backtest.optimize`; needs class attributes as optimize params
- `VS_8000_Strategy/VS_8001_20260204_NVDA/2 Strategy/S8001_2_ConvertFromGemini.py` — **REFERENCE** for manual loop pattern, ATR stop, N-bar exit, force close
- `VS_0006_data/ibMarketData.py` — data format: `datetime, date, time, high, low, close, open, volume, symbolName`

---

## Verification

1. `backtest.py`: `backtestStrategy(strategyS8002BBV1, NVDA_5min_csv)` runs without error, prints stats, opens plot.
2. `otherTest.py`: `lookAheadBiasTest` returns `isLookAhead=False`; `monkeyTest` returns `monkeyTestBetter=False`.
3. `monteCarloSimulation.py`: `monteCarloSimulation1(strategyS8002BBV1, csv, 1000, 100000, 110)` prints percentiles.
4. `walkForwardTest.py`: `walkTestStrategy1(strategyS8002BBV1, csv, ...)` runs IS/OOS windows.
5. Manual: confirm BB cross signals align with `cCrossUpBBTop`/`cCrossDownBBBottom` definition in `S6002_1`.

---

## Decisions

- Live trade (item 4) **EXCLUDED** per user.
- Stop-limit 0.06 offset **EXCLUDED** from backtest (market stop only); live-only.
- Position size = 110 NVDA shares per trade (fixed).
- Force close at 15:55; entry window 09:30–14:55.
- ATR stop uses previous bar ATR (`atr7.shift(1)`) per spec "previous bar of ATR(7)".
- Pure-pandas ATR/BB (no `talib` dependency), matching `S6002_1` style.

---

## Further considerations

1. `walkForwardTest.optimize` needs ≤2 params; recommend exposing `bb_period` + `atr_mult` as optimize params (`hold_bars` fixed at 5 per spec).
2. `backtesting.py` `size` is in units; confirm 110 shares works with `cash=100000` (NVDA ~$800-900 → ~$99k per trade, near cash limit). May need to raise initial capital in test calls.
