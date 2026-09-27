# VS_8101_autotrade.py — Specification & Reference

> File: `VS_0008_liveTrade/IB/autotrade/VS_8101_autotrade.py`
>
> Python port of the AmiBroker AFL trade-execution function `doTrade00`
> (`amibroker/autoTrade.afl` lines 1480–2176) as function `doTrade0`, using
> `ib_insync`. Consolidated from `L8001_autotrade_kimi3_V1.md`,
> `VS_8101_autotrade.md`, and `VS_8101_autotradeV1.md`.

**Safety:** test against paper TWS/Gateway only (port 7497 / 4002); never
transmit real orders from tests. `autoTrade.afl` remains the single source of
truth for the trading logic.

---

## 1. Requirement History

- **V1 — Source conversion** (per `autotradeURV1.txt` / `L8001_autotrade_UR_V1.txt`):
  `IB\autotrade\amibroker\autoTrade.afl` is an AmiBroker AFL program; convert
  function `doTrade00` to a Python program, function `doTrade0`.
- **V2 — Requirement document sync**: merge `L8001_autotrade_kimi3_V1.md` into
  the requirement md (this consolidation carries that forward).
- **V3 — Function verification**: `doTrade0` must connect to IB (default port
  4002, IB Gateway; port is an input parameter) with error handling and retry,
  place market and limit orders, place a single order / order with stop loss /
  bracket order, adjust input price to `ib_controller_tick_size` for CME
  futures, and return `1` if any order action was placed, else `0`.

Out of scope (the UR only asks for `doTrade00`): `doTrade4`, `doTickTrade`,
`closeAllOrderByMktPrice`, `closeOrdersByLMTPrice`.

## 2. Module Structure

| Item | Kind | Role |
|------|------|------|
| `AutoTrade` | class | Holds the connected `ib_insync.IB` plus the ported AFL helper functions |
| `doTrade0` | module function | Faithful port of AFL `doTrade00`; main entry point |
| `do_trade_stock0` | module function | Wrapper calling `doTrade0` with `symbol_type=2` (stock) |
| `do_trade_future0` | module function | Wrapper calling `doTrade0` with `symbol_type=1` (future) |
| `_place_bracket` | module function | Parent `transmit=False`; children chained via `parentId`; last child placed carries `transmit=True` |
| `_stop_child` | module function | `StopLimitOrder` when `stp_lmt_order_price` is set, else plain `StopOrder` |
| `_round_child_prices` | module function | Tick-size normalization (round-to-nearest) IB Controller performs |

## 3. `AutoTrade` Class

```python
AutoTrade(host="127.0.0.1", port=4002, client_id=1, auto_connect=True,
          max_open_position=15)
```

Holds `self.ib` (`ib_insync.IB`), connects immediately unless
`auto_connect=False`. Instance state replacing AFL `StaticVar`s
(single-process assumption): `max_open_position` (default 15), `ibvar_qta`
(default 20), `semaphore1` (default 0).

| Method | AFL source | Behavior |
|--------|-----------|----------|
| `ensure_connected(host, port, client_id, retries=3, sleep_sec=2)` | `IBcStatus` check | Connection gate: returns `True` once connected; retries `retries` times with `sleep_sec` between attempts; returns `False` after all fail |
| `_set_delayed_market_data()` | — | Calls `reqMarketDataType(3)` so accounts without live subscription get delayed data and quote snapshots don't raise errors 354/10168 |
| `write_line(message)` | `writeline(s1)` | Appends `<timestamp>;VS_8101_autotrade.py;<message>` to `TWSTrade111.log` (creates folder if needed) and prints it |
| `reconnect()` | `Reconnect` | `disconnect()` then `connect()` with stored host/port/clientId |
| `get_nearest_round_to_price(price, tick, round_type)` | `getNearestRoundToPrice` (line 1441) | Static. `round_type`: 1 = nearest, 2 = round up, 3 = round down (see quirks in §9) |
| `cancel_pending_order(symbol)` | `cancelPendingOrder` (line 125) | Cancels open orders whose symbol matches exactly or shares a common prefix ≥ 2 chars; only `PreSubmitted`/`Submitted`; returns `True` if anything was cancelled (caller then sleeps 0.5 s) |
| `position_size(symbol)` | `ibc.GetPositionSize` | Net position for `symbol` from `ib.positions()` (+long / −short / 0) |
| `get_position_size_with_retry(symbol, sell, cover)` | lines 1718–1754 | When sell/cover is requested but size reads 0: retry twice with 1 s sleeps, then `reconnect()` and retry twice more, then dump all positions to the log |
| `log_order_status()` | lines 1770–1793 | Logs execution list (filled trades: order id, symbol, filled qty, avg price, status) and pending list (open trades) |
| `get_rt_data(contract, wait_sec=8.0)` | `GetRTData("Last"/"Ask"/"Bid")` | Last/Ask/Bid snapshot via `reqMktData`, polling up to `wait_sec` in 0.5 s steps for the first valid ticks, then cancels the subscription. **Logged only** — order pricing uses `limit_price` (the AFL bid/ask limit logic is commented out) |

Compatibility shims: `_ib_is_connected` supports both ib_insync ≥ 1.0
(`is_connected()`) and 0.9.x (`isConnected()`); `_fill_quantity` supports both
`fill.quantity` (1.0) and `fill.execution.shares` (0.9.x).

## 4. `doTrade0` — Main Entry Point

```python
def doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price,
             stop_loss_price, profit_take_price, num_contracts,
             ib_controller_tick_size, is_intra_day, is_mkt_order,
             max_num_tick_size4_buy_short_slippage,
             max_num_tick_size4_stp_lmt_order_slippage, symbol_type,
             *, port=4002, host="127.0.0.1", client_id=1,
             connect_retries=3, connect_sleep_sec=2):
    # returns 1 if any order action was placed, else 0
```

### Parameters

| Parameter | Type | Meaning |
|-----------|------|---------|
| `auto_trade` | `AutoTrade` / `IB` / `None` | `None` → an `AutoTrade` is created; a raw `ib_insync.IB` is wrapped. Disconnected → `ensure_connected` reconnects with retry logic |
| `contract` | ib_insync `Contract` | `Stock`/`Future`; auto-qualified via `ib.qualifyContracts` when it has no `conId` (ib_insync needs exchange/currency/expiry that the AFL IB Controller resolves implicitly) |
| `buy` / `sell` / `short` / `cover` | bool | Signal flags (AFL `lastvalue()` of arrays). `sell` requires an existing long; `cover` an existing short |
| `limit_price` | float | Reference/limit price |
| `stop_loss_price` | float | Stop-loss trigger price |
| `profit_take_price` | float | Profit-target price; `<= 0` disables the profit-take child |
| `num_contracts` | int | Order quantity |
| `ib_controller_tick_size` | float | Tick size × 10000 (AFL convention); `/ 10000` gives the actual tick |
| `is_intra_day` | bool | `True` → TIF `Day`, `False` → TIF `GTC` |
| `is_mkt_order` | bool | `True` → market parent order, else limit |
| `max_num_tick_size4_buy_short_slippage` | int | Ticks of slippage added to the entry limit price |
| `max_num_tick_size4_stp_lmt_order_slippage` | int | Ticks for the stop-limit offset; `>= 1` converts the stop child to STP LMT, `< 1` keeps a plain STP |
| `symbol_type` | int | 1 = future, 2 = stock (AFL `sybmolType` typo normalized) |
| `port` / `host` / `client_id` | keyword-only | IB connection params; defaults `4002` / `127.0.0.1` / `1` (IB Gateway) |
| `connect_retries` / `connect_sleep_sec` | keyword-only | Retry attempts (default 3) and seconds between them (default 2) |

### Return value

| Value | Meaning |
|-------|---------|
| `1` | At least one order action was placed |
| `0` | No order action was placed (includes connection failure) |

### Price and tick math

- `tick = ib_controller_tick_size / 10000`;
  `max_allow_slippage = max_num_tick_size4_buy_short_slippage * tick`.
- `LIMIT_PRICE_BUY = limit_price + slippage`;
  `LIMIT_PRICE_SELL = limit_price - slippage`.
- **Buy side:** stop-loss rounds down (type 3), profit-take rounds up
  (type 2), `stp_lmt_order_price = round_down(stop_loss_price_raw − n*tick)`.
- **Short side mirrors:** stop rounds up, profit rounds down,
  `stp_lmt_order_price = round_up(stop_loss_price_raw + n*tick)`.
- `stp_lmt_order_price` is computed only when
  `max_num_tick_size4_stp_lmt_order_slippage >= 1`; otherwise 0 ⇒ plain STP.

### Guards

- **Stop-loss validation:** for limit orders, a buy requires
  `stop_loss_price < limit_price` and a short requires
  `stop_loss_price > limit_price`; an invalid stop blocks new buy/short
  entries (every new position must be followed by a stop-loss order).
- **Conflicting signals:** `buy` + `sell` together disable both; same for
  `short` + `cover`.
- **Connection:** if not connected after retries, log and return 0.

### Order logic — six paths

| # | Condition | Action |
|---|-----------|--------|
| 1 | `sell`, not `short`, position > 0 | Cancel pending orders (sleep 0.5 s if any cancelled), sell entire position MKT or LMT at `LIMIT_PRICE_SELL` |
| 2 | `cover`, not `buy`, position < 0 | Cancel pending orders, cover entire position MKT or LMT at `LIMIT_PRICE_BUY` |
| 3 | `buy`, position == 0 | Buy bracket: parent MKT/LMT at `LIMIT_PRICE_BUY` + STP/STP-LMT stop child + optional LMT profit-take child |
| 4 | `short`, position == 0 | Short bracket, mirrored |
| 5 | `buy` + `cover`, position < 0 | Reversal: parent BUY 2×`num_contracts`; children SELL 1× (plain STP stop + optional LMT profit-take) |
| 6 | `short` + `sell`, position > 0 | Reversal, mirrored: parent SELL 2×; children BUY 1× |

Bracket mechanics (`_place_bracket`): parent placed with `transmit=False`,
children chained via `parentId`; the **last child placed** carries
`transmit=True` — the profit-take child when present, otherwise the stop
child itself (AFL semantics). Reversal paths always use a plain `StopOrder`
child (AFL places STP with limit 0).

State write-back: quota starts at 20, +1 on exits, −1 on entries; after each
action `ibvar_qta = max(quota, 0)` and `semaphore1 = 0` are written back;
`semaphore1` is also released when no action fired. Note quota is **not** an
entry gate inside `doTrade00` (as in the AFL).

## 5. Wrappers

```python
def do_trade_stock0(...)   # doTrade0 with symbol_type=2 (stock)
def do_trade_future0(...)  # doTrade0 with symbol_type=1 (future)
```

## 6. Logging

All events are appended to
`C:\Project\ProjectLife\VSCode Algo Workspace DataFile\trade_log\TWSTrade111.log`
in the AFL `writeline` format `<timestamp>;<program name>;<message>`
(`VS_8101_autotrade.py` as program name; AFL writes `autoTrade.afl`), and
echoed to stdout.

## 7. AFL → Python Mapping

| AFL (`autoTrade.afl`) | Python (`VS_8101_autotrade.py`) |
|---|---|
| `writeline(s1)` → append `TWSTrade111.log` | `AutoTrade.write_line(msg)` |
| `GetTradingInterface("IB")` / `ibc.IsConnected()` | `ib_insync.IB` / `_ib_is_connected(ib)` |
| `GetRTData("Last"/"Ask"/"Bid")` | `get_rt_data` via `ib.reqMktData` (logged only) |
| `ibc.GetPositionSize(symbol)` | `ib.positions()` lookup |
| `ibc.GetPositionList()` | `ib.positions()` |
| `ibc.GetPendingList(0/1, "")` | `ib.openTrades()` |
| `ibc.GetExecList(0, "")` + `GetExecInfo` | `ib.trades()` (filled) |
| `ibc.GetStatus(orderId)` | `trade.orderStatus.status` |
| `ibc.CancelOrder(orderId)` | `ib.cancelOrder(order)` |
| `ibc.PlaceOrder(..., "LMT"/"MKT"/"STP", ..., parentId)` | `ib.placeOrder(contract, LimitOrder/MarketOrder/StopOrder/StopLimitOrder)` with `parentId` |
| `getNearestRoundToPrice(price, tick, type)` | `AutoTrade.get_nearest_round_to_price` |
| `cancelPendingOrder(ibc, symbol)` | `AutoTrade.cancel_pending_order(symbol)` |
| `StaticVarGet/Set("MAX_OPEN_POSITION"/"IBVAR_QTA"/"semaphore1")` | instance attributes (single-process) |
| `ThreadSleep(ms)` | `ib.sleep(ms/1000)` |
| `doTrade00(...)` | `doTrade0(...)` |
| `doTradeStock0` / `doTradeFuture0` | `do_trade_stock0` / `do_trade_future0` |

## 8. Usage Example (paper)

```python
from ib_insync import Stock
from VS_8101_autotrade import AutoTrade, doTrade0

at = AutoTrade(host="127.0.0.1", port=4002, client_id=1)  # IB Gateway paper
contract = Stock("AAPL", "SMART", "USD")
at.ib.qualifyContracts(contract)

rs = doTrade0(at, contract,
              buy=True, sell=False, short=False, cover=False,
              limit_price=180.50, stop_loss_price=178.00,
              profit_take_price=185.00,
              num_contracts=10, ib_controller_tick_size=100,  # 0.01 tick
              is_intra_day=True, is_mkt_order=False,
              max_num_tick_size4_buy_short_slippage=3,
              max_num_tick_size4_stp_lmt_order_slippage=2,
              symbol_type=2)
at.ib.sleep(2)

# auto_trade=None also works: connects to the IB Gateway (default port 4002)
# with retry logic before trading
rs = doTrade0(None, contract, True, False, False, False,
              180.50, 178.00, 185.00, 10, 100, True, False, 3, 2, 2)
```

## 9. Known Differences / Limitations

- Real-time quotes (`GetRTData`) are **not** used for order pricing in the
  current AFL logic (limit price drives everything); quotes are only logged.
- AFL symbol-prefix matching in `cancel_pending_order` (common prefix ≥ 2
  chars) is preserved but can match unrelated symbols sharing a 2-char
  prefix — review before relying on it.
- No cross-process semaphore; run one trader process per TWS/Gateway
  clientId.
- `get_nearest_round_to_price` type 1 ("nearest"): the AFL comment says
  "round to closest" but the code only rounds up when the ceil multiple is
  strictly closer; when the floor multiple is closer it returns the original
  price unchanged (e.g. tick 0.25: 256.55 stays 256.55, while 256.45 →
  256.50). Ported faithfully — do not "fix" without a user decision.
- Type-2 rounding always moves up one tick even for on-tick prices
  (4020.0 → 4020.25); ported faithfully.
- `symbol_type` is accepted for AFL signature parity but does not change
  behavior in the Python port (contract object already carries the type).

## 10. Relevant Files

- `VS_0008_liveTrade/IB/autotrade/amibroker/autoTrade.afl` — source of truth
  (`doTrade00` lines 1480–2176, `cancelPendingOrder` line 125,
  `getNearestRoundToPrice` line 1441, `writeline`)
- `VS_0008_liveTrade/test/convert_autoTrade_by_gemini.py` — ib_insync
  bracket-order patterns (structural reference only)
- `VS_0008_liveTrade/displayLiveInfo/displayLiveInfo.py` — connection params
  convention (port 7497 TWS paper)
- `VS_0008_liveTrade/IB/autotrade/autotradeURV1.txt` /
  `L8001_autotrade_UR_V1.txt` — requirement sources (identical 2 lines)

## 11. Verification

1. `python -m py_compile VS_8101_autotrade.py` passes.
2. Unit-check `get_nearest_round_to_price` (tick 0.25): 256.45 → 256.50
   type 1; 256.55 → 256.55 type 1 (AFL quirk, §9); 256.55 → 256.75 type 2;
   256.45 → 256.25 type 3.
3. Connection parity: `auto_trade=None` auto-connects to the IB Gateway
   (default 4002); failures retry `connect_retries` times and return 0
   without orders; a provided disconnected instance reconnects.
4. Order-construction parity vs AFL: six order paths (incl. 2×-qty reversal
   parent with 1×-qty children), entry type per `is_mkt_order`, Day/GTC per
   `is_intra_day`, limit sign per side, stop/profit rounding directions per
   side, stop child = `StopOrder` vs `StopLimitOrder` per stp-slippage
   threshold, transmit flags, cancel-pending before exits, conflict guards,
   stop-loss validation gating new entries.
5. Manual: run against IB Gateway paper (4002) or TWS paper (7497) with a
   test contract and confirm bracket orders appear in TWS.
