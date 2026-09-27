# Plan: Convert AFL `doTrade00` to Python `autotrade.py`

**TL;DR** — Port `doTrade00` from [autoTrade.afl](amibroker/autoTrade.afl) into a new `autotrade.py` (same folder) exposing function `doTrade0`, using `ib_insync` (the IB library already used in this workspace). Also generate a markdown doc describing the conversion. Reference for ib_insync patterns: `VS_0008_liveTrade/test/convert_autoTrade_by_gemini.py`.

## Requirement (from L8001_autotrade_UR_V1.txt)

> `IB\autotrade\amibroker\autoTrade.afl` is an AmiBroker AFL program.
> Convert function `doTrade00` in AFL program `autoTrade.afl` to Python program `IB\autotrade\autotrade.py`, function `doTrade0`.

## Steps

1. **Create `VS_0008_liveTrade/IB/autotrade/autotrade.py`** with:
   - `AutoTrade` class — holds the `ib_insync.IB` connection (default `127.0.0.1:7497`), `write_line()` logger appending to `TWSTrade111.log` in the AFL format (`timestamp;autotrade.py;msg`), and helpers:
     - `get_nearest_round_to_price(price, tick, type)` — port of AFL `getNearestRoundToPrice` (type 1 nearest / 2 up / 3 down).
     - `cancel_pending_order(symbol)` — port of AFL `cancelPendingOrder` (match exact or common prefix >= 2 chars; cancel only `PreSubmitted`/`Submitted`; sleep 500 ms if anything cancelled).
     - `get_position_size_with_retry(contract, symbol)` — port of the sell/cover retry logic (2 retries x 1 s, then `Reconnect` + 2 more retries, dump position list).
     - `log_order_status()` — port of the execution-list / pending-list dump.
   - Module-level **`doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price, stop_loss_price, profit_take_price, num_contracts, ib_controller_tick_size, is_intra_day, is_mkt_order, max_num_tick_size4_buy_short_slippage, max_num_tick_size4_stp_lmt_order_slippage, symbol_type)`** — faithful port of `doTrade00`:
     - stop-loss validation (LMT buy requires SL < limit; short requires SL > limit; invalid SL blocks entries)
     - conflicting-signal filters (buy+sell, short+cover)
     - tick math: `tick = ib_controller_tick_size / 10000`, slippage = `max_num_tick_size4_buy_short_slippage * tick`, `LIMIT_PRICE_BUY/SELL = limit +/- slippage`
     - tick rounding of SL/PT/stpLmt prices per side (buy: SL down, PT up, stpLmt = SL0 - n*tick down; short: mirrored)
     - TIF `DAY` if `is_intra_day` else `GTC`
     - six order paths: sell-only, cover-only, buy (bracket), short (bracket), buy+cover (2x qty parent, bracket for 1x qty), short+sell (mirror)
     - bracket orders via `parentId` chaining; parent `transmit=False`, last child `transmit=True`; STP becomes `StopLimitOrder` when `stpLmtOrderPrice` is set, else `StopOrder`
     - returns 1 if an order was placed, else 0
   - Convenience wrappers `do_trade_stock0(...)` / `do_trade_future0(...)` mirroring the AFL wrappers (`symbol_type` 2 / 1).

2. **Markdown documentation** (this file) — requirement summary, AFL->Python mapping table, `doTrade0` parameter reference, usage example (paper TWS), and known differences/limitations.

## AFL -> Python mapping

| AFL (autoTrade.afl) | Python (autotrade.py) |
|---|---|
| `writeline(s1)` -> append `TWSTrade111.log` | `AutoTrade.write_line(msg)` |
| `GetTradingInterface("IB")` / `ibc.IsConnected()` | `ib_insync.IB` / `ib.isConnected()` |
| `GetRTData("Last"/"Ask"/"Bid")` | `ib.reqMktData` snapshot (logged only; prices come from `limit_price` as in current AFL) |
| `ibc.GetPositionSize(symbol)` | `ib.positions()` lookup / `ib.reqPositions` |
| `ibc.GetPositionList()` | `ib.positions()` |
| `ibc.GetPendingList(0/1, "")` | `ib.openOrders()` / `ib.openTrades()` |
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

## `doTrade0` parameter reference

| Param | Type | Meaning |
|---|---|---|
| `auto_trade` | `AutoTrade` | connected instance |
| `contract` | ib_insync `Contract` | qualified `Stock`/`Future` |
| `buy` / `sell` / `short` / `cover` | bool | signal flags (AFL `lastvalue()` of arrays) |
| `limit_price` | float | reference price; 0 => market order semantics |
| `stop_loss_price` | float | stop-loss trigger price |
| `profit_take_price` | float | profit-target price; <= 0 disables profit taking |
| `num_contracts` | int | order quantity |
| `ib_controller_tick_size` | float | tick size x 10000 (AFL convention; /10000 => actual tick) |
| `is_intra_day` | bool | True => TIF `DAY`, False => `GTC` |
| `is_mkt_order` | bool | True => market parent order |
| `max_num_tick_size4_buy_short_slippage` | int | ticks of slippage added to entry limit |
| `max_num_tick_size4_stp_lmt_order_slippage` | int | ticks for stop-limit offset; < 1 => plain stop |
| `symbol_type` | int | 1 = future, 2 = stock |

Returns `1` if an order was placed, else `0`.

## Relevant files

- `VS_0008_liveTrade/IB/autotrade/amibroker/autoTrade.afl` — source of truth (`doTrade00` ~L1470-2150, `cancelPendingOrder`, `getNearestRoundToPrice`, `writeline`)
- `VS_0008_liveTrade/test/convert_autoTrade_by_gemini.py` — ib_insync bracket-order patterns to reuse
- `VS_0008_liveTrade/displayLiveInfo/displayLiveInfo.py` (L112) — connection params convention (port 7497 TWS paper)

## Usage example (paper TWS)

```python
from ib_insync import Stock
from autotrade import AutoTrade, doTrade0

at = AutoTrade(host='127.0.0.1', port=7497, client_id=1)  # paper TWS
contract = Stock('AAPL', 'SMART', 'USD')
at.ib.qualifyContracts(contract)

rs = doTrade0(at, contract,
              buy=True, sell=False, short=False, cover=False,
              limit_price=180.50, stop_loss_price=178.00, profit_take_price=185.00,
              num_contracts=10, ib_controller_tick_size=100,  # 0.01 tick
              is_intra_day=True, is_mkt_order=False,
              max_num_tick_size4_buy_short_slippage=3,
              max_num_tick_size4_stp_lmt_order_slippage=2,
              symbol_type=2)
at.ib.sleep(2)
```

## Verification

1. `python -m py_compile autotrade.py`
2. Dry-run `get_nearest_round_to_price` against the AFL doc examples (tick 0.25: 256.45->256.50 type 1; 256.55->256.75 type 2; 256.45->256.25 type 3)
3. Manual: run against TWS paper (7497) with a test contract and confirm bracket orders appear in TWS

## Decisions

- `contract` (ib_insync `Stock`/`Future`) is passed instead of a bare symbol string — ib_insync needs exchange/currency/expiry that AFL's IB controller resolves implicitly. Wrappers keep symbol-based calls convenient.
- Signals are plain booleans (AFL `lastvalue()` of arrays).
- AFL `StaticVar` semaphore/quota is process-global in AmiBroker; in Python it becomes an instance attribute (single-process assumption).
- Out of scope: `doTrade4`, `doTickTrade`, `closeAllOrderByMktPrice`, `closeOrdersByLMTPrice` — the UR only asks for `doTrade00`.

## Known differences / limitations

- Real-time quote fetch (`GetRTData`) is not needed for order pricing in current AFL logic (limit price drives everything); quotes are only logged.
- AFL symbol-prefix matching in `cancelPendingOrder` (common prefix >= 2 chars) is preserved but should be reviewed — it can match unrelated symbols sharing a 2-char prefix.
- No cross-process semaphore; run one trader process per TWS clientId.
