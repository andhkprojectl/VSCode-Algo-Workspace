## Requirement: autoTrade V1 (per autotradeURV1.txt / L8001_autotrade_UR_V1.txt)

Convert the AmiBroker AFL trade-execution function `doTrade00` to Python function
`doTrade0` in a new program `IB\autotrade\autotrade.py`, using ib_insync.

**Original requirement (verbatim)**
1. IB\autotrade\amibroker\autoTrade.afl is amibroker afl program
2. Convert function doTrade00 in afl program autoTrade.afl to python program
   IB\autotrade\autotrade.py, function doTrade0

**Conversion scope (verified in autoTrade.afl)**
- Source: `doTrade00`, autoTrade.afl lines 1480-2178 (~700 lines), signature:
  (symbol1, buy1, sell1, short1, cover1, limitPrice, stopLossPrice0,
  profitTakePrice0, numContracts, ibControllerTickSize, isIntraDay, isMktOrder,
  maxNumTickSize4BuyShortSlippage, maxNumTickSize4StpLmtOrderSlippage, sybmolType)
- Flow: connection first - `auto_trade` may be None (an `AutoTrade` is created and
  connected to the IB Gateway, default port 4002, with retry logic:
  `connect_retries` attempts, `connect_sleep_sec` between attempts; a raw
  ib_insync `IB` instance is also accepted and wrapped), and `ensure_connected`
  reconnects with the same retry logic when disconnected; position size via
  GetPositionSize (+long/-short/0) with the retry+reconnect rule when sell/cover
  is requested but size reads 0; exits first (`cancel_pending_order` then MKT/LMT
  sell/cover of the current position), then bracket entries: parent LMT/MKT
  (transmit=false) + child STP stop-loss + child LMT profit-take (transmit=true)
  when profit-take is valid (`profit_take_price > 0`), else the STP child alone
  (transmit=true); `is_intra_day` selects Day vs GTC; reversal (short+sell /
  buy+cover with an existing opposite position) places a 2x-qty parent order while
  bracket children stay 1x qty; `doTrade0` returns 1 if any order action was
  placed, else 0.
- Price preparation: `tick = ib_controller_tick_size / 10000`; buy limit =
  `limit_price + ticks*slippage`, sell/cover limit = `limit_price - ticks*slippage`;
  LONG: stop rounds down, profit rounds up, `stp_lmt_order_price =
  round(stop_loss_price - ticks*stp_slippage)` when stp slippage >= 1 else 0
  (pure STP); SHORT mirrors (stop rounds up, profit rounds down, `+`).
  Last/Ask/Bid are fetched in the AFL but bid/ask limit logic is commented out -
  active code uses `limit_price +- slippage` only.
- Guards to preserve: invalid stop-loss blocks new buy/short; buy+sell /
  short+cover conflicts disable both sides; `MAX_OPEN_POSITION` static (default 15)
  and `quota = 20` - quota is NOT an entry gate inside doTrade00; after each
  action the AFL writes back `IBVAR_QTA` and resets `semaphore1` (cross-instance
  state - in Python keep as an instance attribute or omit with a documented note);
  `writeline` logging (`begin2[symbol]` ...); `sybmolType` (AFL typo, keep the
  name) = 1 future, 2 stock.
- Helpers to port: `get_nearest_round_to_price(price, tick_size, round_type)`
  (autoTrade.afl line 1441; round_type 1 = nearest, 2 = round up, 3 = round down),
  `cancel_pending_order` (line 125; match exact symbol or common prefix >= 2
  chars; cancel only PreSubmitted/Submitted orders; 500 ms sleep if anything was
  cancelled) and the execution/pending list status dump (`log_order_status`).

**Implementation steps**
1. Create `VS_0008_LiveTrade/IB/autotrade/autotrade.py` with an `AutoTrade` class:
   holds the connected `ib_insync.IB` (default host 127.0.0.1, port 4002 IB
   Gateway), `write_line(message)` appending `<timestamp>;autotrade.py;<message>`
   to `TWSTrade111.log` (AFL format: `Now();autoTrade.afl;<msg>`),
   `ensure_connected` (connect/retry gate: `connect_retries` attempts with
   `connect_sleep_sec` between attempts), and methods
   `get_nearest_round_to_price`, `cancel_pending_order`,
   `get_position_size_with_retry` (2x 1s-sleep retries, reconnect, 2x retries,
   then log all positions), `log_order_status` (execution/pending list dump).
2. Module-level function `doTrade0(auto_trade, contract, buy, sell, short, cover,
   limit_price, stop_loss_price, profit_take_price, num_contracts,
   ib_controller_tick_size, is_intra_day, is_mkt_order,
   max_num_tick_size4_buy_short_slippage, max_num_tick_size4_stp_lmt_order_slippage,
   symbol_type, *, port=4002, host="127.0.0.1", client_id=1,
   connect_retries=3, connect_sleep_sec=2)` - faithful port of doTrade00.
   `auto_trade=None` auto-connects with retry logic (default port 4002 IB
   Gateway); a raw ib_insync `IB` instance is accepted and wrapped. `contract`
   is a qualified ib_insync `Stock`/`Future` (ib_insync needs
   exchange/currency/expiry that the AFL IB controller resolves implicitly);
   `buy/sell/short/cover` are current-bar scalar booleans (the AFL receives
   arrays and uses `lastvalue()`).
3. Order construction: six paths - sell-only, cover-only, buy (bracket),
   short (bracket), buy+cover (2x-qty parent, 1x-qty bracket children),
   short+sell (mirror). Bracket = parent + Stop/Stop-Limit child + Limit child
   (OCA) via parentId chaining; parent transmit=False, last child transmit=True;
   the stop child is a `StopLimitOrder` when `stp_lmt_order_price` is set,
   else a plain `StopOrder`; `is_intra_day` picks Day vs GTC.
4. Port helpers as `AutoTrade` methods: `get_nearest_round_to_price`,
   `cancel_pending_order`, `get_position_size_with_retry`, `log_order_status`.
5. State: positions via ib.positions(); MAX_OPEN_POSITION (15) and quota (20)
   as instance attributes; AFL `StaticVar` IBVAR_QTA/semaphore becomes an
   instance attribute (single-process assumption); check `ib.is_connected()`
   where the AFL checks `ibc.IsConnected()`.
6. Convenience wrappers `do_trade_stock0(...)` / `do_trade_future0(...)`
   mirroring the AFL wrappers (symbol_type 2 / 1).
7. Logging: mirror writeline - append `<timestamp>;autotrade.py;<message>` to
   `TWSTrade111.log` (AFL writes `Now();autoTrade.afl;<msg>`).
8. Return: `doTrade0` returns 1 if any order action was placed, else 0.
9. Structural reference only:
   `VS_0008_LiveTrade/test/convert_autoTrade_by_gemini.py` (AutoTradeIB.do_trade_00);
   `autoTrade.afl` remains the single source of truth.

**AFL -> Python mapping**
| AFL (autoTrade.afl) | Python (autotrade.py) |
|---|---|
| `writeline(s1)` -> append TWSTrade111.log | `AutoTrade.write_line(msg)` |
| `GetTradingInterface("IB")` / `ibc.IsConnected()` | `ib_insync.IB` / `ib.isConnected()` |
| `GetRTData("Last"/"Ask"/"Bid")` | `ib.reqMktData` snapshot (logged only; prices come from `limit_price` as in current AFL) |
| `ibc.GetPositionSize(symbol)` | `ib.positions()` lookup |
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

**`doTrade0` parameter reference**
| Param | Type | Meaning |
|---|---|---|
| `auto_trade` | `AutoTrade` / ib_insync `IB` / None | connected instance; None (or disconnected) => auto-connect with retry logic |
| `contract` | ib_insync `Contract` | qualified `Stock`/`Future` |
| `buy` / `sell` / `short` / `cover` | bool | signal flags (AFL `lastvalue()` of arrays); sell requires an existing long, cover an existing short |
| `limit_price` | float | reference price; used for LMT orders |
| `stop_loss_price` | float | stop-loss trigger price |
| `profit_take_price` | float | profit-target price; <= 0 disables profit taking |
| `num_contracts` | int | order quantity |
| `ib_controller_tick_size` | float | tick size x 10000 (AFL convention; /10000 => actual tick) |
| `is_intra_day` | bool | True => TIF `DAY`, False => `GTC` |
| `is_mkt_order` | bool | True => market parent order |
| `max_num_tick_size4_buy_short_slippage` | int | ticks of slippage added to entry limit |
| `max_num_tick_size4_stp_lmt_order_slippage` | int | ticks for stop-limit offset; < 1 => plain stop |
| `symbol_type` | int | 1 = future, 2 = stock |
| `port` | int | IB connection port; default 4002 (IB Gateway) |
| `host` | str | IB connection host; default 127.0.0.1 |
| `client_id` | int | IB clientId; default 1 |
| `connect_retries` | int | connection attempts when (re)connecting; default 3 |
| `connect_sleep_sec` | float | seconds between connection attempts; default 2 |

Returns `1` if an order was placed, else `0`.

**Relevant files**
- `VS_0008_LiveTrade/IB/autotrade/amibroker/autoTrade.afl` - source of truth
  (`doTrade00` lines 1480-2178, `cancelPendingOrder`, `getNearestRoundToPrice`,
  `writeline`)
- `VS_0008_LiveTrade/test/convert_autoTrade_by_gemini.py` - ib_insync
  bracket-order patterns to reuse
- `VS_0008_LiveTrade/displayLiveInfo/displayLiveInfo.py` (L112) - connection
  params convention (port 7497 TWS paper)
- `VS_0008_LiveTrade/IB/autotrade/autotradeURV1.txt` /
  `VS_0008_LiveTrade/IB/autotrade/L8001_autotrade_UR_V1.txt` - requirement
  sources (identical 2 lines)
- `VS_0008_LiveTrade/IB/autotrade/L8001_autotrade_kimi3_V1.md` - companion
  kimi3 plan (merged into this document)

**Safety**
- Test against paper TWS/Gateway only (port 7497 / 4002); never transmit real
  orders from tests.

**Usage example (IB Gateway paper)**
```python
from ib_insync import Stock
from autotrade import AutoTrade, doTrade0

at = AutoTrade(host='127.0.0.1', port=4002, client_id=1)  # IB Gateway paper
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

# auto_trade=None also works: connects to the IB Gateway (default port 4002)
# with retry logic before trading
rs = doTrade0(None, contract,
              buy=True, sell=False, short=False, cover=False,
              limit_price=180.50, stop_loss_price=178.00, profit_take_price=185.00,
              num_contracts=10, ib_controller_tick_size=100,
              is_intra_day=True, is_mkt_order=False,
              max_num_tick_size4_buy_short_slippage=3,
              max_num_tick_size4_stp_lmt_order_slippage=2,
              symbol_type=2)
```

**Verification**
1. `python -m py_compile autotrade.py` passes.
2. `doTrade0` exists with the full parameter set; unit-check
   `get_nearest_round_to_price` (tick 0.25): 256.45 -> 256.50 type 1
   (ceil closer); 256.55 -> 256.55 type 1 (floor closer - AFL quirk, see
   limitations); 256.55 -> 256.75 type 2 up; 256.45 -> 256.25 type 3 down.
3. Connection parity: `auto_trade=None` auto-connects to the IB Gateway on the
   default port 4002; connection failures retry `connect_retries` times and
   return 0 without orders; a provided disconnected instance reconnects.
3. Order-construction parity checklist vs AFL: six order paths (incl. 2x-qty
   reversal parent with 1x-qty children), entry type per `is_mkt_order`,
   Day/GTC per `is_intra_day`, buy/sell limit `limit_price +/- ticks*slippage`
   sign per side, stop/profit tick-rounding directions per side, stop child =
   `StopOrder` vs `StopLimitOrder` per stp-slippage threshold, LMT profit-take
   child, STP-only exit when profit-take invalid, transmit flags, cancel-pending
   before exits, conflict guards, stop-loss validation gating new entries.
4. Position/reversal parity: sell only when position > 0, cover only when < 0,
   GetPositionSize-0 retry/reconnect path on sell/cover, return 1/0 semantics,
   IBVAR_QTA/semaphore handling documented.
5. Manual: run against IB Gateway paper (4002) or TWS paper (7497) with a test
   contract and confirm bracket orders appear in TWS.

**Decisions**
- `contract` (ib_insync `Stock`/`Future`) is passed instead of a bare symbol
  string - ib_insync needs exchange/currency/expiry that AFL's IB controller
  resolves implicitly. Wrappers keep symbol-based calls convenient.
- Signals are plain booleans (AFL `lastvalue()` of arrays).
- AFL `StaticVar` semaphore/quota is process-global in AmiBroker; in Python it
  becomes an instance attribute (single-process assumption).
- Out of scope: `doTrade4`, `doTickTrade`, `closeAllOrderByMktPrice`,
  `closeOrdersByLMTPrice` - the UR only asks for `doTrade00`.

**Known differences / limitations**
- Real-time quote fetch (`GetRTData`) is not needed for order pricing in current
  AFL logic (limit price drives everything); quotes are only logged.
- AFL symbol-prefix matching in `cancelPendingOrder` (common prefix >= 2 chars)
  is preserved but should be reviewed - it can match unrelated symbols sharing a
  2-char prefix.
- No cross-process semaphore; run one trader process per TWS clientId.
- `getNearestRoundToPrice` type 1: the AFL comment says "round to closest" but
  the code only rounds up when the ceil multiple is strictly closer; when the
  floor multiple is closer it returns the original price unchanged (e.g. 256.55
  stays 256.55). Ported faithfully - do not "fix" without a user decision.
- AFL type-2 rounding always moves up one tick even for on-tick prices
  (4020.0 -> 4020.25); ported faithfully.
