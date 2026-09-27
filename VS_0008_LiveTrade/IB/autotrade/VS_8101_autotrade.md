# VS_8101_autotrade.py — Requirements & Specification

> File: `VS_0008_liveTrade/IB/autotrade/VS_8101_autotrade.py`
>
> Python port of the AmiBroker AFL trade-execution function `doTrade00`
> (`autoTrade.afl` lines 1480-2176) as function `doTrade0`.

**Safety:** test against paper TWS/Gateway only (port 7497 / 4002); never
transmit real orders from tests. `autoTrade.afl` remains the single source
of truth for the trading logic.

---

## V1 — Source Conversion

`IB\autotrade\amibroker\autoTrade.afl` is an AmiBroker AFL program.

Convert function `doTrade00` in AFL program `autoTrade.afl` to Python program
`IB\autotrade\autotrade.py`, function `doTrade0`.

---

## V2 — Requirement Document Sync

Compare md file `.kilo\plans\1785245983679-autotrade-requirement-md.md` with
`L8001_autotrade_kimi3_V1.md`.

If anything exists in `L8001_autotrade_kimi3_V1.md` but is missing in file
`1785245983679-autotrade-requirement-md.md`, add it to
`1785245983679-autotrade-requirement-md.md`.

---

## V3 — Function `doTrade0` Verification

Check function `doTrade0` in `VS_8101_autotrade.py`, make sure it is able to:

- Connect to IB, port **default 4002** to IB Gateway; port is an input parameter of `doTrade0`
- Handle connection errors and retry logic
- Place orders to IB, supporting **market order** and **limit order**
- Place a **single order**, an order with a **stop loss order**, or a **bracket order**
- For CME futures, adjust input price to match `ib_controller_tick_size`
- Return `1` if any order action was placed, else `0`

### Function Signature

```python
def doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price,
             stop_loss_price, profit_take_price, num_contracts,
             ib_controller_tick_size, is_intra_day, is_mkt_order,
             max_num_tick_size4_buy_short_slippage,
             max_num_tick_size4_stp_lmt_order_slippage, symbol_type,
             *, port=4002, host="127.0.0.1", client_id=1,
             connect_retries=3, connect_sleep_sec=2):
```

### Parameters

| Parameter | Description |
|---|---|
| `auto_trade` | IB object (`AutoTrade` instance or raw `ib_insync.IB`); if `None`, auto connect with retry logic |
| `contract` | ib_insync contract (e.g. `Future`, `Stock`); auto-qualified via `ib.qualifyContracts` if no `conId` |
| `buy` | `1` = place buy order |
| `sell` | `1` = place sell order; requires an existing long position before placing |
| `short` | `1` = place short order |
| `cover` | `1` = place cover order; requires an existing short position before placing |
| `limit_price` | Limit order price |
| `stop_loss_price` | Stop loss order price |
| `profit_take_price` | Take profit price for bracket order (`> 0` enables the profit-take child) |
| `num_contracts` | Number of contracts for future |
| `ib_controller_tick_size` | Tick size for price adjustment (divided by 10000 internally) |
| `is_intra_day` | `1` = intra day order (TIF `Day`), `0` = end of day order (TIF `GTC`) |
| `is_mkt_order` | `1` = market order, otherwise limit order |
| `max_num_tick_size4_buy_short_slippage` | Maximum slippage allowed (in ticks) for limit price |
| `max_num_tick_size4_stp_lmt_order_slippage` | Maximum slippage allowed (in ticks) for stop limit order price (`>= 1` converts stop child to STP LMT) |
| `symbol_type` | `1` = future, `2` = stock |
| `port` | IB Gateway port, default `4002` (keyword-only) |
| `host` | IB Gateway host, default `127.0.0.1` (keyword-only) |
| `client_id` | IB client id, default `1` (keyword-only) |
| `connect_retries` | Connection retry attempts, default `3` (keyword-only) |
| `connect_sleep_sec` | Seconds between connection retries, default `2` (keyword-only) |

### Return Value

| Value | Meaning |
|---|---|
| `1` | At least one order action was placed |
| `0` | No order action was placed (includes connection failure) |

### Order Logic

| # | Condition | Action |
|---|---|---|
| 1 | `sell` and not `short` and position > 0 | Cancel pending orders, place sell order (MKT or LMT at `LIMIT_PRICE_SELL`) |
| 2 | `cover` and not `buy` and position < 0 | Cancel pending orders, place cover order (MKT or LMT at `LIMIT_PRICE_BUY`) |
| 3 | `buy` and position == 0 | Place buy parent with stop loss child (STP or STP LMT) and optional profit-take child (bracket) |
| 4 | `short` and position == 0 | Place short parent with stop loss child and optional profit-take child (bracket) |
| 5 | `buy` + `cover` and position < 0 | Reversal: buy 2x contracts bracket, children sized 1x |
| 6 | `short` + `sell` and position > 0 | Reversal: short 2x contracts bracket, children sized 1x |

Additional rules:

- If `buy` and `sell` are both set, both are disabled; same for `short` + `cover`
- New positions require a valid stop loss: for LMT orders, buy stop must be < limit price, short stop must be > limit price
- Bracket order: parent placed with `transmit=False`, children chained via `parentId`; the last child carries `transmit=True`
- Sell/cover with a position reading of 0 retries the position query, then reconnects and retries again

---

## Wrappers

```python
def do_trade_stock0(...)   # calls doTrade0 with symbol_type=2 (stock)
def do_trade_future0(...)  # calls doTrade0 with symbol_type=1 (future)
```

## Logging

All events are appended to `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\trade_log\TWSTrade111.log`
in the format `<timestamp>;<program name>;<message>` (AFL `writeline` equivalent).
