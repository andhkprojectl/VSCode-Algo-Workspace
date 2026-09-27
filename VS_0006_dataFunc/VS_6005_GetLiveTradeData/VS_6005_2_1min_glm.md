# getLiveTradeIB1MinData — Requirements & Specification

> File: `VS_0006_dataFunc/VS_6005_GetLiveTradeData/VS_6005_1_1min_glm.py`
>
> Consolidates 1-minute CME future bars from MariaDB and Interactive Brokers
> into a single DataFrame.

---

## 1. Purpose

Function `getLiveTradeIB1MinData`: consolidate data from different sources
(MariaDB `ticker1Min` table + IB `reqHistoricalData` / `reqMktData`) into
one dataframe `df`.

## 2. Output

Dataframe `df`:

- Index: `DatetimeIndex` named `datetime` (US/Eastern naive, matching MariaDB
  `ticker1Min` stamps), sorted ascending, deduplicated
- Columns: `open`, `high`, `low`, `close`, `volume`

## 3. Input Parameters

```python
def getLiveTradeIB1MinData(symbolName, conn1=None, start_data=None,
                           end_Date=None, isCurrBarMktData=False,
                           isCallOnce=True):
```

| Parameter | Type / Default | Description |
|---|---|---|
| `symbolName` | str | Symbol name (e.g. `"ES"`); resolved to the current front-month CME future via `ContFuture` |
| `conn1` | `None` | Optional shared IB / DB connection; when `None`, connect with retry logic (TWS 7497, then IB Gateway 4002) |
| `start_data` | str / datetime | Start date |
| `end_Date` | str / datetime | End date; a bare date (00:00:00) is treated as that day's 23:59:59 |
| `isCurrBarMktData` | bool, default `False` | Source of the current day's in-progress bar |
| `isCallOnce` | bool, default `True` | Single fetch vs. live streaming mode |

### `isCurrBarMktData`

| Value | Behavior |
|---|---|
| `True` | Current day current bar obtained from `reqMktData` (OHLC seeded with last price, volume 0) |
| `False` | Current day current bar obtained from `reqHistoricalData` |

### `isCallOnce`

| Value | Behavior |
|---|---|
| `True` | Call `getLiveTradeIB1MinData` 1 time; fetched bars upserted into `ticker1Min`, then df returned |
| `False` | `reqHistoricalData` runs with `keepUpToDate=True`; callback `onBarUpdate` registered via `bars.updateEvent += onBarUpdate`. If a bar exists in df, update the record; if not, insert it. df rows missing in `ticker1Min` are synced to MariaDB every 5 minutes (excluding the newest df bar). df is returned immediately — keep the process running and the IB connection open to receive updates |

---

## 4. Update Logic

When `getLiveTradeIB1MinData` is called:

1. **Load from DB** — if df has no value, load via
   `VS_6004_GetDataFromMariaDb.load_1Min_data_from_db` using symbol,
   start_date and end_date from the input parameters, into df.
2. **Completeness check** — verify every 1-minute datetime slot between
   start_date and `min(now, end_Date)` exists in df. Only CME Globex trading
   hours (ET) count:
   - Sunday: from 18:00
   - Monday–Thursday: all day except the 17:00–18:00 maintenance break
   - Friday: until 17:00
   - Saturday: closed
   - Exchange holidays are **not** accounted for
3. **Complete** — if all bars exist (or df already reaches end_Date), return
   df as-is, sorted ascending.
4. **Incomplete** — if max datetime in df is still < end_Date:
   - Save existing max datetime in df to `old_max_datetime`
   - Fetch OHLCV from IB using `reqHistoricalData` (1 min, `TRADES`,
     `useRTH=False`), period from (df max datetime − 5 minutes) to
     end_Date 23:59 (capped at now), symbol from input parameter; merge
     into df (dedupe, keep latest)
   - If `isCallOnce == True`: upsert df rows into MariaDB table
     `ticker1Min` from (`old_max_datetime` − 5 minutes) up to **but
     excluding** the newest df bar (`INSERT ... ON DUPLICATE KEY UPDATE`)
   - If `isCallOnce == False`: keep `keepUpToDate=True` request open;
     `onBarUpdate` keeps df live and inserts rows missing in `ticker1Min`
     every 5 minutes, excluding the newest df bar
5. Return df.

---

## 5. Supporting Helpers (implementation reference)

| Helper | Role |
|---|---|
| `_ensure_ib(conn1)` | Return a connected `ib_insync.IB`; 3 retry attempts across ports 7497 / 4002, raises `ConnectionError` on failure |
| `_use_delayed_market_data(ib)` | `reqMarketDataType(3)` — delayed data when live is not subscribed (avoids errors 354/10168) |
| `_front_contract(ib, symbol)` | Resolve symbol to qualified front-month CME future |
| `_ib_1min_bars(...)` | Completed 1-min TRADES bars as a DataFrame; negative volumes clipped to 0 |
| `_db_upsert_bars(...)` | Upsert df bars into `ticker1Min` (`ON DUPLICATE KEY UPDATE`) |
| `_db_insert_missing(...)` | Insert only df rows whose `datetime1` does not exist in `ticker1Min` |
| `_current_bar(...)` | In-progress bar from `reqMktData`, OHLC seeded with last price |
| `_expected_minutes(...)` | Yield 1-minute stamps inside CME Globex hours |
| `_missing_minutes(...)` | Expected market minutes in range missing from df |

## 6. CLI

```bash
python VS_6005_1_1min_glm.py --symbol ES --start 2026-06-01 --end 2026-06-20 \
    [--curr-bar-mkt-data] [--no-call-once]
```

Prints the last 10 rows of the consolidated df.
