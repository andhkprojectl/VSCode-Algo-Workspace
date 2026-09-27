"""
VS_6005_1_1min_glm.py
=====================
getLiveTradeIB1MinData(symbolName, conn1=None, start_data=None, end_Date=None,
                       isCurrBarMktData=False, isCallOnce=True)

Consolidates 1-minute CME future bars into one DataFrame (V2):

1. MariaDB ticker1Min bars for [start_data, end_Date] are loaded into df
   (VS_6004_GetDataFromMariaDb.load_1Min_data_from_db).
2. Completeness check: every 1-minute slot between start_data and
   min(now, end_Date) that falls inside CME Globex trading hours (ET:
   Sun from 18:00, Mon-Thu all day with a 17:00-18:00 maintenance break,
   Fri until 17:00; holidays NOT accounted for) must exist in df.
   When everything is present, df is returned as-is.
3. Otherwise IB reqHistoricalData (1-min, TRADES, useRTH=False) fetches
   from (df max datetime - 5 minutes) to end_Date 23:59 and the fetched
   bars are merged into df.
4. isCurrBarMktData=True: the in-progress current bar is taken from
   reqMktData; False (default): it comes from the historical bars.
5. isCallOnce=True: fetched bars are upserted into MariaDB ticker1Min
   from (old max datetime - 5 minutes) up to (but excluding) the newest
   df bar, then df is returned.
   isCallOnce=False: reqHistoricalData runs with keepUpToDate=True and an
   `onBarUpdate` callback (`bars.updateEvent += onBarUpdate`) keeps df
   live-updated and syncs df rows missing in ticker1Min into the DB every
   5 minutes (excluding the newest df bar). df is returned immediately;
   keep the process running and the IB connection open to receive updates.

Output: DataFrame with a DatetimeIndex named 'datetime' (US/Eastern naive,
matching MariaDB ticker1Min stamps) and columns
[open, high, low, close, volume], sorted ascending, deduplicated.
"""

import math
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import mysql.connector
import pandas as pd
from ib_insync import ContFuture, Future, IB

_DB_DIR = Path(__file__).resolve().parents[1] / "VS_6004_GetDataFromMariaDb"
if str(_DB_DIR) not in sys.path:
    sys.path.insert(0, str(_DB_DIR))

from VS_6004_GetDataFromMariaDb import _load_db_env, load_1Min_data_from_db  # noqa: E402

ET = ZoneInfo("America/New_York")
IB_HOST = "127.0.0.1"
IB_PORTS = (7497, 4002)  # TWS first, IB Gateway fallback
IB_CLIENT_ID = 5
DB_SYNC_INTERVAL_MIN = 5  # live-mode DB sync throttle (isCallOnce=False)

# keeps live keepUpToDate requests and their callbacks alive after return
_LIVE_UPDATES = []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _et_naive(dt):
    """Normalize any datetime to US/Eastern-naive (MariaDB stamp convention)."""
    dt = pd.Timestamp(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert(ET).tz_localize(None)
    return dt.to_pydatetime()


def _ensure_ib(conn1):
    """Return a connected ib_insync IB; connect with retries when needed.
    Tries TWS (7497) then IB Gateway (4002)."""
    ib = conn1 if conn1 is not None else IB()
    last_error = None
    for attempt in range(1, 4):
        try:
            if ib.isConnected():
                _use_delayed_market_data(ib)
                return ib
            for port in IB_PORTS:
                try:
                    print(f"IB connect attempt {attempt}/3 to {IB_HOST}:{port} "
                          f"clientId={IB_CLIENT_ID}")
                    ib.connect(IB_HOST, port, clientId=IB_CLIENT_ID)
                    if ib.isConnected():
                        _use_delayed_market_data(ib)
                        print(f"IB connect OK to {IB_HOST}:{port}")
                        return ib
                except Exception as e:
                    last_error = e
                    print(f"IB connect error attempt {attempt}/3 port {port}: [{e}]")
        except Exception as e:
            last_error = e
            print(f"IB connect error attempt {attempt}/3: [{e}]")
        ib.sleep(2)
    raise ConnectionError(f"cannot connect IB after 3 attempts "
                          f"({IB_HOST}: {', '.join(str(p) for p in IB_PORTS)}): "
                          f"[{last_error}]")


def _use_delayed_market_data(ib):
    """Accounts without a live subscription: request delayed market data so
    reqMktData does not raise errors 354/10168."""
    try:
        ib.reqMarketDataType(3)  # 3 = delayed data when live is not subscribed
    except Exception:
        pass


def _front_contract(ib, symbol_name):
    """Current front-month CME future for the symbol (qualifies it)."""
    cont = ContFuture(symbol_name, "CME")
    ib.qualifyContracts(cont)
    contract = Future(localSymbol=cont.localSymbol, exchange="CME")
    ib.qualifyContracts(contract)
    return contract


def _ib_1min_bars(ib, contract, end_et_naive, duration):
    """Completed 1-min TRADES bars ending at end_et_naive, as a DataFrame
    with an ET-naive DatetimeIndex and columns [open, high, low, close, volume]."""
    # TWS reads a naive endDateTime in the TWS timezone setting - always send
    # an aware UTC stamp so the requested end time is unambiguous
    end_ib = end_et_naive.replace(tzinfo=ET).astimezone(timezone.utc)
    bars = ib.reqHistoricalData(
        contract, endDateTime=end_ib, durationStr=duration,
        barSizeSetting="1 min", whatToShow="TRADES", useRTH=False, formatDate=1)
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    rows = [(b.date, b.open, b.high, b.low, b.close, b.volume) for b in bars]
    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    df["volume"] = df["volume"].clip(lower=0.0)  # delayed feeds use -1 for unknown
    return df


def _db_connect():
    _load_db_env()
    import os
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )


def _db_upsert_bars(symbol_name, df, conn1=None):
    """Upsert df bars into MariaDB ticker1Min (IBDb isOverride pattern)."""
    if df.empty:
        return 0
    own_conn = conn1 is None
    conn = _db_connect() if own_conn else conn1
    try:
        cur = conn.cursor()
        sql = ("INSERT INTO ticker1Min (ticker, datetime1, open, high, low, close, volume) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s) "
               "ON DUPLICATE KEY UPDATE open = VALUES(open), high = VALUES(high), "
               "low = VALUES(low), close = VALUES(close), volume = VALUES(volume)")
        records = [(symbol_name, idx.strftime("%Y-%m-%d %H:%M:%S"), r.open, r.high,
                    r.low, r.close, r.volume) for idx, r in df.iterrows()]
        cur.executemany(sql, records)
        conn.commit()
        return cur.rowcount
    finally:
        if own_conn:
            conn.close()


def _db_insert_missing(symbol_name, df, conn1=None):
    """Insert only the df rows whose datetime1 does not exist in ticker1Min."""
    if df.empty:
        return 0
    own_conn = conn1 is None
    conn = _db_connect() if own_conn else conn1
    try:
        cur = conn.cursor()
        lo = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
        hi = df.index.max().strftime("%Y-%m-%d %H:%M:%S")
        cur.execute("SELECT datetime1 FROM ticker1Min WHERE ticker = %s "
                    "AND datetime1 BETWEEN %s AND %s", (symbol_name, lo, hi))
        existing = {row[0] for row in cur.fetchall()}
        missing = df[~pd.to_datetime(df.index).isin(existing)]
        if missing.empty:
            return 0
        sql = ("INSERT INTO ticker1Min (ticker, datetime1, open, high, low, close, volume) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s)")
        records = [(symbol_name, idx.strftime("%Y-%m-%d %H:%M:%S"), r.open, r.high,
                    r.low, r.close, r.volume) for idx, r in missing.iterrows()]
        cur.executemany(sql, records)
        conn.commit()
        return len(records)
    finally:
        if own_conn:
            conn.close()


def _current_bar(ib, contract, now_et):
    """In-progress bar from reqMktData: OHLC seeded with the last price."""
    price = None
    for _ in range(3):  # first snapshot may error 354 before delayed kicks in
        ticker = ib.reqMktData(contract, "", False, False)
        ib.sleep(1.0)
        price = ticker.last
        if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
            price = ticker.close
        if price is not None and not (isinstance(price, float) and math.isnan(price)) \
                and price > 0:
            break
    if price is None or (isinstance(price, float) and math.isnan(price)) or price <= 0:
        return None
    dt = now_et.replace(second=0, microsecond=0)
    return pd.DataFrame(
        {"open": [float(price)], "high": [float(price)], "low": [float(price)],
         "close": [float(price)], "volume": [0.0]},
        index=pd.DatetimeIndex([dt], name="datetime"))


def _expected_minutes(start_dt, end_dt):
    """1-minute stamps in [start, end] inside CME Globex hours (ET):
    Sun from 18:00, Mon-Thu all day except 17:00-18:00, Fri until 17:00,
    Saturday closed. Exchange holidays are not accounted for."""
    cur = start_dt.floor("min")
    end = end_dt.floor("min")
    while cur <= end:
        wd = cur.weekday()  # Mon=0 .. Sun=6
        t = cur.time()
        if wd == 6:
            ok = t >= time(18, 0)          # Sunday: session opens 18:00 ET
        elif wd == 5:
            ok = False                     # Saturday: closed
        elif wd == 4:
            ok = t < time(17, 0)           # Friday: session ends 17:00 ET
        else:
            ok = not (time(17, 0) <= t < time(18, 0))  # daily maintenance
        if ok:
            yield cur
        cur += timedelta(minutes=1)


def _missing_minutes(df_index, start_dt, horizon):
    """Expected market minutes in [start, horizon] missing from df."""
    existing = pd.DatetimeIndex(df_index)
    return [m for m in _expected_minutes(start_dt, horizon) if m not in existing]


# ---------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------
def getLiveTradeIB1MinData(symbolName, conn1=None, start_data=None, end_Date=None,
                           isCurrBarMktData=False, isCallOnce=True):
    """Consolidated 1-min bars for symbolName (see module docstring).

    start_data / end_Date accept str or datetime-like; a bare end date is
    treated as its 23:59:59. Returns a DataFrame indexed by datetime
    (ET-naive) with columns [open, high, low, close, volume].
    """
    start_dt = pd.Timestamp(start_data) if start_data is not None else None
    end_dt = pd.Timestamp(end_Date) if end_Date is not None else None
    if end_dt is not None and (end_dt.hour == 0 and end_dt.minute == 0
                               and end_dt.second == 0):
        end_dt = end_dt + timedelta(hours=23, minutes=59, seconds=59)

    now_et = _et_naive(datetime.now(ET))

    # 1. df from MariaDB for the requested range
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if start_dt is not None:
        df = load_1Min_data_from_db(symbolName, str(start_dt), str(end_dt))
    df.index = pd.DatetimeIndex(df.index, name="datetime")
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = pd.Series(dtype=float)

    # 2. completeness: every market minute up to min(now, end) must exist
    horizon = min(end_dt, pd.Timestamp(_et_naive(datetime.now(ET)))) \
        if end_dt is not None else pd.Timestamp(_et_naive(datetime.now(ET)))
    missing = _missing_minutes(df.index, start_dt, horizon)
    if not missing:
        return df.sort_index()
    # already reached the requested end: leftover gaps are non-trading minutes
    if not df.empty and df.index.max() >= end_dt:
        return df.sort_index()

    # 3. fetch from IB
    ib = _ensure_ib(conn1)
    contract = _front_contract(ib, symbolName)

    old_max = df.index.max() if not df.empty else None
    fetch_start = (old_max - timedelta(minutes=5)) if old_max is not None else start_dt
    fetch_start = fetch_start.floor("min")
    end_fetch_et = min(end_dt, now_et)  # never request beyond now (TWS quirk)
    days = max(1, (end_fetch_et - fetch_start).days + 1)
    duration = f"{days} D"

    if isCallOnce:
        bars = _ib_1min_bars(ib, contract, end_fetch_et, duration)
        parts = [p for p in (df, bars) if not p.empty]
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="last")].sort_index()
    else:
        # live mode: the initial request stays open and streams updates
        bars = ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
            formatDate=1, keepUpToDate=True)
        for idx, row in bars.iterrows():
            dt = _et_naive(idx)
            df.loc[dt] = [row.open, row.high, row.low, row.close,
                          max(row.volume, 0.0)]
        df.sort_index(inplace=True)

        live_state = {"last_sync": None}

        def onBarUpdate(updated_bars, has_new_bar):
            now = _et_naive(datetime.now(ET))
            for b in updated_bars:
                dt = _et_naive(b.date)
                df.loc[dt] = [b.open, b.high, b.low, b.close, max(b.volume, 0.0)]
            df.sort_index(inplace=True)
            last_sync = live_state["last_sync"]
            if last_sync is not None and (now - last_sync) < timedelta(
                    minutes=DB_SYNC_INTERVAL_MIN):
                return
            live_state["last_sync"] = now
            if df.empty:
                return
            newest = df.index.max()
            # note: conn1 is the IB connection - DB helpers open their own
            # MariaDB connection
            _db_insert_missing(symbolName, df[df.index < newest])

        bars.updateEvent += onBarUpdate
        _LIVE_UPDATES.append((bars, onBarUpdate, df))

    # 4. current bar (in-progress minute) from reqMktData when requested
    if isCurrBarMktData and end_dt > pd.Timestamp(_et_naive(datetime.now(ET))):
        current_bar = _current_bar(ib, contract, _et_naive(datetime.now(ET)))
        if current_bar is not None:
            for idx, row in current_bar.iterrows():
                df.loc[idx] = row
            df.sort_index(inplace=True)

    # 5. persist to MariaDB (isCallOnce only; exclude the newest df bar)
    if isCallOnce and not df.empty:
        newest = df.index.max()
        up_start = old_max - timedelta(minutes=5) if old_max is not None \
            else fetch_start - timedelta(minutes=5)
        rows = df[(df.index >= up_start) & (df.index < newest)]
        _db_upsert_bars(symbolName, rows)

    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="getLiveTradeIB1MinData")
    parser.add_argument("--symbol", default="ES")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--curr-bar-mkt-data", action="store_true")
    parser.add_argument("--no-call-once", dest="call_once", action="store_false")
    args = parser.parse_args()
    df = getLiveTradeIB1MinData(args.symbol, None, args.start, args.end,
                                args.curr_bar_mkt_data, args.call_once)
    print(df.tail(10))


if __name__ == "__main__":
    main()
