"""
test_S4003_live.py
==================
Offline unit tests for S4003_live.py (TASKS.md T4.1 verification).
NO IB connection, NO MariaDB writes, NO real orders:
  - doTrade0 is stubbed (records kwargs, never touches IB)
  - test A scripts strategy signals via the injectable prepare_fn
  - test B replays a real MariaDB slice through the real strategyS4003
    (skipped automatically if the DB is unreachable)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from S4003_live import (  # noqa: E402
    S4003LiveTrade, SIGNAL_LOG_NAME, DOTRADE0_LOG_NAME, DEBUG_LOG_NAME)
from S4003_strategy import strategyS4003  # noqa: E402

ATR = 2.0          # scripted atr14
STOP_PT = 3.0      # scripted max3BarAtr14_1
PM = strategyS4003.profit_mult              # 1.5


class TradeStub:
    """doTrade0 replacement: records every call, returns 1."""

    def __init__(self):
        self.calls = []

    def __call__(self, auto_trade, contract, buy, sell, short, cover,
                 limit_price, stop_loss_price, profit_take_price, num_contracts,
                 ib_controller_tick_size, is_intra_day, is_mkt_order,
                 max_buy_short_slip, max_stp_lmt_slip, symbol_type,
                 *, port, host, client_id, **kw):
        self.calls.append(dict(
            auto_trade=auto_trade,
            buy=buy, sell=sell, short=short, cover=cover,
            limit_price=limit_price, stop_loss_price=stop_loss_price,
            profit_take_price=profit_take_price, num_contracts=num_contracts,
            ib_controller_tick_size=ib_controller_tick_size,
            is_mkt_order=is_mkt_order,
            port=port, host=host, client_id=client_id,
            symbol_type=symbol_type))
        return 1


def make_config(tmp_path, live_trade):
    return {
        "symbolName": "ES", "futureExchange": "CME", "tableName": "ticker1Min",
        "tradeHost": "127.0.0.1", "tradePort": 4002, "clientId": 7,
        "positionSize": 1, "initCapital": 80000.0, "pointValue": 50.0,
        "liveTrade": live_trade, "isDebug": False, "symbolType": 1,
        "tickSize": 0.25, "isIntraDay": True, "isMktOrder": True,
        "maxNumTickSize4BuyShortSlippage": 4,
        "maxNumTickSize4StpLmtOrderSlippage": 4,
        "outputFolder": str(tmp_path),
    }


def scripted_prepare(signals_by_pos):
    """prepare_fn returning buySig/shortSig at scripted row positions."""

    def _prepare(df):
        out = df.copy()
        out["Open"] = out["open"]
        out["High"] = out["high"]
        out["Low"] = out["low"]
        out["Close"] = out["close"]
        n = len(out)
        out["buySig"] = 0
        out["shortSig"] = 0
        for pos, side in signals_by_pos.items():
            if pos < n:
                out.iloc[pos, out.columns.get_loc(
                    "buySig" if side == "buy" else "shortSig")] = 1
        out["atr14"] = ATR
        out["max3BarAtr14_1"] = STOP_PT
        return out

    return _prepare


def flat_df(n=400, start="2025-08-01 09:00"):
    idx = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5,
                         "close": 100.0, "volume": 10.0}, index=idx)


def feed(trader, df, upto):
    """Replay bars 0..upto: appending bar k makes bar k-1 the last closed one."""
    trader.bars_df = df.iloc[: upto + 1].copy()
    return trader.evaluate_latest_bar()


# ---------------------------------------------------------------------------
# A. signal -> doTrade0 flag/price mapping with scripted signals
# ---------------------------------------------------------------------------
def test_mapping_observe_mode(tmp_path):
    stub = TradeStub()
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=stub,
                            prepare_fn=scripted_prepare({150: "buy", 160: "short"}))
    df = flat_df()

    # --- BUY scripted at row 150: fires when bar 150 is the latest ---
    act = feed(trader, df, 150)
    assert act["event"] == "BUY" and act["pos_after"] == 1
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (1, 0, 0, 0)
    assert act["limit_price"] == pytest.approx(100.0)                # market at Open
    assert act["stop_loss"] == pytest.approx(100.0 - STOP_PT)        # 97.00
    assert act["profit_take"] == pytest.approx(100.0 + PM * STOP_PT)  # 104.50
    assert stub.calls == []                       # OBSERVE: doTrade0 never called

    # --- SHORT while long, scripted at row 160: force reversal ---
    act = feed(trader, df, 160)
    assert act["event"] == "SHORT+SELL" and act["pos_after"] == -1
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (0, 1, 1, 0)
    assert act["limit_price"] == pytest.approx(100.0)                # market at Open
    assert act["stop_loss"] == pytest.approx(100.0 + STOP_PT)        # 103.00
    assert act["profit_take"] == pytest.approx(100.0 - PM * STOP_PT)  # 95.50
    assert stub.calls == []

    # --- N-bar stop: short entered at bar 161, stop_period1=15 -> COVER at 175 ---
    act = feed(trader, df, 174)
    assert act is None
    act = feed(trader, df, 175)
    assert act["event"] == "COVER (nbar)" and act["pos_after"] == 0
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (0, 0, 0, 1)
    assert act["limit_price"] == pytest.approx(100.0)
    assert act["stop_loss"] == 0.0 and act["profit_take"] == 0.0
    assert stub.calls == []

    # signal log written with the three events
    log = (tmp_path / SIGNAL_LOG_NAME).read_text()
    assert "BUY;" in log and "SHORT+SELL" in log and "COVER (nbar)" in log
    assert "OBSERVE" in log


def test_mapping_live_mode_calls_doTrade0(tmp_path):
    stub = TradeStub()
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=True),
                            do_trade_fn=stub,
                            prepare_fn=scripted_prepare({150: "short", 160: "buy"}))
    trader.ib = object()
    df = flat_df()

    act = feed(trader, df, 150)   # SHORT entry (flat)
    assert act["event"] == "SHORT"
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (0, 0, 1, 0)

    act = feed(trader, df, 160)   # BUY while short -> buy+cover reversal
    assert act["event"] == "BUY+COVER"
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (1, 0, 0, 1)

    assert len(stub.calls) == 2
    c0, c1 = stub.calls
    assert (c0["short"], c0["cover"]) == (1, 0)
    assert c0["limit_price"] == pytest.approx(100.0)
    assert c0["stop_loss_price"] == pytest.approx(100.0 + STOP_PT)
    assert c0["profit_take_price"] == pytest.approx(100.0 - PM * STOP_PT)
    assert (c1["buy"], c1["cover"]) == (1, 1)
    for c in (c0, c1):
        assert c["auto_trade"] is trader.ib
        assert c["port"] == 4002 and c["host"] == "127.0.0.1"
        assert c["client_id"] == 7 and c["symbol_type"] == 1
        assert c["num_contracts"] == 1
        assert c["ib_controller_tick_size"] == 2500
        assert c["is_mkt_order"] is True
    assert strategyS4003.entry_price_mode == "market"
    assert strategyS4003.num_contracts == 1
    assert strategyS4003.initial_capital == 80000.0
    assert strategyS4003.point_value == 50.0

    log = (tmp_path / SIGNAL_LOG_NAME).read_text()
    assert "mode=LIVE" in log and "doTrade0=[1]" in log


def test_debug_log_redirect(tmp_path):
    S4003LiveTrade(make_config(tmp_path, live_trade=False), do_trade_fn=TradeStub())
    import ibAutotrade
    assert ibAutotrade.LOG_FILE == str(tmp_path / DOTRADE0_LOG_NAME)


def test_trade_debug_log_writes_warmup_and_order_details(tmp_path):
    cfg = make_config(tmp_path, live_trade=False)
    cfg["isDebug"] = True
    trader = S4003LiveTrade(cfg, do_trade_fn=TradeStub(),
                            prepare_fn=scripted_prepare({150: "buy"}))
    df = flat_df()

    assert feed(trader, df, 10) is None
    act = feed(trader, df, 150)
    assert act["event"] == "BUY"

    lines = (tmp_path / DEBUG_LOG_NAME).read_text().splitlines()
    assert len(lines) == 2
    assert "HOLD (warm-up)" in lines[0]
    assert "OHLCV=" in lines[0] and '"open": 100.0' in lines[0]
    assert '"event": "NONE"' in lines[0]
    assert f"bar=[{df.index[150]}]" in lines[1]
    assert '"event": "BUY"' in lines[1]
    assert '"buy": 1' in lines[1]
    assert '"isMktOrder": true' in lines[1]
    assert '"ibControllerTickSize": 2500' in lines[1]
    assert '"numContracts": 1' in lines[1]
    assert '"doTrade0Result": "SKIPPED"' in lines[1]


def test_trade_debug_log_disabled(tmp_path):
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            prepare_fn=scripted_prepare({150: "buy"}))
    feed(trader, flat_df(), 150)
    assert not (tmp_path / DEBUG_LOG_NAME).exists()


# ---------------------------------------------------------------------------
# B. real strategy + real MariaDB slice, bar-by-bar replay, stubbed doTrade0
# ---------------------------------------------------------------------------
def test_real_strategy_db_replay(tmp_path):
    pytest.importorskip("mysql.connector")
    try:
        from VS_6004_GetDataFromMariaDb import load_1Min_data_from_db
        df = load_1Min_data_from_db("ES", "2025-08-01 00:00:00",
                                    "2025-08-01 23:59:59")
    except Exception as e:
        pytest.skip(f"MariaDB unreachable: {e}")
    if len(df) < 500:
        pytest.skip("not enough bars in DB slice")

    stub = TradeStub()
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=stub)  # real strategyS4003._prepare
    actions = []
    step = 200
    trader.bars_df = df.iloc[:step].copy()
    for k in range(step, len(df)):
        trader.bars_df = df.iloc[: k + 1]
        act = trader.evaluate_latest_bar()
        if act:
            actions.append(act)

    events = [a["event"] for a in actions]
    print(f"\nDB replay: {len(df)} bars, {len(actions)} events: {events}")
    assert len(actions) > 0, "real strategy produced no events on the DB slice"
    assert stub.calls == [], "doTrade0 called in OBSERVE mode"
    # every event mapped to exactly one side
    for a in actions:
        assert sum([a["buy"], a["sell"], a["short"], a["cover"]]) >= 1
    log = (tmp_path / SIGNAL_LOG_NAME).read_text()
    for a in actions:
        assert a["event"].split(" ")[0] in log


# ---------------------------------------------------------------------------
# C. FR-6 init gap backfill (IB -> df + ticker1Min), stubbed fetch/save
# ---------------------------------------------------------------------------
def _gap_fetch_result(symbol, start_dt, n=5):
    """Fake ibMarketData.getAllTypesTicketDataWithTimeFromIB result: n 1-min
    bars starting at start_dt, in the (date, time, ...) ingest output shape."""
    idx = pd.date_range(start_dt, periods=n, freq="min")
    return pd.DataFrame({
        "datetime": idx.strftime("%m/%d/%Y %H:%M"),
        "date": idx.strftime("%Y-%m-%d"),
        "time": idx.strftime("%H:%M:%S"),
        "open": 101.0, "high": 101.5, "low": 100.5, "close": 101.0,
        "volume": 5.0, "symbolName": symbol,
    }, index=idx.strftime("%Y-%m-%d %H:%M:%S"))


def test_backfill_gap_fills_df_and_db(tmp_path):
    saved = []
    fetched = []

    def fetch_stub(conn1, symbolName, startDate, startTime, endDate, endTime,
                   **kw):
        fetched.append(dict(startDate=startDate, startTime=startTime,
                            endDate=endDate, endTime=endTime))
        start_dt = pd.Timestamp(f"{startDate} {startTime[:2]}:{startTime[2:]}")
        end_dt = pd.Timestamp(f"{endDate} {endTime[:2]}:{endTime[2:]}")
        n = int((end_dt - start_dt).total_seconds() // 60) + 1
        return _gap_fetch_result(symbolName, start_dt, n)

    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            fetch_history_fn=fetch_stub,
                            save_df_fn=lambda df, **kw: saved.append((df, kw)))
    now = pd.Timestamp("2025-08-01 10:20:30")      # prev minute = 10:19
    # in-memory df ends at 10:15 -> gap 10:16..10:19 (4 bars)
    trader.bars_df = flat_df(n=76, start="2025-08-01 09:00")
    assert trader.bars_df.index.max() == pd.Timestamp("2025-08-01 10:15")
    trader.ib = object()                            # any non-None connection

    n = trader.backfill_gap_from_ib(now=now)
    assert n == 4
    assert trader.bars_df.index.max() == pd.Timestamp("2025-08-01 10:19")
    assert len(trader.bars_df) == 80
    # saved to DB with the ingest output shape and the configured table
    assert len(saved) == 1 and saved[0][1]["tableName"] == "ticker1Min"
    assert saved[0][1]["isOverride"] is True
    for col in ("symbolName", "date", "time", "open", "high", "low",
                "close", "volume"):
        assert col in saved[0][0].columns
    assert fetched[0]["startDate"] == "20250801" and fetched[0]["startTime"] == "1016"
    assert fetched[0]["endDate"] == "20250801" and fetched[0]["endTime"] == "1019"


def test_backfill_gap_not_needed_when_current(tmp_path):
    fetched = []
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            fetch_history_fn=lambda *a, **k: fetched.append(1),
                            save_df_fn=lambda df, **kw: None)
    trader.bars_df = flat_df(n=76, start="2025-08-01 09:00")   # last bar 10:15
    trader.ib = object()
    assert trader.backfill_gap_from_ib(
        now=pd.Timestamp("2025-08-01 10:15:45")) == 0           # prev min 10:14 < last
    assert trader.backfill_gap_from_ib(
        now=pd.Timestamp("2025-08-01 10:16:05")) == 0           # prev min 10:15 == last
    assert trader._prev_minute_et(
        pd.Timestamp("2025-08-01 14:16:05", tz="UTC")) == pd.Timestamp("2025-08-01 10:15")
    assert fetched == []


# ---------------------------------------------------------------------------
# D. FR-6 periodic flush: last 12 minutes, current minute bar excluded
# ---------------------------------------------------------------------------
def test_flush_recent_bars_window(tmp_path):
    saved = []
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            save_df_fn=lambda df, **kw: saved.append(df) or len(df))
    trader.bars_df = flat_df(n=31, start="2025-08-01 09:50")    # 09:50..10:20
    n = trader.flush_recent_bars_to_db(now=pd.Timestamp("2025-08-01 10:20:30"))
    assert n == 12                                               # 10:08..10:19
    assert len(saved) == 1
    df = saved[0]
    assert df["date"].iloc[0] == "2025-08-01"
    assert df["time"].iloc[0] == "10:08:00"
    assert df["time"].iloc[-1] == "10:19:00"                     # 10:20 excluded
    assert (df["symbolName"] == "ES").all()


def test_maybe_flush_db_cadence(tmp_path):
    saved = []
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            save_df_fn=lambda df, **kw: saved.append(df) or len(df))
    trader.bars_df = flat_df(n=31, start="2025-08-01 09:50")
    t0 = pd.Timestamp("2025-08-01 10:20:30")
    trader.maybe_flush_db(now=t0)    # first call flushes
    trader.maybe_flush_db(now=t0 + pd.Timedelta(minutes=3))   # within 8 min -> skipped
    assert len(saved) == 1
    trader.maybe_flush_db(now=t0 + pd.Timedelta(minutes=9))   # timer elapsed -> flush
    assert len(saved) == 2


def test_backfill_single_minute_gap_trims_overfetch(tmp_path):
    saved = []
    fetched = []

    def fetch_stub(conn1, symbolName, startDate, startTime, endDate, endTime,
                   **kw):
        fetched.append(dict(endDate=endDate, endTime=endTime))
        start_dt = pd.Timestamp(f"{startDate} {startTime[:2]}:{startTime[2:]}")
        return _gap_fetch_result(symbolName, start_dt, 2)

    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            fetch_history_fn=fetch_stub,
                            save_df_fn=lambda df, **kw: saved.append((df, kw)))
    trader.bars_df = flat_df(n=79, start="2025-08-01 09:00")  # last bar 10:18
    trader.ib = object()

    n = trader.backfill_gap_from_ib(now=pd.Timestamp("2025-08-01 10:20:30"))
    assert n == 1
    assert fetched[0]["endTime"] == "1020"  # ingest helper needs a non-zero range
    assert trader.bars_df.index.max() == pd.Timestamp("2025-08-01 10:19")
    assert len(saved) == 1 and len(saved[0][0]) == 1
    assert saved[0][0]["time"].iloc[0] == "10:19:00"


def test_flush_recent_bars_accepts_utc_timestamp(tmp_path):
    saved = []
    trader = S4003LiveTrade(make_config(tmp_path, live_trade=False),
                            do_trade_fn=TradeStub(),
                            save_df_fn=lambda df, **kw: saved.append(df) or len(df))
    trader.bars_df = flat_df(n=31, start="2025-08-01 09:50")
    now_utc = pd.Timestamp("2025-08-01 14:20:30", tz="UTC")  # 10:20:30 ET (EDT)
    assert trader.flush_recent_bars_to_db(now=now_utc) == 12
    assert saved[0]["time"].iloc[-1] == "10:19:00"


def test_tick_size_drives_ib_controller_tick_size(tmp_path):
    cfg = make_config(tmp_path, live_trade=False)
    cfg["tickSize"] = 0.5
    cfg["ibControllerTickSize"] = 9999
    trader = S4003LiveTrade(cfg, do_trade_fn=TradeStub())
    assert trader.ib_controller_tick_size == 5000


def test_execution_details_saved_and_deferred_until_order_registration(tmp_path):
    saved = []
    trader = S4003LiveTrade(
        make_config(tmp_path, live_trade=False), do_trade_fn=TradeStub(),
        save_execution_fn=lambda record, **kw: saved.append((record, kw)) or 1)
    trader.submitted_order_ids.add(42)
    contract = SimpleNamespace(symbol="ES")
    trade = SimpleNamespace(contract=contract)
    execution = SimpleNamespace(
        orderId=42, execId="EXEC-42", time=pd.Timestamp("2025-08-01 14:30", tz="UTC"),
        side="BOT", shares=1, price=6000.25, exchange="CME", acctNumber="DU123")
    fill = SimpleNamespace(execution=execution, contract=contract)

    record = trader.on_exec_details(trade, fill)
    assert record["strategy"] == "S4003"
    assert record["executionTimeUtc"] == pd.Timestamp("2025-08-01 14:30").to_pydatetime()
    assert len(saved) == 1
    assert saved[0][1]["tableName"] == "tradeExecution"
    assert saved[0][0]["side"] == "BOT" and saved[0][0]["quantity"] == 1.0

    assert trader.on_exec_details(trade, fill) is None  # duplicate execId
    assert len(saved) == 1

    pending_execution = SimpleNamespace(
        orderId=43, execId="EXEC-43", time=pd.Timestamp("2025-08-01 14:31", tz="UTC"),
        side="SLD", shares=1, price=6001.25, exchange="CME", acctNumber="DU123")
    pending_fill = SimpleNamespace(execution=pending_execution, contract=contract)
    trader.on_exec_details(trade, pending_fill)
    assert len(saved) == 1 and 43 in trader.pending_executions

    trader.ib = SimpleNamespace(
        trades=lambda: [SimpleNamespace(order=SimpleNamespace(orderId=43))])
    assert trader._register_new_orders(before=set()) == [43]
    assert len(saved) == 2 and saved[1][0]["execId"] == "EXEC-43"
    assert 43 not in trader.pending_executions
