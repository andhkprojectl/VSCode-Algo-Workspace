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

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from S4003_live import S4003LiveTrade, SIGNAL_LOG_NAME, DOTRADE0_LOG_NAME  # noqa: E402
from S4003_strategy import strategyS4003  # noqa: E402

ATR = 2.0          # scripted atr14
STOP_PT = 3.0      # scripted max3BarAtr14_1
FRAC = strategyS4003.entry_limit_atr_frac   # 4.0
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
            buy=buy, sell=sell, short=short, cover=cover,
            limit_price=limit_price, stop_loss_price=stop_loss_price,
            profit_take_price=profit_take_price, num_contracts=num_contracts,
            port=port, host=host, client_id=client_id,
            symbol_type=symbol_type))
        return 1


def make_config(tmp_path, live_trade):
    return {
        "symbolName": "ES", "futureExchange": "CME", "tableName": "ticker1Min",
        "tradeHost": "127.0.0.1", "tradePort": 4002, "clientId": 7,
        "positionSize": 1, "liveTrade": live_trade, "symbolType": 1,
        "ibControllerTickSize": 2500, "isIntraDay": True, "isMktOrder": False,
        "maxNumTickSize4BuyShortSlippage": 4,
        "maxNumTickSize4StpLmtOrderSlippage": 4,
        "outputFolder": str(tmp_path),
    }


def scripted_prepare(signals_by_pos):
    """prepare_fn returning buySig/shortSig at scripted row positions."""

    def _prepare(df):
        out = df.copy()
        out["Open"] = out["open"]
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
    assert act["limit_price"] == pytest.approx(100.0 - ATR / FRAC)   # 99.50
    assert act["stop_loss"] == pytest.approx(100.0 - STOP_PT)        # 97.00
    assert act["profit_take"] == pytest.approx(100.0 + PM * STOP_PT)  # 104.50
    assert stub.calls == []                       # OBSERVE: doTrade0 never called

    # --- SHORT while long, scripted at row 160: force reversal ---
    act = feed(trader, df, 160)
    assert act["event"] == "SHORT+SELL" and act["pos_after"] == -1
    assert (act["buy"], act["sell"], act["short"], act["cover"]) == (0, 1, 1, 0)
    assert act["limit_price"] == pytest.approx(100.0 + ATR / FRAC)   # 100.50
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
    assert c0["limit_price"] == pytest.approx(100.0 + ATR / FRAC)
    assert c0["stop_loss_price"] == pytest.approx(100.0 + STOP_PT)
    assert c0["profit_take_price"] == pytest.approx(100.0 - PM * STOP_PT)
    assert (c1["buy"], c1["cover"]) == (1, 1)
    for c in (c0, c1):
        assert c["port"] == 4002 and c["host"] == "127.0.0.1"
        assert c["client_id"] == 7 and c["symbol_type"] == 1
        assert c["num_contracts"] == 1

    log = (tmp_path / SIGNAL_LOG_NAME).read_text()
    assert "mode=LIVE" in log and "doTrade0=[1]" in log


def test_debug_log_redirect(tmp_path):
    S4003LiveTrade(make_config(tmp_path, live_trade=False), do_trade_fn=TradeStub())
    import ibAutotrade
    assert ibAutotrade.LOG_FILE == str(tmp_path / DOTRADE0_LOG_NAME)


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
