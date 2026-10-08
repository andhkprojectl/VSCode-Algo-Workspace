"""
S4003_live.py
=============
Live trade for VS_4003_20260827_ES (TASKS.md T4.1).

- Connects to IB TWS/Gateway (config `Live Trade`: tradeHost/tradePort/clientId)
  and calls reqHistoricalData(keepUpToDate=True) for 1-min ES bars.
- On every bar change the updateEvent callback appends the bar to a rolling
  DataFrame (seeded from MariaDB ticker1Min via
  VS_6004_GetDataFromMariaDb.load_1Min_data_from_db) and calls the strategy
  class strategyS4003 from `2 Strategy/S4003_strategy.py` (no imports from
  `1 Prelimary Test`).
- Signal -> order mapping (doTrade0 in VS_0008_liveTrade/ibAutotrade.py):
      BUY   entry (flat)            -> buy=1                         + bracket stop/profit
      SHORT entry (flat)            -> short=1                       + bracket stop/profit
      SHORT while long (force exit) -> sell=1, short=1  (doTrade0 reversal branch)
      BUY   while short (force exit)-> buy=1, cover=1   (doTrade0 reversal branch)
      N-bar stop exit (long)        -> sell=1
      N-bar stop exit (short)       -> cover=1
  Entry prices mirror the AFL-faithful engine (strategyS4003._simulate):
      buy   limit = Open - atr14[prev]/e1, stop = Open - max3BarAtr14_1,
            profit = Open + profitMult * max3BarAtr14_1
      short limit = Open + atr14[prev]/e1, stop = Open + max3BarAtr14_1,
            profit = Open - profitMult * max3BarAtr14_1
  where Open is the just-opened bar's open (entry fills at next-bar Open).

SAFETY: config `Live Trade`.liveTrade=false -> OBSERVE/paper mode. Signals are
computed and written to the signal log, position state is tracked, but
doTrade0 is NEVER called. Orders are only placed with liveTrade=true (flip
only after backtest/walkforward/other tests pass; live ports 4001/7496 must
not be used before then).

RF model hook: config `Live Trade`.modelFile points at the RandomForest gate
pkl from the backtest program. The gate is intentionally NOT wired yet (the
requirement file L4003_1_randomForest_glm.txt is empty - no spec for how the
RF probability should gate live entries). See _passes_rf_gate(): it currently
returns True so the base strategy signals drive orders. Wire the model there
once the gating rule is specified.

Outputs (config `Live Trade`.outputFolder):
  S4003_live_signals.log     - one line per buy/short/sell/cover event
  S4003_doTrade0_debug.log   - doTrade0 debug log (ibAutotrade.LOG_FILE is
                               redirected here at runtime)

Import-safe: no IB/DB connection and no config access at import time.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_STRATEGY_DIR = Path(__file__).resolve().parents[1] / "2 Strategy"
for _p in (str(_STRATEGY_DIR),
           str(_ROOT / "VS_0008_liveTrade"),
           str(_ROOT / "VS_0006_dataFunc" / "VS_6004_GetDataFromMariaDb")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_strategy import strategyS4003, loadStrategyConfig  # noqa: E402
import ibAutotrade  # noqa: E402
from ibAutotrade import doTrade0  # noqa: E402
from VS_6004_GetDataFromMariaDb import load_1Min_data_from_db  # noqa: E402

ET = ZoneInfo("America/New_York")
PROGRAM_NAME = "S4003_live"
SIGNAL_LOG_NAME = "S4003_live_signals.log"
DOTRADE0_LOG_NAME = "S4003_doTrade0_debug.log"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def loadLiveTradeConfig(config_path=None) -> dict:
    """Return the `Live Trade` section of the local strategy config."""
    if config_path is None:
        config_path = (_ROOT / "VS_4000_strategy" / "VS_4003_20260827_ES"
                       / "config.yaml")
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("Live Trade") or {}


def _et_naive(dt):
    """Normalize any datetime to US/Eastern-naive (MariaDB stamp convention,
    same as VS_6005_1_1min_glm.py)."""
    dt = pd.Timestamp(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert(ET).tz_localize(None)
    return dt.to_pydatetime()


# ---------------------------------------------------------------------------
# live trader
# ---------------------------------------------------------------------------
class S4003LiveTrade:
    """Rolling-bars live trader for strategyS4003.

    do_trade_fn / prepare_fn are injectable so the signal->order mapping can
    be unit-tested offline with a stubbed doTrade0 and scripted signals.
    """

    MIN_BARS = 150  # warm-up guard: percentile(100) + vwap50 need history

    def __init__(self, config=None, do_trade_fn=None, prepare_fn=None):
        self.config = loadLiveTradeConfig() if config is None else (config or {})
        cfg = self.config

        self.symbol = cfg.get("symbolName", "ES")
        self.future_exchange = cfg.get("futureExchange", "CME")
        self.table_name = cfg.get("tableName", "ticker1Min")
        self.start_date = cfg.get("startDate", "")
        self.end_date = cfg.get("endDate", "")
        self.trade_host = cfg.get("tradeHost", "127.0.0.1")
        self.trade_port = int(cfg.get("tradePort", 4002))
        self.client_id = int(cfg.get("clientId", 1))
        self.position_size = int(cfg.get("positionSize", 1))
        self.live_trade = bool(cfg.get("liveTrade", False))
        self.model_file = cfg.get("modelFile", "")
        # doTrade0 order parameters
        self.symbol_type = int(cfg.get("symbolType", 1))
        self.ib_controller_tick_size = int(cfg.get("ibControllerTickSize", 2500))
        self.is_intra_day = bool(cfg.get("isIntraDay", True))
        self.is_mkt_order = bool(cfg.get("isMktOrder", False))
        self.max_tick_buy_short_slip = int(cfg.get("maxNumTickSize4BuyShortSlippage", 4))
        self.max_tick_stp_lmt_slip = int(cfg.get("maxNumTickSize4StpLmtOrderSlippage", 4))

        self.output_folder = cfg.get("outputFolder") or str(
            _ROOT / "VS_4000_strategy" / "VS_4003_20260827_ES" / "6 liveTrade")
        os.makedirs(self.output_folder, exist_ok=True)
        self.signal_log_path = os.path.join(self.output_folder, SIGNAL_LOG_NAME)
        # route the doTrade0 debug log into the live-trade output folder
        ibAutotrade.LOG_FILE = os.path.join(self.output_folder, DOTRADE0_LOG_NAME)

        # apply Strategy section params to the class (walk-forward tuned values
        # can likewise be applied by setting strategyS4003 class attrs first)
        strategyS4003.applyStrategyConfig(loadStrategyConfig())

        self.do_trade_fn = doTrade0 if do_trade_fn is None else do_trade_fn
        self.prepare_fn = prepare_fn or strategyS4003._prepare

        self.bars_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.position = 0            # +1 long, -1 short, 0 flat (strategy-side view)
        self.entry_bar_dt = None     # datetime of the bar the position was entered
        self.last_eval_bar_dt = None  # dedup: evaluate each new bar once
        self.ib = None
        self.contract = None
        self.live_bars = None

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def write_signal_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp};{PROGRAM_NAME};{message}\n"
        os.makedirs(self.output_folder, exist_ok=True)
        with open(self.signal_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)
        print(line.strip())

    # ------------------------------------------------------------------
    # RF model gate hook (NOT wired - see module docstring)
    # ------------------------------------------------------------------
    def _passes_rf_gate(self, side, sig_row) -> bool:
        """RandomForest entry gate. modelFile is loaded nowhere yet because
        L4003_1_randomForest_glm.txt is empty (no gating rule specified).
        Returns True so base strategy signals pass through. To wire: load the
        pkl once in __init__, build the feature vector from sig_row here and
        return model.predict(...) == take-trade."""
        return True

    # ------------------------------------------------------------------
    # initial history from MariaDB
    # ------------------------------------------------------------------
    def load_initial_history(self):
        start = str(self.start_date) if self.start_date else None
        end = str(self.end_date) if self.end_date else None
        df = load_1Min_data_from_db(self.symbol, start, end)
        df.index = pd.DatetimeIndex(df.index)
        self.bars_df = df[["open", "high", "low", "close", "volume"]].sort_index()
        self.write_signal_log(
            f"initial history loaded. symbol=[{self.symbol}] bars=[{len(self.bars_df)}] "
            f"range=[{self.bars_df.index.min()}]..[{self.bars_df.index.max()}]")

    # ------------------------------------------------------------------
    # contract (ibAutotrade symbol_type: 1 = future, 2 = stock)
    # ------------------------------------------------------------------
    def build_contract(self, ib):
        if self.symbol_type == 1:
            from ib_insync import ContFuture, Future
            cont = ContFuture(self.symbol, self.future_exchange)
            ib.qualifyContracts(cont)
            contract = Future(localSymbol=cont.localSymbol,
                              exchange=self.future_exchange)
            ib.qualifyContracts(contract)
            return contract
        from ib_insync import Stock
        contract = Stock(self.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        return contract

    # ------------------------------------------------------------------
    # IB streaming: reqHistoricalData(keepUpToDate=True) + bar-change callback
    # ------------------------------------------------------------------
    def start_stream(self):
        from ib_insync import IB
        if self.bars_df.empty:
            self.load_initial_history()
        self.ib = IB()
        self.ib.connect(self.trade_host, self.trade_port, clientId=self.client_id)
        self.contract = self.build_contract(self.ib)

        if not self.bars_df.empty:
            days = max(1, (datetime.now() - self.bars_df.index.max()).days + 1)
        else:
            days = 2
        days = min(days, 5)
        bars = self.ib.reqHistoricalData(
            self.contract, endDateTime="", durationStr=f"{days} D",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
            formatDate=1, keepUpToDate=True)
        for b in bars:
            self.bars_df.loc[_et_naive(b.date)] = [
                b.open, b.high, b.low, b.close, max(b.volume, 0.0)]
        self.bars_df = self.bars_df[~self.bars_df.index.duplicated(keep="last")]
        self.bars_df.sort_index(inplace=True)
        bars.updateEvent += self.on_bar_update
        self.live_bars = bars
        self.write_signal_log(
            f"stream started. {self.trade_host}:{self.trade_port} "
            f"clientId={self.client_id} contract=[{self.contract.localSymbol}] "
            f"liveTrade=[{self.live_trade}] mode=[{'LIVE' if self.live_trade else 'OBSERVE'}]")

    def on_bar_update(self, bars, has_new_bar):
        """ib_insync updateEvent callback. Sync bars into the rolling df;
        evaluate the strategy only when a new bar opens (previous bar closed)."""
        for b in bars:
            self.bars_df.loc[_et_naive(b.date)] = [
                b.open, b.high, b.low, b.close, max(b.volume, 0.0)]
        if not has_new_bar:
            return
        self.bars_df.sort_index(inplace=True)
        self.evaluate_latest_bar()

    # ------------------------------------------------------------------
    # strategy evaluation on the latest bar change
    # ------------------------------------------------------------------
    def evaluate_latest_bar(self):
        """Evaluate signals for the just-opened bar (last df row). Entry
        signals come from the last CLOSED bar (buySig/shortSig are shifted,
        so the partial current bar never contaminates them). Returns the
        action dict taken, or None."""
        df = self.bars_df
        if len(df) < 2:
            return None
        cur_dt = df.index[-1]
        if cur_dt == self.last_eval_bar_dt:
            return None
        self.last_eval_bar_dt = cur_dt
        if len(df) < self.MIN_BARS:
            return None

        sig = self.prepare_fn(df)
        last = sig.iloc[-1]
        cur_open = float(last["Open"])
        buy_sig = int(last["buySig"]) == 1
        short_sig = int(last["shortSig"]) == 1

        atr14_prev = float(sig["atr14"].iloc[-2])
        stop_pt = float(last["max3BarAtr14_1"])
        if np.isnan(stop_pt):
            stop_pt = atr14_prev
        if np.isnan(atr14_prev):
            atr14_prev = stop_pt
        atr_off = atr14_prev / strategyS4003.entry_limit_atr_frac
        profit_mult = strategyS4003.profit_mult
        stop_period1 = int(strategyS4003.stop_period1)

        pos_before = self.position
        action = None

        if buy_sig and short_sig and pos_before == 0:
            self.write_signal_log(
                f"AMBIGUOUS buy+short same bar, skipped. bar=[{cur_dt}]")
            return None

        # ---- entries / force reversals ----
        if buy_sig and self._passes_rf_gate("BUY", last):
            if pos_before == 0:
                action = self._take("BUY", cur_dt, buy=1, sell=0, short=0, cover=0,
                                    limit_price=cur_open - atr_off,
                                    stop_loss=cur_open - stop_pt,
                                    profit_take=cur_open + profit_mult * stop_pt,
                                    new_pos=1)
            elif pos_before == -1:
                action = self._take("BUY+COVER", cur_dt, buy=1, sell=0, short=0, cover=1,
                                    limit_price=cur_open - atr_off,
                                    stop_loss=cur_open - stop_pt,
                                    profit_take=cur_open + profit_mult * stop_pt,
                                    new_pos=1)
            else:  # already long: consecutive buy shifts the N-bar anchor (AFL)
                self.entry_bar_dt = cur_dt
                self.write_signal_log(f"BUY anchor shift (already long). bar=[{cur_dt}]")

        if action is None and short_sig and self._passes_rf_gate("SHORT", last):
            if pos_before == 0:
                action = self._take("SHORT", cur_dt, buy=0, sell=0, short=1, cover=0,
                                    limit_price=cur_open + atr_off,
                                    stop_loss=cur_open + stop_pt,
                                    profit_take=cur_open - profit_mult * stop_pt,
                                    new_pos=-1)
            elif pos_before == 1:
                action = self._take("SHORT+SELL", cur_dt, buy=0, sell=1, short=1, cover=0,
                                    limit_price=cur_open + atr_off,
                                    stop_loss=cur_open + stop_pt,
                                    profit_take=cur_open - profit_mult * stop_pt,
                                    new_pos=-1)
            else:
                self.entry_bar_dt = cur_dt
                self.write_signal_log(f"SHORT anchor shift (already short). bar=[{cur_dt}]")

        # ---- N-bar stop exit at Open ----
        if action is None and pos_before != 0 and self.entry_bar_dt is not None:
            bars_held = len(df) - 1 - df.index.get_loc(self.entry_bar_dt)
            if bars_held >= stop_period1:
                if pos_before == 1:
                    action = self._take("SELL (nbar)", cur_dt, buy=0, sell=1,
                                        short=0, cover=0, limit_price=cur_open,
                                        stop_loss=0.0, profit_take=0.0, new_pos=0)
                else:
                    action = self._take("COVER (nbar)", cur_dt, buy=0, sell=0,
                                        short=0, cover=1, limit_price=cur_open,
                                        stop_loss=0.0, profit_take=0.0, new_pos=0)

        return action

    # ------------------------------------------------------------------
    # order dispatch (gated by liveTrade)
    # ------------------------------------------------------------------
    def _take(self, event, bar_dt, *, buy, sell, short, cover,
              limit_price, stop_loss, profit_take, new_pos):
        pos_before = self.position
        mode = "LIVE" if self.live_trade else "OBSERVE(liveTrade=false)"
        action = {
            "event": event, "bar_dt": bar_dt, "buy": buy, "sell": sell,
            "short": short, "cover": cover, "limit_price": limit_price,
            "stop_loss": stop_loss, "profit_take": profit_take,
            "pos_before": pos_before, "pos_after": new_pos, "mode": mode,
        }
        rs = 0
        if self.live_trade:
            # only place orders when explicitly enabled in config
            try:
                rs = int(self.do_trade_fn(
                    None, self.contract, buy, sell, short, cover,
                    float(limit_price), float(stop_loss), float(profit_take),
                    self.position_size, self.ib_controller_tick_size,
                    self.is_intra_day, self.is_mkt_order,
                    self.max_tick_buy_short_slip, self.max_tick_stp_lmt_slip,
                    self.symbol_type,
                    port=self.trade_port, host=self.trade_host,
                    client_id=self.client_id))
            except Exception as e:  # never let a broker error kill the stream
                self.write_signal_log(f"doTrade0 ERROR. [{e}]")
                rs = -1
        action["doTrade0_rs"] = rs if self.live_trade else "SKIPPED"
        self.write_signal_log(
            f"{event};bar=[{bar_dt}];pos={pos_before}->{new_pos};"
            f"buy={buy};sell={sell};short={short};cover={cover};"
            f"limit={limit_price:.2f};stop={stop_loss:.2f};profit={profit_take:.2f};"
            f"mode={mode};doTrade0=[{action['doTrade0_rs']}]")
        # strategy-side position tracking (stop/profit exits are handled by the
        # broker-side bracket children; nbar/force exits are handled above)
        self.position = new_pos
        self.entry_bar_dt = bar_dt if new_pos != 0 else None
        return action

    # ------------------------------------------------------------------
    def stop(self):
        if self.ib is not None:
            try:
                self.ib.disconnect()
            except Exception:
                pass
        self.write_signal_log("stopped.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    trader = S4003LiveTrade()
    trader.start_stream()
    mode = "LIVE" if trader.live_trade else "OBSERVE (liveTrade=false - no orders)"
    print(f"S4003 live loop running [{mode}]. Ctrl+C to stop.")
    try:
        while True:
            trader.ib.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        trader.stop()


if __name__ == "__main__":
    main()
