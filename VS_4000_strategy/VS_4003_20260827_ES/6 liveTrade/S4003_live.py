"""
S4003_live.py
=============
Live trade for VS_4003_20260827_ES (TASKS.md T4.1, PRD.md FR-6).

- Initializes a rolling in-memory DataFrame with the last `initMonths` months
  of MariaDB ticker1Min bars, backfills any gap to the previous completed
  minute from IB using the ingest.py fetch path, and upserts those bars to DB.
- Connects to IB TWS/Gateway and subscribes with
  reqHistoricalData(barSizeSetting='1 min', keepUpToDate=True). Each new-bar
  callback updates the rolling DataFrame and evaluates the strategy.
- Signal evaluation reuses strategyS4003._prepare and _simulate_fast from
  S4003_walkforward.py with fixed Live Trade w1/n1/e1 values. No code is
  imported from `1 Prelimary Test`.
- With isMktOrder=true (the FR-6 configuration), the live engine and doTrade0
  both use market-entry semantics at the entry bar Open. Stop/profit brackets
  retain the walk-forward anchors. If isMktOrder=false, entry reference prices
  use the walk-forward ATR limit offset instead.
- The engine's final-row 'eod' trade represents a still-open position. Order
  actions come from the engine/tracked-position diff: BUY/SHORT entries,
  BUY+COVER or SHORT+SELL reversals, and SELL/COVER for N-bar or force exits.
  Stop/profit exits are left to the existing broker bracket.
- Every dbFlushEveryMin minutes, the last dbFlushLookbackMin minutes of bars,
  excluding the current minute, are upserted from memory to ticker1Min.
- Actual IB execution events are stored idempotently in the configured MariaDB
  execution table. The existing IB connection is reused for order dispatch.

SAFETY: config `Live Trade`.liveTrade=false -> OBSERVE mode. Signals and debug
records are written and position state is tracked, but doTrade0 is NEVER
called. Enable orders only after backtest/walkforward/other tests pass; live
ports 4001/7496 must not be used before paper validation.

RF model hook: modelFile is reserved for the backtest RandomForest gate. The
gate is intentionally NOT wired because L4003_1_randomForest_glm.txt has no
gating specification; _passes_rf_gate currently returns True.

Outputs (config `Live Trade`.outputFolder):
  S4003_live_signals.log     - buy/short/sell/cover and execution events
  S4003_doTrade0_debug.log   - doTrade0/IB order debug log
  trade_debug.log            - when isDebug=true, per-new-bar OHLCV, signals,
                               complete order parameters/results, and fills

Import-safe: no IB/DB connection and no config access at import time.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_STRATEGY_DIR = Path(__file__).resolve().parents[1] / "2 Strategy"
_WALKFORWARD_DIR = Path(__file__).resolve().parents[1] / "4 Walk Forward Test"
for _p in (str(_STRATEGY_DIR), str(_WALKFORWARD_DIR),
           str(_ROOT / "VS_0008_liveTrade"),
           str(_ROOT / "VS_0006_dataFunc" / "VS_6004_GetDataFromMariaDb"),
           str(_ROOT / "VS_0006_dataFunc" / "VS_6000_DataSource"),
           str(_ROOT / "VS_0007_dbAndFile" / "mariaDB")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_strategy import strategyS4003, loadStrategyConfig  # noqa: E402
import ibAutotrade  # noqa: E402
from ibAutotrade import doTrade0  # noqa: E402
from VS_6004_GetDataFromMariaDb import load_1Min_data_from_db  # noqa: E402
import ibMarketData  # noqa: E402  (ingest.py fetch logic, FR-6 init step 2)
import mariaDb  # noqa: E402      (saveDfToTicker5Min upsert, FR-6 init/flush)
# the walkforward's engine (bit-identical transcription of
# strategyS4003._simulate); importing S4003_walkforward also applies its fast
# _lin_reg_slope/_bars_since_min shims process-locally, so
# strategyS4003._prepare runs faster too
from S4003_walkforward import _simulate_fast  # noqa: E402

ET = ZoneInfo("America/New_York")
PROGRAM_NAME = "S4003_live"
SIGNAL_LOG_NAME = "S4003_live_signals.log"
DOTRADE0_LOG_NAME = "S4003_doTrade0_debug.log"
DEBUG_LOG_NAME = "trade_debug.log"
_ENV_PATH = _ROOT / "VS_0002_config" / ".env"


def _load_db_env():
    """Load DB credentials from VS_0002_config/.env (mariaDb.saveDfToTicker5Min
    reads DB_* via os.getenv at call time)."""
    try:
        from dotenv import load_dotenv
        if _ENV_PATH.exists():
            load_dotenv(dotenv_path=str(_ENV_PATH), override=False)
    except ImportError:
        pass


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

    Signal evaluation follows `4 Walk Forward Test/S4003_walkforward.py`:
    strategyS4003._prepare + _simulate_fast (imported from
    S4003_walkforward.py) on a rolling warm-up window with entries gated to
    the live-start bar and fixed w1/n1/e1 from config. do_trade_fn /
    prepare_fn are injectable so the signal->order mapping can be
    unit-tested offline with a stubbed doTrade0 and scripted signals.
    """

    MIN_BARS = 150  # warm-up guard: percentile(100) + vwap50 need history

    def __init__(self, config=None, do_trade_fn=None, prepare_fn=None,
                 fetch_history_fn=None, save_df_fn=None,
                 ensure_execution_table_fn=None, save_execution_fn=None):
        _load_db_env()
        self.config = loadLiveTradeConfig() if config is None else (config or {})
        cfg = self.config

        self.symbol = cfg.get("symbolName", "ES")
        self.ticker_type = cfg.get("tickerType", "FU")
        self.future_exchange = cfg.get("futureExchange", "CME")
        self.table_name = cfg.get("tableName", "ticker1Min")
        self.execution_table_name = cfg.get("executionTableName", "tradeExecution")
        self.init_months = int(cfg.get("initMonths", 6))
        self.is_con_future = bool(cfg.get("isConFuture", True))
        self.future_expire_date = cfg.get("futureExpireDate")
        self.db_flush_every_min = int(cfg.get("dbFlushEveryMin", 8))
        self.db_flush_lookback_min = int(cfg.get("dbFlushLookbackMin", 12))
        self.trade_host = cfg.get("tradeHost", "127.0.0.1")
        self.trade_port = int(cfg.get("tradePort", 4002))
        self.client_id = int(cfg.get("clientId", 1))
        self.position_size = int(cfg.get("positionSize", 1))
        self.live_trade = bool(cfg.get("liveTrade", False))
        self.is_debug = bool(cfg.get("isDebug", False))
        self.model_file = cfg.get("modelFile", "")
        # doTrade0 order parameters
        self.symbol_type = int(cfg.get("symbolType", 1))
        if cfg.get("tickSize") is not None:
            self.ib_controller_tick_size = int(round(float(cfg["tickSize"]) * 10000))
        else:
            self.ib_controller_tick_size = int(cfg.get("ibControllerTickSize", 2500))
        self.is_intra_day = bool(cfg.get("isIntraDay", True))
        self.is_mkt_order = bool(cfg.get("isMktOrder", False))
        self.max_tick_buy_short_slip = int(cfg.get("maxNumTickSize4BuyShortSlippage", 4))
        self.max_tick_stp_lmt_slip = int(cfg.get("maxNumTickSize4StpLmtOrderSlippage", 4))

        self.output_folder = cfg.get("outputFolder") or str(
            _ROOT / "VS_4000_strategy" / "VS_4003_20260827_ES" / "6 liveTrade")
        os.makedirs(self.output_folder, exist_ok=True)
        self.signal_log_path = os.path.join(self.output_folder, SIGNAL_LOG_NAME)
        self.debug_log_path = os.path.join(self.output_folder, DEBUG_LOG_NAME)
        # route the doTrade0 debug log into the live-trade output folder
        ibAutotrade.LOG_FILE = os.path.join(self.output_folder, DOTRADE0_LOG_NAME)

        # walkforward _set_params pattern: Strategy section first, then the
        # fixed w1/n1/e1 from the Live Trade section override (TASKS.md T4.1)
        strategyS4003.applyStrategyConfig(loadStrategyConfig())
        self.w1 = float(cfg.get("w1", strategyS4003.profit_mult))
        self.n1 = int(cfg.get("n1", strategyS4003.stop_period1))
        self.e1 = float(cfg.get("e1", strategyS4003.entry_limit_atr_frac))
        self.lookback_bars = int(cfg.get("lookbackBars", 3000))
        strategyS4003.profit_mult = self.w1
        strategyS4003.stop_period1 = self.n1
        strategyS4003.entry_limit_atr_frac = self.e1
        strategyS4003.num_contracts = self.position_size
        strategyS4003.initial_capital = float(
            cfg.get("initCapital", strategyS4003.initial_capital))
        strategyS4003.entry_price_mode = "market" if self.is_mkt_order else "limit"
        if cfg.get("pointValue") is not None:  # FR-7: Live Trade pointValue
            strategyS4003.point_value = float(cfg["pointValue"])

        self.do_trade_fn = doTrade0 if do_trade_fn is None else do_trade_fn
        self.prepare_fn = prepare_fn or strategyS4003._prepare
        # FR-6 init/flush helpers (injectable for offline tests)
        self.fetch_history_fn = (ibMarketData.getAllTypesTicketDataWithTimeFromIB
                                 if fetch_history_fn is None else fetch_history_fn)
        self.save_df_fn = (mariaDb.saveDfToTicker5Min
                           if save_df_fn is None else save_df_fn)
        self.ensure_execution_table_fn = (
            mariaDb.ensureTradeExecutionTable
            if ensure_execution_table_fn is None else ensure_execution_table_fn)
        self.save_execution_fn = (mariaDb.saveTradeExecution
                                  if save_execution_fn is None else save_execution_fn)

        self.bars_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.position = 0            # +1 long, -1 short, 0 flat (strategy-side view)
        self.open_trade = None       # engine trade dict of the adopted open position
        self.seen_closed = set()     # (type, entry_dt, exit_dt) of processed closes
        self.trade_start_ts = None   # entries gated to >= this ts (set in start_stream)
        self.last_eval_bar_dt = None  # dedup: evaluate each new bar once
        self.last_db_flush = None    # wall-clock of the last periodic DB flush
        self.submitted_order_ids = set()
        self.pending_executions = {}
        self.saved_execution_ids = set()
        self._last_engine_event = "HOLD"
        self._last_engine_exit = None
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

    def write_debug_log(self, message):
        if not self.is_debug:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp};{PROGRAM_NAME};{message}\n"
        os.makedirs(self.output_folder, exist_ok=True)
        with open(self.debug_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    def _write_bar_debug(self, bar_dt, raw_buy=0, raw_short=0, action=None):
        if not self.is_debug or self.bars_df.empty:
            return
        bar = self.bars_df.iloc[-1]
        buy = int(action["buy"]) if action else int(raw_buy)
        short = int(action["short"]) if action else int(raw_short)
        sell = int(action["sell"]) if action else int(self._last_engine_exit == "SELL")
        cover = int(action["cover"]) if action else int(self._last_engine_exit == "COVER")
        event = action["event"] if action else self._last_engine_event
        ohlcv = {key: float(bar[key]) for key in ("open", "high", "low", "close", "volume")}
        if action:
            order = dict(action["order_params"])
            order.update({"event": action["event"],
                          "doTrade0Result": action["doTrade0_rs"],
                          "orderIds": action["order_ids"]})
        else:
            order = {"event": "NONE", "doTrade0Result": "NOT_CALLED"}
        signal = {"buy": buy, "sell": sell, "short": short, "cover": cover,
                  "event": event}
        self.write_debug_log(
            f"bar=[{bar_dt}];OHLCV={json.dumps(ohlcv, sort_keys=True)};"
            f"signal={json.dumps(signal, sort_keys=True, default=str)};"
            f"order={json.dumps(order, sort_keys=True, default=str)};"
            f"position=[{self.position}]")

    # ------------------------------------------------------------------
    # RF model gate hook (NOT wired - see module docstring)
    # ------------------------------------------------------------------
    def _passes_rf_gate(self, sig) -> bool:
        """RandomForest entry gate. modelFile is loaded nowhere yet because
        L4003_1_randomForest_glm.txt is empty (no gating rule specified).
        Returns True so base strategy signals pass through. To wire: load the
        pkl once in __init__, build the feature vector from the last row of
        sig here and return model.predict(...) == take-trade. When it returns
        False, the just-opened bar's entry signals are masked before the
        engine runs, so engine state and orders stay consistent."""
        return True

    # ------------------------------------------------------------------
    # initial history from MariaDB (FR-6 init step 1: last `initMonths` months)
    # ------------------------------------------------------------------
    def load_initial_history(self):
        end = pd.Timestamp.now(tz=ET).tz_localize(None)
        start = end - pd.DateOffset(months=self.init_months)
        df = load_1Min_data_from_db(self.symbol, str(start), str(end))
        df.index = pd.DatetimeIndex(df.index)
        self.bars_df = df[["open", "high", "low", "close", "volume"]].sort_index()
        self.write_signal_log(
            f"initial history loaded. symbol=[{self.symbol}] bars=[{len(self.bars_df)}] "
            f"initMonths=[{self.init_months}] "
            f"range=[{self.bars_df.index.min()}]..[{self.bars_df.index.max()}]")

    # ------------------------------------------------------------------
    # FR-6 init step 2: IB gap backfill (ingest.py logic) -> df + ticker1Min
    # ------------------------------------------------------------------
    @staticmethod
    def _prev_minute_et(now=None):
        """Previous completed 1-min bar stamp, US/Eastern-naive (DB stamp
        convention, same as ibMarketData.DB_TZ)."""
        now = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now)
        if now.tzinfo is None:
            now = now.tz_localize(ET)
        else:
            now = now.tz_convert(ET)
        return ((now - pd.Timedelta(minutes=1)).floor("min").tz_localize(None))

    def backfill_gap_from_ib(self, now=None):
        """If the last in-memory bar is older than the previous minute, fetch
        the missing bars from IB (same function ingest.py uses,
        ibMarketData.getAllTypesTicketDataWithTimeFromIB, reusing the live
        connection) for [last bar + 1 min, previous minute], then save the
        result to BOTH the in-memory dataframe and MariaDB ticker1Min.
        Returns the number of bars merged (0 when there is no gap)."""
        if self.ib is None:
            return 0
        prev_min = self._prev_minute_et(now)
        if self.bars_df.empty:
            self.write_signal_log(
                "gap backfill skipped: in-memory dataframe empty "
                "(stream seed will cover recent bars)")
            return 0
        last_bar = self.bars_df.index.max()
        if last_bar >= prev_min:
            self.write_signal_log(
                f"gap backfill not needed. last bar=[{last_bar}] "
                f"prev minute=[{prev_min}]")
            return 0
        start = last_bar + pd.Timedelta(minutes=1)
        fetch_end = prev_min + pd.Timedelta(minutes=1) if start == prev_min else prev_min
        self.write_signal_log(
            f"gap backfill from IB. [{start}]..[{prev_min}] (ingest.py logic)")
        df = self.fetch_history_fn(
            conn1=self.ib, symbolName=self.symbol,
            startDate=start.strftime("%Y%m%d"), startTime=start.strftime("%H%M"),
            endDate=fetch_end.strftime("%Y%m%d"), endTime=fetch_end.strftime("%H%M"),
            period1=1, tickerType=self.ticker_type,
            isConFuture=self.is_con_future,
            futureExpireDate=self.future_expire_date,
            futureExchange=self.future_exchange)
        if df is None or df.empty:
            self.write_signal_log("gap backfill: IB returned no bars")
            return 0
        idx = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        keep = (idx >= start) & (idx <= prev_min)
        df = df.loc[keep].copy()
        if df.empty:
            self.write_signal_log("gap backfill: IB returned no bars in the requested gap")
            return 0
        idx = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        # persist to MariaDB first (same save as ingest.py)
        self.save_df_fn(df, tableName=self.table_name, isOverride=True)
        # then merge into the in-memory dataframe (ET-naive index, lowercase cols)
        gap = pd.DataFrame({
            "open": df["open"].to_numpy(float),
            "high": df["high"].to_numpy(float),
            "low": df["low"].to_numpy(float),
            "close": df["close"].to_numpy(float),
            "volume": df["volume"].to_numpy(float),
        }, index=idx)
        self.bars_df = pd.concat([self.bars_df, gap])
        self._trim_rolling_history()
        self.write_signal_log(
            f"gap backfill done. bars=[{len(gap)}] saved to {self.table_name} "
            f"and merged; last bar now=[{self.bars_df.index.max()}]")
        return len(gap)

    def _trim_rolling_history(self):
        if self.bars_df.empty:
            return
        self.bars_df = self.bars_df[~self.bars_df.index.duplicated(keep="last")]
        self.bars_df.sort_index(inplace=True)
        cutoff = self.bars_df.index[-1] - pd.DateOffset(months=self.init_months)
        if self.bars_df.index[0] < cutoff:
            self.bars_df = self.bars_df.loc[self.bars_df.index >= cutoff]

    # ------------------------------------------------------------------
    # FR-6 periodic persist: every dbFlushEveryMin minutes write the last
    # dbFlushLookbackMin minutes of bars (current minute excluded) to MariaDB
    # ------------------------------------------------------------------
    def flush_recent_bars_to_db(self, now=None):
        """Write the last `dbFlushLookbackMin` minutes of bars (current minute
        bar excluded) from the in-memory dataframe to MariaDB tableName
        (upsert). Returns the number of rows sent."""
        if self.bars_df.empty:
            return 0
        now = (pd.Timestamp.now(tz=ET).tz_localize(None)
               if now is None else pd.Timestamp(_et_naive(now)))
        cur_min = now.floor("min")
        win_start = cur_min - pd.Timedelta(minutes=self.db_flush_lookback_min)
        win = self.bars_df[(self.bars_df.index >= win_start)
                           & (self.bars_df.index < cur_min)]
        if win.empty:
            return 0
        df_db = pd.DataFrame({
            "symbolName": self.symbol,
            "date": win.index.strftime("%Y-%m-%d"),
            "time": win.index.strftime("%H:%M:%S"),
            "open": win["open"].to_numpy(float),
            "high": win["high"].to_numpy(float),
            "low": win["low"].to_numpy(float),
            "close": win["close"].to_numpy(float),
            "volume": win["volume"].to_numpy(float),
        })
        n = self.save_df_fn(df_db, tableName=self.table_name, isOverride=True)
        self.write_signal_log(
            f"db flush. saved [{n}] bars [{win.index.min()}]..[{win.index.max()}] "
            f"to {self.table_name}")
        return n

    def maybe_flush_db(self, now=None):
        """Timer gate for flush_recent_bars_to_db (every dbFlushEveryMin
        minutes). Flush errors are logged, never raised into the stream."""
        now = (pd.Timestamp.now(tz=ET).tz_localize(None)
               if now is None else pd.Timestamp(_et_naive(now)))
        if (self.last_db_flush is not None
                and (now - self.last_db_flush).total_seconds()
                < self.db_flush_every_min * 60):
            return
        self.last_db_flush = now
        try:
            self.flush_recent_bars_to_db(now=now)
        except Exception as e:
            self.write_signal_log(f"db flush ERROR. [{e}]")

    # ------------------------------------------------------------------
    # contract (ibAutotrade symbol_type: 1 = future, 2 = stock)
    # ------------------------------------------------------------------
    def build_contract(self, ib):
        if self.symbol_type == 1:
            from ib_insync import ContFuture, Future
            if not self.is_con_future:
                if not self.future_expire_date:
                    raise ValueError("futureExpireDate is required when isConFuture=false")
                contract = Future(self.symbol, str(self.future_expire_date),
                                  self.future_exchange, includeExpired=True)
                ib.qualifyContracts(contract)
                return contract
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
        if self.live_trade:
            self.ensure_execution_table_fn(tableName=self.execution_table_name)
            self.ib.execDetailsEvent += self.on_exec_details
        # FR-6 init step 2: fill any DB->now gap from IB, save to df + ticker1Min
        self.backfill_gap_from_ib()

        if not self.bars_df.empty:
            now_et = pd.Timestamp.now(tz=ET).tz_localize(None)
            days = max(1, (now_et - self.bars_df.index.max()).days + 1)
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
        self._trim_rolling_history()
        # gate entries to the live stream start (walkforward window-start
        # gating): signals from before program start are never traded; an
        # engine trade opened on a seeded bar is ghost-adopted in _reconcile
        if not self.bars_df.empty:
            self.trade_start_ts = self.bars_df.index[-1]
        bars.updateEvent += self.on_bar_update
        self.live_bars = bars
        self.write_signal_log(
            f"stream started. {self.trade_host}:{self.trade_port} "
            f"clientId={self.client_id} contract=[{self.contract.localSymbol}] "
            f"w1={self.w1} n1={self.n1} e1={self.e1} "
            f"tradeStart=[{self.trade_start_ts}] isDebug=[{self.is_debug}] "
            f"isMktOrder=[{self.is_mkt_order}] "
            f"liveTrade=[{self.live_trade}] mode=[{'LIVE' if self.live_trade else 'OBSERVE'}]")

    def on_bar_update(self, bars, has_new_bar):
        """ib_insync updateEvent callback. Sync bars into the rolling df;
        evaluate the strategy only when a new bar opens (previous bar closed)."""
        for b in bars:
            self.bars_df.loc[_et_naive(b.date)] = [
                b.open, b.high, b.low, b.close, max(b.volume, 0.0)]
        if not has_new_bar:
            return
        self._trim_rolling_history()
        self.evaluate_latest_bar()
        self.maybe_flush_db()

    # ------------------------------------------------------------------
    # strategy evaluation on the latest bar change (walkforward engine path)
    # ------------------------------------------------------------------
    def evaluate_latest_bar(self):
        """Evaluate signals for the just-opened bar (last df row) the way
        S4003_walkforward.py does: prepare the last `lookbackBars` rows, run
        _simulate_fast (the walkforward's engine) with entries gated to the
        live-start bar, and derive order actions from the engine's
        trade-list diff. Entry signals come from the last CLOSED bar
        (buySig/shortSig are shifted), so the partial current bar never
        contaminates them. Returns the action dict taken, or None."""
        df = self.bars_df
        if df.empty:
            return None
        cur_dt = df.index[-1]
        if cur_dt == self.last_eval_bar_dt:
            return None
        self.last_eval_bar_dt = cur_dt
        self._last_engine_event = "HOLD"
        self._last_engine_exit = None
        if len(df) < self.MIN_BARS:
            self._last_engine_event = "HOLD (warm-up)"
            self._write_bar_debug(cur_dt)
            return None

        sig = self.prepare_fn(df.iloc[-self.lookback_bars:])
        if not self._passes_rf_gate(sig):
            # mask the just-opened bar's entries so engine state and orders
            # stay consistent (hook is unwired - always True today)
            sig.iloc[-1, sig.columns.get_loc("buySig")] = 0
            sig.iloc[-1, sig.columns.get_loc("shortSig")] = 0
        # same per-window inputs as S4003_walkforward._window_data: python
        # lists with entry signals before trade_start_ts zeroed (the mask
        # strategyS4003._simulate applies for trade_start_ts)
        dts = sig.index
        if self.trade_start_ts is not None:
            pre = int(np.count_nonzero(np.asarray(dts < self.trade_start_ts)))
        else:
            pre = 0
        buy0 = sig["buySig"].to_numpy(int).tolist()
        shr0 = sig["shortSig"].to_numpy(int).tolist()
        for i in range(pre):
            buy0[i] = 0
            shr0[i] = 0
        raw_buy = buy0[-1]
        raw_short = shr0[-1]
        if raw_buy:
            self._last_engine_event = "BUY"
        elif raw_short:
            self._last_engine_event = "SHORT"
        trades, _eq = _simulate_fast(
            sig["Open"].tolist(), sig["High"].tolist(), sig["Low"].tolist(),
            sig["Close"].tolist(), sig["atr14"].tolist(),
            sig["max3BarAtr14_1"].tolist(), buy0, shr0, dts, pre,
            strategyS4003.profit_mult, strategyS4003.stop_period1,
            strategyS4003.entry_limit_atr_frac,
            strategyS4003.entry_price_mode == "limit",
            strategyS4003.commission_per_trade, strategyS4003.point_value,
            strategyS4003.num_contracts, strategyS4003.initial_capital)
        action = self._reconcile(sig, trades, cur_dt)
        self._write_bar_debug(cur_dt, raw_buy, raw_short, action)
        return action

    # ------------------------------------------------------------------
    # engine-state diff -> order actions
    # ------------------------------------------------------------------
    def _reconcile(self, sig, trades, cur_dt):
        """Diff the engine's trade list against the tracked strategy-side
        position and dispatch the matching orders. Returns the action dict
        taken, or None."""
        prev = self.open_trade
        open_tr = None
        new_closed = []
        for t in trades:
            if t["reason"] == "eod":
                open_tr = t            # engine force-closes any open position at
                continue               # the last row -> 'eod' means "open now"
            key = (t["type"], t["entry_dt"], t["exit_dt"])
            if key not in self.seen_closed:
                self.seen_closed.add(key)
                new_closed.append(t)

        close = None
        if prev is not None:
            still_open = (open_tr is not None
                          and open_tr["type"] == prev["type"]
                          and open_tr["entry_dt"] == prev["entry_dt"])
            if still_open:
                self._last_engine_event = f"HOLD {prev['type']}"
                return None
            close = next((t for t in new_closed
                          if t["type"] == prev["type"]
                          and t["entry_dt"] == prev["entry_dt"]), None)
            if close is None:
                # trade vanished from the sim: the entry bar completed and the
                # engine voided the trade (AFL same-bar cancel, Buy[i]=0)
                self._last_engine_event = f"VOID {prev['type']} (same-bar cancel)"
                self.write_signal_log(
                    f"engine voided {prev['type']} entry=[{prev['entry_dt']}] "
                    f"(same-bar cancel); broker bracket governs the real fill. "
                    f"bar=[{cur_dt}]")
                self._adopt(None)
            elif close["reason"] in ("stop", "profit"):
                self._last_engine_exit = "SELL" if prev["type"] == "Long" else "COVER"
                self._last_engine_event = (
                    f"{self._last_engine_exit} ({close['reason']}) via broker bracket")
                self.write_signal_log(
                    f"{prev['type']} exit [{close['reason']}] handled by broker "
                    f"bracket. entry=[{prev['entry_dt']}] "
                    f"exit=[{close['exit_dt']}] price=[{close['exit_price']:.2f}]")
                self._adopt(None)
            elif open_tr is None:
                if prev.get("ghost"):
                    self._last_engine_event = (
                        f"GHOST {prev['type']} ended ({close['reason']})")
                    self.write_signal_log(
                        f"ghost {prev['type']} ended [{close['reason']}], no "
                        f"order (entry was never dispatched). "
                        f"entry=[{prev['entry_dt']}]")
                    self._adopt(None)
                    return None
                cur_open = float(sig["Open"].iloc[-1])
                action = self._exit_order(prev["type"], close["reason"],
                                          cur_dt, cur_open)
                self.open_trade = None
                return action
            # else: reversal - the exit leg rides with the new entry below

        if open_tr is None:
            return None

        side = "BUY" if open_tr["type"] == "Long" else "SHORT"
        if open_tr["entry_dt"] != cur_dt:
            # engine opened this trade on an earlier bar (startup gate or
            # missed bars): cannot place a retroactive order - track it as a
            # ghost so its later exits never emit orders either
            self._last_engine_event = f"{side} (ghost-adopted; no order)"
            self.write_signal_log(
                f"engine {side} entry=[{open_tr['entry_dt']}] predates the "
                f"live bar [{cur_dt}] - ghost-adopted, no orders")
            self._adopt(open_tr, ghost=True)
            return None

        o, stop_pt, atr_off = self._entry_prices(sig, open_tr["entry_dt"])
        w1 = strategyS4003.profit_mult
        if open_tr["type"] == "Long":
            limit_price = o if self.is_mkt_order else o - atr_off
            stop_loss = o - stop_pt
            profit_take = o + w1 * stop_pt
        else:
            limit_price = o if self.is_mkt_order else o + atr_off
            stop_loss = o + stop_pt
            profit_take = o - w1 * stop_pt

        # exit leg only when the broker really still holds the old side
        # (force/nbar close; stop/profit exits are broker-bracket business)
        exit_leg = (prev is not None and not prev.get("ghost")
                    and close is not None
                    and close["reason"] not in ("stop", "profit"))
        if open_tr["type"] == "Long":
            action = self._take("BUY+COVER" if exit_leg else "BUY", cur_dt,
                                buy=1, sell=0, short=0,
                                cover=1 if exit_leg else 0,
                                limit_price=limit_price, stop_loss=stop_loss,
                                profit_take=profit_take, new_pos=1)
        else:
            action = self._take("SHORT+SELL" if exit_leg else "SHORT", cur_dt,
                                buy=0, sell=1 if exit_leg else 0, short=1,
                                cover=0,
                                limit_price=limit_price, stop_loss=stop_loss,
                                profit_take=profit_take, new_pos=-1)
        self.open_trade = dict(open_tr, ghost=False)
        return action

    def _exit_order(self, trade_type, reason, cur_dt, cur_open):
        """Explicit exit order for engine closes the broker bracket does not
        perform (nbar / force without re-entry)."""
        if trade_type == "Long":
            return self._take(f"SELL ({reason})", cur_dt, buy=0, sell=1,
                              short=0, cover=0, limit_price=cur_open,
                              stop_loss=0.0, profit_take=0.0, new_pos=0)
        return self._take(f"COVER ({reason})", cur_dt, buy=0, sell=0,
                          short=0, cover=1, limit_price=cur_open,
                          stop_loss=0.0, profit_take=0.0, new_pos=0)

    def _adopt(self, trade, ghost=False):
        """Sync strategy-side position tracking with the engine state."""
        if trade is None:
            self.open_trade = None
            self.position = 0
        else:
            self.open_trade = dict(trade, ghost=ghost)
            self.position = 1 if trade["type"] == "Long" else -1

    @staticmethod
    def _entry_prices(sig, entry_dt):
        """(entry-bar Open, stop points, limit offset) at the engine's entry
        bar - the same anchors the engine uses (Open, max3BarAtr14_1,
        atr14[prev]/e1)."""
        pos = sig.index.searchsorted(entry_dt)
        o = float(sig["Open"].iloc[pos])
        stop_pt = float(sig["max3BarAtr14_1"].iloc[pos])
        atr_prev = float(sig["atr14"].iloc[pos - 1]) if pos > 0 else float("nan")
        if np.isnan(stop_pt):
            stop_pt = atr_prev
        if np.isnan(atr_prev):
            atr_prev = stop_pt
        return o, stop_pt, atr_prev / strategyS4003.entry_limit_atr_frac

    def _current_order_ids(self):
        trades_fn = getattr(self.ib, "trades", None)
        if not callable(trades_fn):
            return set()
        try:
            return {int(t.order.orderId) for t in trades_fn()}
        except Exception as e:
            self.write_signal_log(f"order-id snapshot ERROR. [{e}]")
            return set()

    def _register_new_orders(self, before):
        new_ids = self._current_order_ids() - before
        self.submitted_order_ids.update(new_ids)
        for order_id in new_ids:
            for record in self.pending_executions.pop(order_id, []):
                self._save_execution_record(record)
        return sorted(new_ids)

    # ------------------------------------------------------------------
    # order dispatch (gated by liveTrade)
    # ------------------------------------------------------------------
    def _take(self, event, bar_dt, *, buy, sell, short, cover,
              limit_price, stop_loss, profit_take, new_pos):
        pos_before = self.position
        mode = "LIVE" if self.live_trade else "OBSERVE(liveTrade=false)"
        order_params = {
            "contract": str(getattr(self.contract, "localSymbol", self.contract)),
            "buy": buy, "sell": sell, "short": short, "cover": cover,
            "limitPrice": float(limit_price),
            "stopLossPrice": float(stop_loss),
            "profitTakePrice": float(profit_take),
            "numContracts": self.position_size,
            "ibControllerTickSize": self.ib_controller_tick_size,
            "isIntraDay": self.is_intra_day,
            "isMktOrder": self.is_mkt_order,
            "maxNumTickSize4BuyShortSlippage": self.max_tick_buy_short_slip,
            "maxNumTickSize4StpLmtOrderSlippage": self.max_tick_stp_lmt_slip,
            "symbolType": self.symbol_type,
            "host": self.trade_host, "port": self.trade_port,
            "clientId": self.client_id,
        }
        action = {
            "event": event, "bar_dt": bar_dt, "buy": buy, "sell": sell,
            "short": short, "cover": cover, "limit_price": limit_price,
            "stop_loss": stop_loss, "profit_take": profit_take,
            "pos_before": pos_before, "pos_after": new_pos, "mode": mode,
            "order_params": order_params, "order_ids": [],
        }
        self._last_engine_event = event
        rs = 0
        before_order_ids = self._current_order_ids() if self.live_trade else set()
        if self.live_trade:
            # only place orders when explicitly enabled in config
            try:
                rs = int(self.do_trade_fn(
                    self.ib, self.contract, buy, sell, short, cover,
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
            finally:
                action["order_ids"] = self._register_new_orders(before_order_ids)
        action["doTrade0_rs"] = rs if self.live_trade else "SKIPPED"
        self.write_signal_log(
            f"{event};bar=[{bar_dt}];pos={pos_before}->{new_pos};"
            f"buy={buy};sell={sell};short={short};cover={cover};"
            f"limit={limit_price:.2f};stop={stop_loss:.2f};profit={profit_take:.2f};"
            f"mode={mode};doTrade0=[{action['doTrade0_rs']}];"
            f"orderIds={action['order_ids']}")
        # strategy-side position tracking (stop/profit exits are handled by the
        # broker-side bracket children; nbar/force exits are handled above)
        self.position = new_pos
        return action

    def _save_execution_record(self, record):
        exec_id = record["execId"]
        if exec_id in self.saved_execution_ids:
            return 0
        try:
            saved = self.save_execution_fn(
                record, tableName=self.execution_table_name)
        except Exception as e:
            self.write_signal_log(f"execution DB save ERROR. execId=[{exec_id}] [{e}]")
            return 0
        self.saved_execution_ids.add(exec_id)
        self.write_signal_log(
            f"execution saved. execId=[{exec_id}] orderId=[{record['orderId']}] "
            f"{record['side']} {record['quantity']} {record['ticker']} "
            f"@ {record['price']} utc=[{record['executionTimeUtc']}]")
        self.write_debug_log(f"execution={json.dumps(record, sort_keys=True, default=str)}")
        return saved

    def on_exec_details(self, trade, fill):
        """Persist an IB fill after its order is associated with this trader."""
        execution = getattr(fill, "execution", None)
        if execution is None:
            return None
        contract = getattr(fill, "contract", None) or getattr(trade, "contract", None)
        ticker = str(getattr(contract, "symbol", ""))
        if ticker != self.symbol:
            return None
        exec_id = str(getattr(execution, "execId", ""))
        if not exec_id or exec_id in self.saved_execution_ids:
            return None
        order_id_value = getattr(execution, "orderId", None)
        if order_id_value is None:
            order_id_value = trade.order.orderId
        order_id = int(order_id_value)
        execution_time = getattr(execution, "time", None) or getattr(fill, "time", None)
        when = (pd.Timestamp.now(tz="UTC") if execution_time is None
                else pd.Timestamp(execution_time))
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        else:
            when = when.tz_convert("UTC")
        quantity = getattr(fill, "quantity", None)
        if quantity is None:
            quantity = getattr(execution, "shares", 0)
        price = getattr(fill, "price", None)
        if price is None:
            price = getattr(execution, "price", 0)
        record = {
            "strategy": "S4003",
            "ticker": ticker,
            "orderId": order_id,
            "execId": exec_id,
            "executionTimeUtc": when.tz_localize(None).to_pydatetime(),
            "side": str(getattr(execution, "side", "")),
            "quantity": float(quantity),
            "price": float(price),
            "exchangeName": str(getattr(execution, "exchange", "")),
            "account": str(getattr(execution, "acctNumber", "")),
        }
        if order_id not in self.submitted_order_ids:
            self.pending_executions.setdefault(order_id, []).append(record)
            self.write_debug_log(
                f"execution deferred until order registration. execId=[{exec_id}] "
                f"orderId=[{order_id}]")
            return record
        self._save_execution_record(record)
        return record

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
            trader.maybe_flush_db()   # FR-6: periodic bars -> ticker1Min
    except KeyboardInterrupt:
        pass
    finally:
        trader.stop()


if __name__ == "__main__":
    main()
