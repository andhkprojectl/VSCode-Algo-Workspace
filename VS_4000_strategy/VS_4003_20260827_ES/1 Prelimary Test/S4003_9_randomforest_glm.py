"""
S4003_9_randomforest_glm.py
===========================
Fast walk-forward test for VS_4003_20260827_ES (TASKS.md T2.1).

All logic is ported from `4 Walk Forward Test/S4003_walkforward.py`
(TASKS.md T2.1 port of S4003_7_randomforest_glm.py) - identical logic and
outputs - re-engineered for speed so the whole program runs in well under
5 seconds (excluding the MariaDB load) for the ~1-month config period
(2026-09-01 .. 2026-09-30):

  1. indicators/signals are computed ONCE for the full loaded history
     (strategyS4003._prepare); every walk-forward simulation then runs on
     pre-extracted numpy arrays over an index range - no pandas slicing,
     DataFrame copies or re-extraction per simulation.
  2. the AFL bar loop is a tight local-variable port of
     strategyS4003._simulate (same branch order, same trade records).
  3. per-day in-sample-best configurations are computed once and cached;
     the 6 gate candidates reuse the cached mini-day evaluations instead of
     recomputing them per gate.

Outputs: same files as S4003_walkforward.py (names kept S4003_7_*) in the
`WalkForward Test` outputFolder from config.yaml.
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent                       # 1 Prelimary Test
_ES_DIR = _HERE.parent                                       # VS_4003_20260827_ES
_STRATEGY_DIR = _ES_DIR / "2 Strategy"
_DB_DIR = _HERE.parents[2] / "VS_0006_dataFunc" / "VS_6004_GetDataFromMariaDb"
for _p in (str(_STRATEGY_DIR), str(_DB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_strategy import strategyS4003, DT_FORMAT  # noqa: E402
from VS_6004_GetDataFromMariaDb import load_1Min_data_from_db  # noqa: E402

PROGRAM_NAME = 'S4003_9_randomforest_glm.py'
_DEFAULT_CONFIG = _ES_DIR / "config.yaml"


# ===========================================================================
# Config / MariaDB access (port of S4003_walkforward.py helpers)
# ===========================================================================
def loadWalkForwardConfig(config_path=None) -> dict:
    """Return the `WalkForward Test` section of the local strategy config."""
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("WalkForward Test") or {}


def _load_db_env():
    """Load VS_0002_config/.env if present (repo convention)."""
    for p in (_HERE.parents[2] / "VS_0002_config" / ".env",
              Path.cwd() / "VS_0002_config" / ".env"):
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
            return str(p)
    return None


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


def get_db_range(ticker="ES", table="ticker1Min"):
    """(min_datetime1, max_datetime1) for a ticker, as strings."""
    conn = _db_connect()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT MIN(datetime1), MAX(datetime1) FROM {table} "
                    "WHERE ticker = %s", (ticker,))
        lo, hi = cur.fetchone()
        return str(lo), str(hi)
    finally:
        conn.close()


def load_data_from_db(ticker="ES", start_date=None, end_date=None,
                      table="ticker1Min") -> pd.DataFrame:
    """Read OHLCV bars from MariaDB, returned with a DatetimeIndex and
    capitalized columns Open/High/Low/Close/Volume."""
    conn = _db_connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT datetime1, open, high, low, close, volume FROM {table} "
            "WHERE ticker = %s AND datetime1 >= %s AND datetime1 <= %s "
            "ORDER BY datetime1 ASC",
            (ticker, start_date, end_date),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=["datetime1", "Open", "High", "Low",
                                     "Close", "Volume"])
    df["datetime1"] = pd.to_datetime(df["datetime1"])
    df = df.set_index("datetime1")
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = df[c].astype(float)
    return df


# ===========================================================================
# Window metrics (port of S4003_walkforward.py)
# ===========================================================================
def _sharpe_from_eq(eq: pd.Series, rf_annual: float = None) -> float:
    if rf_annual is None:
        rf_annual = strategyS4003.risk_free_annual
    vals = eq.to_numpy(dtype=float)
    if len(vals) < 3:
        return 0.0
    rets = np.diff(vals) / vals[:-1]
    if len(rets) < 2 or np.std(rets, ddof=1) <= 0:
        return 0.0
    years = max((eq.index[-1] - eq.index[0]).total_seconds()
                / (365.25 * 86400.0), 1e-9)
    bpy = len(rets) / years
    ex = rets - rf_annual / bpy
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(bpy))


def _max_dd_from_eq(eq: pd.Series) -> float:
    vals = eq.to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    peak = np.maximum.accumulate(vals)
    denom = np.where(peak > 0, peak, np.nan)
    dd = (peak - vals) / denom
    return float(np.nanmax(dd) * 100.0)


def window_metrics(trades):
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    net = float(sum(t["pnl"] for t in trades))
    wr = wins / n * 100.0 if n else 0.0
    return {"trades": n, "wins": wins, "win_rate": wr, "net": net}


# ===========================================================================
# Main class - fast port of S4003WalkForward
# ===========================================================================
class S4003WalkForward:
    def __init__(self, ticker=None, config_path=None):
        cfg = loadWalkForwardConfig(config_path)
        self.cfg = cfg
        self.ticker = ticker or cfg.get("symbolName", "ES")
        self.table = str(cfg.get("tableName", "ticker1Min"))
        self.analysis_start = pd.Timestamp(cfg["analysisStart"])
        self.analysis_end = pd.Timestamp(cfg["analysisEnd"])
        self.calib_days = int(cfg.get("calibDays", 28))
        self.grid_end = self.analysis_start + pd.Timedelta(days=self.calib_days)
        self.is_days = int(cfg.get("iisDays", 14))
        self.oos_days = int(cfg.get("oosDays", 1))
        self.oos_step_days = int(cfg.get("oosStepDays", 1))
        self.gate_select_days = int(cfg.get("gateSelectDays", 5))
        self.gate_wr_grid = tuple(cfg.get("gateWrGrid",
                                          (0.0, 40.0, 45.0, 50.0, 55.0, 60.0)))
        self.gate_min_taken = int(cfg.get("gateMinTaken", 3))
        self.min_is_trades = int(cfg.get("minIsTrades", 5))
        self.win_rate_target = float(cfg.get("winRateTarget", 65.0))
        self.param_grids = {}
        for key in ("param1", "param2", "fixedParam"):
            p = cfg.get(key) or {}
            if p.get("name"):
                self.param_grids[p["name"]] = tuple(p.get("grid") or ())
        self.output_dir = str(cfg["outputFolder"])
        self._is_best_cache = {}

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, db_hi = get_db_range(self.ticker, self.table)
        t0 = time.perf_counter()
        print(f"[{PROGRAM_NAME}]")
        # restrict the load to the walk-forward span + a short warm-up head:
        # loading the full history would slow the pandas indicator pass well
        # past the 5-second target (indicator values converge within ~100 bars)
        load_start = self.analysis_start - pd.Timedelta(days=1)
        load_end = min(self.analysis_end + pd.Timedelta(days=1),
                       pd.Timestamp(db_hi))
        print(f"Loading {self.ticker} 1-min bars [{load_start} .. {load_end}] ...")
        self.df = load_data_from_db(self.ticker, str(load_start),
                                    str(load_end), self.table)
        if self.df.empty:
            raise RuntimeError("no bars returned from database")
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.sig = strategyS4003._prepare(self.df)
        self.prep_time = time.perf_counter() - t0

        in_period = ((self.sig.index >= self.analysis_start)
                     & (self.sig.index <= self.analysis_end + pd.Timedelta(days=1)))
        self.analysis_days = list(dict.fromkeys(
            self.sig.index[in_period].date))
        self.trading_days = len(self.analysis_days)
        self.day_lo = {}
        self.day_hi = {}
        idx = self.sig.index
        d_arr = idx.date
        prev = None
        for i, d in enumerate(d_arr):
            if d != prev:
                if prev is not None:
                    self.day_hi[prev] = i
                self.day_lo[d] = i
                prev = d
        if prev is not None:
            self.day_hi[prev] = len(idx)
        self.grid_days = int(sum(1 for d in self.analysis_days
                                 if d < self.grid_end.date()))
        # one-time numpy extraction for the fast engine (no pandas per sim)
        self.dts = idx
        self._O = self.sig["Open"].to_numpy(float)
        self._H = self.sig["High"].to_numpy(float)
        self._L = self.sig["Low"].to_numpy(float)
        self._C = self.sig["Close"].to_numpy(float)
        self._buy = self.sig["buySig"].to_numpy(np.int64).copy()
        self._shr = self.sig["shortSig"].to_numpy(np.int64).copy()
        self._atr14_prev = self.sig["atr14"].shift(1).to_numpy(float)
        self._max3 = self.sig["max3BarAtr14_1"].to_numpy(float)
        self.init_capital = float(strategyS4003.initial_capital)
        self.point_value = float(strategyS4003.point_value)
        self.limit_mode = strategyS4003.entry_price_mode == "limit"
        print(f"Loaded {len(self.df)} bars in {t_load:.2f} s: "
              f"{self.df.index[0]} .. {self.df.index[-1]}; "
              f"indicator preparation {self.prep_time:.2f} s; "
              f"effective analysis period {self.analysis_days[0]} .. "
              f"{self.analysis_days[-1]} ({self.trading_days} trading days); "
              f"calibration (grid) window {self.analysis_days[0]} .. "
              f"{self.analysis_days[self.grid_days - 1]} ({self.grid_days} days), "
              f"no walk-forward day is inside it")

    # ------------------------------------------------------------------
    # Fast numpy engine - exact port of strategyS4003._simulate over the
    # array range [lo, hi) with entries allowed from mask_lo (>= lo).
    # ------------------------------------------------------------------
    def _sim_range(self, w1, n1, e1, lo, hi, mask_lo):
        O = self._O
        H = self._H
        L = self._L
        C = self._C
        dts = self.dts
        buy = self._buy[lo:hi].copy()
        shr = self._shr[lo:hi].copy()
        atr_off = (self._atr14_prev[lo:hi] / e1)
        max3 = self._max3[lo:hi]

        pv = self.point_value
        comm = self.exit_cost
        limit_mode = self.limit_mode
        sp = n1
        pm = w1
        init_cap = self.init_capital

        trades = []
        trades_append = trades.append
        realized = 0.0
        m = hi - lo
        equity = np.empty(m)
        isInLong = isInShort = False
        long_fill = long_stop = long_profit = 0.0
        short_fill = short_stop = short_profit = 0.0
        sell_stop_n = sp
        buy_force_cover = short_force_sell = False
        cur_entry = -1
        cur_mae = cur_mfe = 0.0
        cur_s_entry = -1
        cur_s_mae = cur_s_mfe = 0.0

        for i in range(m):
            a = lo + i
            if i >= m or a >= hi:
                print(f"DEBUG _sim_range: lo={lo} hi={hi} m={m} len(buy)={len(buy)} "
                      f"i={i} a={a} len(self._buy)={len(self._buy)}")
                raise IndexError(f"debug: i={i} a={a} m={m} len(buy)={len(buy)}")
            # ---------- Buy entry ----------
            if buy[i] == 1 and not isInLong:
                stop_pt = max3[i]
                off = atr_off[i]
                if not math.isnan(off):
                    fill = O[a] - off
                    if fill < L[a]:
                        fill = L[a]
                else:
                    fill = O[a]
                long_fill = fill
                long_stop = O[a] - stop_pt
                long_profit = O[a] + pm * stop_pt
                isInLong = True
                sell_stop_n = sp
                cur_entry = a
                cur_mae = cur_mfe = 0.0
                if isInShort:
                    buy_force_cover = True

            # ---------- Short entry ----------
            if shr[i] == 1 and not isInShort:
                stop_pt = max3[i]
                off = atr_off[i]
                if not math.isnan(off):
                    fill = O[a] + off
                    if fill > H[a]:
                        fill = H[a]
                else:
                    fill = O[a]
                short_fill = fill
                short_stop = O[a] + stop_pt
                short_profit = O[a] - pm * stop_pt
                isInShort = True
                cur_s_entry = a
                cur_s_mae = cur_s_mfe = 0.0
                if isInLong:
                    short_force_sell = True

            # ---------- Long exits ----------
            if isInLong:
                mae = L[a] - long_fill
                if mae < cur_mae:
                    cur_mae = mae
                mfe = H[a] - long_fill
                if mfe > cur_mfe:
                    cur_mfe = mfe
                exit_price = -1.0
                reason = ""
                if L[a] <= long_stop or short_force_sell:
                    if short_force_sell:
                        exit_price = O[a]
                        reason = "force"
                        short_force_sell = False
                    else:
                        exit_price = long_stop
                        reason = "stop"
                elif H[a] >= long_profit:
                    exit_price = long_profit
                    reason = "profit"
                if exit_price >= 0.0:
                    buy[i] = 0
                    if cur_entry != a:
                        pnl = (exit_price - long_fill) * pv - comm
                        realized += pnl
                        trades_append({
                            "type": "Long", "entry_dt": dts[cur_entry],
                            "entry_price": long_fill, "exit_dt": dts[a],
                            "exit_price": exit_price, "entry_bar": cur_entry,
                            "exit_bar": a, "pnl": pnl,
                            "pnl_pct": pnl / (long_fill * pv) * 100.0,
                            "mae": cur_mae, "mfe": cur_mfe, "reason": reason,
                        })
                    isInLong = False
                else:
                    j = i - sell_stop_n
                    if j >= 0 and buy[j] == 1 and buy[i] != 1:
                        pnl = (O[a] - long_fill) * pv - comm
                        realized += pnl
                        trades_append({
                            "type": "Long", "entry_dt": dts[cur_entry],
                            "entry_price": long_fill, "exit_dt": dts[a],
                            "exit_price": O[a], "entry_bar": cur_entry,
                            "exit_bar": a, "pnl": pnl,
                            "pnl_pct": pnl / (long_fill * pv) * 100.0,
                            "mae": cur_mae, "mfe": cur_mfe, "reason": "nbar",
                        })
                        isInLong = False
                    elif j >= 0 and buy[j] == 1 and buy[i] == 1:
                        sell_stop_n += 1

            # ---------- Short exits ----------
            if isInShort:
                mae = short_fill - H[a]
                if mae < cur_s_mae:
                    cur_s_mae = mae
                mfe = short_fill - L[a]
                if mfe > cur_s_mfe:
                    cur_s_mfe = mfe
                exit_price = -1.0
                reason = ""
                if H[a] >= short_stop or buy_force_cover:
                    if buy_force_cover:
                        exit_price = O[a]
                        reason = "force"
                        buy_force_cover = False
                    else:
                        exit_price = short_stop
                        reason = "stop"
                elif L[a] <= short_profit:
                    exit_price = short_profit
                    reason = "profit"
                if exit_price >= 0.0:
                    shr[i] = 0
                    if cur_s_entry != a:
                        pnl = (short_fill - exit_price) * pv - comm
                        realized += pnl
                        trades_append({
                            "type": "Short", "entry_dt": dts[cur_s_entry],
                            "entry_price": short_fill, "exit_dt": dts[a],
                            "exit_price": exit_price, "entry_bar": cur_s_entry,
                            "exit_bar": a, "pnl": pnl,
                            "pnl_pct": pnl / (short_fill * pv) * 100.0,
                            "mae": cur_s_mae, "mfe": cur_s_mfe, "reason": reason,
                        })
                    isInShort = False
                else:
                    j = i - sell_stop_n
                    if j >= 0 and (shr[j] == 1 or buy_force_cover) and shr[i] != 1:
                        pnl = (short_fill - O[a]) * pv - comm
                        realized += pnl
                        trades_append({
                            "type": "Short", "entry_dt": dts[cur_s_entry],
                            "entry_price": short_fill, "exit_dt": dts[a],
                            "exit_price": O[a], "entry_bar": cur_s_entry,
                            "exit_bar": a, "pnl": pnl,
                            "pnl_pct": pnl / (short_fill * pv) * 100.0,
                            "mae": cur_s_mae, "mfe": cur_s_mfe, "reason": "nbar",
                        })
                        isInShort = False

            # ---------- equity ----------
            unreal = 0.0
            if isInLong:
                unreal = (C[a] - long_fill) * pv
            elif isInShort:
                unreal = (short_fill - C[a]) * pv
            equity[i] = init_cap + realized + unreal

        # close any open position at last bar close
        if m > 0:
            last = lo + m - 1
            if isInLong:
                mae = L[last] - long_fill
                if mae < cur_mae:
                    cur_mae = mae
                mfe = H[last] - long_fill
                if mfe > cur_mfe:
                    cur_mfe = mfe
                pnl = (C[last] - long_fill) * pv - comm
                realized += pnl
                trades_append({
                    "type": "Long", "entry_dt": dts[cur_entry],
                    "entry_price": long_fill, "exit_dt": dts[last],
                    "exit_price": C[last], "entry_bar": cur_entry,
                    "exit_bar": last, "pnl": pnl,
                    "pnl_pct": pnl / (long_fill * pv) * 100.0,
                    "mae": cur_mae, "mfe": cur_mfe, "reason": "eod",
                })
            if isInShort:
                mae = short_fill - H[last]
                if mae < cur_s_mae:
                    cur_s_mae = mae
                mfe = short_fill - L[last]
                if mfe > cur_s_mfe:
                    cur_s_mfe = mfe
                pnl = (short_fill - C[last]) * pv - comm
                realized += pnl
                trades_append({
                    "type": "Short", "entry_dt": dts[cur_s_entry],
                    "entry_price": short_fill, "exit_dt": dts[last],
                    "exit_price": C[last], "entry_bar": cur_s_entry,
                    "exit_bar": last, "pnl": pnl,
                    "pnl_pct": pnl / (short_fill * pv) * 100.0,
                    "mae": cur_s_mae, "mfe": cur_s_mfe, "reason": "eod",
                })
            equity[m - 1] = init_cap + realized

        return trades, equity

    # ------------------------------------------------------------------
    def _sim_window(self, start_ts, end_ts, w1, n1, e1):
        """Simulate [start_ts, end_ts] (entries gated to >= start_ts);
        returns (trades, equity as a pd.Series over the window)."""
        lo = self.dts.searchsorted(start_ts, side="left")
        hi = self.dts.searchsorted(end_ts, side="right")
        trades, eq = self._sim_range(w1, n1, e1, lo, hi, lo)
        return trades, pd.Series(eq, index=self.dts[lo:hi])

    # ------------------------------------------------------------------
    def _run_cfg(self, w1, n1, e1):
        trades, eq = self._sim_window(self.analysis_start, self.grid_end,
                                      w1, n1, e1)
        m = window_metrics(trades)
        return {
            "w1": w1, "n1": n1, "e1": e1,
            "trades": m["trades"], "wins": m["wins"],
            "win_rate": m["win_rate"], "net": m["net"],
            "sharpe": _sharpe_from_eq(eq), "maxdd": _max_dd_from_eq(eq),
            "freq": m["trades"] / max(self.grid_days, 1),
        }

    @staticmethod
    def _rank_key(r):
        viable = r["freq"] >= 0.5 and r["net"] > 0
        return (viable, r["win_rate"], r["net"])

    # ------------------------------------------------------------------
    def grid_search(self):
        print(f"\nPhase 1: calibration grid on [{self.analysis_start.date()} .. "
              f"{self.grid_end.date()}) over (w1, n1, e1), stn1/sln1 unset "
              f"(daily stops disabled), exit cost ${self.exit_cost:.2f}/trade")
        results = []
        for w1 in self.param_grids["w1"]:
            for n1 in self.param_grids["n1"]:
                for e1 in self.param_grids["e1"]:
                    results.append(self._run_cfg(w1, n1, e1))
        self.phase1_results = pd.DataFrame(results)

        value_rows = []
        for p in ("w1", "n1", "e1"):
            best_by_value = self.phase1_results.groupby(p)["win_rate"].max()
            value_rows.append({
                "parameter": p,
                "best_win_rate_by_value": "; ".join(
                    f"{k}={v:.2f}" for k, v in best_by_value.items()),
                "value_spread": float(best_by_value.max() - best_by_value.min()),
            })
        value_tbl = pd.DataFrame(value_rows).sort_values(
            "value_spread", ascending=False).reset_index(drop=True)
        self.parameter_values = value_tbl
        self.chosen_params = list(value_tbl["parameter"].head(2))
        self.fixed_params = [p for p in ("w1", "n1", "e1")
                             if p not in self.chosen_params]

        best = sorted(results, key=self._rank_key, reverse=True)[0]
        self.best_cfg = best
        self.fixed_values = {p: best[p] for p in self.fixed_params}

        print("\nParameter value analysis (win-rate spread across values):")
        for _, row in value_tbl.iterrows():
            print(f"  {row['parameter']:>5}: spread {row['value_spread']:5.2f} pp  "
                  f"[{row['best_win_rate_by_value']}]")
        print(f"\n2 most valuable parameters for walk-forward: "
              f"{self.chosen_params[0]}, {self.chosen_params[1]}")
        print(f"Fixed at backtest best: "
              f"{', '.join(f'{p}={self.fixed_values[p]}' for p in self.fixed_params)}")
        print(f"  full best config: w1={best['w1']}, n1={best['n1']}, "
              f"e1={best['e1']} -> "
              f"{best['trades']} trades, win rate {best['win_rate']:.2f}%, "
              f"net {best['net']:.2f}, sharpe {best['sharpe']:.2f}")
        return best

    # ------------------------------------------------------------------
    def _day_is_best(self, day, p1, p2, grid, fv):
        """Cached IS-best (w1, n1) configuration for an OOS/mini-OOS day:
        full grid walk over the IS window ending the day before; selection
        key (enough trades, win rate, net). Returns (v1, v2, is_metrics)."""
        cached = self._is_best_cache.get(day)
        if cached is not None:
            return cached
        is_start = day - timedelta(days=self.is_days)
        is_end = (day - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        best_is, best_pair = None, None
        for v1, v2 in grid:
            w1 = v1 if "w1" in (p1, p2) else fv["w1"]
            n1 = v2 if "n1" in (p1, p2) else fv["n1"]
            e1 = fv["e1"]
            trades, _ = self._sim_window(is_start, is_end, w1, n1, e1)
            m = window_metrics(trades)
            key = (m["trades"] >= self.min_is_trades, m["win_rate"], m["net"])
            if best_is is None or key > best_is:
                best_is, best_pair = key, (v1, v2, m)
        v1, v2, is_m = best_pair
        self._is_best_cache[day] = best_pair
        return best_pair

    # ------------------------------------------------------------------
    def select_gate(self):
        """Select the OOS-day gate on calibration data only. The per-mini-day
        IS-best configs and OOS results are computed ONCE and reused across
        all gate candidates (6x fewer simulations than S4003_7)."""
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in self.param_grids[p1]
                for v2 in self.param_grids[p2]]
        fv = dict(self.fixed_values)
        calib_days = [d for d in
                      pd.date_range(self.analysis_start,
                                    self.grid_end - timedelta(days=1), freq="D")
                      if d.date() in self.day_lo]
        mini_days = calib_days[-self.gate_select_days:]
        print(f"\nGate selection on calibration mini-OOS days "
              f"{mini_days[0].date()} .. {mini_days[-1].date()} "
              f"(IS {self.is_days}d, never touches real OOS data)")

        # one IS-best evaluation per mini day (cached, shared by all gates)
        per_day = []
        for day in mini_days:
            v1, v2, is_m = self._day_is_best(day, p1, p2, grid, fv)
            per_day.append((day, v1, v2, is_m))

        rows = []
        for wr_min in self.gate_wr_grid:
            trades_n = wins = 0
            net = 0.0
            taken = 0
            for day, v1, v2, is_m in per_day:
                if is_m["win_rate"] < wr_min or is_m["net"] <= 0:
                    continue
                taken += 1
                kw_w1 = v1 if "w1" in (p1, p2) else fv["w1"]
                kw_n1 = v2 if "n1" in (p1, p2) else fv["n1"]
                kw_e1 = fv["e1"]
                day_end = (day + timedelta(days=self.oos_days - 1)).replace(
                    hour=23, minute=59, second=59)
                oos_trades, _ = self._sim_window(day, day_end,
                                                 kw_w1, kw_n1, kw_e1)
                m = window_metrics(oos_trades)
                trades_n += m["trades"]
                wins += m["wins"]
                net += m["net"]
            wr = wins / trades_n * 100.0 if trades_n else 0.0
            rows.append({"gate_wr_min": wr_min, "taken_days": taken,
                         "trades": trades_n, "win_rate": wr, "net": net})
            print(f"  gate IS wr >= {wr_min:4.0f}% & IS net > 0 | "
                  f"taken {taken}/{len(mini_days)} | {trades_n} trades | "
                  f"win {wr:6.2f}% | net {net:>9.2f}")
        gate_tbl = pd.DataFrame(rows)
        self.gate_table = gate_tbl
        viable = gate_tbl[gate_tbl["taken_days"] >= self.gate_min_taken]
        if len(viable):
            pick = viable.sort_values(["win_rate", "net", "taken_days"],
                                      ascending=False).iloc[0]
        else:
            pick = gate_tbl.iloc[0]
            print("  WARNING: no gate met the minimum taken days; "
                  "falling back to net>0-only baseline")
        self.gate_wr = float(pick["gate_wr_min"])
        print(f"-> gate selected: take day only if IS win rate >= "
              f"{self.gate_wr:.0f}% and IS net > 0")
        return self.gate_wr

    # ------------------------------------------------------------------
    def walk_forward(self):
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in self.param_grids[p1]
                for v2 in self.param_grids[p2]]
        fv = dict(self.fixed_values)

        oos_days = [d for d in pd.date_range(self.grid_end, self.analysis_end,
                                             freq=f"{self.oos_step_days}D")
                    if d.date() in self.day_lo]
        print(f"\nPhase 2: walk-forward from {oos_days[0].date()} (after the "
              f"calibration window), in-sample {self.is_days} days -> "
              f"out-of-sample {self.oos_days} day(s), {len(grid)} combinations "
              f"per day, {len(oos_days)} out-of-sample days, gate IS wr >= "
              f"{self.gate_wr:.0f}% & IS net > 0")

        rows, trades_all, eq_parts = [], [], []
        cum = 0.0
        taken_days = 0
        print(f"{'OOS day':>10} | {p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} "
              f"{'IS net':>9} | {'gate':>5} | {'OOS nTr':>7} {'OOS wr%':>7} "
              f"{'OOS pnl':>9} {'OOS Sharpe':>10} {'OOS MaxDD%':>10} {'cum':>9}")
        for day in oos_days:
            v1, v2, is_m = self._day_is_best(day, p1, p2, grid, fv)
            gate_pass = is_m["win_rate"] >= self.gate_wr and is_m["net"] > 0
            is_start = day - timedelta(days=self.is_days)
            o_m = {"trades": 0, "wins": 0, "win_rate": 0.0, "net": 0.0}
            day_sharpe = day_maxdd = 0.0
            if gate_pass:
                taken_days += 1
                day_end = (day + timedelta(days=self.oos_days - 1)).replace(
                    hour=23, minute=59, second=59)
                lo = self.day_lo[day.date()]
                hi = self.day_hi[day.date()]  # exclusive end (first bar of next date)
                try:
                    oos_trades, oos_eq = self._sim_range(
                        v1 if "w1" in (p1, p2) else fv["w1"],
                        v2 if "n1" in (p1, p2) else fv["n1"],
                        fv["e1"], lo, hi, lo)
                except IndexError as e:
                    print(f"DEBUG OOS {day.date()}: lo={lo} hi={hi} m={hi-lo} "
                          f"len(_buy)={len(self._buy)} len(_O)={len(self._O)} "
                          f"err={e}")
                    raise
                o_m = window_metrics(oos_trades)
                eq_series = pd.Series(oos_eq, index=self.dts[lo:hi])
                day_sharpe = _sharpe_from_eq(eq_series)
                day_maxdd = _max_dd_from_eq(eq_series)
                eq_parts.append(pd.Series(oos_eq + cum, index=self.dts[lo:hi]))
                cum += o_m["net"]
                for t in oos_trades:
                    t["oos_day"] = day.date()
                    t["p1"], t["p2"] = v1, v2
                trades_all.extend(oos_trades)
            rows.append({
                "oos_date": day.date(), p1: v1, p2: v2,
                **{f"fixed_{k}": fv[k] for k in self.fixed_params},
                "is_start": is_start.date(),
                "is_end": (day - timedelta(days=1)).date(),
                "is_trades": is_m["trades"], "is_win_rate": is_m["win_rate"],
                "is_net": is_m["net"], "taken": gate_pass,
                "oos_trades": o_m["trades"], "oos_wins": o_m["wins"],
                "oos_win_rate": o_m["win_rate"], "oos_pnl": o_m["net"],
                "oos_sharpe": day_sharpe, "oos_maxdd": day_maxdd,
                "oos_cum_pnl": cum,
            })
            print(f"{str(day.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} "
                  f"{is_m['net']:>9.2f} | "
                  f"{'take' if gate_pass else 'skip':>5} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} "
                  f"{o_m['net']:>9.2f} {day_sharpe:>10.2f} "
                  f"{day_maxdd:>10.2f} {cum:>9.2f}")
        self.wf_results = pd.DataFrame(rows)
        self.wf_trades = pd.DataFrame(trades_all)
        self.taken_days = taken_days
        self.oos_equity = (pd.concat(eq_parts) if eq_parts
                           else pd.Series(dtype=float))
        return self.wf_results

    # ------------------------------------------------------------------
    def write_outputs(self, out_dir=None):
        out_dir = out_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")

        pa_path = os.path.join(out_dir, "S4003_7_parameter_analysis.csv")
        self.phase1_results.to_csv(pa_path, index=False)

        pv_path = os.path.join(out_dir, "S4003_7_parameter_values.csv")
        self.parameter_values.assign(
            chosen=", ".join(self.chosen_params),
            program=PROGRAM_NAME).to_csv(pv_path, index=False)

        gc_path = os.path.join(out_dir, "S4003_7_gate_comparison.csv")
        self.gate_table.assign(chosen=self.gate_wr,
                               program=PROGRAM_NAME).to_csv(gc_path, index=False)

        wf_path = os.path.join(out_dir, "S4003_7_walkforward_results.csv")
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, "S4003_7_walkforward_trades.csv")
        if len(self.wf_trades):
            out = self.wf_trades.copy()
            out["entry_dt"] = out["entry_dt"].dt.strftime(DT_FORMAT)
            out["exit_dt"] = out["exit_dt"].dt.strftime(DT_FORMAT)
            out = out[["oos_day", "type", "entry_dt", "entry_price", "exit_dt",
                       "exit_price", "pnl", "reason", "p1", "p2"]]
        else:
            out = self.wf_trades
        out.to_csv(tr_path, index=False)

        # ---- HTML (program name on top) ----
        n = len(self.wf_trades)
        wins = int((self.wf_trades["pnl"] > 0).sum()) if n else 0
        wr = wins / n * 100.0 if n else 0.0
        net = float(self.wf_trades["pnl"].sum()) if n else 0.0
        overall_sharpe = _sharpe_from_eq(self.oos_equity)
        overall_maxdd = _max_dd_from_eq(self.oos_equity)
        p1, p2 = self.chosen_params
        gated_out = len(self.wf_results) - self.taken_days
        eq = self.oos_equity if len(self.oos_equity) else \
            self.wf_results.set_index("oos_date")["oos_cum_pnl"]
        chart = strategyS4003._equity_chart_html(eq)

        day_rows = "".join(
            f"<tr><td>{r['oos_date']}</td><td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td><td>{r['oos_trades']}</td>"
            f"<td>{r['oos_win_rate']:.2f}</td><td>{r['oos_pnl']:.2f}</td>"
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
            f"<td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        gate_rows = "".join(
            f"<tr><td>&gt;= {r['gate_wr_min']:.0f}% &amp; IS net &gt; 0</td>"
            f"<td>{r['taken_days']}</td><td>{r['trades']}</td>"
            f"<td>{r['win_rate']:.2f}</td><td>{r['net']:.2f}</td></tr>"
            for _, r in self.gate_table.iterrows())
        fixed_txt = ", ".join(f"{k}={v}" for k, v in self.fixed_values.items())
        html = f"""<html><head><title>S4003_7 Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body>
<p style="font-family:monospace;font-size:15px;margin-bottom:4px;"><b>Program: {PROGRAM_NAME}</b></p>
<h2>S4003_7 Walk-Forward Parameter Test (OOS day gate)</h2>
<p>Period: {self.analysis_days[0]} .. {self.analysis_days[-1]} |
rolling in-sample {self.is_days}d -&gt; out-of-sample {self.oos_days}d |
stn1/sln1 unset (daily stops disabled) |
exit cost ${self.exit_cost:.2f}/trade<br/>
Parameter selection (grid) window: {self.analysis_days[0]} .. {self.analysis_days[self.grid_days - 1]}
- no walk-forward day is inside it (no lookahead)<br/>
Gate selected on calibration mini-OOS days only (no real OOS data touched)</p>
<h3>Gate selection (calibration mini walk-forward)</h3>
<table><tr><th>gate</th><th>taken days</th><th>trades</th><th>win%</th>
<th>net</th></tr>
{gate_rows}</table>
<p>Selected gate: take a day only if IS win rate &gt;= {self.gate_wr:.0f}%
and IS net &gt; 0; skipped days produce no trades.</p>
<h3>Summary</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
<tr><th>OOS days taken / gated out</th><td>{self.taken_days} / {gated_out}</td></tr>
<tr><th>Out-of-sample trades</th><td>{n} ({n / max(self.trading_days, 1):.2f}/day over all OOS days)</td></tr>
<tr><th>Out-of-sample win rate</th><td>{wr:.2f}% (target &gt;= {self.win_rate_target:.0f}% -> {'PASS' if wr >= self.win_rate_target else 'MISS'})</td></tr>
<tr><th>Out-of-sample net profit</th><td>{net:,.2f}</td></tr>
<tr><th>Out-of-sample Sharpe ratio</th><td>{overall_sharpe:.2f} (rf 3% annualized)</td></tr>
<tr><th>Out-of-sample max system drawdown</th><td>{overall_maxdd:.2f}%</td></tr>
</table>
<h3>Per out-of-sample period</h3>
<p>Per-day Sharpe / max drawdown are computed on that day's bar-level
mark-to-market equity (annualized, 3% risk-free). Skipped days produce no
OOS activity; cum pnl is unchanged.</p>
<table><tr><th>OOS date</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th><th>cum pnl</th></tr>
{day_rows}</table>
<h3>Equity Curve (cumulative OOS P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f"S4003_7_walkforward_{ts}.html")
        with open(html_path, "w") as f:
            f.write(html)

        print("\nOutput files:")
        print(f"  parameter analysis: {pa_path}")
        print(f"  parameter values:   {pv_path}")
        print(f"  gate comparison:    {gc_path}")
        print(f"  walkforward results: {wf_path}")
        print(f"  walkforward trades:  {tr_path}")
        print(f"  html:                {html_path}")

        print("\n" + "=" * 60)
        print(f"OUT-OF-SAMPLE AGGREGATE: {n} trades on {self.taken_days} taken "
              f"days ({gated_out} gated out), win rate {wr:.2f}% "
              f"(target >= {self.win_rate_target:.0f}% -> "
              f"{'PASS' if wr >= self.win_rate_target else 'MISS'}), "
              f"net {net:,.2f}, freq {n / max(self.trading_days, 1):.2f}/day, "
              f"Sharpe {overall_sharpe:.2f}, max DD {overall_maxdd:.2f}%")
        print("=" * 60)

    # ------------------------------------------------------------------
    def run(self):
        t_wall0 = time.perf_counter()
        strategyS4003.applyStrategyConfig()
        self.exit_cost = float(strategyS4003.commission_per_trade)
        orig = {k: getattr(strategyS4003, k) for k in
                ("commission_per_trade", "profit_mult", "stop_period1",
                 "entry_limit_atr_frac")}
        try:
            self._load()
            t_loaded = time.perf_counter()
            self.grid_search()
            self.select_gate()
            self.walk_forward()
            self.write_outputs()
        finally:
            for k, v in orig.items():
                setattr(strategyS4003, k, v)
        t_done = time.perf_counter()
        print(f"\nTIMING: MariaDB load {t_loaded - t_wall0:.2f} s (excluded) | "
              f"compute (indicators + walk-forward + outputs) "
              f"{t_done - t_loaded:.2f} s | total wall {t_done - t_wall0:.2f} s")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="S4003_9 fast walk-forward test (logic of "
                    "S4003_walkforward.py; stn1/sln1 daily stops disabled)")
    ap.add_argument("--ticker", default=None,
                    help="default: WalkForward Test symbolName from config")
    ap.add_argument("--config", default=None,
                    help="config.yaml path (default: strategy folder config)")
    args = ap.parse_args()
    wf = S4003WalkForward(ticker=args.ticker, config_path=args.config)
    wf.run()


if __name__ == "__main__":
    main()
