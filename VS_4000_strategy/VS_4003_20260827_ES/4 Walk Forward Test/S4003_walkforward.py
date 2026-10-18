"""
S4003_walkforward.py
====================
Walk-forward test for VS_4003_20260827_ES (TASKS.md T2.1).

Port of `1 Prelimary Test/S4003_7_randomforest_glm.py` onto the strategy
class strategyS4003 from `2 Strategy/S4003_strategy.py`; no imports from
`1 Prelimary Test`. Rolling walk-forward: a calibration window (grid over
w1/n1/e1 + OOS-day gate selection over gateWrGrid on gateSelectDays
mini-OOS days, gateMinTaken min taken days), then iisDays-day in-sample /
oosDays-day out-of-sample folds stepping oosStepDays days, win-rate target
winRateTarget, minIsTrades min in-sample trades. stn1/sln1 daily stops stay
unset (disabled), as in S4003_7.

All parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml`
section `WalkForward Test` (signal/account parameters from section
`Strategy` via strategyS4003.applyStrategyConfig). Data from MariaDB,
table ticker1Min. In-sample windows are iisDays trading days (not calendar
days). `endDateIsOOS: 1` (default): analysisEnd is the last OOS day.
`endDateIsOOS: 0`: analysisEnd is the last IS day; the results CSV/HTML
gets one extra final row with a blank OOS date carrying the w1/n1 selected
for the next trading day (its OOS columns are blank - no bars in MariaDB
yet).

Speed optimizations (ported from
`1 Prelimary Test/S4003_9_randomforest_kimi.py`; output CSV values are
bit-identical to the pre-optimization version):
  1. Warm-up truncation: bars are loaded from
     max(db_min_datetime, analysisStart - WARMUP_DAYS calendar days) instead
     of the database minimum date. ATR(Wilder) is an IIR recursion whose
     start-up difference underflows float64 after ~500 bars; every other
     indicator has a <= 100-bar window, so WARMUP_DAYS=10 (several thousand
     bars) of warm-up makes every value read by the walk-forward windows
     bit-identical to loading full history.
  2. In-process acceleration of the two rolling-apply indicator helpers
     (_lin_reg_slope, _bars_since_min) with stride-trick numpy equivalents
     that keep the exact per-window operation order -> bit-identical Series.
     Patched into the S4003_strategy module namespace for this process only;
     S4003_strategy.py on disk is untouched.
  3. Gate selection no longer re-runs the IS grid search per gate
     candidate: the IS-best config of a mini-OOS day does not depend on the
     gate threshold, so each day's IS grid + OOS simulation is computed once
     and reused for every gateWrGrid candidate (390 -> 65 simulations for
     the default config; gate table rows are identical).
  4. _simulate_window runs a line-by-line transcription of
     strategyS4003._simulate (_simulate_fast) on per-window cached python
     lists instead of pandas/numpy containers: the slice view, the
     pre-window signal zeroing (same mask _simulate applies for
     trade_start_ts) and the array->list conversions are computed once per
     window and reused for every (w1, n1, e1) grid combination.
  5. The HTML equity chart uses the strategy's dependency-free SVG renderer
     (strategyS4003._equity_chart_svg, the same fallback _equity_chart_html
     uses when matplotlib is absent) instead of importing matplotlib.
  6. mysql.connector / python-dotenv are imported lazily inside the DB
     helpers.

Outputs (same names/format as S4003_7_randomforest_glm.py, written to the
`WalkForward Test` outputFolder):
  S4003_7_parameter_analysis.csv   calibration (w1, n1, e1) grid results
  S4003_7_parameter_values.csv     win-rate spread per parameter
  S4003_7_gate_comparison.csv      gate-candidate mini walk-forward results
  S4003_7_walkforward_results.csv  per-OOS-day metrics incl. Sharpe/MaxDD
  S4003_7_walkforward_trades.csv   every out-of-sample trade (taken days)
  S4003_7_walkforward_<ts>.html    program name on top + gate info +
                                   per-OOS-period table + equity chart
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_ROOT = Path(__file__).resolve().parents[3]
_STRATEGY_DIR = Path(__file__).resolve().parents[1] / '2 Strategy'
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

import S4003_strategy as _strategy_mod
from S4003_strategy import strategyS4003, DT_FORMAT

PROGRAM_NAME = 'S4003_walkforward.py'
VALUE_PARAMS = ('w1', 'n1', 'e1')
_DEFAULT_CONFIG = (_ROOT / 'VS_4000_strategy' / 'VS_4003_20260827_ES'
                   / 'config.yaml')
WARMUP_DAYS = 10   # warm-up calendar days loaded before analysisStart (>> the
                   # ~500 bars ATR(Wilder) needs to converge bit-exactly plus
                   # the <= 100-bar indicator windows plus lookbackBars)


# ===========================================================================
# Acceleration shims (bit-identical, process-local; S4003_strategy.py on
# disk is untouched). Both replace pandas rolling-apply helpers with numpy
# sliding-window equivalents that keep the same per-window operation order,
# so every output value is exactly the same.
# ===========================================================================
def _lin_reg_slope_fast(s: pd.Series, period: int) -> pd.Series:
    """Bit-identical fast _lin_reg_slope: rolling OLS slope of the last
    `period` values (AmiBroker LinRegSlope)."""
    y = s.to_numpy(dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=s.index)
    t = np.arange(period, dtype=float)
    tm = t.mean()
    denom = float(((t - tm) ** 2).sum())
    w = np.lib.stride_tricks.sliding_window_view(y, period)
    with np.errstate(all='ignore'):
        m = w.mean(axis=1)
        res = ((w - m[:, None]) * (t - tm)).sum(axis=1) / denom
    res[np.isnan(w).any(axis=1)] = np.nan
    out[period - 1:] = res
    return pd.Series(out, index=s.index)


def _bars_since_min_fast(s: pd.Series, period: int) -> pd.Series:
    """Bit-identical fast _bars_since_min (AmiBroker LLVBars): bars passed
    since the lowest value in the window (partial windows allowed, oldest
    occurrence wins ties, 0 = current bar)."""
    y = s.to_numpy(dtype=float)
    n = len(y)
    out = np.empty(n, dtype=float)
    for i in range(min(period - 1, n)):      # partial head windows
        w = y[:i + 1]
        out[i] = float(len(w) - 1 - np.argmin(w))
    if n >= period:
        w = np.lib.stride_tricks.sliding_window_view(y, period)
        out[period - 1:] = (period - 1 - np.argmin(w, axis=1)).astype(float)
    return pd.Series(out, index=s.index)


_strategy_mod._lin_reg_slope = _lin_reg_slope_fast
_strategy_mod._bars_since_min = _bars_since_min_fast


# ===========================================================================
# Config / MariaDB access (self-contained; credentials in VS_0002_config/.env)
# mysql.connector / dotenv are imported lazily so their import time belongs
# to the MariaDB fetch.
# ===========================================================================
def loadWalkForwardConfig(config_path=None) -> dict:
    """Return the `WalkForward Test` section of the local strategy config."""
    config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('WalkForward Test') or {}


def _load_db_env():
    """Load VS_0002_config/.env if present (repo convention)."""
    from dotenv import load_dotenv
    for p in (_ROOT / 'VS_0002_config' / '.env',
              Path.cwd() / 'VS_0002_config' / '.env'):
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
            return str(p)
    return None


def _db_connect():
    import mysql.connector
    _load_db_env()
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )


def get_db_range(ticker='ES', table='ticker1Min'):
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


def load_data_from_db(ticker='ES', start_date=None, end_date=None,
                      table='ticker1Min') -> pd.DataFrame:
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
    df = pd.DataFrame(rows, columns=['datetime1', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime1'] = pd.to_datetime(df['datetime1'])
    df = df.set_index('datetime1')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = df[c].astype(float)
    return df


# ===========================================================================
# Window metrics (ported from S4003_5_randomforest_glm.py)
# ===========================================================================
def _sharpe_from_eq(eq: pd.Series, rf_annual: float = None) -> float:
    """Annualized Sharpe of a bar-level equity series (risk-free from the
    strategy config, 3% default), same methodology as
    strategyS4003._compute_stats."""
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
    """Max system drawdown (%) of a bar-level equity series."""
    vals = eq.to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    peak = np.maximum.accumulate(vals)
    denom = np.where(peak > 0, peak, np.nan)
    dd = (peak - vals) / denom
    return float(np.nanmax(dd) * 100.0)


def window_metrics(trades):
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    net = float(sum(t['pnl'] for t in trades))
    wr = wins / n * 100.0 if n else 0.0
    return {'trades': n, 'wins': wins, 'win_rate': wr, 'net': net}


# ===========================================================================
# Fast AFL-faithful trade simulation loop
# ===========================================================================
def _simulate_fast(O, H, L, C, atr14, max3, buy, shr, dts, start_pos,
                   profit_mult, stop_period1, entry_limit_atr_frac, limit_mode,
                   commission_per_trade, point_value, num_contracts,
                   initial_capital):
    """Line-by-line transcription of strategyS4003._simulate (same order of
    operations -> bit-identical trades and equity) running on python lists
    instead of pandas/numpy containers. O/H/L/C/atr14/max3 are lists of
    floats covering the window (including the lookbackBars warm-up head);
    buy/shr are mutable list copies of the window entry signals with bars
    before the window start already zeroed (exactly what
    _simulate(trade_start_ts=...) does); start_pos is the number of zeroed
    head bars; dts is the window DatetimeIndex. Returns (trades, equity
    list); in_market is not computed (callers discard it).
    """
    n = len(O)
    ef = entry_limit_atr_frac
    pv = point_value
    comm = commission_per_trade
    ct = num_contracts
    sp = stop_period1
    pm = profit_mult
    ic = initial_capital

    trades = []
    realized = 0.0
    equity = [0.0] * n

    isInLong = isInShort = False
    long_fill = long_stop = long_profit = 0.0
    short_fill = short_stop = short_profit = 0.0
    sell_stop_n = sp
    buy_force_cover = short_force_sell = False
    cur = None      # [entry_bar, mae, mfe] of the open long trade
    cur_s = None    # [entry_bar, mae, mfe] of the open short trade

    for i in range(n):
        # ---------- Buy entry ----------
        if buy[i] == 1 and not isInLong:
            stop_pt = max3[i]
            a = atr14[i - 1] / ef if i >= 1 else np.nan
            if limit_mode and not (a != a):
                long_fill = O[i] - a
                if long_fill < L[i]:
                    long_fill = L[i]            # max(O - atr_off, Low)
            else:
                long_fill = O[i]
            long_stop = O[i] - stop_pt
            long_profit = O[i] + pm * stop_pt
            isInLong = True
            sell_stop_n = sp
            cur = [i, 0.0, 0.0]
            if isInShort:
                buy_force_cover = True

        # ---------- Short entry ----------
        if shr[i] == 1 and not isInShort:
            stop_pt = max3[i]
            a = atr14[i - 1] / ef if i >= 1 else np.nan
            if limit_mode and not (a != a):
                short_fill = O[i] + a
                if short_fill > H[i]:
                    short_fill = H[i]           # min(O + atr_off, High)
            else:
                short_fill = O[i]
            short_stop = O[i] + stop_pt
            short_profit = O[i] - pm * stop_pt
            isInShort = True
            cur_s = [i, 0.0, 0.0]
            if isInLong:
                short_force_sell = True

        # ---------- Long exits ----------
        if isInLong and cur is not None:
            v = L[i] - long_fill
            if v < cur[1]:
                cur[1] = v                      # mae = min(mae, L - fill)
            v = H[i] - long_fill
            if v > cur[2]:
                cur[2] = v                      # mfe = max(mfe, H - fill)
            exit_price = None
            if (L[i] <= long_stop) or short_force_sell:
                if short_force_sell:
                    exit_price = O[i]; reason = 'force'; short_force_sell = False
                else:
                    exit_price = long_stop; reason = 'stop'
            elif H[i] >= long_profit:
                exit_price = long_profit; reason = 'profit'
            if exit_price is not None:
                buy[i] = 0      # AFL cancels same-bar entries (Buy[i] = 0)
                if cur[0] != i:
                    pnl = (exit_price - long_fill) * pv * ct - comm
                    realized += pnl
                    trades.append({
                        'type': 'Long', 'entry_dt': dts[cur[0]],
                        'entry_price': long_fill, 'exit_dt': dts[i],
                        'exit_price': exit_price, 'entry_bar': cur[0],
                        'exit_bar': i, 'pnl': pnl,
                        'pnl_pct': pnl / (long_fill * pv * ct) * 100.0,
                        'mae': cur[1], 'mfe': cur[2], 'reason': reason})
                isInLong = False
                cur = None
            else:
                j = i - sell_stop_n
                if j >= 0 and buy[j] == 1 and buy[i] != 1:
                    # N-bar stop at Open
                    pnl = (O[i] - long_fill) * pv * ct - comm
                    realized += pnl
                    trades.append({
                        'type': 'Long', 'entry_dt': dts[cur[0]],
                        'entry_price': long_fill, 'exit_dt': dts[i],
                        'exit_price': O[i], 'entry_bar': cur[0],
                        'exit_bar': i, 'pnl': pnl,
                        'pnl_pct': pnl / (long_fill * pv * ct) * 100.0,
                        'mae': cur[1], 'mfe': cur[2], 'reason': 'nbar'})
                    isInLong = False
                    cur = None
                else:
                    if j >= 0 and buy[j] == 1 and buy[i] == 1:
                        sell_stop_n += 1

        # ---------- Short exits ----------
        if isInShort and cur_s is not None:
            v = short_fill - H[i]
            if v < cur_s[1]:
                cur_s[1] = v
            v = short_fill - L[i]
            if v > cur_s[2]:
                cur_s[2] = v
            exit_price = None
            if (H[i] >= short_stop) or buy_force_cover:
                if buy_force_cover:
                    exit_price = O[i]; reason = 'force'; buy_force_cover = False
                else:
                    exit_price = short_stop; reason = 'stop'
            elif L[i] <= short_profit:
                exit_price = short_profit; reason = 'profit'
            if exit_price is not None:
                shr[i] = 0
                if cur_s[0] != i:
                    pnl = (short_fill - exit_price) * pv * ct - comm
                    realized += pnl
                    trades.append({
                        'type': 'Short', 'entry_dt': dts[cur_s[0]],
                        'entry_price': short_fill, 'exit_dt': dts[i],
                        'exit_price': exit_price, 'entry_bar': cur_s[0],
                        'exit_bar': i, 'pnl': pnl,
                        'pnl_pct': pnl / (short_fill * pv * ct) * 100.0,
                        'mae': cur_s[1], 'mfe': cur_s[2], 'reason': reason})
                isInShort = False
                cur_s = None
            else:
                j = i - sp
                if j >= 0 and (shr[j] == 1 or buy_force_cover) and shr[i] != 1:
                    pnl = (short_fill - O[i]) * pv * ct - comm
                    realized += pnl
                    trades.append({
                        'type': 'Short', 'entry_dt': dts[cur_s[0]],
                        'entry_price': short_fill, 'exit_dt': dts[i],
                        'exit_price': O[i], 'entry_bar': cur_s[0],
                        'exit_bar': i, 'pnl': pnl,
                        'pnl_pct': pnl / (short_fill * pv * ct) * 100.0,
                        'mae': cur_s[1], 'mfe': cur_s[2], 'reason': 'nbar'})
                    isInShort = False
                    cur_s = None

        # ---------- equity ----------
        unreal = 0.0
        if isInLong:
            unreal = (C[i] - long_fill) * pv * ct
        elif isInShort:
            unreal = (short_fill - C[i]) * pv * ct
        equity[i] = ic + realized + unreal

    # close any open position at last bar close (AmiBroker end-of-data behavior)
    if n > 0:
        if isInLong and cur is not None:
            v = L[n - 1] - long_fill
            if v < cur[1]:
                cur[1] = v
            v = H[n - 1] - long_fill
            if v > cur[2]:
                cur[2] = v
            pnl = (C[n - 1] - long_fill) * pv * ct - comm
            realized += pnl
            trades.append({
                'type': 'Long', 'entry_dt': dts[cur[0]],
                'entry_price': long_fill, 'exit_dt': dts[n - 1],
                'exit_price': C[n - 1], 'entry_bar': cur[0],
                'exit_bar': n - 1, 'pnl': pnl,
                'pnl_pct': pnl / (long_fill * pv * ct) * 100.0,
                'mae': cur[1], 'mfe': cur[2], 'reason': 'eod'})
        if isInShort and cur_s is not None:
            v = short_fill - H[n - 1]
            if v < cur_s[1]:
                cur_s[1] = v
            v = short_fill - L[n - 1]
            if v > cur_s[2]:
                cur_s[2] = v
            pnl = (short_fill - C[n - 1]) * pv * ct - comm
            realized += pnl
            trades.append({
                'type': 'Short', 'entry_dt': dts[cur_s[0]],
                'entry_price': short_fill, 'exit_dt': dts[n - 1],
                'exit_price': C[n - 1], 'entry_bar': cur_s[0],
                'exit_bar': n - 1, 'pnl': pnl,
                'pnl_pct': pnl / (short_fill * pv * ct) * 100.0,
                'mae': cur_s[1], 'mfe': cur_s[2], 'reason': 'eod'})
        equity[n - 1] = ic + realized

    return trades, equity


# ===========================================================================
# Main class (port of S4003V7WalkForward)
# ===========================================================================
class S4003WalkForward:
    def __init__(self, ticker=None, config_path=None):
        cfg = loadWalkForwardConfig(config_path)
        self.cfg = cfg
        self.ticker = ticker or cfg.get('symbolName', 'ES')
        self.table = str(cfg.get('tableName', 'ticker1Min'))
        self.analysis_start = pd.Timestamp(cfg['analysisStart'])
        self.analysis_end = pd.Timestamp(cfg['analysisEnd'])
        self.calib_days = int(cfg.get('calibDays', 28))
        self.grid_end = self.analysis_start + pd.Timedelta(days=self.calib_days)
        self.is_days = int(cfg.get('iisDays', 14))
        self.oos_days = int(cfg.get('oosDays', 1))
        self.oos_step_days = int(cfg.get('oosStepDays', 1))
        self.rolling_window = bool(cfg.get('rollingWindow', True))
        self.end_date_is_oos = bool(int(cfg.get('endDateIsOOS', 1)))
        self.gate_select_days = int(cfg.get('gateSelectDays', 5))
        self.gate_wr_grid = tuple(cfg.get('gateWrGrid',
                                          (0.0, 40.0, 45.0, 50.0, 55.0, 60.0)))
        self.gate_min_taken = int(cfg.get('gateMinTaken', 3))
        self.min_is_trades = int(cfg.get('minIsTrades', 5))
        self.win_rate_target = float(cfg.get('winRateTarget', 65.0))
        self.lookback_bars = int(cfg.get('lookbackBars', 100))
        self.output_dir = str(cfg['outputFolder'])
        self.param_grids = {}
        for key in ('param1', 'param2', 'fixedParam'):
            p = cfg.get(key) or {}
            if p.get('name'):
                self.param_grids[p['name']] = tuple(p.get('grid') or ())
        self.exit_cost = float(strategyS4003.commission_per_trade)
        self.timings = {}
        self._win_cache = {}

    # ------------------------------------------------------------------
    def _set_params(self, w1, n1, e1):
        strategyS4003.profit_mult = w1
        strategyS4003.stop_period1 = n1
        strategyS4003.entry_limit_atr_frac = e1
        strategyS4003.commission_per_trade = self.exit_cost

    def _window_data(self, sig, start_ts, end_ts):
        """Per-window cached inputs for _simulate_fast: the iloc slice view
        (lookbackBars bars kept before start for engine warm-up) converted
        once to python lists, with entry signals before start_ts zeroed -
        the same mask strategyS4003._simulate applies for trade_start_ts, so
        trades/equity are identical. Reused for every (w1, n1, e1) grid
        combination simulated on the same window."""
        key = (start_ts, end_ts)
        wd = self._win_cache.get(key)
        if wd is None:
            lo = sig.index.searchsorted(start_ts)
            lo = max(0, lo - self.lookback_bars)
            hi = sig.index.searchsorted(end_ts, side='right')
            sl = sig.iloc[lo:hi]
            dts = sl.index
            pre = int(np.count_nonzero(np.asarray(dts < start_ts)))
            buy0 = sl['buySig'].to_numpy(int).tolist()
            shr0 = sl['shortSig'].to_numpy(int).tolist()
            for i in range(pre):
                buy0[i] = 0
                shr0[i] = 0
            wd = {'dts': dts, 'pre': pre,
                  'O': sl['Open'].tolist(), 'H': sl['High'].tolist(),
                  'L': sl['Low'].tolist(), 'C': sl['Close'].tolist(),
                  'atr14': sl['atr14'].tolist(),
                  'max3': sl['max3BarAtr14_1'].tolist(),
                  'buy0': buy0, 'shr0': shr0}
            self._win_cache[key] = wd
        return wd

    def _simulate_window(self, sig, start_ts, end_ts, w1, n1, e1, stn1, sln1):
        """Run the strategy engine on [start_ts, end_ts] with entries gated
        to the window (lookbackBars bars kept before start for engine
        warm-up). stn1/sln1 daily stops stay unset (disabled), as in
        S4003_7. Returns (trades list, bar-level equity pd.Series).

        Runs _simulate_fast on the cached per-window lists (bit-identical
        trades/equity vs strategyS4003._simulate)."""
        if stn1 is not None or sln1 is not None:
            raise ValueError('stn1/sln1 daily stops are disabled in this port')
        self._set_params(w1, n1, e1)
        wd = self._window_data(sig, start_ts, end_ts)
        trades, equity = _simulate_fast(
            wd['O'], wd['H'], wd['L'], wd['C'], wd['atr14'], wd['max3'],
            wd['buy0'][:], wd['shr0'][:], wd['dts'], wd['pre'],
            strategyS4003.profit_mult, strategyS4003.stop_period1,
            strategyS4003.entry_limit_atr_frac,
            strategyS4003.entry_price_mode == 'limit',
            strategyS4003.commission_per_trade, strategyS4003.point_value,
            strategyS4003.num_contracts, strategyS4003.initial_capital)
        pre = wd['pre']
        eq_series = pd.Series(equity[pre:], index=wd['dts'][pre:])
        return trades, eq_series

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, _db_hi = get_db_range(self.ticker, self.table)
        # warm-up truncation (speed): analysisStart - WARMUP_DAYS calendar
        # days of history is enough for every indicator read by the
        # walk-forward windows to be bit-identical to full history (see
        # module docstring); fall back to the db minimum when it is later.
        warm_start = self.analysis_start - pd.Timedelta(days=WARMUP_DAYS)
        load_start = max(pd.Timestamp(db_lo), warm_start)
        print(f"[{PROGRAM_NAME}]")
        print(f"Loading {self.ticker} 1-min bars [{load_start} .. "
              f"{self.analysis_end}] (db min {db_lo}, warm-up "
              f"{WARMUP_DAYS}d) ...")
        t_db = time.perf_counter()
        self.df = load_data_from_db(self.ticker, str(load_start),
                                    str(self.analysis_end), self.table)
        self.timings['db_fetch'] = time.perf_counter() - t_db
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        t0 = time.perf_counter()
        self.sig = strategyS4003._prepare(self.df)
        self.timings['prepare'] = time.perf_counter() - t0
        in_period = ((self.sig.index >= self.analysis_start)
                     & (self.sig.index <= self.analysis_end))
        self.analysis_days = np.unique(np.asarray(self.sig.index[in_period].date))
        self.trading_days = len(self.analysis_days)
        self.grid_days = int(np.sum(self.analysis_days < self.grid_end.date()))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. {self.df.index[-1]}; "
              f"effective analysis period {self.analysis_days[0]} .. "
              f"{self.df.index[-1].date()} ({self.trading_days} trading days); "
              f"calibration (grid) window {self.analysis_days[0]} .. "
              f"{self.analysis_days[self.grid_days - 1]} ({self.grid_days} days), "
              f"no walk-forward day is inside it")

    # ------------------------------------------------------------------
    def _run_cfg(self, w1, n1, e1):
        trades, eq = self._simulate_window(self.sig, self.analysis_start,
                                           self.grid_end, w1, n1, e1, None, None)
        m = window_metrics(trades)
        return {
            'w1': w1, 'n1': n1, 'e1': e1,
            'trades': m['trades'], 'wins': m['wins'],
            'win_rate': m['win_rate'], 'net': m['net'],
            'sharpe': _sharpe_from_eq(eq), 'maxdd': _max_dd_from_eq(eq),
            'freq': m['trades'] / max(self.grid_days, 1),
        }

    @staticmethod
    def _rank_key(r):
        viable = (r['freq'] >= 0.5 and r['net'] > 0)
        return (viable, r['win_rate'], r['net'])

    # ------------------------------------------------------------------
    def grid_search(self):
        """Calibration-window (w1, n1, e1) grid. stn1/sln1 daily stops are
        never set (both stay disabled). Everything derived here uses only
        data before the first walk-forward out-of-sample day (no lookahead)."""
        print(f"\nPhase 1: calibration grid on [{self.analysis_start.date()} .. "
              f"{self.grid_end.date()}) over (w1, n1, e1), stn1/sln1 unset (daily "
              f"stops disabled), exit cost ${self.exit_cost:.2f}/trade")
        results = []
        for w1 in self.param_grids['w1']:
            for n1 in self.param_grids['n1']:
                for e1 in self.param_grids['e1']:
                    results.append(self._run_cfg(w1, n1, e1))
        self.phase1_results = pd.DataFrame(results)

        value_rows = []
        for p in VALUE_PARAMS:
            best_by_value = self.phase1_results.groupby(p)['win_rate'].max()
            value_rows.append({
                'parameter': p,
                'best_win_rate_by_value': '; '.join(
                    f"{k}={v:.2f}" for k, v in best_by_value.items()),
                'value_spread': float(best_by_value.max() - best_by_value.min()),
            })
        value_tbl = pd.DataFrame(value_rows).sort_values(
            'value_spread', ascending=False).reset_index(drop=True)
        self.parameter_values = value_tbl
        self.chosen_params = list(value_tbl['parameter'].head(2))
        self.fixed_params = [p for p in VALUE_PARAMS if p not in self.chosen_params]

        best = sorted(results, key=self._rank_key, reverse=True)[0]
        self.best_cfg = best
        self.fixed_values = {p: best[p] for p in self.fixed_params}

        print('\nParameter value analysis (win-rate spread across values):')
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
    def _is_window(self, day):
        """In-sample window for an OOS day: the last iisDays trading days
        strictly before `day` (trading days per analysis_days, not calendar
        days). Returns (is_start 00:00:00, is_end 23:59:59)."""
        prev = self.analysis_days[self.analysis_days < day.date()]
        win = prev[-self.is_days:]
        if len(win) == 0:
            raise RuntimeError(f'no in-sample trading days before {day.date()}')
        return (pd.Timestamp(win[0]),
                pd.Timestamp(win[-1]).replace(hour=23, minute=59, second=59))

    def _is_best_cfg(self, day, p1, p2, grid, fv):
        """IS-best (w1/n1/e1) configuration for an OOS day: full grid walk
        over the IS window (last iisDays trading days before `day`);
        selection key is (enough trades, win rate, net).
        Returns (v1, v2, is_metrics)."""
        is_start, is_end = self._is_window(day)
        best_is, best_pair = None, None
        for v1, v2 in grid:
            kw = dict(fv)
            kw[p1], kw[p2] = v1, v2
            trades, _ = self._simulate_window(self.sig, is_start, is_end,
                                              kw['w1'], kw['n1'], kw['e1'],
                                              None, None)
            m = window_metrics(trades)
            key = (m['trades'] >= self.min_is_trades, m['win_rate'], m['net'])
            if best_is is None or key > best_is:
                best_is, best_pair = key, (v1, v2, m)
        v1, v2, is_m = best_pair
        return v1, v2, is_m

    # ------------------------------------------------------------------
    def select_gate(self):
        """Select the OOS-day gate on calibration data only: mini
        walk-forward over the last gateSelectDays trading days of the
        calibration window. A day is taken when its IS-selected config has
        IS win rate >= wr_min AND IS net > 0. wr_min = 0.0 is the
        net>0-only baseline.

        Speed: a day's IS-best config and its OOS-day simulation do not
        depend on the gate candidate wr_min, so they are computed once per
        day and reused for every candidate (identical gate table)."""
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in self.param_grids[p1]
                for v2 in self.param_grids[p2]]
        fv = dict(self.fixed_values)
        calib_days = [d for d in
                      pd.date_range(self.analysis_start,
                                    self.grid_end - timedelta(days=1), freq='D')
                      if d.date() in set(self.analysis_days)]
        mini_days = calib_days[-self.gate_select_days:]
        print(f"\nGate selection on calibration mini-OOS days "
              f"{mini_days[0].date()} .. {mini_days[-1].date()} "
              f"(IS {self.is_days}d, never touches real OOS data)")
        day_results = []
        for day in mini_days:
            v1, v2, is_m = self._is_best_cfg(day, p1, p2, grid, fv)
            kw = dict(fv)
            kw[p1], kw[p2] = v1, v2
            day_end = (day + timedelta(days=self.oos_days - 1)).replace(
                hour=23, minute=59, second=59)
            oos_trades, _ = self._simulate_window(
                self.sig, day, day_end, kw['w1'], kw['n1'], kw['e1'],
                None, None)
            day_results.append((is_m, window_metrics(oos_trades)))
        rows = []
        for wr_min in self.gate_wr_grid:
            trades_n = wins = 0
            net = 0.0
            taken = 0
            for is_m, m in day_results:
                if is_m['win_rate'] < wr_min or is_m['net'] <= 0:
                    continue
                taken += 1
                trades_n += m['trades']
                wins += m['wins']
                net += m['net']
            wr = wins / trades_n * 100.0 if trades_n else 0.0
            rows.append({'gate_wr_min': wr_min, 'taken_days': taken,
                         'trades': trades_n, 'win_rate': wr, 'net': net})
            print(f"  gate IS wr >= {wr_min:4.0f}% & IS net > 0 | "
                  f"taken {taken}/{len(mini_days)} | {trades_n} trades | "
                  f"win {wr:6.2f}% | net {net:>9.2f}")
        gate_tbl = pd.DataFrame(rows)
        self.gate_table = gate_tbl
        viable = gate_tbl[gate_tbl['taken_days'] >= self.gate_min_taken]
        if len(viable):
            pick = viable.sort_values(['win_rate', 'net', 'taken_days'],
                                      ascending=False).iloc[0]
        else:
            pick = gate_tbl.iloc[0]
            print("  WARNING: no gate met the minimum taken days; "
                  "falling back to net>0-only baseline")
        self.gate_wr = float(pick['gate_wr_min'])
        print(f"-> gate selected: take day only if IS win rate >= "
              f"{self.gate_wr:.0f}% and IS net > 0 "
              f"(maximizes calibration mini-OOS win rate)")
        return self.gate_wr

    # ------------------------------------------------------------------
    def walk_forward(self):
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in self.param_grids[p1]
                for v2 in self.param_grids[p2]]
        fv = dict(self.fixed_values)

        oos_days = [d for d in pd.date_range(self.grid_end, self.analysis_end,
                                             freq=f'{self.oos_step_days}D')
                    if d.date() in set(self.analysis_days)]
        print(f"\nPhase 2: walk-forward from {oos_days[0].date()} (after the "
              f"calibration window), in-sample {self.is_days} trading days -> "
              f"out-of-sample {self.oos_days} day(s), {len(grid)} combinations "
              f"per window, {len(oos_days)} out-of-sample days, gate IS wr >= "
              f"{self.gate_wr:.0f}% & IS net > 0"
              + ("" if self.end_date_is_oos else
                 "; endDateIsOOS=0: extra final row selects parameters for "
                 "the next trading day after analysisEnd (OOS columns blank)"))

        rows, trades_all, eq_segments = [], [], []
        cum = 0.0
        taken_days = 0
        print(f"{'OOS day':>10} | {p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} "
              f"{'IS net':>9} | {'gate':>5} | {'OOS nTr':>7} {'OOS wr%':>7} "
              f"{'OOS pnl':>9} {'OOS Sharpe':>10} {'OOS MaxDD%':>10} {'cum':>9}")
        for day in oos_days:
            v1, v2, is_m = self._is_best_cfg(day, p1, p2, grid, fv)
            gate_pass = (is_m['win_rate'] >= self.gate_wr) and (is_m['net'] > 0)
            is_start, is_end = self._is_window(day)
            o_m = {'trades': 0, 'wins': 0, 'win_rate': 0.0, 'net': 0.0}
            day_sharpe = day_maxdd = 0.0
            if gate_pass:
                taken_days += 1
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                day_end = (day + timedelta(days=self.oos_days - 1)).replace(
                    hour=23, minute=59, second=59)
                oos_trades, oos_eq = self._simulate_window(
                    self.sig, day, day_end, kw['w1'], kw['n1'], kw['e1'],
                    None, None)
                o_m = window_metrics(oos_trades)
                day_sharpe = _sharpe_from_eq(oos_eq)
                day_maxdd = _max_dd_from_eq(oos_eq)
                eq_segments.append(oos_eq + cum)
                cum += o_m['net']
                for t in oos_trades:
                    t['oos_day'] = day.date()
                    t['p1'], t['p2'] = v1, v2
                trades_all.extend(oos_trades)
            rows.append({
                'oos_date': day.date(), p1: v1, p2: v2,
                **{f'fixed_{k}': fv[k] for k in self.fixed_params},
                'is_start': is_start.date(), 'is_end': is_end.date(),
                'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                'is_net': is_m['net'], 'taken': gate_pass,
                'oos_trades': o_m['trades'], 'oos_wins': o_m['wins'],
                'oos_win_rate': o_m['win_rate'], 'oos_pnl': o_m['net'],
                'oos_sharpe': day_sharpe, 'oos_maxdd': day_maxdd,
                'oos_cum_pnl': cum,
            })
            print(f"{str(day.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{'take' if gate_pass else 'skip':>5} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{day_sharpe:>10.2f} {day_maxdd:>10.2f} {cum:>9.2f}")
        # endDateIsOOS=0: analysisEnd is the last IS day, so append one row
        # selecting w1/n1 for the next trading day after analysisEnd. That
        # day has no bars in MariaDB yet, so OOS columns stay blank.
        self.next_day_params = None
        if not self.end_date_is_oos:
            next_day = self.analysis_end.normalize() + pd.Timedelta(days=1)
            v1, v2, is_m = self._is_best_cfg(next_day, p1, p2, grid, fv)
            is_start, is_end = self._is_window(next_day)
            gate_pass = (is_m['win_rate'] >= self.gate_wr) and (is_m['net'] > 0)
            rows.append({
                'oos_date': '', p1: v1, p2: v2,
                **{f'fixed_{k}': fv[k] for k in self.fixed_params},
                'is_start': is_start.date(), 'is_end': is_end.date(),
                'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                'is_net': is_m['net'], 'taken': gate_pass,
                'oos_trades': '', 'oos_wins': '', 'oos_win_rate': '',
                'oos_pnl': '', 'oos_sharpe': '', 'oos_maxdd': '',
                'oos_cum_pnl': cum,
            })
            self.next_day_params = {
                p1: v1, p2: v2, **fv,
                'is_start': is_start.date(), 'is_end': is_end.date(),
                'gate_take': gate_pass,
            }
            print(f"{'(next)':>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{'take' if gate_pass else 'skip':>5} | endDateIsOOS=0: "
                  f"parameters for the next trading day after "
                  f"{self.analysis_end.date()} (no bars in DB yet, "
                  f"OOS columns blank)")
        self.wf_results = pd.DataFrame(rows)
        self.wf_trades = pd.DataFrame(trades_all)
        self.taken_days = taken_days
        self.oos_equity = (pd.concat(eq_segments) if eq_segments
                           else pd.Series(dtype=float))
        return self.wf_results

    # ------------------------------------------------------------------
    def write_outputs(self, out_dir=None):
        out_dir = out_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')

        pa_path = os.path.join(out_dir, 'S4003_7_parameter_analysis.csv')
        self.phase1_results.to_csv(pa_path, index=False)

        pv_path = os.path.join(out_dir, 'S4003_7_parameter_values.csv')
        self.parameter_values.assign(
            chosen=', '.join(self.chosen_params),
            program=PROGRAM_NAME).to_csv(pv_path, index=False)

        gc_path = os.path.join(out_dir, 'S4003_7_gate_comparison.csv')
        self.gate_table.assign(chosen=self.gate_wr,
                               program=PROGRAM_NAME).to_csv(gc_path, index=False)

        wf_path = os.path.join(out_dir, 'S4003_7_walkforward_results.csv')
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, 'S4003_7_walkforward_trades.csv')
        if len(self.wf_trades):
            out = self.wf_trades.copy()
            out['entry_dt'] = out['entry_dt'].dt.strftime(DT_FORMAT)
            out['exit_dt'] = out['exit_dt'].dt.strftime(DT_FORMAT)
            out = out[['oos_day', 'type', 'entry_dt', 'entry_price', 'exit_dt',
                       'exit_price', 'pnl', 'reason', 'p1', 'p2']]
        else:
            out = self.wf_trades
        out.to_csv(tr_path, index=False)

        # ---- HTML (program name on top) ----
        n = len(self.wf_trades)
        wins = int((self.wf_trades['pnl'] > 0).sum()) if n else 0
        wr = wins / n * 100.0 if n else 0.0
        net = float(self.wf_trades['pnl'].sum()) if n else 0.0
        overall_sharpe = _sharpe_from_eq(self.oos_equity)
        overall_maxdd = _max_dd_from_eq(self.oos_equity)
        p1, p2 = self.chosen_params
        gated_out = (len(self.wf_results) - self.taken_days
                     - (1 if self.next_day_params else 0))
        eq = self.oos_equity if len(self.oos_equity) else \
            self.wf_results.set_index('oos_date')['oos_cum_pnl']
        # speed: dependency-free SVG chart (the fallback _equity_chart_html
        # itself uses when matplotlib is absent); importing matplotlib and
        # rendering a PNG costs ~1-2 s.
        chart = strategyS4003._equity_chart_svg(eq)

        def _wf_row_html(r):
            if r['oos_date'] == '':   # endDateIsOOS=0 next-day params row
                return (f"<tr><td></td><td>{r[p1]}</td><td>{r[p2]}</td>"
                        f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
                        f"<td>{r['is_net']:.2f}</td>"
                        f"<td>{'take' if r['taken'] else 'skip'}</td>"
                        f"<td></td><td></td><td></td><td></td><td></td>"
                        f"<td>{r['oos_cum_pnl']:.2f}</td></tr>")
            return (f"<tr><td>{r['oos_date']}</td><td>{r[p1]}</td><td>{r[p2]}</td>"
                    f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
                    f"<td>{r['is_net']:.2f}</td>"
                    f"<td>{'take' if r['taken'] else 'skip'}</td>"
                    f"<td>{r['oos_trades']}</td><td>{r['oos_win_rate']:.2f}</td>"
                    f"<td>{r['oos_pnl']:.2f}</td>"
                    f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
                    f"<td>{r['oos_cum_pnl']:.2f}</td></tr>")

        rows_html = ''.join(_wf_row_html(r)
                            for _, r in self.wf_results.iterrows())
        next_day_note = ''
        if self.next_day_params:
            ndp = self.next_day_params
            next_day_note = (
                f"<p>Row with blank OOS date (endDateIsOOS=0): parameters "
                f"selected for the next trading day after "
                f"{self.analysis_end.date()} - {p1}={ndp[p1]}, {p2}={ndp[p2]}, "
                f"IS {ndp['is_start']} .. {ndp['is_end']}, gate: "
                f"{'take' if ndp['gate_take'] else 'skip'}. OOS columns are "
                f"blank: that day has no bars in MariaDB yet.</p>")
        gate_rows = ''.join(
            f"<tr><td>&gt;= {r['gate_wr_min']:.0f}% &amp; IS net &gt; 0</td>"
            f"<td>{r['taken_days']}</td><td>{r['trades']}</td>"
            f"<td>{r['win_rate']:.2f}</td><td>{r['net']:.2f}</td></tr>"
            for _, r in self.gate_table.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        html = f"""<html><head><title>S4003_7 Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body>
<p style="font-family:monospace;font-size:15px;margin-bottom:4px;"><b>Program: {PROGRAM_NAME}</b></p>
<h2>S4003_7 Walk-Forward Parameter Test (OOS day gate)</h2>
<p>Period: {self.analysis_days[0]} .. {self.df.index[-1].date()} |
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
mark-to-market equity (annualized, 3% risk-free). Skipped days show no
OOS activity; cum pnl is unchanged.</p>
{next_day_note}
<table><tr><th>OOS date</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th><th>gate</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th><th>cum pnl</th></tr>
{rows_html}</table>
<h3>Equity Curve (cumulative OOS P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f'S4003_7_walkforward_{ts}.html')
        with open(html_path, 'w') as f:
            f.write(html)

        print('\nOutput files:')
        print(f'  parameter analysis: {pa_path}')
        print(f'  parameter values:   {pv_path}')
        print(f'  gate comparison:    {gc_path}')
        print(f'  walkforward results: {wf_path}')
        print(f'  walkforward trades:  {tr_path}')
        print(f'  html:                {html_path}')

        print('\n' + '=' * 60)
        print(f"OUT-OF-SAMPLE AGGREGATE: {n} trades on {self.taken_days} taken "
              f"days ({gated_out} gated out), win rate {wr:.2f}% "
              f"(target >= {self.win_rate_target:.0f}% -> "
              f"{'PASS' if wr >= self.win_rate_target else 'MISS'}), "
              f"net {net:,.2f}, freq {n / max(self.trading_days, 1):.2f}/day, "
              f"Sharpe {overall_sharpe:.2f}, max DD {overall_maxdd:.2f}%")
        print('=' * 60)

    # ------------------------------------------------------------------
    def run(self):
        t_run0 = time.perf_counter()
        strategyS4003.applyStrategyConfig()
        self.exit_cost = float(strategyS4003.commission_per_trade)
        orig = {k: getattr(strategyS4003, k) for k in
                ('commission_per_trade', 'profit_mult', 'stop_period1',
                 'entry_limit_atr_frac')}
        try:
            self._load()
            t0 = time.perf_counter()
            self.grid_search()
            self.timings['phase1_grid'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.select_gate()
            self.timings['gate_select'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.walk_forward()
            self.timings['walk_forward'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            self.write_outputs()
            self.timings['write_outputs'] = time.perf_counter() - t0
        finally:
            for k, v in orig.items():
                setattr(strategyS4003, k, v)
        t_total = time.perf_counter() - t_run0
        t_no_db = t_total - self.timings.get('db_fetch', 0.0)
        print(f"\nRuntime breakdown (s): "
              + ' | '.join(f"{k} {v:.2f}" for k, v in self.timings.items()))
        print(f"Total runtime: {t_total:.2f}s | excluding MariaDB fetch: "
              f"{t_no_db:.2f}s")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='S4003 walk-forward test with OOS day gate '
                    '(stn1/sln1 daily stops disabled); parameters from '
                    'config.yaml section "WalkForward Test"')
    ap.add_argument('--ticker', default=None,
                    help='default: WalkForward Test symbolName from config')
    ap.add_argument('--config', default=None,
                    help='config.yaml path (default: strategy folder config)')
    args = ap.parse_args()
    S4003WalkForward(ticker=args.ticker, config_path=args.config).run()


if __name__ == '__main__':
    main()
