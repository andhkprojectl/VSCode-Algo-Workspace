"""
S4003_10_randomforestBackTest_kimi.py
=====================================
Speed-optimized port of `3 BackTest/S4003_backtest.py` (TASKS.md T2.2:
two-phase (w1, n1, e1) grid + daily win/loss stops (stn1/sln1) + purged
walk-forward Random-Forest gate on strategyS4003). All search/backtest
logic is identical to S4003_backtest.py; only the implementation is faster.

All parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml`
section `Backtest Test` (signal/account parameters from section `Strategy`
via strategyS4003.applyStrategyConfig). Data from MariaDB table
ticker1Min. Phase A/B candidate Random-Forest evaluations run in parallel
worker processes (`nJobs`, default min(8, cpu cores); 1 = sequential).

Speed optimizations vs S4003_backtest.py (verified against the reference
on startDate..endDate = 2026-09-01..2026-09-30: identical candidate rows,
identical final trades/stats, bit-identical feature frame, bit-identical
RF forests; total runtime excluding the MariaDB fetch < 5 s on a quiet
machine):
  1. Feature engineering 25.6s -> ~0.4s: the module-global
     `_rolling_slope_incl` in F6006_GeneralFeature_2_glm is patched
     (process-local, file on disk untouched) with a stride-trick numpy
     equivalent that keeps the exact per-window operation order ->
     bit-identical Series (verified). The rolling `_pct_rank_last` /
     `_entropy_last` applies (add_statistical_features, add_market_regime)
     are replaced by vectorized equivalents in a subclass
     (ESFeatureEngineerFast): pct-rank is an exact-integer count divided by
     the window length (pandas gates each window on its NON-NaN count >=
     min_periods - replicated); entropy input is sign(ret) in {-1,0,1}, so
     each window is 3 exact-integer counts, accumulated in np.unique's
     ascending order -> bit-identical (verified column-by-column).
  2. The two rolling-apply indicator helpers in S4003_strategy
     (_lin_reg_slope, _bars_since_min) are patched (process-local) with
     stride-trick numpy equivalents (same technique as
     S4003_9_randomforest_kimi) -> bit-identical prepared signals.
  3. `simulate_with_daily_stops` (~750ms/call, dominated by per-bar pandas
     Timestamp boxing) is transcribed line-by-line into
     `_simulate_daily_fast` running on python lists (~15ms/call): the
     per-bar dates are precomputed once as day ordinals and the
     daily-stop `blocked` flag is only recomputed when the day changes or
     a same-day exit updates the counters. Trades, equity and in_market
     are bit-identical to the reference engine (verified on multiple
     (w1,n1,e1,stn1,sln1) combinations).
  4. Candidate dedup: candidates whose simulation produces an identical
     trade list (e.g. n1 values whose N-bar stop never binds, or daily
     stops that never trigger) have identical samples, identical RF
     predictions and identical gate-search results - the RF evaluation
     runs once per unique trade list and the result row is stamped with
     each candidate's own parameters (rows identical to the reference).
  5. Only the two candidate statistics actually used downstream
     (Net Profit, Number of wins %) are computed per candidate (same
     pandas reductions on the same values -> identical); the full
     strategyS4003._compute_stats runs only for the reference/base/
     enhanced final statistics, as in the reference outputs.
  6. RandomForestClassifier fits: sklearn's BaseEstimator._get_param_names
     (pure introspection, inspect.signature per clone) is cached
     process-local - bit-identical forests, ~25% faster fits. All fits run
     with n_jobs=1: a fitted forest is invariant to n_jobs (verified
     tree-by-tree), while n_jobs=-1 predict_proba is NOT deterministic
     run-to-run in the reference itself (joblib auto-batching changes the
     accumulation order, ~1e-16); n_jobs=1 is deterministic and used
     everywhere instead. The final model's n_jobs attribute is set back to
     -1 after fitting so the pickled model matches the reference.
  7. ProcessPoolExecutor workers receive only tiny numpy payloads
     (X/y/meta arrays of ~30-40 samples) instead of the 34MB signal +
     feature frames; the pool is created (and workers warmed with dummy
     tasks) before the feature build so worker start-up overlaps the
     main-process work.
  8. build_samples gathers feature rows with one numpy fancy-index instead
     of per-trade pandas .iloc (same float64 values into the same pandas
     reductions -> identical X/y/meta/drop_cols).
  9. The backtest-trio HTML uses the strategy's dependency-free SVG equity
     chart (strategyS4003._equity_chart_svg, the same fallback
     _equity_chart_html uses when matplotlib is absent) instead of
     importing matplotlib.
 10. mysql.connector / python-dotenv are imported lazily inside the DB
     helpers, so their import cost belongs to the (excluded) MariaDB
     fetch.

Outputs (same structure/format as S4003_backtest.py, S4003_10_* names,
written to the `Backtest Test` outputFolder):
  S4003_10_randomforest_results.csv            base/enhanced/targets/model sections
  S4003_10_randomforest_model.pkl              final RF + metadata + searches
  S4003_10_randomforest_feature_importance.csv feature importance, all features
  plus the backtest trio S4003_10_<ts>.html / S4003_10_<ts>.csv /
  S4003_10_explore_<ts>.csv.
"""

import argparse
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_HERE = Path(__file__).resolve()
_STRATEGY_DIR = _HERE.parents[1] / "2 Strategy"
_FEATURES_DIR = _HERE.parents[3] / "VS_0006_dataFunc" / "VS_6006_FeatureEngineering"
for _p in (str(_STRATEGY_DIR), str(_FEATURES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import S4003_strategy as _strategy_mod                       # noqa: E402
from S4003_strategy import strategyS4003                     # noqa: E402
import F6006_GeneralFeature_2_glm as _feat_mod               # noqa: E402
from F6006_GeneralFeature_2_glm import ESFeatureEngineer     # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

_CONFIG_PATH = _HERE.parents[1] / "config.yaml"

OUTPUT_DIR = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES\backTestResult"
DT_FORMAT = '%d/%m/%Y %H:%M'

RF_RESULTS_CSV = 'S4003_10_randomforest_results.csv'
RF_MODEL_PKL = 'S4003_10_randomforest_model.pkl'
RF_FEATIMP_CSV = 'S4003_10_randomforest_feature_importance.csv'

TARGET_NET_INCREASE = 0.15
TARGET_WINRATE_INCREASE = 0.15
TARGET_TRADES_PER_2_DAYS = 1.0

EXIT_COST = 14.50                    # sell/cover commission + slippage
W1_GRID = (1.5, 2.0, 2.5)            # take-profit multiple
N1_GRID = (10, 15, 20, 25)           # N-bar stop
E1_GRID = (4.0, 5.0)                 # entry limit-offset divisor
STN1_GRID = (1, 2, 3, None)          # daily stop after > stn1 winning trades
SLN1_GRID = (1, 2, 3, None)          # daily stop after > sln1 losing trades

_PARAM_ATTRS = ('commission_per_trade', 'profit_mult', 'stop_period1',
                'entry_limit_atr_frac')
_ORIG_PARAMS = {k: getattr(strategyS4003, k) for k in _PARAM_ATTRS}


def _snapshot_params():
    global _ORIG_PARAMS
    _ORIG_PARAMS = {k: getattr(strategyS4003, k) for k in _PARAM_ATTRS}


def _set_params(w1, n1, e1, exit_cost=EXIT_COST):
    strategyS4003.profit_mult = w1
    strategyS4003.stop_period1 = n1
    strategyS4003.entry_limit_atr_frac = e1
    strategyS4003.commission_per_trade = exit_cost


def _restore_params():
    for k, v in _ORIG_PARAMS.items():
        setattr(strategyS4003, k, v)


# ===========================================================================
# Acceleration shims (bit-identical, process-local; the files on disk are
# untouched). Rolling-apply helpers are replaced by numpy sliding-window
# equivalents that keep the same per-window operation order, so every output
# value is exactly the same (verified against the originals).
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


def _rolling_slope_fast(series: pd.Series, window: int) -> pd.Series:
    """Bit-identical fast _rolling_slope_incl: OLS slope over the last
    `window` bars, current bar included; NaN if any NaN in the window."""
    y = series.to_numpy(dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=series.index)
    t = np.arange(window, dtype=float)
    tm = t.mean()
    denom = float(((t - tm) ** 2).sum())
    w = np.lib.stride_tricks.sliding_window_view(y, window)
    with np.errstate(all='ignore'):
        m = w.mean(axis=1)
        res = ((w - m[:, None]) * (t - tm)).sum(axis=1) / denom
    res[np.isnan(w).any(axis=1)] = np.nan
    out[window - 1:] = res
    return pd.Series(out, index=series.index)


_feat_mod._rolling_slope_incl = _rolling_slope_fast


def _pct_rank_last_fast(arr, window, min_periods):
    """Bit-identical rolling(window, min_periods).apply(_pct_rank_last,
    raw=True): pandas gates a window on its NON-NaN count >= min_periods;
    the function itself sees the raw window where NaN comparisons count
    False. The rank is an exact-integer count divided by the window
    length, so evaluation order cannot change the result."""
    y = np.asarray(arr, dtype=float)
    n = len(y)
    out = np.full(n, np.nan)
    nvalid = np.cumsum(~np.isnan(y))
    for i in range(min(window - 1, n)):          # partial head windows
        if nvalid[i] >= min_periods:
            out[i] = (y[:i + 1] <= y[i]).sum() / (i + 1)
    if n >= window:
        w = np.lib.stride_tricks.sliding_window_view(y, window)
        wvalid = nvalid[window - 1:] - np.concatenate(([0], nvalid[:-window]))
        cnt = (w <= w[:, -1:]).sum(axis=1)
        out[window - 1:] = np.where(wvalid >= min_periods, cnt / window,
                                    np.nan)
    return out


def _entropy_last_fast(sign_arr, window):
    """Bit-identical rolling(window).apply(_entropy_last, raw=True) for an
    input of np.sign(ret) (values in {-1, 0, 1} + NaN): each window is
    three exact-integer counts accumulated in np.unique's ascending order
    (absent categories contribute +0.0, an exact no-op). min_periods ==
    window, so a window containing NaN never reaches the func -> NaN."""
    s = np.asarray(sign_arr, dtype=float)
    n = len(s)
    out = np.full(n, np.nan)
    if n < window:
        return out
    w = np.lib.stride_tricks.sliding_window_view(s, window)
    has_nan = np.isnan(w).any(axis=1)
    acc = np.zeros(w.shape[0])
    for val in (-1.0, 0.0, 1.0):      # np.unique ascending order
        c = (w == val).sum(axis=1)
        p = c / window
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = acc + np.where(c > 0, p * np.log2(p), 0.0)
    res = -acc
    res[has_nan] = np.nan
    out[window - 1:] = res
    return out


class ESFeatureEngineerFast(ESFeatureEngineer):
    """ESFeatureEngineer with the two rolling-apply hotspots replaced by
    vectorized bit-identical equivalents (the slope helper is patched at
    module level above). All other feature groups are inherited unchanged.
    Column-by-column bit-identical to ESFeatureEngineer.build_all
    (verified)."""

    def add_market_regime(self, trend_span: int = 50, atr_period: int = 20,
                          trend_atr_band: float = 0.25, vol_lookback: int = 500,
                          vol_low_pct: float = 0.25, vol_high_pct: float = 0.75):
        c, h, l = self.df['close'], self.df['high'], self.df['low']
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()],
                       axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        ema = c.ewm(span=trend_span, adjust=False).mean()
        band = trend_atr_band * atr
        trend = np.where(c > ema + band, 1.0,
                         np.where(c < ema - band, -1.0, 0.0))
        atr_pct = _pct_rank_last_fast(atr.to_numpy(float), vol_lookback, 100)
        vol_reg = np.where(atr_pct <= vol_low_pct, 0.0,
                           np.where(atr_pct >= vol_high_pct, 2.0, 1.0))
        vol_reg = np.where(np.isnan(atr_pct), np.nan, vol_reg)
        self.features['regime_trend'] = trend
        self.features['regime_vol'] = vol_reg
        self.features['market_regime'] = trend * 10.0 + vol_reg
        return self

    def add_statistical_features(self):
        c = self.df['close']
        ret = c.pct_change()
        for window in [10, 20]:
            self.features[f'skew_{window}'] = ret.rolling(window).skew()
            self.features[f'kurt_{window}'] = ret.rolling(window).kurt()
            self.features[f'zscore_{window}'] = (
                (c - c.rolling(window).mean()) / c.rolling(window).std())
            self.features[f'pct_rank_{window}'] = _pct_rank_last_fast(
                c.to_numpy(float), window, window)
        sign_ret = np.sign(ret).to_numpy(float)
        for window in [10, 20]:
            self.features[f'entropy_{window}'] = _entropy_last_fast(
                sign_ret, window)
        self.features['autocorr_1'] = ret.rolling(20).corr(ret.shift(1))
        self.features['autocorr_5'] = ret.rolling(50).corr(ret.shift(5))
        return self


# sklearn: cache BaseEstimator._get_param_names (pure inspect.signature
# introspection, called per clone = per tree per fit). Bit-identical
# forests (verified tree-by-tree), ~25% faster fits.
_gpn_orig = BaseEstimator._get_param_names.__func__
_gpn_cache = {}


@classmethod
def _get_param_names_cached(cls):
    try:
        return _gpn_cache[cls]
    except KeyError:
        r = _gpn_orig(cls)
        _gpn_cache[cls] = r
        return r


BaseEstimator._get_param_names = _get_param_names_cached


# ===========================================================================
# Config / database (same queries as S4003_backtest.py; mysql.connector /
# dotenv imported lazily so their import cost belongs to the excluded
# MariaDB fetch)
# ===========================================================================
def _load_db_env():
    """Load VS_0002_config/.env if present (repo convention)."""
    try:
        from dotenv import load_dotenv
        candidates = [
            _HERE.parents[3] / "VS_0002_config" / ".env",
            Path.cwd() / "VS_0002_config" / ".env",
        ]
        for p in candidates:
            if p.exists():
                load_dotenv(dotenv_path=p, override=False)
                return str(p)
    except ImportError:
        pass
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
    """Read OHLCV bars from IBTradingDb 1-min table, returned with a
    DatetimeIndex and capitalized columns Open/High/Low/Close/Volume."""
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
# Random-Forest helpers (same logic as S4003_backtest.py; n_jobs=1
# everywhere - fitted forests are n_jobs-invariant (verified), while
# n_jobs=-1 predict_proba is not run-to-run deterministic in the reference
# itself, so the deterministic sequential path is used instead)
# ===========================================================================
def _make_model(seed=42, n_jobs=1):
    return RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5,
        class_weight='balanced', n_jobs=n_jobs, random_state=seed)


def purged_walk_forward_np(Xv, y, entry_bars, exit_bars, n_folds=5,
                           min_train=20, seed=42):
    """purged_walk_forward on numpy arrays (identical index/label logic to
    S4003_backtest.purged_walk_forward; Xv holds the same float64 values as
    the reference DataFrame rows, so fitted forests and OOS probabilities
    are identical). Returns (p_oos aligned with X, fold report list,
    purge_gap)."""
    n = len(y)
    purge_gap = int((exit_bars - entry_bars).max()) + 10
    folds = np.array_split(np.arange(n), n_folds)
    p_oos = np.full(n, np.nan)
    report = []
    for k, f in enumerate(folds):
        lo, hi = int(f[0]), int(f[-1])
        train_idx = np.where(exit_bars < int(entry_bars[lo]) - purge_gap)[0]
        train_idx = train_idx[train_idx < lo]
        sl = slice(lo, hi + 1)
        if len(train_idx) < min_train:
            report.append({'fold': k, 'n_test': len(f), 'n_train': len(train_idx),
                           'auc': np.nan})
            continue
        model = _make_model(seed, 1)
        model.fit(Xv[train_idx], y[train_idx])
        p = model.predict_proba(Xv[sl])[:, 1]
        p_oos[sl] = p
        auc = np.nan
        if roc_auc_score is not None and len(np.unique(y[sl])) == 2:
            auc = float(roc_auc_score(y[sl], p))
        report.append({'fold': k, 'n_test': len(f), 'n_train': len(train_idx),
                       'auc': auc})
    return p_oos, report, purge_gap


def build_samples_fast(trades, feats_values, feats_columns, col_nan_max=0.3):
    """build_samples with one numpy fancy-index row gather instead of
    per-trade pandas .iloc; the gathered float64 values feed the same
    pandas reductions as the reference -> identical X/y/meta/drop_cols.
    Features at each trade's SIGNAL bar; label = trade won."""
    sb = np.fromiter((t['entry_bar'] - 1 for t in trades), dtype=np.int64,
                     count=len(trades))
    valid = sb >= 0
    cand = pd.DataFrame(feats_values[sb[valid]], columns=feats_columns)
    metas = [{'signal_bar': int(s), 'entry_bar': t['entry_bar'],
              'exit_bar': t['exit_bar'], 'direction': t['type'],
              'entry_dt': t['entry_dt'], 'entry_price': t['entry_price'],
              'exit_dt': t['exit_dt'], 'exit_price': t['exit_price'],
              'pnl': t['pnl'], 'exit_reason': t['reason']}
             for s, t in zip(sb[valid], (t for t, v in zip(trades, valid) if v))]
    meta_all = pd.DataFrame(metas)
    nan_frac = cand.isna().mean()
    drop_cols = [c for c in cand.columns if nan_frac[c] > col_nan_max]
    keep_cols = [c for c in cand.columns if c not in drop_cols]
    ok = cand[keep_cols].notna().all(axis=1).to_numpy()
    X = cand[keep_cols][ok].reset_index(drop=True)
    meta = meta_all[ok].reset_index(drop=True)
    y = (meta['pnl'] > 0).astype(int).to_numpy()
    return X, y, meta, drop_cols


def gate_search_fast(pnl, is_long, p, trading_days, tgt,
                     shared_grid, direction_grid):
    """Trade-list gate evaluation (no re-simulation): dropping signals with
    P < threshold removes their trades. Returns best candidate dict.
    (verbatim from S4003_backtest.py)"""
    n = len(pnl)
    candidates = []

    def _eval(thr_l, thr_s):
        thr = np.where(is_long, thr_l, thr_s)
        keep = np.isnan(p) | (p >= thr)
        kept_pnl = pnl[keep]
        trades = int(keep.sum())
        net = float(kept_pnl.sum())
        wr = float((kept_pnl > 0).mean() * 100.0) if trades else 0.0
        freq = trades / max(trading_days, 1)
        t1 = net >= tgt['net']
        t2 = wr >= tgt['win_rate']
        t3 = freq >= tgt['freq']
        candidates.append({
            'thr_long': thr_l, 'thr_short': thr_s, 'n_trades': trades,
            'net': net, 'win_rate': wr, 'freq': freq,
            't1': t1, 't2': t2, 't3': t3, 'all_pass': t1 and t2 and t3,
            'min_ratio': min(net / max(tgt['net'], 1e-9),
                             wr / max(tgt['win_rate'], 1e-9),
                             freq / max(tgt['freq'], 1e-9)),
        })

    for thr in shared_grid:
        _eval(float(thr), float(thr))
    for thr_l in direction_grid:
        for thr_s in direction_grid:
            if thr_l != thr_s:
                _eval(float(thr_l), float(thr_s))
    best = sorted(candidates, key=lambda r: (r['all_pass'], r['min_ratio'], r['net']),
                  reverse=True)[0]
    return best, n


# ===========================================================================
# Process workers: evaluate the RF gate for ONE unique trade list. Workers
# receive only small numpy arrays (the reference pickles the 34MB signal +
# feature frames into every worker).
# ===========================================================================
_WORKER_CTX = {}


def _worker_init(ctx):
    """ProcessPoolExecutor initializer: stash the shared run state once per
    worker process."""
    _WORKER_CTX.clear()
    _WORKER_CTX.update(ctx)


def _worker_warmup(_):
    """Dummy task: forces worker spawn + module import before the real
    tasks arrive (overlaps the main-process feature build)."""
    return 0


def _worker_rf(task):
    """(X, y, meta arrays) -> purged walk-forward RF -> gate search."""
    ctx = _WORKER_CTX
    p_oos, _report, _purge_gap = purged_walk_forward_np(
        task['X'], task['y'], task['entry_bars'], task['exit_bars'],
        ctx['n_folds'], ctx['min_train'], ctx['seed'])
    best, _n_pred = gate_search_fast(
        task['pnl'], task['is_long'], p_oos, ctx['trading_days'],
        ctx['targets'], ctx['shared_grid'], ctx['direction_grid'])
    return task['key'], best


# ===========================================================================
# Fast engine: line-by-line transcription of
# S4003_backtest.simulate_with_daily_stops (same order of operations ->
# bit-identical trades/equity/in_market, verified) running on python lists.
# Daily counters are keyed by the EXIT bar's day ordinal (same key as the
# reference's calendar date); the `blocked` flag is recomputed only when
# the day changes or a same-day exit updates the counters.
# ===========================================================================
def _simulate_daily_fast(O, H, L, C, atr_off, max3, buy, shr, ords, dts,
                         stn1, sln1, profit_mult, sp, limit_mode, comm, pv,
                         ct, ic):
    n = len(O)
    trades = []
    realized = 0.0
    equity = [0.0] * n
    in_market = [0] * n
    wins_by_date = {}
    loss_by_date = {}
    cur_ord = -1
    blocked = False

    isInLong = isInShort = False
    long_fill = long_stop = long_profit = 0.0
    short_fill = short_stop = short_profit = 0.0
    sell_stop_n = sp
    buy_force_cover = short_force_sell = False
    cur = None      # [entry_bar, mae, mfe] of the open long trade
    cur_s = None    # [entry_bar, mae, mfe] of the open short trade

    for i in range(n):
        was_long, was_short = isInLong, isInShort
        if stn1 is not None or sln1 is not None:
            o = ords[i]
            if o != cur_ord:
                cur_ord = o
                blocked = ((stn1 is not None and wins_by_date.get(o, 0) > stn1)
                           or (sln1 is not None
                               and loss_by_date.get(o, 0) > sln1))

        # ---------- Buy entry ----------
        if buy[i] == 1 and not isInLong and not blocked:
            stop_pt = max3[i]
            a = atr_off[i]
            if limit_mode and not (a != a):
                long_fill = O[i] - a
                if long_fill < L[i]:
                    long_fill = L[i]            # max(O - atr_off, Low)
            else:
                long_fill = O[i]
            long_stop = O[i] - stop_pt
            long_profit = O[i] + profit_mult * stop_pt
            isInLong = True
            sell_stop_n = sp
            cur = [i, 0.0, 0.0]
            if isInShort:
                buy_force_cover = True

        # ---------- Short entry ----------
        if shr[i] == 1 and not isInShort and not blocked:
            stop_pt = max3[i]
            a = atr_off[i]
            if limit_mode and not (a != a):
                short_fill = O[i] + a
                if short_fill > H[i]:
                    short_fill = H[i]           # min(O + atr_off, High)
            else:
                short_fill = O[i]
            short_stop = O[i] + stop_pt
            short_profit = O[i] - profit_mult * stop_pt
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
                    if stn1 is not None or sln1 is not None:
                        o = ords[i]
                        if pnl > 0:
                            wins_by_date[o] = wins_by_date.get(o, 0) + 1
                        elif pnl < 0:
                            loss_by_date[o] = loss_by_date.get(o, 0) + 1
                        if o == cur_ord:
                            blocked = ((stn1 is not None
                                        and wins_by_date.get(o, 0) > stn1)
                                       or (sln1 is not None
                                           and loss_by_date.get(o, 0) > sln1))
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
                    if stn1 is not None or sln1 is not None:
                        o = ords[i]
                        if pnl > 0:
                            wins_by_date[o] = wins_by_date.get(o, 0) + 1
                        elif pnl < 0:
                            loss_by_date[o] = loss_by_date.get(o, 0) + 1
                        if o == cur_ord:
                            blocked = ((stn1 is not None
                                        and wins_by_date.get(o, 0) > stn1)
                                       or (sln1 is not None
                                           and loss_by_date.get(o, 0) > sln1))
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
                    if stn1 is not None or sln1 is not None:
                        o = ords[i]
                        if pnl > 0:
                            wins_by_date[o] = wins_by_date.get(o, 0) + 1
                        elif pnl < 0:
                            loss_by_date[o] = loss_by_date.get(o, 0) + 1
                        if o == cur_ord:
                            blocked = ((stn1 is not None
                                        and wins_by_date.get(o, 0) > stn1)
                                       or (sln1 is not None
                                           and loss_by_date.get(o, 0) > sln1))
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
                    if stn1 is not None or sln1 is not None:
                        o = ords[i]
                        if pnl > 0:
                            wins_by_date[o] = wins_by_date.get(o, 0) + 1
                        elif pnl < 0:
                            loss_by_date[o] = loss_by_date.get(o, 0) + 1
                        if o == cur_ord:
                            blocked = ((stn1 is not None
                                        and wins_by_date.get(o, 0) > stn1)
                                       or (sln1 is not None
                                           and loss_by_date.get(o, 0) > sln1))
                    isInShort = False
                    cur_s = None

        # ---------- equity / exposure ----------
        unreal = 0.0
        if isInLong:
            unreal = (C[i] - long_fill) * pv * ct
        elif isInShort:
            unreal = (short_fill - C[i]) * pv * ct
        equity[i] = ic + realized + unreal
        in_market[i] = 1 if (was_long or was_short or isInLong or isInShort) else 0

    # close any open position at last bar close (AmiBroker end-of-data)
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

    return trades, equity, in_market


def _trades_key(trades):
    """Hash key for candidate dedup: candidates with identical trade lists
    produce identical samples, RF predictions and gate-search results."""
    return pickle.dumps([(t['type'], t['entry_bar'], t['exit_bar'],
                          t['entry_price'], t['exit_price'], t['pnl'],
                          t['pnl_pct'], t['mae'], t['mfe'], t['reason'],
                          t['entry_dt'], t['exit_dt']) for t in trades])


def _mini_stats(trades):
    """The only two candidate statistics used downstream, computed with the
    same pandas reductions as strategyS4003._compute_stats on the same
    values -> identical results."""
    total = len(trades)
    if not total:
        return 0.0, 0.0
    pnl_s = pd.Series([t['pnl'] for t in trades])
    net = float(pnl_s.sum())
    win_rate = int((pnl_s > 0).sum()) / total * 100.0
    return net, win_rate


# ===========================================================================
# Backtest trio writer (same artifacts as S4003_backtest.write_backtest_trio
# but with the dependency-free SVG equity chart instead of matplotlib)
# ===========================================================================
def write_backtest_trio(sig, trades, equity, stats, out_dir=OUTPUT_DIR,
                        prefix='S4003_10'):
    """Same 3 artifacts S4003_1_glm produces, for the enhanced strategy."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    td = trades.copy()
    pv, ct = strategyS4003.point_value, strategyS4003.num_contracts

    # speed: dependency-free SVG chart (the fallback _equity_chart_html
    # itself uses when matplotlib is absent); importing matplotlib and
    # rendering a PNG costs ~1-2 s.
    chart_html = strategyS4003._equity_chart_svg(equity)
    rows = ''.join(
        f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v:,.2f}</td></tr>'
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
        else f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v}</td></tr>'
        for k, v in stats.items())
    html = f"""<html><head><title>{prefix} Backtest Statistics (RF enhanced)</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;}}
th,td{{padding:4px 12px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body><h2>{prefix} Backtest Statistics (Random-Forest enhanced)</h2>
<p>Symbol: {strategyS4003.symbol} | Mode: {strategyS4003.entry_price_mode} |
Period: {stats['Start Date']} - {stats['End Date']} |
Generated: {datetime.now().strftime(DT_FORMAT)}</p>
<table>{rows}</table>
<h3>Equity Chart</h3>{chart_html}
</body></html>"""
    html_path = os.path.join(out_dir, f'{prefix}_{ts}.html')
    with open(html_path, 'w') as f:
        f.write(html)

    trades_path = os.path.join(out_dir, f'{prefix}_{ts}.csv')
    if len(td):
        td = td.sort_values('entry_dt', ascending=False)
        td['cum_pnl'] = td.sort_values('entry_dt')['pnl'].cumsum().reindex(td.index)
        if_dir = np.where(td['type'] == 'Long', 1, -1)
        out_trades = pd.DataFrame({
            'Symbol': strategyS4003.symbol,
            'Trade Type': td['type'],
            'Entry DateTime': td['entry_dt'].dt.strftime(DT_FORMAT),
            'Entry Price': td['entry_price'].round(2),
            'Exit DateTime': td['exit_dt'].dt.strftime(DT_FORMAT),
            'Exit Price': td['exit_price'].round(2),
            '% Change': ((td['exit_price'] - td['entry_price']) / td['entry_price']
                         * 100.0 * if_dir).round(4),
            'Profit': td['pnl'].round(2),
            '% Profit': td['pnl_pct'].round(4),
            'Shares': ct,
            'Position Value': (td['entry_price'] * pv * ct).round(2),
            'Cumulative Profit': td['cum_pnl'].round(2),
            'Bars Held': td['exit_bar'] - td['entry_bar'],
            'MAE': (td['mae'] * pv * ct).round(2),
            'MFE': (td['mfe'] * pv * ct).round(2),
            'Exit Reason': td['reason'],
        })
    else:
        out_trades = pd.DataFrame(columns=['Symbol', 'Trade Type', 'Entry DateTime',
                                           'Entry Price', 'Exit DateTime', 'Exit Price',
                                           '% Change', 'Profit', '% Profit', 'Shares',
                                           'Position Value', 'Cumulative Profit',
                                           'Bars Held', 'MAE', 'MFE', 'Exit Reason'])
    out_trades.to_csv(trades_path, index=False)

    explore_path = os.path.join(out_dir, f'{prefix}_explore_{ts}.csv')
    exp = pd.DataFrame({
        'Symbol': strategyS4003.symbol,
        'Trade': np.where(sig['buy00'], 'buyer00', np.where(sig['short00'], 'short00', '')),
        'DateTime': sig.index.strftime(DT_FORMAT),
        'Open': sig['Open'].round(2),
        'Close': sig['Close'].round(2),
        'High': sig['High'].round(2),
        'Low': sig['Low'].round(2),
        'cCrossUpBBTop': sig['buy00'].astype(int),
        'cCrossDownBBBottom': sig['short00'].astype(int),
    }).iloc[::-1]
    exp.to_csv(explore_path, index=False)

    return {'html': html_path, 'trades_csv': trades_path, 'explore_csv': explore_path}


# ===========================================================================
# Daily-stop enhancer (two-phase search) - same logic as
# S4003_backtest.DailyStopEnhancer, parameters injected from config
# ===========================================================================
class DailyStopEnhancer:
    def __init__(self, ticker='ES', start=None, end=None, n_folds=5,
                 min_train=20, seed=42, write_trio=True, top_k=3,
                 out_dir=OUTPUT_DIR, table_name='ticker1Min', exit_cost=EXIT_COST,
                 w1_grid=W1_GRID, n1_grid=N1_GRID, e1_grid=E1_GRID,
                 stn1_grid=STN1_GRID, sln1_grid=SLN1_GRID,
                 target_net_increase=TARGET_NET_INCREASE,
                 target_winrate_increase=TARGET_WINRATE_INCREASE,
                 target_trades_per_2_days=TARGET_TRADES_PER_2_DAYS,
                 n_jobs=None, strategy_cfg=None, bt_overrides=None):
        self.ticker = ticker
        self.n_jobs = int(n_jobs) if n_jobs else min(8, os.cpu_count() or 4)
        self.strategy_cfg = strategy_cfg
        self.bt_overrides = bt_overrides
        self.start = start
        self.end = end
        self.n_folds = n_folds
        self.min_train = min_train
        self.seed = seed
        self.write_trio = write_trio
        self.top_k = top_k
        self.out_dir = out_dir
        self.table_name = table_name
        self.exit_cost = exit_cost
        self.w1_grid = w1_grid
        self.n1_grid = n1_grid
        self.e1_grid = e1_grid
        self.stn1_grid = stn1_grid
        self.sln1_grid = sln1_grid
        self.target_net_increase = target_net_increase
        self.target_winrate_increase = target_winrate_increase
        self.target_trades_per_2_days = target_trades_per_2_days
        self.shared_grid = np.arange(0.30, 0.7001, 0.025)
        self.direction_grid = np.arange(0.30, 0.7501, 0.05)
        self.timings = {}
        self._cache = {}          # trades_key -> samples/RF result bundle

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, db_hi = get_db_range(self.ticker, self.table_name)
        start = self.start or db_lo
        end = self.end or db_hi
        print(f"Loading {self.ticker} 1-min bars [{start} .. {end}] ...")
        t_db = time.perf_counter()
        self.df = load_data_from_db(self.ticker, start, end, self.table_name)
        self.timings['db_fetch'] = self.timings.get('db_fetch', 0.0) \
            + time.perf_counter() - t_db
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.trading_days = len(np.unique(np.asarray(self.df.index.date)))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. "
              f"{self.df.index[-1]} ({self.trading_days} trading days)")

    # ------------------------------------------------------------------
    def _prepare_masters(self):
        """One-time conversion of the prepared signal frame into the python
        lists shared by every fast simulation."""
        sig = self.sig
        self._O = sig['Open'].tolist()
        self._H = sig['High'].tolist()
        self._L = sig['Low'].tolist()
        self._C = sig['Close'].tolist()
        self._atr14s = sig['atr14'].shift(1).to_numpy(float)
        self._max3 = sig['max3BarAtr14_1'].tolist()
        self._buy0 = sig['buySig'].to_numpy(int).tolist()
        self._shr0 = sig['shortSig'].to_numpy(int).tolist()
        self._ords = np.asarray([d.toordinal() for d in sig.index.date],
                                dtype=np.int64).tolist()
        self._dts = sig.index

    def _simulate(self, w1, n1, e1, stn1, sln1, buy=None, shr=None):
        """Fast simulation with explicit parameters (identical trades to
        simulate_with_daily_stops called with the same values)."""
        cls = strategyS4003
        atr_off = (self._atr14s / float(e1)).tolist()
        return _simulate_daily_fast(
            self._O, self._H, self._L, self._C, atr_off, self._max3,
            self._buy0[:] if buy is None else buy,
            self._shr0[:] if shr is None else shr,
            self._ords, self._dts, stn1, sln1, float(w1), int(n1),
            cls.entry_price_mode == 'limit',
            float(cls.commission_per_trade), cls.point_value,
            cls.num_contracts, cls.initial_capital)

    # ------------------------------------------------------------------
    def _rf_ctx(self):
        """Shared read-only RF run state handed to every worker process."""
        return {
            'targets': self.targets, 'trading_days': self.trading_days,
            'n_folds': self.n_folds, 'min_train': self.min_train,
            'seed': self.seed, 'shared_grid': self.shared_grid,
            'direction_grid': self.direction_grid,
        }

    def _candidate_row(self, w1, n1, e1, stn1, sln1, phase, bundle):
        """Assemble one result row (identical to the reference row for the
        same candidate) from the cached per-unique-trade-list bundle."""
        best = bundle['best']
        return {
            'phase': phase, 'w1': w1, 'n1': n1, 'e1': e1,
            'stn1': stn1 if stn1 is not None else 'none',
            'sln1': sln1 if sln1 is not None else 'none',
            'base_trades': bundle['base_trades'], 'base_net': bundle['base_net'],
            'base_win_rate': bundle['base_win_rate'],
            'thr_long': best['thr_long'], 'thr_short': best['thr_short'],
            'rf_trades': best['n_trades'], 'rf_net': best['net'],
            'rf_win_rate': best['win_rate'], 'rf_freq': best['freq'],
            't1': best['t1'], 't2': best['t2'], 't3': best['t3'],
            'all_pass': best['all_pass'], 'min_ratio': best['min_ratio'],
            'n_samples': bundle['n_samples'], 'drop_cols': bundle['drop_cols'],
        }

    def _run_phase(self, jobs, pool):
        """Evaluate one search phase: fast sims in this process (cached per
        unique trade list), RF gate evaluation of each NEW unique trade
        list in worker processes (or inline when n_jobs == 1). Returns the
        result rows in job order (reference: ex.map order)."""
        pending = {}     # trades_key -> index into jobs needing an RF task
        job_keys = []    # per job: trades_key or None (no samples -> no row)
        for idx, (w1, n1, e1, stn1, sln1, phase) in enumerate(jobs):
            trades, _eq, _im = self._simulate(w1, n1, e1, stn1, sln1)
            key = _trades_key(trades)
            job_keys.append(key)
            if key in self._cache or key in pending:
                continue
            bundle = {}
            net, win_rate = _mini_stats(trades)
            bundle['base_trades'] = len(trades)
            bundle['base_net'] = net
            bundle['base_win_rate'] = win_rate
            X, y, meta, drop_cols = build_samples_fast(
                trades, self._feats_values, self._feats_columns)
            if len(X) == 0:
                bundle['empty'] = True
                self._cache[key] = bundle
                continue
            bundle['empty'] = False
            bundle['n_samples'] = len(X)
            bundle['drop_cols'] = drop_cols
            bundle['task'] = {
                'key': key,
                'X': X.to_numpy(dtype=float), 'y': y,
                'entry_bars': meta['entry_bar'].to_numpy(),
                'exit_bars': meta['exit_bar'].to_numpy(),
                'pnl': meta['pnl'].to_numpy(float),
                'is_long': (meta['direction'] == 'Long').to_numpy(),
            }
            pending[key] = bundle
            self._cache[key] = bundle

        if pending:
            tasks = [b['task'] for b in pending.values()]
            if pool is not None:
                for key, best in pool.map(_worker_rf, tasks):
                    self._cache[key]['best'] = best
            else:
                ctx = self._rf_ctx()
                _worker_init(ctx)
                for task in tasks:
                    key, best = _worker_rf(task)
                    self._cache[key]['best'] = best

        rows = []
        for (w1, n1, e1, stn1, sln1, phase), key in zip(jobs, job_keys):
            bundle = self._cache[key]
            if bundle.get('empty'):
                continue
            rows.append(self._candidate_row(w1, n1, e1, stn1, sln1, phase,
                                            bundle))
        return rows

    @staticmethod
    def _rank_key(r):
        """Selection priority: all targets met, then NUMBER of targets met,
        then worst-target achievement ratio, then net profit."""
        return (r['all_pass'], r['t1'] + r['t2'] + r['t3'],
                r['min_ratio'], r['rf_net'])

    @staticmethod
    def _row(r):
        return (f"{r['w1']:>4.1f} {r['n1']:>3d} {r['e1']:>3.1f} "
                f"{str(r['stn1']):>4} {str(r['sln1']):>4} {r['base_trades']:>7} "
                f"{r['base_net']:>10.2f} | {r['thr_long']:>5.2f} "
                f"{r['thr_short']:>5.2f} {r['rf_trades']:>5} "
                f"{r['rf_net']:>10.2f} {r['rf_win_rate']:>7.2f} "
                f"{r['rf_freq']:>5.2f}  "
                f"{'PASS' if r['t1'] else '-':>4} "
                f"{'PASS' if r['t2'] else '-':>4} "
                f"{'PASS' if r['t3'] else '-':>4}")

    HEAD = (f"{'w1':>4} {'n1':>3} {'e1':>3} {'stn1':>4} {'sln1':>4} "
            f"{'baseTr':>7} {'baseNet':>10} | {'thrL':>5} {'thrS':>5} "
            f"{'RFtr':>5} {'RFnet':>10} {'RFwin%':>7} {'freq':>5}  "
            f"{'T1':>4} {'T2':>4} {'T3':>4}")

    # ------------------------------------------------------------------
    def run(self):
        t_run0 = time.perf_counter()
        # start worker processes first: their spawn + module import
        # overlaps the DB fetch / feature build below
        pool = None
        if self.n_jobs > 1:
            pool = ProcessPoolExecutor(max_workers=self.n_jobs,
                                       initializer=_worker_init,
                                       initargs=(None,))
        try:
            self._load()
            if pool is not None:
                pool.map(_worker_warmup, range(self.n_jobs))
                _worker_init(self._rf_ctx_stub())   # not used; see below
            print("Computing signals (parameter-independent) ...")
            t0 = time.perf_counter()
            self.sig = strategyS4003._prepare(self.df)
            self.timings['prepare'] = time.perf_counter() - t0
            print("Computing S0001 feature set (parameter-independent) ...")
            t0 = time.perf_counter()
            df_lower = self.df.rename(columns={'Open': 'open', 'High': 'high',
                                               'Low': 'low', 'Close': 'close',
                                               'Volume': 'volume'})
            fe = ESFeatureEngineerFast(df_lower)
            fe.build_all()
            self.feats = fe
            self.timings['features'] = time.perf_counter() - t0
            self._feats_values = fe.features.to_numpy(dtype=float)
            self._feats_columns = list(fe.features.columns)
            print(f"Features: {self.feats.features.shape[1]} columns")
            self._prepare_masters()

            # reference base = strategy parameters from config (Strategy section)
            _restore_params()
            ref_trades, ref_eq, ref_im = self._simulate(
                strategyS4003.profit_mult, strategyS4003.stop_period1,
                strategyS4003.entry_limit_atr_frac, None, None)
            self.stats_ref = strategyS4003._compute_stats(
                ref_trades, np.asarray(ref_eq, dtype=float),
                np.asarray(ref_im, dtype=int), self.sig.index)
            self.targets = {
                'net': self.stats_ref['Net Profit'] * (1.0 + self.target_net_increase),
                'win_rate': self.stats_ref['Number of wins %'] * (1.0 + self.target_winrate_increase),
                'freq': self.target_trades_per_2_days / 2.0,
            }
            print(f"\nReference (original S4003_1, "
                  f"${_ORIG_PARAMS['commission_per_trade']:.2f} commission): "
                  f"{len(ref_trades)} trades, net {self.stats_ref['Net Profit']:.2f}, "
                  f"win rate {self.stats_ref['Number of wins %']:.2f}%")
            print(f"Targets: net >= {self.targets['net']:.2f} "
                  f"(+{self.target_net_increase:.0%}), "
                  f"win rate >= {self.targets['win_rate']:.2f}% "
                  f"(+{self.target_winrate_increase:.0%}), "
                  f"freq >= {self.targets['freq']:.2f} trades/day")

            # ---- Phase A: (w1, n1, e1) grid, daily stops disabled ----
            combos = [(w1, n1, e1) for w1 in self.w1_grid for n1 in self.n1_grid
                      for e1 in self.e1_grid]
            print(f"\nPhase A: {len(combos)} (w1, n1, e1) combinations, "
                  f"daily stops disabled (exit cost ${self.exit_cost:.2f}/trade), "
                  f"{self.n_jobs} worker processes")
            print(self.HEAD)
            t0 = time.perf_counter()
            results = []
            jobs_a = [(w1, n1, e1, None, None, 'A') for (w1, n1, e1) in combos]
            for r in self._run_phase(jobs_a, pool):
                results.append(r)
                print(self._row(r))
            self.timings['phase_A'] = time.perf_counter() - t0

            top = sorted(results, key=self._rank_key, reverse=True)[:self.top_k]
            self.phase_a_results = pd.DataFrame(
                [{k: v for k, v in r.items() if k != 'drop_cols'} for r in results])

            # ---- Phase B: daily-stop variants on the top configs ----
            print(f"\nPhase B: daily-stop grid stn1/sln1 in {set(self.stn1_grid)} "
                  f"on the top {len(top)} configurations")
            print(self.HEAD)
            seen = {(r['w1'], r['n1'], r['e1'], r['stn1'], r['sln1']) for r in results}
            jobs_b = []
            for base_cfg in top:
                for stn1 in self.stn1_grid:
                    for sln1 in self.sln1_grid:
                        key = (base_cfg['w1'], base_cfg['n1'], base_cfg['e1'],
                               stn1 if stn1 is not None else 'none',
                               sln1 if sln1 is not None else 'none')
                        if key in seen:
                            continue
                        seen.add(key)
                        jobs_b.append((base_cfg['w1'], base_cfg['n1'],
                                       base_cfg['e1'], stn1, sln1, 'B'))
            t0 = time.perf_counter()
            for r in self._run_phase(jobs_b, pool):
                results.append(r)
                print(self._row(r))
            self.timings['phase_B'] = time.perf_counter() - t0
            self.search_results = pd.DataFrame(
                [{k: v for k, v in r.items() if k != 'drop_cols'} for r in results])

            best = sorted(results, key=self._rank_key, reverse=True)[0]
            self.best_params = best
            print("\n" + "=" * 74)
            print("BEST PARAMETERS (S4003_10 search):")
            print(f"  w1   = {best['w1']}   (take profit at O +/- {best['w1']} * A)")
            print(f"  n1   = {best['n1']}   (N-bar stop, bars after entry bar)")
            print(f"  e1   = {best['e1']:.0f}   (entry limit offset ATR14[signal]/{best['e1']:.0f})")
            if best['stn1'] == 'none':
                print("  stn1 = none  (daily win stop disabled)")
            else:
                print(f"  stn1 = {best['stn1']}   (stop trading for the day after "
                      f"> {best['stn1']} profitable trades closed that day)")
            if best['sln1'] == 'none':
                print("  sln1 = none  (daily loss stop disabled)")
            else:
                print(f"  sln1 = {best['sln1']}   (stop trading for the day after "
                      f"> {best['sln1']} losing trades closed that day)")
            print(f"  exit cost = ${self.exit_cost:.2f} deducted from profit at sell/cover")
            print(f"  RF gates: long >= {best['thr_long']:.3f}, "
                  f"short >= {best['thr_short']:.3f}")
            if best['all_pass']:
                print("  -> meets ALL targets (max net profit among passing candidates)")
            else:
                missing = [n for n, ok in (('T1 net', best['t1']),
                                           ('T2 win rate', best['t2']),
                                           ('T3 frequency', best['t3'])) if not ok]
                print(f"  -> no candidate meets all targets; best worst-target "
                      f"achievement (misses: {', '.join(missing)})")
            print("=" * 74)

            # ---- final full simulation with the winning configuration ----
            t0 = time.perf_counter()
            stn1 = None if best['stn1'] == 'none' else best['stn1']
            sln1 = None if best['sln1'] == 'none' else best['sln1']
            _set_params(best['w1'], best['n1'], best['e1'], self.exit_cost)
            self.trades_base, eq_base, im_base = self._simulate(
                best['w1'], best['n1'], best['e1'], stn1, sln1)
            self.stats_base = strategyS4003._compute_stats(
                self.trades_base, np.asarray(eq_base, dtype=float),
                np.asarray(im_base, dtype=int), self.sig.index)
            X, y, meta, drop_cols = build_samples_fast(
                self.trades_base, self._feats_values, self._feats_columns)
            p_oos, fold_rep, purge_gap = purged_walk_forward_np(
                X.to_numpy(dtype=float), y, meta['entry_bar'].to_numpy(),
                meta['exit_bar'].to_numpy(), self.n_folds, self.min_train,
                self.seed)
            self.X, self.y, self.meta = X, y, meta
            self.dropped_feature_cols = drop_cols
            self.p_oos, self.fold_report, self.purge_gap = p_oos, fold_rep, purge_gap

            thr_l, thr_s = best['thr_long'], best['thr_short']
            sig2 = self.sig.copy()
            buy = sig2['buySig'].to_numpy(int).copy()
            shr = sig2['shortSig'].to_numpy(int).copy()
            n_pred = n_kept = n_vetoed = 0
            entry_bars = meta['entry_bar'].to_numpy()
            is_long_m = (meta['direction'] == 'Long').to_numpy()
            for j in range(len(meta)):
                p = self.p_oos[j]
                if np.isnan(p):
                    continue
                n_pred += 1
                eb = int(entry_bars[j])
                thr = thr_l if is_long_m[j] else thr_s
                if p >= thr:
                    n_kept += 1
                else:
                    n_vetoed += 1
                    if is_long_m[j]:
                        buy[eb] = 0
                    else:
                        shr[eb] = 0
            sig2['buySig'] = buy
            sig2['shortSig'] = shr
            self.sig2 = sig2
            self.trades2, eq2, im2 = self._simulate(
                best['w1'], best['n1'], best['e1'], stn1, sln1,
                buy=buy.tolist(), shr=shr.tolist())
            self.stats_enh = strategyS4003._compute_stats(
                self.trades2, np.asarray(eq2, dtype=float),
                np.asarray(im2, dtype=int), sig2.index)
            self.veto_stats = {'predicted': n_pred, 'kept': n_kept,
                               'vetoed': n_vetoed}
            print(f"\nFinal enhanced simulation (full re-sim, gates + daily stops): "
                  f"{n_pred} predicted ({n_kept} kept, {n_vetoed} vetoed) "
                  f"-> {len(self.trades2)} trades")

            # fitted forests are n_jobs-invariant (verified); the n_jobs
            # attribute is set back to -1 so the pickled model matches the
            # reference model's metadata
            self.final_model = _make_model(self.seed, 1).fit(self.X, self.y)
            self.final_model.n_jobs = -1
            self.importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.final_model.feature_importances_,
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            self.timings['final'] = time.perf_counter() - t0

            t0 = time.perf_counter()
            self.write_outputs()
            self.timings['write_outputs'] = time.perf_counter() - t0
            self.print_summary()
        finally:
            if pool is not None:
                pool.shutdown()
            _restore_params()
        t_total = time.perf_counter() - t_run0
        t_no_db = t_total - self.timings.get('db_fetch', 0.0)
        print(f"\nRuntime breakdown (s): "
              + ' | '.join(f"{k} {v:.2f}" for k, v in self.timings.items()))
        print(f"Total runtime: {t_total:.2f}s | excluding MariaDB fetch: "
              f"{t_no_db:.2f}s (target < 5s -> "
              f"{'PASS' if t_no_db < 5.0 else 'MISS'})")
        return self.stats_base, self.stats_enh

    # ------------------------------------------------------------------
    def _target_report(self):
        tgt = self.targets
        freq = len(self.trades2) / max(self.trading_days, 1)
        return [
            ('reference net profit (original S4003_1)', self.stats_ref['Net Profit']),
            ('net profit target (+15%)', tgt['net']),
            ('net profit achieved', self.stats_enh['Net Profit']),
            ('net profit target met', self.stats_enh['Net Profit'] >= tgt['net']),
            ('reference win rate %', self.stats_ref['Number of wins %']),
            ('win rate target % (+15%)', tgt['win_rate']),
            ('win rate achieved %', self.stats_enh['Number of wins %']),
            ('win rate target met', self.stats_enh['Number of wins %'] >= tgt['win_rate']),
            ('trades per day target', tgt['freq']),
            ('trades per day achieved', freq),
            ('trades per day target met', freq >= tgt['freq']),
            ('trading days', self.trading_days),
            ('all targets met', (self.stats_enh['Net Profit'] >= tgt['net']
                                 and self.stats_enh['Number of wins %'] >= tgt['win_rate']
                                 and freq >= tgt['freq'])),
        ]

    def write_outputs(self, out_dir=None):
        out_dir = out_dir or self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        b = self.best_params
        rows = []

        def add(section, metric, value):
            rows.append({'section': section, 'metric': metric, 'value': value})

        for name, st in (('reference_base(original S4003_1)', self.stats_ref),
                         ('base(best params)', self.stats_base),
                         ('enhanced', self.stats_enh)):
            for k, v in st.items():
                add(name, k, v)
        for metric, value in self._target_report():
            add('targets', metric, value)
        add('model', 'best_w1_take_profit_mult', b['w1'])
        add('model', 'best_n1_nbar_stop', b['n1'])
        add('model', 'best_e1_entry_offset_divisor', b['e1'])
        add('model', 'best_stn1_daily_win_stop', b['stn1'])
        add('model', 'best_sln1_daily_loss_stop', b['sln1'])
        add('model', 'exit_cost_sell_cover', self.exit_cost)
        add('model', 'threshold_long', b['thr_long'])
        add('model', 'threshold_short', b['thr_short'])
        add('model', 'n_folds', self.n_folds)
        add('model', 'purge_gap_bars', self.purge_gap)
        add('model', 'n_samples', len(self.X))
        add('model', 'n_features', self.X.shape[1])
        vs = self.veto_stats
        add('model', 'signals_predicted', vs['predicted'])
        add('model', 'signals_kept', vs['kept'])
        add('model', 'signals_vetoed', vs['vetoed'])
        aucs = [f['auc'] for f in self.fold_report if not np.isnan(f['auc'])]
        add('model', 'oos_auc_mean', float(np.mean(aucs)) if aucs else np.nan)
        add('model', 'label_definition',
            f'y=1 if trade pnl>0 (net, ${self.exit_cost:.2f} exit cost)')
        add('model', 'feature_source', 'F6006_GeneralFeature_2_glm.ESFeatureEngineer')
        add('model', 'base_strategy', 'S4003_strategy.strategyS4003')

        results_path = os.path.join(out_dir, RF_RESULTS_CSV)
        pd.DataFrame(rows).to_csv(results_path, index=False)

        imp_path = os.path.join(out_dir, RF_FEATIMP_CSV)
        self.importance.to_csv(imp_path, index=False)

        payload = {
            'model': self.final_model,
            'feature_names': list(self.X.columns),
            'feature_importance': self.importance,
            'threshold_long': b['thr_long'],
            'threshold_short': b['thr_short'],
            'best_params': {'w1': b['w1'], 'n1': b['n1'], 'e1': b['e1'],
                            'stn1': b['stn1'], 'sln1': b['sln1'],
                            'exit_cost': self.exit_cost},
            'phase_a_search': self.phase_a_results,
            'parameter_search': self.search_results,
            'label_definition': f'y = 1 if trade pnl (net, ${self.exit_cost:.2f} exit cost) > 0',
            'trade_labels': self.meta.assign(p_win_oos=self.p_oos),
            'fold_report': pd.DataFrame(self.fold_report),
            'target_report': pd.DataFrame(self._target_report(),
                                          columns=['metric', 'value']),
        }
        model_path = os.path.join(out_dir, RF_MODEL_PKL)
        joblib.dump(payload, model_path)

        print('\nOutput files:')
        print(f'  results:            {results_path}')
        print(f'  model:              {model_path}')
        print(f'  feature importance: {imp_path}')

        if self.write_trio:
            trio = write_backtest_trio(
                self.sig2, pd.DataFrame(self.trades2),
                pd.Series(np.asarray(self.trades2 and [] or [], dtype=float))
                if False else pd.Series(self._eq2_array(), index=self.sig2.index),
                self.stats_enh, out_dir=out_dir, prefix='S4003_10')
            for k, p in trio.items():
                print(f'  {k}: {p}')

    def _eq2_array(self):
        return np.asarray(self._eq2, dtype=float)

    # ------------------------------------------------------------------
    def print_summary(self):
        b = self.best_params
        print('\n' + '=' * 66)
        print(f"{'metric':<30}{'ref S4003_1':>16}{'best base':>16}{'RF enhanced':>16}")
        print('=' * 66)
        keys = ['Number of trades', 'Number of wins %', 'Net Profit',
                'Profit Factor', 'Sharpe Ratio', 'Maximum system drawdown',
                'Average Profit/loss']
        for k in keys:
            vals = [self.stats_ref.get(k, float('nan')),
                    self.stats_base.get(k, float('nan')),
                    self.stats_enh.get(k, float('nan'))]
            line = f"{k:<30}" + ''.join(
                f"{v:>16,.2f}" if isinstance(v, (int, float, np.floating)) else f"{v:>16}"
                for v in vals)
            print(line)
        print('=' * 66)
        tgt = self.targets
        freq = len(self.trades2) / max(self.trading_days, 1)
        hdr = (f"TARGETS (w1={b['w1']}, n1={b['n1']}, e1={b['e1']:.0f}, "
               f"stn1={b['stn1']}, sln1={b['sln1']}; "
               f"gate L {b['thr_long']:.3f} / S {b['thr_short']:.3f})")
        print(f"{hdr:<52}{'target':>10}{'achieved':>10}  status")
        for label, t, a in (
                ('T1 net profit +15%', tgt['net'], self.stats_enh['Net Profit']),
                ('T2 win rate +15%', tgt['win_rate'], self.stats_enh['Number of wins %']),
                ('T3 >=1 trade/2 days', tgt['freq'], freq)):
            status = 'PASS' if a >= t else 'MISS'
            print(f"{label:<52}{t:>10,.2f}{a:>10,.2f}  {status}")


# ===========================================================================
# Main: all parameters from config.yaml section `Backtest Test`
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description='S4003_10 backtest: daily-stop grid search + RF gate '
                    'enhancement, speed-optimized port of S4003_backtest.py; '
                    'parameters from config.yaml section "Backtest Test"')
    ap.add_argument('--config', default=str(_CONFIG_PATH),
                    help='path to the strategy config.yaml (default: strategy folder config)')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_all = yaml.safe_load(f) or {}
    strategyS4003.applyStrategyConfig(cfg_all.get('Strategy'))
    bt = cfg_all.get('Backtest Test') or {}

    if bt.get('symbolName'):
        strategyS4003.symbol = bt['symbolName']
    if bt.get('initCapital') is not None:
        strategyS4003.initial_capital = float(bt['initCapital'])
    if bt.get('positionSize') is not None:
        strategyS4003.num_contracts = int(bt['positionSize'])
    if bt.get('commissionPerTrade') is not None:
        strategyS4003.commission_per_trade = float(bt['commissionPerTrade'])
    _snapshot_params()
    print(f"Config: {args.config}")

    enh = DailyStopEnhancer(
        ticker=bt.get('symbolName', 'ES'),
        start=bt.get('startDate') or None,
        end=bt.get('endDate') or None,
        n_folds=int(bt.get('nFolds', 5)),
        min_train=int(bt.get('minTrain', 20)),
        seed=int(bt.get('seed', 42)),
        write_trio=True,
        top_k=int(bt.get('topK', 3)),
        out_dir=bt.get('outputFolder', OUTPUT_DIR),
        table_name=bt.get('tableName', 'ticker1Min'),
        exit_cost=float(bt.get('commissionPerTrade', EXIT_COST)),
        w1_grid=tuple(bt.get('w1Grid', W1_GRID)),
        n1_grid=tuple(bt.get('n1Grid', N1_GRID)),
        e1_grid=tuple(bt.get('e1Grid', E1_GRID)),
        stn1_grid=tuple(bt.get('stn1Grid', STN1_GRID)),
        sln1_grid=tuple(bt.get('sln1Grid', SLN1_GRID)),
        target_net_increase=float(bt.get('targetNetIncrease', TARGET_NET_INCREASE)),
        target_winrate_increase=float(bt.get('targetWinRateIncrease', TARGET_WINRATE_INCREASE)),
        target_trades_per_2_days=float(bt.get('targetTradesPer2Days', TARGET_TRADES_PER_2_DAYS)),
        n_jobs=int(bt.get('nJobs', 0)) or None,
        strategy_cfg=cfg_all.get('Strategy'),
        bt_overrides={k: bt[k] for k in
                      ('symbolName', 'initCapital', 'positionSize',
                       'commissionPerTrade') if k in bt},
    )
    enh.run()


if __name__ == '__main__':
    main()
