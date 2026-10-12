"""
S4003_backtest.py
=================
TASKS.md T2.2 - backtest for VS_4003_20260827_ES.

Port of the backtest function in
`1 Prelimary Test/S4003_4_randomforest_glm.py` (DailyStopEnhancer: two-phase
(w1, n1, e1) grid + daily win/loss stops (stn1/sln1) + purged walk-forward
Random-Forest gate), calling the strategy class `strategyS4003` from
`2 Strategy/S4003_strategy.py` (T1.1). No imports from `1 Prelimary Test`.

All parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml`
section `Backtest Test`. Data from MariaDB table ticker1Min. Phase A/B
candidate evaluations run in parallel worker processes (`nJobs`, default
min(8, cpu cores); 1 = sequential); each candidate passes its parameters
to the simulator explicitly, so results are identical to sequential runs.

Outputs (config outputFolder, same names/format as S4003_4_randomforest_glm.py):
  S4003_4_randomforest_results.csv            base/enhanced/targets/model sections
  S4003_4_randomforest_model.pkl              final RF + metadata + searches
  S4003_4_randomforest_feature_importance.csv feature importance, all features
  plus the backtest trio S4003_4_<ts>.html / S4003_4_<ts>.csv /
  S4003_4_explore_<ts>.csv.
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml
import mysql.connector
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

from S4003_strategy import strategyS4003, loadStrategyConfig  # noqa: E402
from F6006_GeneralFeature_2_glm import ESFeatureEngineer       # noqa: E402

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

_CONFIG_PATH = _HERE.parents[1] / "config.yaml"

OUTPUT_DIR = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES\backTestResult"
DT_FORMAT = '%d/%m/%Y %H:%M'

RF_RESULTS_CSV = 'S4003_4_randomforest_results.csv'
RF_MODEL_PKL = 'S4003_4_randomforest_model.pkl'
RF_FEATIMP_CSV = 'S4003_4_randomforest_feature_importance.csv'

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
# Config / database (ported from S4003_1_glm / S4003_3_randomforest_glm;
# table name parametrized)
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


def get_db_range(ticker='ES', table='ticker1Min'):
    """(min_datetime1, max_datetime1) for a ticker, as strings."""
    _load_db_env()
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )
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
    _load_db_env()
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )
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
# Random-Forest helpers (verbatim port from S4003_3_randomforest_glm.py)
# ===========================================================================
def _make_model(seed=42, n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5,
        class_weight='balanced', n_jobs=n_jobs, random_state=seed)


def purged_walk_forward(X, y, meta, n_folds=5, min_train=20, seed=42,
                        n_jobs=-1):
    """Expanding past-only walk-forward with label-horizon purge.
    Returns (p_oos aligned with X, fold report list)."""
    n = len(X)
    entry_bars = meta['entry_bar'].to_numpy()
    exit_bars = meta['exit_bar'].to_numpy()
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
        model = _make_model(seed, n_jobs)
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_proba(X.iloc[sl])[:, 1]
        p_oos[sl] = p
        auc = np.nan
        if roc_auc_score is not None and len(np.unique(y[sl])) == 2:
            auc = float(roc_auc_score(y[sl], p))
        report.append({'fold': k, 'n_test': len(f), 'n_train': len(train_idx),
                       'auc': auc})
    return p_oos, report, purge_gap


def build_samples(trades, feats, col_nan_max=0.3):
    """Features at each trade's SIGNAL bar; label = trade won.
    Sparse features dropped; returns (X, y, meta, dropped_cols)."""
    rows, metas = [], []
    for t in trades:
        sb = t['entry_bar'] - 1
        if sb < 0:
            continue
        rows.append(feats.iloc[sb])
        metas.append({'signal_bar': sb, 'entry_bar': t['entry_bar'],
                      'exit_bar': t['exit_bar'], 'direction': t['type'],
                      'entry_dt': t['entry_dt'], 'entry_price': t['entry_price'],
                      'exit_dt': t['exit_dt'], 'exit_price': t['exit_price'],
                      'pnl': t['pnl'], 'exit_reason': t['reason']})
    cand = pd.DataFrame(rows)
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
    P < threshold removes their trades. Returns best candidate dict."""
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
# Backtest trio writer (ported from S4003_2_randomforest_glm.py onto
# strategyS4003)
# ===========================================================================
def write_backtest_trio(sig, trades, equity, stats, out_dir=OUTPUT_DIR,
                        prefix='S4003_4'):
    """Same 3 artifacts S4003_1_glm produces, for the enhanced strategy."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    td = trades.copy()
    pv, ct = strategyS4003.point_value, strategyS4003.num_contracts

    chart_html = strategyS4003._equity_chart_html(equity)
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
# Engine with daily win/loss stops (verbatim port from
# S4003_4_randomforest_glm.py: strategyS4003._simulate loop + daily entry
# blocking)
# ===========================================================================
def simulate_with_daily_stops(sig, stn1=None, sln1=None, contracts=None,
                              w1=None, n1=None, e1=None, exit_cost=None):
    """AFL-faithful S4003 trade loop plus daily trading stops.

    Daily counters are keyed by the EXIT bar's calendar date; new entries on
    a day are blocked once wins_that_day > stn1 (profitable trades) or
    losses_that_day > sln1 (pnl < 0 trades). None disables a stop.
    w1/n1/e1/exit_cost override the strategyS4003 class attributes, so
    candidate evaluation does not mutate shared class state (thread-safe).
    Returns (trades list of dict, equity ndarray, in_market ndarray).
    """
    cls = strategyS4003
    ct = cls.num_contracts if contracts is None else contracts
    profit_mult = cls.profit_mult if w1 is None else float(w1)
    sp = cls.stop_period1 if n1 is None else int(n1)
    atr_div = cls.entry_limit_atr_frac if e1 is None else float(e1)
    comm = cls.commission_per_trade if exit_cost is None else float(exit_cost)
    n = len(sig)
    O = sig['Open'].to_numpy(float)
    H = sig['High'].to_numpy(float)
    L = sig['Low'].to_numpy(float)
    C = sig['Close'].to_numpy(float)
    dts = sig.index
    buy = sig['buySig'].to_numpy(int).copy()
    shr = sig['shortSig'].to_numpy(int).copy()
    atr_off = (sig['atr14'].shift(1) / atr_div).to_numpy(float)
    max3 = sig['max3BarAtr14_1'].to_numpy(float)

    pv = cls.point_value
    limit_mode = (cls.entry_price_mode == 'limit')

    trades = []
    realized = 0.0
    equity = np.full(n, cls.initial_capital)
    in_market = np.zeros(n, dtype=int)
    wins_by_date = {}
    loss_by_date = {}

    isInLong = isInShort = False
    long_fill = long_stop = long_profit = 0.0
    short_fill = short_stop = short_profit = 0.0
    sell_stop_n = sp
    buy_force_cover = short_force_sell = False
    cur = None
    cur_s = None

    def _record_long(i, price, reason):
        nonlocal realized, cur
        pnl = (price - long_fill) * pv * ct - comm
        realized += pnl
        trades.append({
            'type': 'Long', 'entry_dt': dts[cur['entry_bar']],
            'entry_price': long_fill, 'exit_dt': dts[i], 'exit_price': price,
            'entry_bar': cur['entry_bar'], 'exit_bar': i,
            'pnl': pnl, 'pnl_pct': pnl / (long_fill * pv * ct) * 100.0,
            'mae': cur['mae'], 'mfe': cur['mfe'], 'reason': reason,
        })
        d = dts[i].date()
        if pnl > 0:
            wins_by_date[d] = wins_by_date.get(d, 0) + 1
        elif pnl < 0:
            loss_by_date[d] = loss_by_date.get(d, 0) + 1

    def _record_short(i, price, reason):
        nonlocal realized, cur_s
        pnl = (short_fill - price) * pv * ct - comm
        realized += pnl
        trades.append({
            'type': 'Short', 'entry_dt': dts[cur_s['entry_bar']],
            'entry_price': short_fill, 'exit_dt': dts[i], 'exit_price': price,
            'entry_bar': cur_s['entry_bar'], 'exit_bar': i,
            'pnl': pnl, 'pnl_pct': pnl / (short_fill * pv * ct) * 100.0,
            'mae': cur_s['mae'], 'mfe': cur_s['mfe'], 'reason': reason,
        })
        d = dts[i].date()
        if pnl > 0:
            wins_by_date[d] = wins_by_date.get(d, 0) + 1
        elif pnl < 0:
            loss_by_date[d] = loss_by_date.get(d, 0) + 1

    for i in range(n):
        was_long, was_short = isInLong, isInShort
        today = dts[i].date()
        blocked = ((stn1 is not None and wins_by_date.get(today, 0) > stn1)
                   or (sln1 is not None and loss_by_date.get(today, 0) > sln1))

        # ---------- Buy entry ----------
        if buy[i] == 1 and not isInLong and not blocked:
            stop_pt = max3[i]
            if limit_mode and not np.isnan(atr_off[i]):
                long_fill = max(O[i] - atr_off[i], L[i])
            else:
                long_fill = O[i]
            long_stop = O[i] - stop_pt
            long_profit = O[i] + profit_mult * stop_pt
            isInLong = True
            sell_stop_n = sp
            cur = {'entry_bar': i, 'mae': 0.0, 'mfe': 0.0}
            if isInShort:
                buy_force_cover = True

        # ---------- Short entry ----------
        if shr[i] == 1 and not isInShort and not blocked:
            stop_pt = max3[i]
            if limit_mode and not np.isnan(atr_off[i]):
                short_fill = min(O[i] + atr_off[i], H[i])
            else:
                short_fill = O[i]
            short_stop = O[i] + stop_pt
            short_profit = O[i] - profit_mult * stop_pt
            isInShort = True
            cur_s = {'entry_bar': i, 'mae': 0.0, 'mfe': 0.0}
            if isInLong:
                short_force_sell = True

        # ---------- Long exits ----------
        if isInLong and cur is not None:
            cur['mae'] = min(cur['mae'], L[i] - long_fill)
            cur['mfe'] = max(cur['mfe'], H[i] - long_fill)
            exit_price = None
            if (L[i] <= long_stop) or short_force_sell:
                if short_force_sell:
                    exit_price, reason = O[i], 'force'
                    short_force_sell = False
                else:
                    exit_price, reason = long_stop, 'stop'
            elif H[i] >= long_profit:
                exit_price, reason = long_profit, 'profit'
            if exit_price is not None:
                buy[i] = 0
                if cur['entry_bar'] != i:
                    _record_long(i, exit_price, reason)
                isInLong = False
                cur = None
            else:
                j = i - sell_stop_n
                if j >= 0 and buy[j] == 1 and buy[i] != 1:
                    _record_long(i, O[i], 'nbar')
                    isInLong = False
                    cur = None
                else:
                    if j >= 0 and buy[j] == 1 and buy[i] == 1:
                        sell_stop_n += 1

        # ---------- Short exits ----------
        if isInShort and cur_s is not None:
            cur_s['mae'] = min(cur_s['mae'], short_fill - H[i])
            cur_s['mfe'] = max(cur_s['mfe'], short_fill - L[i])
            exit_price = None
            if (H[i] >= short_stop) or buy_force_cover:
                if buy_force_cover:
                    exit_price, reason = O[i], 'force'
                    buy_force_cover = False
                else:
                    exit_price, reason = short_stop, 'stop'
            elif L[i] <= short_profit:
                exit_price, reason = short_profit, 'profit'
            if exit_price is not None:
                shr[i] = 0
                if cur_s['entry_bar'] != i:
                    _record_short(i, exit_price, reason)
                isInShort = False
                cur_s = None
            else:
                j = i - sp
                if (j >= 0 and (shr[j] == 1 or buy_force_cover) and shr[i] != 1):
                    _record_short(i, O[i], 'nbar')
                    isInShort = False
                    cur_s = None

        # ---------- equity / exposure ----------
        unreal = 0.0
        if isInLong:
            unreal = (C[i] - long_fill) * pv * ct
        elif isInShort:
            unreal = (short_fill - C[i]) * pv * ct
        equity[i] = cls.initial_capital + realized + unreal
        in_market[i] = 1 if (was_long or was_short or isInLong or isInShort) else 0

    # close any open position at last bar close (AmiBroker end-of-data)
    if n > 0:
        if isInLong and cur is not None:
            cur['mae'] = min(cur['mae'], L[n - 1] - long_fill)
            cur['mfe'] = max(cur['mfe'], H[n - 1] - long_fill)
            _record_long(n - 1, C[n - 1], 'eod')
        if isInShort and cur_s is not None:
            cur_s['mae'] = min(cur_s['mae'], short_fill - H[n - 1])
            cur_s['mfe'] = max(cur_s['mfe'], short_fill - L[n - 1])
            _record_short(n - 1, C[n - 1], 'eod')
        equity[n - 1] = cls.initial_capital + realized

    return trades, equity, in_market


# ===========================================================================
# Candidate evaluation core + process workers (parallel search)
# ===========================================================================
def _evaluate_ctx(ctx, w1, n1, e1, stn1, sln1, phase):
    """One candidate: (params, daily stops) -> sim -> RF -> gate search.
    ctx carries the shared, read-only run state (sig, features, targets,
    grids). Parameters are passed to the simulator explicitly, so no
    strategyS4003 class attributes are mutated - safe to run in worker
    processes."""
    trades, eq, im = simulate_with_daily_stops(
        ctx['sig'], stn1, sln1, w1=w1, n1=n1, e1=e1,
        exit_cost=ctx['exit_cost'])
    stats = strategyS4003._compute_stats(trades, eq, im, ctx['sig'].index)
    X, y, meta, drop_cols = build_samples(trades, ctx['features'])
    if len(X) == 0:
        return None
    p_oos, fold_rep, purge_gap = purged_walk_forward(
        X, y, meta, ctx['n_folds'], ctx['min_train'], ctx['seed'], n_jobs=1)
    pnl = meta['pnl'].to_numpy(float)
    is_long = (meta['direction'] == 'Long').to_numpy()
    best, n_pred = gate_search_fast(
        pnl, is_long, p_oos, ctx['trading_days'], ctx['targets'],
        ctx['shared_grid'], ctx['direction_grid'])
    return {
        'phase': phase, 'w1': w1, 'n1': n1, 'e1': e1,
        'stn1': stn1 if stn1 is not None else 'none',
        'sln1': sln1 if sln1 is not None else 'none',
        'base_trades': len(trades), 'base_net': stats['Net Profit'],
        'base_win_rate': stats['Number of wins %'],
        'thr_long': best['thr_long'], 'thr_short': best['thr_short'],
        'rf_trades': best['n_trades'], 'rf_net': best['net'],
        'rf_win_rate': best['win_rate'], 'rf_freq': best['freq'],
        't1': best['t1'], 't2': best['t2'], 't3': best['t3'],
        'all_pass': best['all_pass'], 'min_ratio': best['min_ratio'],
        'n_samples': len(X), 'drop_cols': drop_cols,
    }


_WORKER_CTX = {}


def _worker_init(ctx, strategy_cfg, bt_overrides):
    """ProcessPoolExecutor initializer: configure the strategy class and
    stash the shared run state once per worker process."""
    strategyS4003.applyStrategyConfig(strategy_cfg)
    ov = bt_overrides or {}
    if ov.get('symbolName'):
        strategyS4003.symbol = ov['symbolName']
    if ov.get('initCapital') is not None:
        strategyS4003.initial_capital = float(ov['initCapital'])
    if ov.get('positionSize') is not None:
        strategyS4003.num_contracts = int(ov['positionSize'])
    if ov.get('commissionPerTrade') is not None:
        strategyS4003.commission_per_trade = float(ov['commissionPerTrade'])
    _WORKER_CTX.clear()
    _WORKER_CTX.update(ctx)


def _worker_evaluate(job):
    return _evaluate_ctx(_WORKER_CTX, *job)


# ===========================================================================
# Daily-stop enhancer (two-phase search) - ported from
# S4003_4_randomforest_glm.py, parameters injected from config
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

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, db_hi = get_db_range(self.ticker, self.table_name)
        start = self.start or db_lo
        end = self.end or db_hi
        print(f"Loading {self.ticker} 1-min bars [{start} .. {end}] ...")
        self.df = load_data_from_db(self.ticker, start, end, self.table_name)
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.trading_days = len(np.unique(np.asarray(self.df.index.date)))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. "
              f"{self.df.index[-1]} ({self.trading_days} trading days)")

    # ------------------------------------------------------------------
    def _build_ctx(self):
        """Shared read-only run state handed to every candidate evaluation
        (and pickled once per worker process)."""
        return {
            'sig': self.sig, 'features': self.feats.features,
            'targets': self.targets, 'trading_days': self.trading_days,
            'n_folds': self.n_folds, 'min_train': self.min_train,
            'seed': self.seed, 'exit_cost': self.exit_cost,
            'shared_grid': self.shared_grid,
            'direction_grid': self.direction_grid,
        }

    def _evaluate(self, w1, n1, e1, stn1, sln1, phase):
        return _evaluate_ctx(self._ctx, w1, n1, e1, stn1, sln1, phase)

    def _run_jobs(self, jobs):
        """Evaluate candidates in result order; worker processes when
        n_jobs > 1 (each worker loads the shared state once via
        _worker_init; RF fits run single-threaded inside workers)."""
        if self.n_jobs > 1:
            with ProcessPoolExecutor(
                    max_workers=self.n_jobs, initializer=_worker_init,
                    initargs=(self._ctx, self.strategy_cfg,
                              self.bt_overrides)) as ex:
                return list(ex.map(_worker_evaluate, jobs))
        return [_evaluate_ctx(self._ctx, *job) for job in jobs]

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
        self._load()
        print("Computing signals (parameter-independent) ...")
        self.sig = strategyS4003._prepare(self.df)
        print("Computing S0001 feature set (parameter-independent) ...")
        df_lower = self.df.rename(columns={'Open': 'open', 'High': 'high',
                                           'Low': 'low', 'Close': 'close',
                                           'Volume': 'volume'})
        fe = ESFeatureEngineer(df_lower)
        fe.build_all()
        self.feats = fe
        print(f"Features: {self.feats.features.shape[1]} columns")

        # reference base = strategy parameters from config (Strategy section)
        _restore_params()
        ref_trades, ref_eq, ref_im = strategyS4003._simulate(self.sig)
        self.stats_ref = strategyS4003._compute_stats(
            ref_trades, ref_eq, ref_im, self.sig.index)
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
        self._ctx = self._build_ctx()
        results = []
        jobs_a = [(w1, n1, e1, None, None, 'A') for (w1, n1, e1) in combos]
        for r in self._run_jobs(jobs_a):
            if r is not None:
                results.append(r)
                print(self._row(r))

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
        for r in self._run_jobs(jobs_b):
            if r is not None:
                results.append(r)
                print(self._row(r))
        self.search_results = pd.DataFrame(
            [{k: v for k, v in r.items() if k != 'drop_cols'} for r in results])

        best = sorted(results, key=self._rank_key, reverse=True)[0]
        self.best_params = best
        print("\n" + "=" * 74)
        print("BEST PARAMETERS (S4003_4 search):")
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
        stn1 = None if best['stn1'] == 'none' else best['stn1']
        sln1 = None if best['sln1'] == 'none' else best['sln1']
        _set_params(best['w1'], best['n1'], best['e1'], self.exit_cost)
        self.trades_base, self.eq_base, self.im_base = simulate_with_daily_stops(
            self.sig, stn1, sln1)
        self.stats_base = strategyS4003._compute_stats(
            self.trades_base, self.eq_base, self.im_base, self.sig.index)
        X, y, meta, drop_cols = build_samples(self.trades_base, self.feats.features)
        p_oos, fold_rep, purge_gap = purged_walk_forward(
            X, y, meta, self.n_folds, self.min_train, self.seed)
        self.X, self.y, self.meta = X, y, meta
        self.dropped_feature_cols = drop_cols
        self.p_oos, self.fold_report, self.purge_gap = p_oos, fold_rep, purge_gap

        thr_l, thr_s = best['thr_long'], best['thr_short']
        sig2 = self.sig.copy()
        buy = sig2['buySig'].to_numpy(int).copy()
        shr = sig2['shortSig'].to_numpy(int).copy()
        n_pred = n_kept = n_vetoed = 0
        for j in range(len(self.meta)):
            p = self.p_oos[j]
            if np.isnan(p):
                continue
            n_pred += 1
            m = self.meta.iloc[j]
            eb = int(m['entry_bar'])
            thr = thr_l if m['direction'] == 'Long' else thr_s
            if p >= thr:
                n_kept += 1
            else:
                n_vetoed += 1
                if m['direction'] == 'Long':
                    buy[eb] = 0
                else:
                    shr[eb] = 0
        sig2['buySig'] = buy
        sig2['shortSig'] = shr
        self.sig2 = sig2
        self.trades2, self.eq2, self.im2 = simulate_with_daily_stops(
            sig2, stn1, sln1)
        self.stats_enh = strategyS4003._compute_stats(
            self.trades2, self.eq2, self.im2, sig2.index)
        self.veto_stats = {'predicted': n_pred, 'kept': n_kept, 'vetoed': n_vetoed}
        print(f"\nFinal enhanced simulation (full re-sim, gates + daily stops): "
              f"{n_pred} predicted ({n_kept} kept, {n_vetoed} vetoed) "
              f"-> {len(self.trades2)} trades")

        self.final_model = _make_model(self.seed).fit(self.X, self.y)
        self.importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.final_model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        self.write_outputs()
        self.print_summary()
        _restore_params()
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
                pd.Series(self.eq2, index=self.sig2.index), self.stats_enh,
                out_dir=out_dir, prefix='S4003_4')
            for k, p in trio.items():
                print(f'  {k}: {p}')

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
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003 backtest (T2.2): daily-stop grid search + RF gate enhancement')
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
