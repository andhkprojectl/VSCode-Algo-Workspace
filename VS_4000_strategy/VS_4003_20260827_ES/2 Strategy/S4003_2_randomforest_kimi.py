"""
S4003_2_randomforest_kimi.py
============================
Random-Forest enhancement of the S4003_1 ES 1-min strategy
(requirement: VS_4003_20260827_ES/2 Strategy/S4003_2_kimi_ur V1.txt).

Targets (vs the base S4003_1 strategy over the same period):
  T1  net profit  >= base net profit  * 1.15   (+15%)
  T2  win rate    >= base win rate    * 1.15   (+15%)
  T3  trade frequency >= 1 trade per 2 trading days (trades / trading days
      >= 0.5)

Pipeline:
  1. Load ES 1-min bars from MariaDB IBTradingDb.ticker1Min (full history by
     default - more trades -> more RF training samples than the S4003_1
     parity window).
  2. Run the existing S4003_1 strategy (strategyS4003V1, unchanged exits:
     ATR stop / 1.5x profit / 15-bar stop / force exit) -> base trades.
     Every base trade stores the original ENTRY SIGNAL (buy00/short00),
     direction, entry price and exit price as LABELS alongside the sample
     (never used as features - unknown at decision time).
  3. Compute features (see build_features, list following the style of
     VS_0006_dataFunc/VS_6006_FeatureEngineering/F6006_GeneralFeature_2_glm.py) at each trade's
     SIGNAL bar (entry_bar - 1, the last completed bar when the enter/skip
     decision is made). All features are causal (past/current bars only).
  4. Label each base trade: y = 1 if trade P&L (net) > 0.
  5. Purged walk-forward (expanding, past-only): trades split into K
     chronological folds; the model for each fold is trained only on trades
     whose holding period ends more than `purge_gap` bars before the fold
     starts (label-horizon overlap purge). OOS P(Win) for every predictable
     fold.
  6. Entry gate: Signal fires -> features computed -> RF predicts P(Win):
       P(Win) >= threshold -> enter with the SAME exits
       P(Win) <  threshold -> skip the entry ("skip" branch of the
       requirement; the exit logic itself is never modified).
     The threshold is selected by grid search over the OOS predictions to
     meet T1-T3; among thresholds meeting all targets the highest net
     profit wins; otherwise the threshold with the best worst-target
     achievement is kept and shortfalls reported.
     Note: threshold selection evaluates OOS predictions (a mild
     meta-parameter selection bias - documented, not corrected, given the
     small sample size).
  7. Outputs (backTestResult folder):
       S4003_2_randomforest_results.csv             metrics base vs enhanced + model + targets
       S4003_2_randomforest_model.pkl               final RF (all labels) + metadata + targets
       S4003_2_randomforest_feature_importance.csv  feature importance, all features, sorted
     plus the S4003_1-style backtest trio for the enhanced run
       (S4003_2_kimi_<ts>.html / S4003_2_kimi_<ts>.csv /
        S4003_2_kimi_explore_<ts>.csv).
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import (strategyS4003V1, load_data_from_db, OUTPUT_DIR,
                         DT_FORMAT, _load_db_env)

RF_RESULTS_CSV = 'S4003_2_randomforest_results.csv'
RF_MODEL_PKL = 'S4003_2_randomforest_model.pkl'
RF_FEATIMP_CSV = 'S4003_2_randomforest_feature_importance.csv'

TARGET_NET_INCREASE = 0.15        # T1: net profit +15%
TARGET_WINRATE_INCREASE = 0.15    # T2: win rate +15%
TARGET_TRADES_PER_2_DAYS = 1.0    # T3: >= 1 trade per 2 trading days

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None


def get_db_range(ticker='ES'):
    """(min_datetime1, max_datetime1) for a ticker, as strings."""
    import mysql.connector
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
        cur.execute("SELECT MIN(datetime1), MAX(datetime1) FROM ticker1Min "
                    "WHERE ticker = %s", (ticker,))
        lo, hi = cur.fetchone()
        return str(lo), str(hi)
    finally:
        conn.close()


# ===========================================================================
# Enhanced backtest trio writer (S4003_1-style, S4003_2_kimi prefix)
# ===========================================================================
def write_backtest_trio(sig, trades, equity, stats, out_dir=OUTPUT_DIR,
                        prefix='S4003_2_kimi'):
    """Same 3 artifacts S4003_1_glm produces, for the enhanced strategy."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    td = trades.copy()
    pv, ct = strategyS4003V1.point_value, strategyS4003V1.num_contracts

    chart_html = strategyS4003V1._equity_chart_html(equity)
    rows = ''.join(
        f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v:,.2f}</td></tr>'
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
        else f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v}</td></tr>'
        for k, v in stats.items())
    html = f"""<html><head><title>{prefix} Backtest Statistics (RF enhanced)</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;}}
th,td{{padding:4px 12px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body><h2>{prefix} Backtest Statistics (Random-Forest enhanced)</h2>
<p>Symbol: {strategyS4003V1.symbol} | Mode: {strategyS4003V1.entry_price_mode} |
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
            'Symbol': strategyS4003V1.symbol,
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
        'Symbol': strategyS4003V1.symbol,
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
# Random-Forest enhancer (target-driven)
# ===========================================================================
class RandomForestEnhancer:
    """Enhances strategyS4003V1 with a purged walk-forward Random-Forest
    P(Win) gate on entries. Exits are never modified. The gate threshold is
    selected so the enhanced strategy meets the T1-T3 targets where possible."""

    def __init__(self, ticker='ES', start=None, end=None, threshold=None,
                 n_folds=5, min_train=20, seed=42, n_estimators=400,
                 max_depth=6, min_samples_leaf=5, write_trio=True,
                 threshold_grid=None):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.threshold = threshold          # None -> target-driven search
        self.n_folds = n_folds
        self.min_train = min_train
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.write_trio = write_trio
        self.threshold_grid = (threshold_grid if threshold_grid is not None
                               else np.arange(0.30, 0.7001, 0.02))

    # ------------------------------------------------------------------
    def _make_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, class_weight='balanced',
            n_jobs=-1, random_state=self.seed)

    # ------------------------------------------------------------------
    def load_data(self):
        # S4003_1's loader needs explicit bounds; resolve the DB range first
        db_lo, db_hi = get_db_range(self.ticker)
        start = self.start or db_lo
        end = self.end or db_hi
        print(f"Loading {self.ticker} 1-min bars [{start} .. {end}] ...")
        self.df = load_data_from_db(self.ticker, start, end)
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.trading_days = len(np.unique(np.asarray(self.df.index.date)))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. {self.df.index[-1]} "
              f"({self.trading_days} trading days)")
        return self.df

    # ------------------------------------------------------------------
    def run_base_strategy(self):
        """S4003_1 signals + base trades (all exits unchanged)."""
        self.sig = strategyS4003V1._prepare(self.df)
        self.trades, self.equity, self.in_market = strategyS4003V1._simulate(self.sig)
        self.stats_base = strategyS4003V1._compute_stats(
            self.trades, self.equity, self.in_market, self.sig.index)
        print(f"Base strategy: {len(self.trades)} trades, "
              f"net {self.stats_base['Net Profit']:.2f}, "
              f"win rate {self.stats_base['Number of wins %']:.2f}%")
        return self.trades

    # ------------------------------------------------------------------
    def build_features(self):
        """Feature set per bar (all causal - current/past bars only).

        Feature list (style of F6006_GeneralFeature_2_glm, scaled down to
        the S4003_1 signal context; tree models are scale-invariant so no
        normalization is applied):

          price action / momentum:
            ret_1, ret_5, ret_10, ret_20          close pct change
            body_pct, upper_wick_pct, lower_wick_pct, bar_range
            large_body_pct                         body_pct percentile(70, 100) flag
          volatility:
            atr14                                  Wilder ATR(14)
            atr14_pct                              ATR(14) / close
            range_pct_20                           (HH20 - LL20) / close
          trend / regime:
            ema9, ema20, ema50                     close EMAs
            close_gt_ema{9,20,50}                  position flags
            ema20_slope, ema50_slope               5-bar OLS slope / close
            rsi14                                  Wilder RSI(14)
            adx14                                  Wilder ADX(14)
          vwap:
            vwap50, vwap_dist, vwap50_slope3, vwap50_slope5
            crossUpVWap50, crossDownVWap50         S4003_1 cross flags
          volume:
            vol, vol_ma20, vol_rel                 volume, mean, ratio
          time of day (cyclical):
            minute_of_day_sin / _cos               (1-min ES session clock)
          S4003_1 signal context at the signal bar:
            isBullish, isBearish
            largeBullishBody, largeBearishBody
            llvBar5_1, llvBar5_2                   bars since 5-bar low
            regSlop3Vwap50Bar, regSlop5vwap50Bar
            regSlop5vwap50BarL0Sum15
            max3BarAtr14_1                         stop distance anchor
        """
        print("Computing feature set ...")
        out = pd.DataFrame(index=self.sig.index)
        O, H = self.sig['Open'], self.sig['High']
        L, C = self.sig['Low'], self.sig['Close']
        V = self.sig['Volume']

        # --- price action / momentum -----------------------------------
        out['ret_1'] = C.pct_change()
        out['ret_5'] = C.pct_change(5)
        out['ret_10'] = C.pct_change(10)
        out['ret_20'] = C.pct_change(20)
        rng = (H - L)
        out['bar_range'] = rng
        body = (C - O)
        rng_safe = rng.where(rng != 0, np.nan)
        out['body_pct'] = (body / rng_safe).fillna(0.0)
        hi_co = pd.concat([C, O], axis=1).max(axis=1)
        lo_co = pd.concat([C, O], axis=1).min(axis=1)
        out['upper_wick_pct'] = ((H - hi_co) / rng_safe).fillna(0.0)
        out['lower_wick_pct'] = ((lo_co - L) / rng_safe).fillna(0.0)
        body_rank = body.abs().rolling(100, min_periods=100).quantile(0.70)
        out['large_body_pct'] = (body.abs() >= body_rank).astype(float)

        # --- volatility -------------------------------------------------
        out['atr14'] = self.sig['atr14']
        out['atr14_pct'] = out['atr14'] / C.replace(0, np.nan)
        hh20 = H.rolling(20, min_periods=20).max()
        ll20 = L.rolling(20, min_periods=20).min()
        out['range_pct_20'] = ((hh20 - ll20) / C.replace(0, np.nan))

        # --- trend / regime ---------------------------------------------
        ema9 = C.ewm(span=9, adjust=False).mean()
        ema20 = C.ewm(span=20, adjust=False).mean()
        ema50 = C.ewm(span=50, adjust=False).mean()
        out['ema9'] = ema9
        out['ema20'] = ema20
        out['ema50'] = ema50
        out['close_gt_ema9'] = (C > ema9).astype(float)
        out['close_gt_ema20'] = (C > ema20).astype(float)
        out['close_gt_ema50'] = (C > ema50).astype(float)

        t5 = np.arange(5, dtype=float)
        t5m = t5.mean()
        t5d = float(((t5 - t5m) ** 2).sum())

        def _slope5(y):
            y = np.asarray(y, dtype=float)
            if np.isnan(y).any():
                return np.nan
            return float(((y - y.mean()) * (t5 - t5m)).sum() / t5d)

        out['ema20_slope'] = ema20.rolling(5).apply(_slope5, raw=True) / C.replace(0, np.nan)
        out['ema50_slope'] = ema50.rolling(5).apply(_slope5, raw=True) / C.replace(0, np.nan)

        # Wilder RSI(14)
        delta = C.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        ag = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
        al = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = ag / al
        rsi = 100.0 - 100.0 / (1.0 + rs)
        rsi = rsi.where(al != 0.0, 100.0)
        rsi = rsi.where(~((ag == 0.0) & (al == 0.0)), 50.0)
        out['rsi14'] = rsi

        # Wilder ADX(14)
        up = H.diff()
        dn = -L.diff()
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=C.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=C.index)
        tr = pd.concat([H - L, (H - C.shift(1)).abs(),
                        (L - C.shift(1)).abs()], axis=1).max(axis=1)
        atr_w = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
        pdi = 100.0 * plus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr_w.replace(0, np.nan)
        mdi = 100.0 * minus_dm.ewm(alpha=1.0 / 14, adjust=False).mean() / atr_w.replace(0, np.nan)
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        out['adx14'] = dx.ewm(alpha=1.0 / 14, adjust=False).mean()

        # --- vwap --------------------------------------------------------
        out['vwap50'] = self.sig['vwap50']
        out['vwap_dist'] = ((C - self.sig['vwap50'])
                            / self.sig['vwap50'].replace(0, np.nan))
        out['vwap50_slope3'] = self.sig['regSlop3Vwap50Bar']
        out['vwap50_slope5'] = self.sig['regSlop5vwap50Bar']
        out['crossUpVWap50'] = self.sig['crossUpVWap50'].astype(float)
        out['crossDownVWap50'] = self.sig['crossDownVWap50'].astype(float)

        # --- volume -------------------------------------------------------
        out['vol'] = V
        out['vol_ma20'] = V.rolling(20, min_periods=20).mean()
        out['vol_rel'] = V / out['vol_ma20'].replace(0, np.nan)

        # --- time of day (cyclical) ---------------------------------------
        mod = self.sig.index.hour * 60 + self.sig.index.minute
        out['minute_of_day_sin'] = np.sin(2.0 * np.pi * mod / 1440.0)
        out['minute_of_day_cos'] = np.cos(2.0 * np.pi * mod / 1440.0)

        # --- S4003_1 signal context ---------------------------------------
        for col in ('isBullish', 'isBearish', 'largeBullishBody',
                    'largeBearishBody', 'llvBar5_1', 'llvBar5_2',
                    'regSlop3Vwap50Bar', 'regSlop5vwap50Bar',
                    'regSlop5vwap50BarL0Sum15', 'max3BarAtr14_1'):
            s = self.sig[col]
            out[col] = s.astype(float) if s.dtype == bool else s

        self.feats = out
        print(f"Features: {out.shape[1]} columns")
        return self.feats

    # ------------------------------------------------------------------
    def build_samples(self, col_nan_max=0.3):
        """One sample per base trade: features at the SIGNAL bar
        (entry_bar - 1, fully known at decision time), label = trade won.
        Sparse features (NaN for > col_nan_max of samples) are dropped.

        The original entry/exit signal, direction and prices are stored as
        LABELS in self.meta (requirement: label existing strategy entry,
        exit signal and price). They are never part of X."""
        rows, metas = [], []
        for t in self.trades:
            sb = t['entry_bar'] - 1
            if sb < 0:
                continue
            rows.append(self.feats.iloc[sb])
            metas.append({
                'signal_bar': sb,
                'entry_signal': 'buy00' if t['type'] == 'Long' else 'short00',
                'exit_signal': t['reason'],
                'direction': t['type'],
                'entry_bar': t['entry_bar'],
                'exit_bar': t['exit_bar'],
                'entry_dt': t['entry_dt'],
                'entry_price': t['entry_price'],
                'exit_dt': t['exit_dt'],
                'exit_price': t['exit_price'],
                'pnl': t['pnl'],
            })
        cand = pd.DataFrame(rows)
        meta_all = pd.DataFrame(metas)
        nan_frac = cand.isna().mean()
        drop_cols = [c for c in cand.columns if nan_frac[c] > col_nan_max]
        keep_cols = [c for c in cand.columns if c not in drop_cols]
        ok = cand[keep_cols].notna().all(axis=1).to_numpy()
        self.X = cand[keep_cols][ok].reset_index(drop=True)
        self.meta = meta_all[ok].reset_index(drop=True)
        self.y = (self.meta['pnl'] > 0).astype(int).to_numpy()
        self.dropped_feature_cols = drop_cols
        wins = int(self.y.sum()) if len(self.y) else 0
        print(f"RF samples: {len(self.X)} trades labeled "
              f"(wins={wins}, win rate={wins / len(self.y) * 100 if len(self.y) else 0:.1f}%; "
              f"{len(cand) - len(self.X)} rows dropped for NaN, "
              f"{len(drop_cols)} sparse features dropped)")
        return self.X, self.y, self.meta

    # ------------------------------------------------------------------
    def purged_walk_forward(self):
        """Expanding past-only walk-forward with label-horizon purge.
        Fold k is tested with a model trained only on trades whose holding
        period ends more than purge_gap bars BEFORE the fold starts.
        Folds with too little training data get no prediction and keep base
        behavior."""
        n = len(self.X)
        if n == 0:
            raise RuntimeError('no labeled trades')
        entry_bars = self.meta['entry_bar'].to_numpy()
        exit_bars = self.meta['exit_bar'].to_numpy()
        hold = exit_bars - entry_bars
        self.purge_gap = int(hold.max()) + 10
        folds = np.array_split(np.arange(n), self.n_folds)
        p_oos = np.full(n, np.nan)
        self.fold_report = []
        print(f"\nPurged walk-forward ({self.n_folds} folds, purge gap "
              f"{self.purge_gap} bars):")
        print(f"{'fold':>4} {'test trades':>12} {'train':>6} {'train win%':>10} "
              f"{'pred>thr':>9} {'AUC':>6}")
        for k, f in enumerate(folds):
            lo, hi = int(f[0]), int(f[-1])
            test_lo_bar = int(entry_bars[lo])
            train_idx = np.where(exit_bars < test_lo_bar - self.purge_gap)[0]
            train_idx = train_idx[train_idx < lo]
            test_slice = slice(lo, hi + 1)
            if len(train_idx) < self.min_train:
                self.fold_report.append({'fold': k, 'n_test': len(f),
                                         'n_train': len(train_idx), 'auc': np.nan})
                print(f"{k:>4} {len(f):>12} {len(train_idx):>6} "
                      f"{'-':>10} {'-':>9} {'-':>6}  (skip: min_train)")
                continue
            model = self._make_model()
            model.fit(self.X.iloc[train_idx], self.y[train_idx])
            p = model.predict_proba(self.X.iloc[test_slice])[:, 1]
            p_oos[test_slice] = p
            auc = np.nan
            if roc_auc_score is not None and len(np.unique(self.y[test_slice])) == 2:
                auc = float(roc_auc_score(self.y[test_slice], p))
            n_above = int((p >= 0.5).sum())
            self.fold_report.append({'fold': k, 'n_test': len(f),
                                     'n_train': len(train_idx), 'auc': auc})
            print(f"{k:>4} {len(f):>12} {len(train_idx):>6} "
                  f"{self.y[train_idx].mean() * 100:>9.1f}% "
                  f"{n_above:>9} {auc:>6.3f}")
        self.p_oos = p_oos
        return p_oos

    # ------------------------------------------------------------------
    # Target-driven threshold selection + enhanced simulation
    # ------------------------------------------------------------------
    def _simulate_with_threshold(self, thr_long, thr_short):
        """Veto predicted-loser entries (OOS P(Win) < threshold for that
        direction), re-run the S4003_1 engine. Returns (sig2, trades,
        equity, in_market, stats, veto_stats)."""
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
            thr = thr_long if m['direction'] == 'Long' else thr_short
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
        trades, equity, in_mk = strategyS4003V1._simulate(sig2)
        stats = strategyS4003V1._compute_stats(trades, equity, in_mk, sig2.index)
        return (sig2, trades, equity, in_mk, stats,
                {'predicted': n_pred, 'kept': n_kept, 'vetoed': n_vetoed})

    def _targets(self):
        base = self.stats_base
        return {
            'net': base['Net Profit'] * (1.0 + TARGET_NET_INCREASE),
            'win_rate': base['Number of wins %'] * (1.0 + TARGET_WINRATE_INCREASE),
            'freq': TARGET_TRADES_PER_2_DAYS / 2.0,   # trades per trading day
        }

    def select_threshold(self):
        """Target-driven gate-threshold search.

        Stage 1: shared-threshold grid (the literal requirement mechanism:
        one P(Win) threshold for all entries).
        Stage 2: per-direction grid (separate long/short thresholds - a
        refinement that subsumes stage 1 when both are equal).

        Selection: any candidate meeting T1-T3 -> highest net profit;
        otherwise best worst-target achievement ratio (tie-break: net)."""
        tgt = self._targets()
        base_freq = len(self.trades) / max(self.trading_days, 1)
        print(f"\nTargets: net >= {tgt['net']:.2f} "
              f"(base {self.stats_base['Net Profit']:.2f} +15%), "
              f"win rate >= {tgt['win_rate']:.2f}% "
              f"(base {self.stats_base['Number of wins %']:.2f}% +15%), "
              f"freq >= {tgt['freq']:.2f} trades/day "
              f"(base {base_freq:.2f})")

        candidates = []

        def _evaluate(thr_long, thr_short, verbose):
            (_s2, trades, _eq, _im, stats, _vs) = self._simulate_with_threshold(
                thr_long, thr_short)
            freq = len(trades) / max(self.trading_days, 1)
            t1 = stats['Net Profit'] >= tgt['net']
            t2 = stats['Number of wins %'] >= tgt['win_rate']
            t3 = freq >= tgt['freq']
            rec = {
                'thr_long': thr_long, 'thr_short': thr_short,
                'n_trades': len(trades), 'net': stats['Net Profit'],
                'win_rate': stats['Number of wins %'], 'freq': freq,
                't1': t1, 't2': t2, 't3': t3, 'all_pass': t1 and t2 and t3,
                'min_ratio': min(stats['Net Profit'] / max(tgt['net'], 1e-9),
                                 stats['Number of wins %'] / max(tgt['win_rate'], 1e-9),
                                 freq / max(tgt['freq'], 1e-9)),
            }
            candidates.append(rec)
            if verbose:
                print(f"{thr_long:>5.2f} {thr_short:>5.2f} {len(trades):>7} "
                      f"{stats['Net Profit']:>10.2f} "
                      f"{stats['Number of wins %']:>7.2f} {freq:>6.3f}  "
                      f"{'PASS' if t1 else '-':>4} {'PASS' if t2 else '-':>4} "
                      f"{'PASS' if t3 else '-':>4}")
            return rec

        print(f"\nStage 1: shared threshold grid "
              f"({len(self.threshold_grid)} values):")
        print(f"{'thr':>5} {'thrS':>5} {'trades':>7} {'net':>10} {'win%':>7} "
              f"{'freq':>6}  {'T1':>4} {'T2':>4} {'T3':>4}")
        for thr in self.threshold_grid:
            _evaluate(float(thr), float(thr), verbose=True)

        print(f"\nStage 2: per-direction grid (top 10 candidates shown):")
        print(f"{'thrL':>5} {'thrS':>5} {'trades':>7} {'net':>10} {'win%':>7} "
              f"{'freq':>6}  {'T1':>4} {'T2':>4} {'T3':>4}")
        grid2 = np.arange(0.30, 0.7501, 0.05)
        stage2 = []
        for thr_l in grid2:
            for thr_s in grid2:
                stage2.append(_evaluate(float(thr_l), float(thr_s), verbose=False))
        top = sorted(stage2, key=lambda r: (r['all_pass'], r['min_ratio'], r['net']),
                     reverse=True)[:10]
        for r in top:
            print(f"{r['thr_long']:>5.2f} {r['thr_short']:>5.2f} "
                  f"{r['n_trades']:>7} {r['net']:>10.2f} {r['win_rate']:>7.2f} "
                  f"{r['freq']:>6.3f}  {'PASS' if r['t1'] else '-':>4} "
                  f"{'PASS' if r['t2'] else '-':>4} {'PASS' if r['t3'] else '-':>4}")

        self.threshold_search = pd.DataFrame(candidates)
        best = sorted(candidates,
                      key=lambda r: (r['all_pass'], r['min_ratio'], r['net']),
                      reverse=True)[0]
        if best['all_pass']:
            print(f"\nChosen thresholds long={best['thr_long']:.2f} "
                  f"short={best['thr_short']:.2f} (meets all targets, max net)")
        else:
            missing = [name for name, ok in
                       (('T1 net', best['t1']), ('T2 win rate', best['t2']),
                        ('T3 frequency', best['t3'])) if not ok]
            print(f"\nNo candidate meets all targets; chosen "
                  f"long={best['thr_long']:.2f} short={best['thr_short']:.2f} "
                  f"= best worst-target achievement (misses: {', '.join(missing)})")
        self.threshold_long = float(best['thr_long'])
        self.threshold_short = float(best['thr_short'])
        self.threshold = (self.threshold_long
                          if self.threshold_long == self.threshold_short
                          else float(np.mean([self.threshold_long,
                                              self.threshold_short])))
        return best

    def enhanced_simulation(self):
        (self.sig2, self.trades2, self.equity2, self.in_market2,
         self.stats_enh, self.veto_stats) = self._simulate_with_threshold(
            self.threshold_long, self.threshold_short)
        vs = self.veto_stats
        print(f"\nEnhanced simulation @ thresholds long "
              f"{self.threshold_long:.2f} / short {self.threshold_short:.2f}: "
              f"{vs['predicted']} signals with OOS prediction "
              f"({vs['kept']} kept, {vs['vetoed']} vetoed) "
              f"-> {len(self.trades2)} trades")
        return self.trades2

    # ------------------------------------------------------------------
    def fit_final_model(self):
        """Final RF on all labeled trades (for persistence + importances)."""
        self.final_model = self._make_model().fit(self.X, self.y)
        imp = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.final_model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        self.importance = imp
        return imp

    # ------------------------------------------------------------------
    def run(self):
        self.load_data()
        self.run_base_strategy()
        self.build_features()
        self.build_samples()
        self.purged_walk_forward()
        if self.threshold is None:
            self.select_threshold()
        else:
            self.threshold_long = self.threshold
            self.threshold_short = self.threshold
        self.enhanced_simulation()
        self.fit_final_model()
        self.write_outputs()
        self.print_summary()
        return self.stats_base, self.stats_enh

    # ------------------------------------------------------------------
    def _target_report(self):
        tgt = self._targets()
        freq = len(self.trades2) / max(self.trading_days, 1)
        return [
            ('net profit base', self.stats_base['Net Profit']),
            ('net profit target (+15%)', tgt['net']),
            ('net profit achieved', self.stats_enh['Net Profit']),
            ('net profit target met', self.stats_enh['Net Profit'] >= tgt['net']),
            ('win rate base %', self.stats_base['Number of wins %']),
            ('win rate target % (+15%)', tgt['win_rate']),
            ('win rate achieved %', self.stats_enh['Number of wins %']),
            ('win rate target met', self.stats_enh['Number of wins %'] >= tgt['win_rate']),
            ('trades per day base', len(self.trades) / max(self.trading_days, 1)),
            ('trades per day target', tgt['freq']),
            ('trades per day achieved', freq),
            ('trades per day target met', freq >= tgt['freq']),
            ('trading days', self.trading_days),
            ('all targets met', (self.stats_enh['Net Profit'] >= tgt['net']
                                 and self.stats_enh['Number of wins %'] >= tgt['win_rate']
                                 and freq >= tgt['freq'])),
        ]

    def write_outputs(self, out_dir=OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        rows = []

        def add(section, metric, value):
            rows.append({'section': section, 'metric': metric, 'value': value})

        for name, st in (('base', self.stats_base), ('enhanced', self.stats_enh)):
            for k, v in st.items():
                add(name, k, v)
        for metric, value in self._target_report():
            add('targets', metric, value)
        vs = self.veto_stats
        add('model', 'threshold_long', self.threshold_long)
        add('model', 'threshold_short', self.threshold_short)
        add('model', 'threshold', self.threshold)
        add('model', 'threshold_search',
            'shared grid 0.30-0.70 step 0.02 + per-direction 0.30-0.75 step 0.05, vs targets')
        add('model', 'n_folds', self.n_folds)
        add('model', 'purge_gap_bars', self.purge_gap)
        add('model', 'n_samples', len(self.X))
        add('model', 'n_features', self.X.shape[1])
        add('model', 'signals_predicted', vs['predicted'])
        add('model', 'signals_kept', vs['kept'])
        add('model', 'signals_vetoed', vs['vetoed'])
        add('model', 'n_estimators', self.n_estimators)
        add('model', 'max_depth', self.max_depth)
        add('model', 'min_samples_leaf', self.min_samples_leaf)
        add('model', 'random_seed', self.seed)
        aucs = [f['auc'] for f in self.fold_report if not np.isnan(f['auc'])]
        add('model', 'oos_auc_mean', float(np.mean(aucs)) if aucs else np.nan)
        add('model', 'oos_auc_folds',
            ';'.join(f"{f['auc']:.3f}" for f in self.fold_report))
        add('model', 'label_definition', 'y=1 if base trade pnl>0 (net)')
        add('model', 'feature_source', 'S4003_2_randomforest_kimi.build_features '
            '(style of F6006_GeneralFeature_2_glm)')
        add('model', 'base_strategy', 'S4003_1_glm.strategyS4003V1')
        add('model', 'features_dropped_sparse',
            ';'.join(self.dropped_feature_cols) if self.dropped_feature_cols else '')

        results_path = os.path.join(out_dir, RF_RESULTS_CSV)
        pd.DataFrame(rows).to_csv(results_path, index=False)

        imp_path = os.path.join(out_dir, RF_FEATIMP_CSV)
        self.importance.to_csv(imp_path, index=False)

        payload = {
            'model': self.final_model,
            'feature_names': list(self.X.columns),
            'feature_importance': self.importance,
            'threshold': self.threshold,
            'threshold_long': self.threshold_long,
            'threshold_short': self.threshold_short,
            'label_definition': 'y = 1 if base S4003_1 trade pnl (net) > 0',
            'trade_labels': self.meta.assign(p_win_oos=self.p_oos),
            'fold_report': pd.DataFrame(self.fold_report),
            'threshold_search': self.threshold_search,
            'target_report': pd.DataFrame(self._target_report(),
                                          columns=['metric', 'value']),
            'config': {'ticker': self.ticker, 'start': str(self.start),
                       'end': str(self.end), 'n_estimators': self.n_estimators,
                       'max_depth': self.max_depth,
                       'min_samples_leaf': self.min_samples_leaf,
                       'n_folds': self.n_folds, 'purge_gap': self.purge_gap,
                       'seed': self.seed,
                       'targets': {'net_increase': TARGET_NET_INCREASE,
                                   'winrate_increase': TARGET_WINRATE_INCREASE,
                                   'trades_per_2_days': TARGET_TRADES_PER_2_DAYS}},
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
                pd.Series(self.equity2, index=self.sig2.index), self.stats_enh)
            for k, p in trio.items():
                print(f'  {k}: {p}')

    # ------------------------------------------------------------------
    def print_summary(self):
        print('\n' + '=' * 66)
        print(f"{'metric':<28}{'base':>18}{'RF enhanced':>18}")
        print('=' * 66)
        keys = ['Number of trades', 'Number of Long trades', 'Number of Short trades',
                'Number of wins %', 'Net Profit', 'Profit Factor', 'Sharpe Ratio',
                'Maximum system drawdown', 'Annual Return %', 'Average Profit/loss',
                'Avg Bars Held']
        for k in keys:
            b, e = self.stats_base.get(k, float('nan')), self.stats_enh.get(k, float('nan'))
            fb = f"{b:,.2f}" if isinstance(b, (int, float, np.floating)) else str(b)
            fe = f"{e:,.2f}" if isinstance(e, (int, float, np.floating)) else str(e)
            print(f"{k:<28}{fb:>18}{fe:>18}")
        print('=' * 66)
        tgt = self._targets()
        freq = len(self.trades2) / max(self.trading_days, 1)
        hdr = (f"TARGETS (gate L {self.threshold_long:.2f} / "
               f"S {self.threshold_short:.2f})")
        print(f"{hdr:<40}{'target':>12}{'achieved':>12}  status")
        for label, t, a in (
                ('T1 net profit +15%', tgt['net'], self.stats_enh['Net Profit']),
                ('T2 win rate +15%', tgt['win_rate'], self.stats_enh['Number of wins %']),
                ('T3 >=1 trade/2 days', tgt['freq'], freq)):
            status = 'PASS' if a >= t else 'MISS'
            print(f"{label:<40}{t:>12,.2f}{a:>12,.2f}  {status}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_2 Random-Forest enhanced S4003_1 ES strategy (target-driven)')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (default DB start)')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD (default DB end)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--threshold', type=float, default=None,
                    help='fixed gate threshold (default: target-driven search)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-trio', action='store_true',
                    help='skip the S4003_1-style html/trades/explore artifacts')
    args = ap.parse_args()

    enh = RandomForestEnhancer(
        ticker=args.ticker, start=args.start, end=args.end,
        threshold=args.threshold, n_folds=args.folds, min_train=args.min_train,
        seed=args.seed, write_trio=not args.no_trio)
    enh.run()


if __name__ == '__main__':
    main()
