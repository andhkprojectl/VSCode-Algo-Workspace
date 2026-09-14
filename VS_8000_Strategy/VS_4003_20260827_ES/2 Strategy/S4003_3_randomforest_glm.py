"""
S4003_3_randomforest_glm.py
===========================
Grid-search enhancement of the S4003_2 Random-Forest strategy
(requirement: VS_8000_Strategy/VS_4003_20260827_ES/2 Strategy/
S4003_3_glm_ur V1.txt).

Targets (vs the ORIGINAL S4003_1 strategy over the same period):
  T1  net profit  >= reference net profit  * 1.15   (+15%)
  T2  win rate    >= reference win rate    * 1.15   (+15%)
  T3  trade frequency >= 1 trade per 2 trading days

Enhancements over S4003_2_randomforest_glm.py:
  C   exit-side commission + slippage: $14.50 deducted from each trade's
      profit at sell/cover - applied to every backtest in this program
      (same amount as the original S4003_1 round-turn commission).
  w1  take-profit multiple: High >= O + w1 * A (long) /
      Low <= O - w1 * A (short), w1 in {1.5, 2.0, 2.5}
  n1  N-bar stop: n1 bars after the entry bar (skip when the current bar
      is another entry signal; long-side anchor still shifts on
      consecutive buy signals), n1 in {10, 15, 20, 25}
  e1  entry limit-offset divisor: long  max(O - ATR14[signal]/e1, Low),
      short min(O + ATR14[signal]/e1, High), e1 in {4, 5}
  where A = max3BarAtr14_1 (highest ATR14 of the previous 3 bars,
  excluding the entry bar) and O = entry-bar Open.

Search design (24 parameter combinations):
  - Signals (_prepare) and the S0001 feature set are parameter-independent
    -> computed once.
  - Per combination: re-run the S4003_1 engine with (w1, n1, e1) and the
    $14.50 exit cost -> base trades -> labels -> purged walk-forward RF ->
    OOS P(Win) per signal -> gate-threshold search (shared + per-direction
    grids) evaluated with fast trade-list arithmetic (net / win rate /
    trade frequency need no equity re-simulation).
  - Winner = any (params, gates) meeting T1-T3 with max net profit, else
    best worst-target achievement; the winner is then fully re-simulated
    (base + gated) for the official outputs.

Outputs (backTestResult folder; file names per requirement):
  S4003_2_randomforest_results.csv            base/enhanced/targets/model sections
  S4003_2_randomforest_model.pkl              final RF + metadata + parameter search
  S4003_2_randomforest_feature_importance.csv feature importance, all features
  plus the S4003_1-style backtest trio with prefix S4003_3
  (S4003_3_<ts>.html / S4003_3_<ts>.csv / S4003_3_explore_<ts>.csv).
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
_FEATURES_DIR = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'VS_0001_GeneralStrategy'))
for _p in (_HERE, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import (strategyS4003V1, load_data_from_db, OUTPUT_DIR,
                         DT_FORMAT, _load_db_env)
from S0001_GeneralFeature_2_glm import ESFeatureEngineer
from S4003_2_randomforest_glm import write_backtest_trio

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

RF_RESULTS_CSV = 'S4003_2_randomforest_results.csv'
RF_MODEL_PKL = 'S4003_2_randomforest_model.pkl'
RF_FEATIMP_CSV = 'S4003_2_randomforest_feature_importance.csv'

TARGET_NET_INCREASE = 0.15
TARGET_WINRATE_INCREASE = 0.15
TARGET_TRADES_PER_2_DAYS = 1.0

EXIT_COST = 14.50                    # sell/cover commission + slippage
W1_GRID = (1.5, 2.0, 2.5)            # take-profit multiple
N1_GRID = (10, 15, 20, 25)           # N-bar stop
E1_GRID = (4.0, 5.0)                 # entry limit-offset divisor

_ORIG_PARAMS = {k: getattr(strategyS4003V1, k) for k in
                ('commission_per_trade', 'profit_mult', 'stop_period1',
                 'entry_limit_atr_frac')}


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


def _set_params(w1, n1, e1, exit_cost=EXIT_COST):
    strategyS4003V1.profit_mult = w1
    strategyS4003V1.stop_period1 = n1
    strategyS4003V1.entry_limit_atr_frac = e1
    strategyS4003V1.commission_per_trade = exit_cost


def _restore_params():
    for k, v in _ORIG_PARAMS.items():
        setattr(strategyS4003V1, k, v)


def _make_model(seed=42):
    return RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=5,
        class_weight='balanced', n_jobs=-1, random_state=seed)


def purged_walk_forward(X, y, meta, n_folds=5, min_train=20, seed=42):
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
        model = _make_model(seed)
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
# Main enhancer
# ===========================================================================
class GridSearchEnhancer:
    def __init__(self, ticker='ES', start=None, end=None, n_folds=5,
                 min_train=20, seed=42, write_trio=True):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.n_folds = n_folds
        self.min_train = min_train
        self.seed = seed
        self.write_trio = write_trio
        self.shared_grid = np.arange(0.30, 0.7001, 0.025)
        self.direction_grid = np.arange(0.30, 0.7501, 0.05)

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, db_hi = get_db_range(self.ticker)
        start = self.start or db_lo
        end = self.end or db_hi
        print(f"Loading {self.ticker} 1-min bars [{start} .. {end}] ...")
        self.df = load_data_from_db(self.ticker, start, end)
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.trading_days = len(np.unique(np.asarray(self.df.index.date)))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. "
              f"{self.df.index[-1]} ({self.trading_days} trading days)")

    # ------------------------------------------------------------------
    def run(self):
        self._load()
        print("Computing signals (parameter-independent) ...")
        self.sig = strategyS4003V1._prepare(self.df)
        print("Computing S0001 feature set (parameter-independent) ...")
        df_lower = self.df.rename(columns={'Open': 'open', 'High': 'high',
                                           'Low': 'low', 'Close': 'close',
                                           'Volume': 'volume'})
        self.feats = ESFeatureEngineer(df_lower)
        self.feats.build_all()
        print(f"Features: {self.feats.features.shape[1]} columns")

        # reference base = ORIGINAL S4003_1 parameters and commission
        _restore_params()
        ref_trades, ref_eq, ref_im = strategyS4003V1._simulate(self.sig)
        self.stats_ref = strategyS4003V1._compute_stats(
            ref_trades, ref_eq, ref_im, self.sig.index)
        tgt = {
            'net': self.stats_ref['Net Profit'] * (1.0 + TARGET_NET_INCREASE),
            'win_rate': self.stats_ref['Number of wins %'] * (1.0 + TARGET_WINRATE_INCREASE),
            'freq': TARGET_TRADES_PER_2_DAYS / 2.0,
        }
        self.targets = tgt
        print(f"\nReference (original S4003_1, ${_ORIG_PARAMS['commission_per_trade']:.2f}"
              f" commission): {len(ref_trades)} trades, "
              f"net {self.stats_ref['Net Profit']:.2f}, "
              f"win rate {self.stats_ref['Number of wins %']:.2f}%")
        print(f"Targets: net >= {tgt['net']:.2f} (+15%), "
              f"win rate >= {tgt['win_rate']:.2f}% (+15%), "
              f"freq >= {tgt['freq']:.2f} trades/day")

        # ---- parameter sweep ----
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        print(f"\nParameter sweep: {len(combos)} combinations "
              f"(exit cost ${EXIT_COST:.2f}/trade at sell/cover)")
        print(f"{'w1':>5} {'n1':>4} {'e1':>4} {'baseTr':>7} {'baseNet':>10} "
              f"{'baseWin%':>9} | {'RFthrL':>6} {'RFthrS':>6} {'RFtr':>5} "
              f"{'RFnet':>10} {'RFwin%':>7} {'freq':>5}  {'T1':>4} {'T2':>4} {'T3':>4}")
        results = []
        for (w1, n1, e1) in combos:
            _set_params(w1, n1, e1)
            trades, _eq, _im = strategyS4003V1._simulate(self.sig)
            stats = strategyS4003V1._compute_stats(trades, _eq, _im, self.sig.index)
            X, y, meta, drop_cols = build_samples(trades, self.feats.features)
            if len(X) == 0:
                continue
            p_oos, fold_rep, purge_gap = purged_walk_forward(
                X, y, meta, self.n_folds, self.min_train, self.seed)
            pnl = meta['pnl'].to_numpy(float)
            is_long = (meta['direction'] == 'Long').to_numpy()
            best, n_pred = gate_search_fast(
                pnl, is_long, p_oos, self.trading_days, tgt,
                self.shared_grid, self.direction_grid)
            results.append({
                'w1': w1, 'n1': n1, 'e1': e1,
                'base_trades': len(trades), 'base_net': stats['Net Profit'],
                'base_win_rate': stats['Number of wins %'],
                'base_freq': len(trades) / self.trading_days,
                'thr_long': best['thr_long'], 'thr_short': best['thr_short'],
                'rf_trades': best['n_trades'], 'rf_net': best['net'],
                'rf_win_rate': best['win_rate'], 'rf_freq': best['freq'],
                't1': best['t1'], 't2': best['t2'], 't3': best['t3'],
                'all_pass': best['all_pass'], 'min_ratio': best['min_ratio'],
                'n_samples': len(X), 'drop_cols': drop_cols,
            })
            print(f"{w1:>5.1f} {n1:>4d} {e1:>4.1f} {len(trades):>7} "
                  f"{stats['Net Profit']:>10.2f} {stats['Number of wins %']:>9.2f} | "
                  f"{best['thr_long']:>6.2f} {best['thr_short']:>6.2f} "
                  f"{best['n_trades']:>5} {best['net']:>10.2f} "
                  f"{best['win_rate']:>7.2f} {best['freq']:>5.2f}  "
                  f"{'PASS' if best['t1'] else '-':>4} "
                  f"{'PASS' if best['t2'] else '-':>4} "
                  f"{'PASS' if best['t3'] else '-':>4}")
        self.search_results = pd.DataFrame(
            [{k: v for k, v in r.items() if k != 'drop_cols'} for r in results])

        best_row = sorted(results, key=lambda r: (r['all_pass'], r['min_ratio'],
                                                  r['rf_net']), reverse=True)[0]
        self.best_params = best_row
        print("\n" + "=" * 70)
        print("BEST PARAMETERS (step 3 grid search):")
        print(f"  w1 = {best_row['w1']}   (take profit at O +/- {best_row['w1']} * A)")
        print(f"  n1 = {best_row['n1']}   (N-bar stop, bars after entry bar)")
        print(f"  e1 = {best_row['e1']}   (entry limit offset ATR14[signal]/{best_row['e1']:.0f})")
        print(f"  exit cost = ${EXIT_COST:.2f} deducted from profit at sell/cover")
        print(f"  RF gates: long >= {best_row['thr_long']:.3f}, "
              f"short >= {best_row['thr_short']:.3f}")
        if best_row['all_pass']:
            print("  -> meets ALL targets (max net profit among passing candidates)")
        else:
            missing = [n for n, ok in (('T1 net', best_row['t1']),
                                       ('T2 win rate', best_row['t2']),
                                       ('T3 frequency', best_row['t3'])) if not ok]
            print(f"  -> no candidate meets all targets; best worst-target "
                  f"achievement (misses: {', '.join(missing)})")
        print("=" * 70)

        # ---- final full simulation with the winning configuration ----
        _set_params(best_row['w1'], best_row['n1'], best_row['e1'])
        self.trades_base, self.eq_base, self.im_base = strategyS4003V1._simulate(self.sig)
        self.stats_base = strategyS4003V1._compute_stats(
            self.trades_base, self.eq_base, self.im_base, self.sig.index)
        X, y, meta, drop_cols = build_samples(self.trades_base, self.feats.features)
        p_oos, fold_rep, purge_gap = purged_walk_forward(
            X, y, meta, self.n_folds, self.min_train, self.seed)
        self.X, self.y, self.meta, self.dropped_feature_cols = X, y, meta, drop_cols
        self.p_oos, self.fold_report, self.purge_gap = p_oos, fold_rep, purge_gap

        thr_l, thr_s = best_row['thr_long'], best_row['thr_short']
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
        self.trades2, self.eq2, self.im2 = strategyS4003V1._simulate(sig2)
        self.stats_enh = strategyS4003V1._compute_stats(
            self.trades2, self.eq2, self.im2, sig2.index)
        self.veto_stats = {'predicted': n_pred, 'kept': n_kept, 'vetoed': n_vetoed}
        print(f"\nFinal enhanced simulation (full re-sim): "
              f"{n_pred} predicted ({n_kept} kept, {n_vetoed} vetoed) "
              f"-> {len(self.trades2)} trades")

        # final model + importance
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

    def write_outputs(self, out_dir=OUTPUT_DIR):
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
        add('model', 'exit_cost_sell_cover', EXIT_COST)
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
        add('model', 'label_definition', 'y=1 if trade pnl>0 (net, $14.50 exit cost)')
        add('model', 'feature_source', 'S0001_GeneralFeature_2_glm.ESFeatureEngineer')
        add('model', 'base_strategy', 'S4003_1_glm.strategyS4003V1')

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
                            'exit_cost': EXIT_COST},
            'parameter_search': self.search_results,
            'label_definition': 'y = 1 if trade pnl (net, $14.50 exit cost) > 0',
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
                prefix='S4003_3')
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
        hdr = (f"TARGETS (w1={b['w1']}, n1={b['n1']}, e1={b['e1']:.0f}; "
               f"gate L {b['thr_long']:.3f} / S {b['thr_short']:.3f})")
        print(f"{hdr:<48}{'target':>10}{'achieved':>10}  status")
        for label, t, a in (
                ('T1 net profit +15%', tgt['net'], self.stats_enh['Net Profit']),
                ('T2 win rate +15%', tgt['win_rate'], self.stats_enh['Number of wins %']),
                ('T3 >=1 trade/2 days', tgt['freq'], freq)):
            status = 'PASS' if a >= t else 'MISS'
            print(f"{label:<48}{t:>10,.2f}{a:>10,.2f}  {status}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_3 parameter grid search + RF gate enhancement')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (default DB start)')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD (default DB end)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-trio', action='store_true',
                    help='skip the S4003_1-style html/trades/explore artifacts')
    args = ap.parse_args()

    enh = GridSearchEnhancer(
        ticker=args.ticker, start=args.start, end=args.end,
        n_folds=args.folds, min_train=args.min_train, seed=args.seed,
        write_trio=not args.no_trio)
    enh.run()


if __name__ == '__main__':
    main()
