"""
S4003_4_linearRegression_kimi.py
================================
Linear-regression variant of the S4003_4 (kimi) strategy enhancer
(requirement: VS_8000_Strategy/VS_4003_20260827_ES/2 Strategy/
S4003_4_kimi_ur_OLS V1.txt).

Targets (vs the ORIGINAL S4003_1 strategy over the same period):
  T1  net profit  >= reference net profit  * 1.15   (+15%)
  T2  win rate    >= reference win rate    * 1.15   (+15%)
  T3  trade frequency >= 1 trade per 2 trading days

Change vs S4003_4_randomforest_kimi.py (step 3 of the requirement):
  The Random-Forest P(Win) gate is replaced by an OLS LinearRegression
  trained on the binary label y = 1{trade pnl > 0}. The raw prediction
  clipped to [0, 1] is used as the P(Win) score; the same purged
  walk-forward and gate-threshold search machinery is kept.
  Features are z-score standardized (fit on the training fold only)
  because linear models are scale-sensitive.

Parameter search (step 4 of the requirement) - two stages:
  Stage 1: (w1, n1, e1) sweep (24 combos) x purged walk-forward linear
           model x gate-threshold search -> best (w1, n1, e1) + gates.
  Stage 2: (stn1, sln1) daily-stop sweep (16 combos) on the gated
           signals with the daily-stop simulator.
  Winner = any configuration meeting T1-T3 with max net profit, else
  best worst-target achievement.

Outputs (backTestResult folder):
  S4003_4_linearRegression_results.csv            ref/base/enhanced/targets/model
  S4003_4_linearRegression_model.pkl              final OLS model + metadata + searches
  S4003_4_linearRegression_feature_importance.csv |standardized coefficient| per feature
  plus the S4003_1-style backtest trio with prefix S4003_4_lr_kimi
  (S4003_4_lr_kimi_<ts>.html / S4003_4_lr_kimi_<ts>.csv /
   S4003_4_lr_kimi_explore_<ts>.csv).
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from S4003_1_glm import (strategyS4003V1, load_data_from_db, OUTPUT_DIR,
                         DT_FORMAT, _load_db_env)
from S4003_2_randomforest_kimi import (RandomForestEnhancer,
                                       write_backtest_trio, get_db_range)
from S4003_3_randomforest_kimi import (
    _set_params, _restore_params, build_samples, gate_search_fast,
    _ORIG_PARAMS, W1_GRID, N1_GRID, E1_GRID,
    TARGET_NET_INCREASE, TARGET_WINRATE_INCREASE, TARGET_TRADES_PER_2_DAYS)
from S4003_4_randomforest_kimi import (_simulate_daily_stop,
                                       STN1_GRID, SLN1_GRID)

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

LR_RESULTS_CSV = 'S4003_4_linearRegression_results.csv'
LR_MODEL_PKL = 'S4003_4_linearRegression_model.pkl'
LR_FEATIMP_CSV = 'S4003_4_linearRegression_feature_importance.csv'


# ===========================================================================
# OLS linear-regression P(Win) model (replaces the Random Forest)
# ===========================================================================
class LinearWinModel:
    """OLS on the binary win label with train-fold z-score standardization.
    predict_p returns the clipped [0,1] prediction used as P(Win)."""

    def __init__(self):
        self.model = LinearRegression(n_jobs=-1)
        self.mu_ = None
        self.sd_ = None

    def fit(self, X, y):
        Xv = np.asarray(X, dtype=float)
        self.mu_ = Xv.mean(axis=0)
        self.sd_ = Xv.std(axis=0)
        self.sd_[self.sd_ == 0.0] = 1.0
        self.model.fit((Xv - self.mu_) / self.sd_, y)
        return self

    def predict_p(self, X):
        Xv = (np.asarray(X, dtype=float) - self.mu_) / self.sd_
        return np.clip(self.model.predict(Xv), 0.0, 1.0)

    def coef_importance(self, feature_names):
        """|standardized coefficient| as the linear-model importance."""
        imp = pd.DataFrame({
            'feature': list(feature_names),
            'coefficient': self.model.coef_,
            'importance': np.abs(self.model.coef_),
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return imp


def purged_walk_forward_lr(X, y, meta, n_folds=5, min_train=20, seed=42):
    """Expanding past-only walk-forward with label-horizon purge (OLS).
    Returns (p_oos aligned with X, fold report list, purge gap)."""
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
        model = LinearWinModel().fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_p(X.iloc[sl])
        p_oos[sl] = p
        auc = np.nan
        if roc_auc_score is not None and len(np.unique(y[sl])) == 2:
            auc = float(roc_auc_score(y[sl], p))
        report.append({'fold': k, 'n_test': len(f), 'n_train': len(train_idx),
                       'auc': auc})
    return p_oos, report, purge_gap


# ===========================================================================
# Main enhancer
# ===========================================================================
class DailyStopEnhancerLR:
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
    def _build_kimi_features(self):
        """Reuse the S4003_2 kimi feature builder (parameter-independent:
        it only reads self.sig, which depends on signals, not on w1/n1/e1)."""
        helper = RandomForestEnhancer.__new__(RandomForestEnhancer)
        helper.sig = self.sig
        helper.build_features()
        return helper.feats

    # ------------------------------------------------------------------
    def _apply_gate(self, thr_l, thr_s):
        """Veto signals with OOS P(Win) below the direction threshold."""
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
        return sig2, {'predicted': n_pred, 'kept': n_kept, 'vetoed': n_vetoed}

    # ------------------------------------------------------------------
    def run(self):
        self._load()
        print("Computing signals (parameter-independent) ...")
        self.sig = strategyS4003V1._prepare(self.df)
        print("Computing kimi feature set (parameter-independent) ...")
        self.feats = self._build_kimi_features()

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
        print(f"\nReference (original S4003_1): {len(ref_trades)} trades, "
              f"net {self.stats_ref['Net Profit']:.2f}, "
              f"win rate {self.stats_ref['Number of wins %']:.2f}%")
        print(f"Targets: net >= {tgt['net']:.2f} (+15%), "
              f"win rate >= {tgt['win_rate']:.2f}% (+15%), "
              f"freq >= {tgt['freq']:.2f} trades/day")

        # ================= Stage 1: (w1, n1, e1) sweep ==================
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        print(f"\nStage 1 parameter sweep: {len(combos)} combinations "
              f"(OLS linear regression gate)")
        print(f"{'w1':>5} {'n1':>4} {'e1':>4} {'baseTr':>7} {'baseNet':>10} "
              f"{'baseWin%':>9} | {'LRthrL':>6} {'LRthrS':>6} {'LRtr':>5} "
              f"{'LRnet':>10} {'LRwin%':>7} {'freq':>5}  {'T1':>4} {'T2':>4} {'T3':>4}")
        results = []
        for (w1, n1, e1) in combos:
            _set_params(w1, n1, e1)
            trades, _eq, _im = strategyS4003V1._simulate(self.sig)
            stats = strategyS4003V1._compute_stats(trades, _eq, _im, self.sig.index)
            X, y, meta, drop_cols = build_samples(trades, self.feats)
            if len(X) == 0:
                continue
            p_oos, fold_rep, purge_gap = purged_walk_forward_lr(
                X, y, meta, self.n_folds, self.min_train, self.seed)
            pnl = meta['pnl'].to_numpy(float)
            is_long = (meta['direction'] == 'Long').to_numpy()
            best = gate_search_fast(
                pnl, is_long, p_oos, self.trading_days, tgt,
                self.shared_grid, self.direction_grid)
            results.append({
                'w1': w1, 'n1': n1, 'e1': e1,
                'base_trades': len(trades), 'base_net': stats['Net Profit'],
                'base_win_rate': stats['Number of wins %'],
                'base_freq': len(trades) / self.trading_days,
                'thr_long': best['thr_long'], 'thr_short': best['thr_short'],
                'lr_trades': best['n_trades'], 'lr_net': best['net'],
                'lr_win_rate': best['win_rate'], 'lr_freq': best['freq'],
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
                                                  r['lr_net']), reverse=True)[0]
        self.best_params = best_row

        # ---- final OLS model for the winning (w1, n1, e1) ----
        _set_params(best_row['w1'], best_row['n1'], best_row['e1'])
        self.trades_base, self.eq_base, self.im_base = strategyS4003V1._simulate(self.sig)
        self.stats_base = strategyS4003V1._compute_stats(
            self.trades_base, self.eq_base, self.im_base, self.sig.index)
        X, y, meta, drop_cols = build_samples(self.trades_base, self.feats)
        p_oos, fold_rep, purge_gap = purged_walk_forward_lr(
            X, y, meta, self.n_folds, self.min_train, self.seed)
        self.X, self.y, self.meta, self.dropped_feature_cols = X, y, meta, drop_cols
        self.p_oos, self.fold_report, self.purge_gap = p_oos, fold_rep, purge_gap

        thr_l, thr_s = best_row['thr_long'], best_row['thr_short']
        self.sig2, self.veto_stats = self._apply_gate(thr_l, thr_s)
        print(f"\nStage 1 winner: w1={best_row['w1']}, n1={best_row['n1']}, "
              f"e1={best_row['e1']:.0f}; LR gate L {thr_l:.3f} / S {thr_s:.3f} "
              f"({self.veto_stats['kept']} kept, {self.veto_stats['vetoed']} vetoed)")

        # ================= Stage 2: (stn1, sln1) daily-stop sweep =======
        ds_combos = [(s1, s2) for s1 in STN1_GRID for s2 in SLN1_GRID]
        print(f"\nStage 2 daily-stop sweep: {len(ds_combos)} combinations "
              f"(on LR-gated signals)")
        print(f"{'stn1':>5} {'sln1':>5} {'trades':>7} {'net':>10} {'win%':>7} "
              f"{'freq':>6}  {'T1':>4} {'T2':>4} {'T3':>4}")
        ds_results = []
        for (stn1, sln1) in ds_combos:
            trades, eq, im = _simulate_daily_stop(self.sig2, stn1, sln1)
            stats = strategyS4003V1._compute_stats(trades, eq, im, self.sig2.index)
            freq = len(trades) / max(self.trading_days, 1)
            t1 = stats['Net Profit'] >= tgt['net']
            t2 = stats['Number of wins %'] >= tgt['win_rate']
            t3 = freq >= tgt['freq']
            ds_results.append({
                'stn1': stn1, 'sln1': sln1, 'n_trades': len(trades),
                'net': stats['Net Profit'],
                'win_rate': stats['Number of wins %'], 'freq': freq,
                't1': t1, 't2': t2, 't3': t3, 'all_pass': t1 and t2 and t3,
                'min_ratio': min(stats['Net Profit'] / max(tgt['net'], 1e-9),
                                 stats['Number of wins %'] / max(tgt['win_rate'], 1e-9),
                                 freq / max(tgt['freq'], 1e-9)),
            })
            print(f"{str(stn1):>5} {str(sln1):>5} {len(trades):>7} "
                  f"{stats['Net Profit']:>10.2f} {stats['Number of wins %']:>7.2f} "
                  f"{freq:>6.3f}  {'PASS' if t1 else '-':>4} "
                  f"{'PASS' if t2 else '-':>4} {'PASS' if t3 else '-':>4}")
        self.daily_stop_search = pd.DataFrame(ds_results)
        best_ds = sorted(ds_results, key=lambda r: (r['all_pass'], r['min_ratio'],
                                                    r['net']), reverse=True)[0]
        self.best_daily_stop = best_ds

        print("\n" + "=" * 70)
        print("BEST PARAMETERS (step 4 grid search, OLS linear regression):")
        print(f"  w1 = {best_row['w1']}   (take profit at O +/- {best_row['w1']} * A)")
        print(f"  n1 = {best_row['n1']}   (N-bar stop, bars after entry bar)")
        print(f"  e1 = {best_row['e1']}   (entry limit offset ATR14[signal]/{best_row['e1']:.0f})")
        print(f"  stn1 = {best_ds['stn1']}   (daily stop after MORE THAN this many wins)")
        print(f"  sln1 = {best_ds['sln1']}   (daily stop after MORE THAN this many losses)")
        print(f"  LR gates: long >= {thr_l:.3f}, short >= {thr_s:.3f}")
        if best_ds['all_pass']:
            print("  -> meets ALL targets (max net profit among passing candidates)")
        else:
            missing = [n for n, ok in (('T1 net', best_ds['t1']),
                                       ('T2 win rate', best_ds['t2']),
                                       ('T3 frequency', best_ds['t3'])) if not ok]
            print(f"  -> no candidate meets all targets; best worst-target "
                  f"achievement (misses: {', '.join(missing)})")
        print("=" * 70)

        # ---- final full simulation with the winning configuration ----
        self.trades2, self.eq2, self.im2 = _simulate_daily_stop(
            self.sig2, best_ds['stn1'], best_ds['sln1'])
        self.stats_enh = strategyS4003V1._compute_stats(
            self.trades2, self.eq2, self.im2, self.sig2.index)
        print(f"\nFinal enhanced simulation (daily stop stn1={best_ds['stn1']}, "
              f"sln1={best_ds['sln1']}): {len(self.trades2)} trades, "
              f"net {self.stats_enh['Net Profit']:.2f}, "
              f"win rate {self.stats_enh['Number of wins %']:.2f}%")

        # final model + coefficient importance
        self.final_model = LinearWinModel().fit(self.X, self.y)
        self.importance = self.final_model.coef_importance(self.X.columns)

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
        d = self.best_daily_stop
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
        add('model', 'model_type', 'OLS LinearRegression on y=1{pnl>0}, '
            'train-fold z-score standardized, prediction clipped to [0,1]')
        add('model', 'best_w1_take_profit_mult', b['w1'])
        add('model', 'best_n1_nbar_stop', b['n1'])
        add('model', 'best_e1_entry_offset_divisor', b['e1'])
        add('model', 'best_stn1_daily_win_stop', d['stn1'])
        add('model', 'best_sln1_daily_loss_stop', d['sln1'])
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
        add('model', 'label_definition', 'y=1 if trade pnl>0 (net)')
        add('model', 'feature_source',
            'S4003_2_randomforest_kimi.build_features (kimi feature set)')
        add('model', 'base_strategy', 'S4003_1_glm.strategyS4003V1')
        add('model', 'features_dropped_sparse',
            ';'.join(self.dropped_feature_cols) if self.dropped_feature_cols else '')

        results_path = os.path.join(out_dir, LR_RESULTS_CSV)
        pd.DataFrame(rows).to_csv(results_path, index=False)

        imp_path = os.path.join(out_dir, LR_FEATIMP_CSV)
        self.importance.to_csv(imp_path, index=False)

        payload = {
            'model': self.final_model,
            'feature_names': list(self.X.columns),
            'feature_importance': self.importance,
            'threshold_long': b['thr_long'],
            'threshold_short': b['thr_short'],
            'best_params': {'w1': b['w1'], 'n1': b['n1'], 'e1': b['e1'],
                            'stn1': d['stn1'], 'sln1': d['sln1']},
            'parameter_search': self.search_results,
            'daily_stop_search': self.daily_stop_search,
            'label_definition': 'y = 1 if trade pnl (net) > 0',
            'trade_labels': self.meta.assign(p_win_oos=self.p_oos),
            'fold_report': pd.DataFrame(self.fold_report),
            'target_report': pd.DataFrame(self._target_report(),
                                          columns=['metric', 'value']),
        }
        model_path = os.path.join(out_dir, LR_MODEL_PKL)
        joblib.dump(payload, model_path)

        print('\nOutput files:')
        print(f'  results:            {results_path}')
        print(f'  model:              {model_path}')
        print(f'  feature importance: {imp_path}')

        if self.write_trio:
            trio = write_backtest_trio(
                self.sig2, pd.DataFrame(self.trades2),
                pd.Series(self.eq2, index=self.sig2.index), self.stats_enh,
                prefix='S4003_4_lr_kimi')
            for k, p in trio.items():
                print(f'  {k}: {p}')

    # ------------------------------------------------------------------
    def print_summary(self):
        b = self.best_params
        d = self.best_daily_stop
        print('\n' + '=' * 66)
        print(f"{'metric':<30}{'ref S4003_1':>16}{'best base':>16}{'LR+dayStop':>16}")
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
               f"stn1={d['stn1']}, sln1={d['sln1']}; "
               f"gate L {b['thr_long']:.3f} / S {b['thr_short']:.3f})")
        print(f"{hdr:<58}{'target':>10}{'achieved':>10}  status")
        for label, t, a in (
                ('T1 net profit +15%', tgt['net'], self.stats_enh['Net Profit']),
                ('T2 win rate +15%', tgt['win_rate'], self.stats_enh['Number of wins %']),
                ('T3 >=1 trade/2 days', tgt['freq'], freq)):
            status = 'PASS' if a >= t else 'MISS'
            print(f"{label:<58}{t:>10,.2f}{a:>10,.2f}  {status}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_4 (kimi, OLS linear regression) parameter grid '
                    'search + LR gate + daily stop')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (default DB start)')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD (default DB end)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-trio', action='store_true',
                    help='skip the S4003_1-style html/trades/explore artifacts')
    args = ap.parse_args()

    enh = DailyStopEnhancerLR(
        ticker=args.ticker, start=args.start, end=args.end,
        n_folds=args.folds, min_train=args.min_train, seed=args.seed,
        write_trio=not args.no_trio)
    enh.run()


if __name__ == '__main__':
    main()
