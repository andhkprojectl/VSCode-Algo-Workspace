"""
S4003_4_randomforest_glm.py
===========================
Daily-stop enhancement of the S4003_3 Random-Forest strategy
(requirement: VS_8000_Strategy/VS_4003_20260827_ES/2 Strategy/
S4003_4_glm_ur V1.txt).

Targets (vs the ORIGINAL S4003_1 strategy over the same period):
  T1  net profit  >= reference net profit  * 1.15   (+15%)
  T2  win rate    >= reference win rate    * 1.15   (+15%)
  T3  trade frequency >= 1 trade per 2 trading days

New parameters added on top of S4003_3_randomforest_glm.py:
  stn1  stop trading for the rest of a calendar day once MORE THAN stn1
        profitable trades have CLOSED that day (None = disabled)
  sln1  stop trading for the rest of a calendar day once MORE THAN sln1
        losing trades (pnl < 0) have CLOSED that day (None = disabled)
  Grids searched: stn1, sln1 in {1, 2, 3, None} on top of the S4003_3
  w1/n1/e1 grid. Daily stops change the trade sequence itself (blocking
  entries frees the state machine), so they are evaluated by re-simulation
  (simulate_with_daily_stops), not by trade-list filtering.

Search design (two phases):
  Phase A: the 24 (w1, n1, e1) combinations with daily stops disabled
           (identical to S4003_3) -> top 3 by target rule.
  Phase B: each top-3 config x {stn1, sln1} variants (15 new combos each,
           (None, None) already covered by phase A); per variant: daily-stop
           simulation -> labels -> purged walk-forward RF -> fast gate
           threshold search.
  Winner: any candidate meeting T1-T3 with max net profit, else best
  worst-target achievement; the winner is fully re-simulated (base and
  gated) for the official outputs.

Outputs (backTestResult folder):
  S4003_4_randomforest_results.csv            base/enhanced/targets/model sections
  S4003_4_randomforest_model.pkl              final RF + metadata + searches
  S4003_4_randomforest_feature_importance.csv feature importance, all features
  plus the S4003_1-style backtest trio with prefix S4003_4
  (S4003_4_<ts>.html / S4003_4_<ts>.csv / S4003_4_explore_<ts>.csv).
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_FEATURES_DIR = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'VS_4001_GeneralStrategy'))
for _p in (_HERE, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import (strategyS4003V1, load_data_from_db, OUTPUT_DIR,
                         DT_FORMAT)
from S0001_GeneralFeature_2_glm import ESFeatureEngineer
from S4003_2_randomforest_glm import write_backtest_trio
from S4003_3_randomforest_glm import (
    get_db_range, _set_params, _restore_params, _make_model,
    purged_walk_forward, build_samples, gate_search_fast,
    EXIT_COST, W1_GRID, N1_GRID, E1_GRID, _ORIG_PARAMS,
    TARGET_NET_INCREASE, TARGET_WINRATE_INCREASE, TARGET_TRADES_PER_2_DAYS)

RF_RESULTS_CSV = 'S4003_4_randomforest_results.csv'
RF_MODEL_PKL = 'S4003_4_randomforest_model.pkl'
RF_FEATIMP_CSV = 'S4003_4_randomforest_feature_importance.csv'

STN1_GRID = (1, 2, 3, None)   # daily stop after > stn1 winning trades
SLN1_GRID = (1, 2, 3, None)   # daily stop after > sln1 losing trades


# ===========================================================================
# Engine with daily win/loss stops (S4003_1 loop + daily entry blocking).
# S4003_1_glm.py itself stays untouched.
# ===========================================================================
def simulate_with_daily_stops(sig, stn1=None, sln1=None, contracts=None):
    """AFL-faithful S4003_1 trade loop plus daily trading stops.

    Daily counters are keyed by the EXIT bar's calendar date; new entries on
    a day are blocked once wins_that_day > stn1 (profitable trades) or
    losses_that_day > sln1 (pnl < 0 trades). None disables a stop.
    Returns (trades list of dict, equity ndarray, in_market ndarray).
    """
    cls = strategyS4003V1
    ct = cls.num_contracts if contracts is None else contracts
    n = len(sig)
    O = sig['Open'].to_numpy(float)
    H = sig['High'].to_numpy(float)
    L = sig['Low'].to_numpy(float)
    C = sig['Close'].to_numpy(float)
    dts = sig.index
    buy = sig['buySig'].to_numpy(int).copy()
    shr = sig['shortSig'].to_numpy(int).copy()
    atr_off = (sig['atr14'].shift(1) / cls.entry_limit_atr_frac).to_numpy(float)
    max3 = sig['max3BarAtr14_1'].to_numpy(float)

    pv = cls.point_value
    comm = cls.commission_per_trade
    limit_mode = (cls.entry_price_mode == 'limit')
    sp = cls.stop_period1

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
            long_profit = O[i] + cls.profit_mult * stop_pt
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
            short_profit = O[i] - cls.profit_mult * stop_pt
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
# Daily-stop enhancer (two-phase search)
# ===========================================================================
class DailyStopEnhancer:
    def __init__(self, ticker='ES', start=None, end=None, n_folds=5,
                 min_train=20, seed=42, write_trio=True, top_k=3):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.n_folds = n_folds
        self.min_train = min_train
        self.seed = seed
        self.write_trio = write_trio
        self.top_k = top_k
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
    def _evaluate(self, w1, n1, e1, stn1, sln1, phase):
        """One candidate: (params, daily stops) -> sim -> RF -> gate search."""
        _set_params(w1, n1, e1)
        if stn1 is None and sln1 is None:
            trades, eq, im = strategyS4003V1._simulate(self.sig)
        else:
            trades, eq, im = simulate_with_daily_stops(self.sig, stn1, sln1)
        stats = strategyS4003V1._compute_stats(trades, eq, im, self.sig.index)
        X, y, meta, drop_cols = build_samples(trades, self.feats.features)
        if len(X) == 0:
            return None
        p_oos, fold_rep, purge_gap = purged_walk_forward(
            X, y, meta, self.n_folds, self.min_train, self.seed)
        pnl = meta['pnl'].to_numpy(float)
        is_long = (meta['direction'] == 'Long').to_numpy()
        best, n_pred = gate_search_fast(
            pnl, is_long, p_oos, self.trading_days, self.targets,
            self.shared_grid, self.direction_grid)
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
        self.sig = strategyS4003V1._prepare(self.df)
        print("Computing S0001 feature set (parameter-independent) ...")
        df_lower = self.df.rename(columns={'Open': 'open', 'High': 'high',
                                           'Low': 'low', 'Close': 'close',
                                           'Volume': 'volume'})
        fe = ESFeatureEngineer(df_lower)
        fe.build_all()
        self.feats = fe
        print(f"Features: {self.feats.features.shape[1]} columns")

        # reference base = ORIGINAL S4003_1 parameters and commission
        _restore_params()
        ref_trades, ref_eq, ref_im = strategyS4003V1._simulate(self.sig)
        self.stats_ref = strategyS4003V1._compute_stats(
            ref_trades, ref_eq, ref_im, self.sig.index)
        self.targets = {
            'net': self.stats_ref['Net Profit'] * (1.0 + TARGET_NET_INCREASE),
            'win_rate': self.stats_ref['Number of wins %'] * (1.0 + TARGET_WINRATE_INCREASE),
            'freq': TARGET_TRADES_PER_2_DAYS / 2.0,
        }
        print(f"\nReference (original S4003_1, "
              f"${_ORIG_PARAMS['commission_per_trade']:.2f} commission): "
              f"{len(ref_trades)} trades, net {self.stats_ref['Net Profit']:.2f}, "
              f"win rate {self.stats_ref['Number of wins %']:.2f}%")
        print(f"Targets: net >= {self.targets['net']:.2f} (+15%), "
              f"win rate >= {self.targets['win_rate']:.2f}% (+15%), "
              f"freq >= {self.targets['freq']:.2f} trades/day")

        # ---- Phase A: S4003_3 grid, daily stops disabled ----
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        print(f"\nPhase A: {len(combos)} (w1, n1, e1) combinations, "
              f"daily stops disabled (exit cost ${EXIT_COST:.2f}/trade)")
        print(self.HEAD)
        results = []
        for (w1, n1, e1) in combos:
            r = self._evaluate(w1, n1, e1, None, None, phase='A')
            if r is not None:
                results.append(r)
                print(self._row(r))

        top = sorted(results, key=self._rank_key, reverse=True)[:self.top_k]
        self.phase_a_results = pd.DataFrame(
            [{k: v for k, v in r.items() if k != 'drop_cols'} for r in results])

        # ---- Phase B: daily-stop variants on the top configs ----
        print(f"\nPhase B: daily-stop grid stn1/sln1 in {{1,2,3,None}} "
              f"on the top {len(top)} configurations")
        print(self.HEAD)
        seen = {(r['w1'], r['n1'], r['e1'], r['stn1'], r['sln1']) for r in results}
        for base_cfg in top:
            for stn1 in STN1_GRID:
                for sln1 in SLN1_GRID:
                    key = (base_cfg['w1'], base_cfg['n1'], base_cfg['e1'],
                           stn1 if stn1 is not None else 'none',
                           sln1 if sln1 is not None else 'none')
                    if key in seen:
                        continue
                    r = self._evaluate(base_cfg['w1'], base_cfg['n1'],
                                       base_cfg['e1'], stn1, sln1, phase='B')
                    if r is not None:
                        results.append(r)
                        seen.add(key)
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
        print(f"  exit cost = ${EXIT_COST:.2f} deducted from profit at sell/cover")
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
        _set_params(best['w1'], best['n1'], best['e1'])
        self.trades_base, self.eq_base, self.im_base = simulate_with_daily_stops(
            self.sig, stn1, sln1)
        self.stats_base = strategyS4003V1._compute_stats(
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
        self.stats_enh = strategyS4003V1._compute_stats(
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
        add('model', 'best_stn1_daily_win_stop', b['stn1'])
        add('model', 'best_sln1_daily_loss_stop', b['sln1'])
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
                            'stn1': b['stn1'], 'sln1': b['sln1'],
                            'exit_cost': EXIT_COST},
            'phase_a_search': self.phase_a_results,
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
                prefix='S4003_4')
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
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_4 daily-stop grid search + RF gate enhancement')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (default DB start)')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD (default DB end)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--top-k', type=int, default=3,
                    help='phase-A configurations carried into the daily-stop phase')
    ap.add_argument('--no-trio', action='store_true',
                    help='skip the S4003_1-style html/trades/explore artifacts')
    args = ap.parse_args()

    enh = DailyStopEnhancer(
        ticker=args.ticker, start=args.start, end=args.end,
        n_folds=args.folds, min_train=args.min_train, seed=args.seed,
        write_trio=not args.no_trio, top_k=args.top_k)
    enh.run()


if __name__ == '__main__':
    main()
