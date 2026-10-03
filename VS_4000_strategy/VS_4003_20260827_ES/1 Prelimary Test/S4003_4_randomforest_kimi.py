"""
S4003_4_randomforest_kimi.py
============================
Daily-stop enhancement of the S4003_3 Random-Forest (kimi) strategy
(requirement: VS_4000_strategy/VS_4003_20260827_ES/1 Prelimary Test/
S4003_4_kimi_ur V1.txt).

Targets (vs the ORIGINAL S4003_1 strategy over the same period):
  T1  net profit  >= reference net profit  * 1.15   (+15%)
  T2  win rate    >= reference win rate    * 1.15   (+15%)
  T3  trade frequency >= 1 trade per 2 trading days

Enhancements over S4003_3_randomforest_kimi.py (step 3 of the requirement):
  stn1  daily win stop: stop opening NEW trades for the rest of a day once
        the number of winning trades closed that day is MORE THAN stn1
        (stn1 in {1, 2, 3, off})
  sln1  daily loss stop: stop opening NEW trades for the rest of a day once
        the number of losing trades closed that day is MORE THAN sln1
        (sln1 in {1, 2, 3, off})
  Open positions still exit normally; only new entries are blocked.

Search design (two stages, keeps runtime sane):
  Stage 1: (w1, n1, e1) sweep identical to S4003_3_randomforest_kimi
           (24 combos x purged walk-forward RF x gate-threshold search)
           -> best (w1, n1, e1) + RF gate thresholds -> gated signals.
  Stage 2: (stn1, sln1) sweep (16 combos) on the GATED signals with the
           daily-stop simulator (_simulate_daily_stop, an exact copy of
           the S4003_1 engine plus the daily entry block).
  Winner = any configuration meeting T1-T3 with max net profit, else best
  worst-target achievement.

Outputs (backTestResult folder):
  S4003_2_randomforest_results.csv            ref/base/enhanced/targets/model
  S4003_2_randomforest_model.pkl              final RF + metadata + searches
  S4003_2_randomforest_feature_importance.csv feature importance, all features
  plus the S4003_1-style backtest trio with prefix S4003_4_kimi
  (S4003_4_kimi_<ts>.html / S4003_4_kimi_<ts>.csv /
   S4003_4_kimi_explore_<ts>.csv).
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from S4003_1_glm import (strategyS4003V1, load_data_from_db, OUTPUT_DIR,
                         DT_FORMAT, _load_db_env)
from S4003_2_randomforest_kimi import (RandomForestEnhancer,
                                       write_backtest_trio, get_db_range)
from S4003_3_randomforest_kimi import (
    _set_params, _restore_params, _make_model, purged_walk_forward,
    build_samples, gate_search_fast, _ORIG_PARAMS,
    W1_GRID, N1_GRID, E1_GRID,
    RF_RESULTS_CSV, RF_MODEL_PKL, RF_FEATIMP_CSV,
    TARGET_NET_INCREASE, TARGET_WINRATE_INCREASE, TARGET_TRADES_PER_2_DAYS)

STN1_GRID = (1, 2, 3, None)          # daily win stop (None = off)
SLN1_GRID = (1, 2, 3, None)          # daily loss stop (None = off)


# ===========================================================================
# S4003_1 engine + daily win/loss stop (exact copy of _simulate plus the
# daily entry block; open positions still exit normally)
# ===========================================================================
def _simulate_daily_stop(sig, stn1=None, sln1=None, contracts=None,
                         trade_start_ts=None):
    """strategyS4003V1._simulate with a daily new-entry stop:
    once the number of winning (resp. losing) trades closed on a calendar
    day is MORE THAN stn1 (resp. sln1), all remaining entry signals of that
    day are cancelled. None disables the respective stop."""
    cls = strategyS4003V1
    ct = cls.num_contracts if contracts is None else contracts
    n = len(sig)
    O = sig['Open'].to_numpy(float)
    H = sig['High'].to_numpy(float)
    L = sig['Low'].to_numpy(float)
    C = sig['Close'].to_numpy(float)
    dts = sig.index
    days = np.asarray(dts.date)
    buy = sig['buySig'].to_numpy(int).copy()
    shr = sig['shortSig'].to_numpy(int).copy()
    if trade_start_ts is not None:
        gate = dts < trade_start_ts
        buy[gate] = 0
        shr[gate] = 0
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

    isInLong = isInShort = False
    long_fill = long_stop = long_profit = 0.0
    short_fill = short_stop = short_profit = 0.0
    sell_stop_n = sp
    buy_force_cover = short_force_sell = False
    cur = None      # open long trade dict
    cur_s = None    # open short trade dict

    # --- daily stop state ---
    cur_day = None
    day_wins = 0
    day_losses = 0
    day_stop = False

    def _day_update(pnl):
        nonlocal day_wins, day_losses, day_stop
        if pnl > 0:
            day_wins += 1
        else:
            day_losses += 1
        if (stn1 is not None and day_wins > stn1) or \
           (sln1 is not None and day_losses > sln1):
            day_stop = True

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
        _day_update(pnl)

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
        _day_update(pnl)

    for i in range(n):
        was_long, was_short = isInLong, isInShort

        # ---------- daily stop: new day resets; stopped day cancels entries --
        if days[i] != cur_day:
            cur_day = days[i]
            day_wins = 0
            day_losses = 0
            day_stop = False
        if day_stop:
            buy[i] = 0
            shr[i] = 0

        # ---------- Buy entry ----------
        if buy[i] == 1 and not isInLong:
            stop_pt = max3[i]
            if limit_mode and not np.isnan(atr_off[i]):
                long_fill = max(O[i] - atr_off[i], L[i])  # AB price-bound clamp to Low
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
        if shr[i] == 1 and not isInShort:
            stop_pt = max3[i]
            if limit_mode and not np.isnan(atr_off[i]):
                short_fill = min(O[i] + atr_off[i], H[i])  # AB price-bound clamp to High
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
                buy[i] = 0  # AFL cancels same-bar entries (Buy[i] = 0)
                if cur['entry_bar'] != i:
                    _record_long(i, exit_price, reason)
                isInLong = False
                cur = None
            else:
                j = i - sell_stop_n
                if j >= 0 and buy[j] == 1 and buy[i] != 1:
                    # N-bar stop at Open
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

    # close any open position at last bar close (AmiBroker end-of-data behavior)
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
# Main enhancer
# ===========================================================================
class DailyStopEnhancerKimi:
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

        # ================= Stage 1: (w1, n1, e1) sweep (as S4003_3) =====
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        print(f"\nStage 1 parameter sweep: {len(combos)} combinations")
        print(f"{'w1':>5} {'n1':>4} {'e1':>4} {'baseTr':>7} {'baseNet':>10} "
              f"{'baseWin%':>9} | {'RFthrL':>6} {'RFthrS':>6} {'RFtr':>5} "
              f"{'RFnet':>10} {'RFwin%':>7} {'freq':>5}  {'T1':>4} {'T2':>4} {'T3':>4}")
        results = []
        for (w1, n1, e1) in combos:
            _set_params(w1, n1, e1)
            trades, _eq, _im = strategyS4003V1._simulate(self.sig)
            stats = strategyS4003V1._compute_stats(trades, _eq, _im, self.sig.index)
            X, y, meta, drop_cols = build_samples(trades, self.feats)
            if len(X) == 0:
                continue
            p_oos, fold_rep, purge_gap = purged_walk_forward(
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

        # ---- final RF for the winning (w1, n1, e1) ----
        _set_params(best_row['w1'], best_row['n1'], best_row['e1'])
        self.trades_base, self.eq_base, self.im_base = strategyS4003V1._simulate(self.sig)
        self.stats_base = strategyS4003V1._compute_stats(
            self.trades_base, self.eq_base, self.im_base, self.sig.index)
        X, y, meta, drop_cols = build_samples(self.trades_base, self.feats)
        p_oos, fold_rep, purge_gap = purged_walk_forward(
            X, y, meta, self.n_folds, self.min_train, self.seed)
        self.X, self.y, self.meta, self.dropped_feature_cols = X, y, meta, drop_cols
        self.p_oos, self.fold_report, self.purge_gap = p_oos, fold_rep, purge_gap

        thr_l, thr_s = best_row['thr_long'], best_row['thr_short']
        self.sig2, self.veto_stats = self._apply_gate(thr_l, thr_s)
        print(f"\nStage 1 winner: w1={best_row['w1']}, n1={best_row['n1']}, "
              f"e1={best_row['e1']:.0f}; RF gate L {thr_l:.3f} / S {thr_s:.3f} "
              f"({self.veto_stats['kept']} kept, {self.veto_stats['vetoed']} vetoed)")

        # ================= Stage 2: (stn1, sln1) daily-stop sweep =======
        ds_combos = [(s1, s2) for s1 in STN1_GRID for s2 in SLN1_GRID]
        print(f"\nStage 2 daily-stop sweep: {len(ds_combos)} combinations "
              f"(on RF-gated signals)")
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
        print("BEST PARAMETERS (step 3 grid search):")
        print(f"  w1 = {best_row['w1']}   (take profit at O +/- {best_row['w1']} * A)")
        print(f"  n1 = {best_row['n1']}   (N-bar stop, bars after entry bar)")
        print(f"  e1 = {best_row['e1']}   (entry limit offset ATR14[signal]/{best_row['e1']:.0f})")
        print(f"  stn1 = {best_ds['stn1']}   (daily stop after MORE THAN this many wins)")
        print(f"  sln1 = {best_ds['sln1']}   (daily stop after MORE THAN this many losses)")
        print(f"  RF gates: long >= {thr_l:.3f}, short >= {thr_s:.3f}")
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
                            'stn1': d['stn1'], 'sln1': d['sln1']},
            'parameter_search': self.search_results,
            'daily_stop_search': self.daily_stop_search,
            'label_definition': 'y = 1 if trade pnl (net) > 0',
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
                prefix='S4003_4_kimi')
            for k, p in trio.items():
                print(f'  {k}: {p}')

    # ------------------------------------------------------------------
    def print_summary(self):
        b = self.best_params
        d = self.best_daily_stop
        print('\n' + '=' * 66)
        print(f"{'metric':<30}{'ref S4003_1':>16}{'best base':>16}{'RF+dayStop':>16}")
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
        description='S4003_4 (kimi) parameter grid search + RF gate + daily stop')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD (default DB start)')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD (default DB end)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--min-train', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-trio', action='store_true',
                    help='skip the S4003_1-style html/trades/explore artifacts')
    args = ap.parse_args()

    enh = DailyStopEnhancerKimi(
        ticker=args.ticker, start=args.start, end=args.end,
        n_folds=args.folds, min_train=args.min_train, seed=args.seed,
        write_trio=not args.no_trio)
    enh.run()


if __name__ == '__main__':
    main()
