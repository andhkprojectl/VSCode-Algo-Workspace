"""
S4003_8_randomforest_glm.py
===========================
Enhancement of S4003_7_randomforest_glm.py
(requirement: VS_4000_strategy/VS_4003_20260827_ES/1 Prelimary Test/
S4003_8_glm_ur_RF V1.txt).

Changes vs S4003_7_randomforest_glm.py:
  1. Rolling-window walk-forward: walk-forward start = 1 year ago from
     today, overall end = 7 days before today. IIS period 60 days, OOS
     period 7 days, step 1 day (windows overlap: consecutive OOS blocks
     shift by 1 day).
     Window i: IIS = [start + i, start + i + 60d),
     OOS = [start + i + 60d, start + i + 67d), next window i += 1 day.
  2. Per window: the (w1, n1) grid is walked over the 60-day IIS and the
     best configuration (enough trades, win rate, net) trades the whole
     7-day OOS block (block-level day gate, same as S4003_7's OOS-day
     gate applied to every day of the block).
  3. Overlap consolidation: an OOS day covered by several windows uses
     the result of the MOST RECENT window covering it (freshest IIS).
     Per-window aggregates are kept in a separate window_results CSV.
  4. Calibration (grid + gate selection) uses the first IIS window
     (the first 60 days of the analysis period) - no lookahead.
  5. stn1/sln1 daily stops stay unset (disabled). Target: OOS win rate
     >= 65% (reported honestly as PASS/MISS).

Outputs (backTestResult folder, structure similar to S4003_7):
  S4003_8_parameter_analysis.csv    calibration (w1, n1, e1) grid results
  S4003_8_parameter_values.csv      win-rate spread per parameter
  S4003_8_gate_comparison.csv       gate-candidate mini walk-forward results
  S4003_8_window_results.csv        per-window aggregates (every window)
  S4003_8_walkforward_results.csv   consolidated per-OOS-day metrics
  S4003_8_walkforward_trades.csv    consolidated OOS trades (taken blocks)
  S4003_8_walkforward_<ts>.html     program name + gate/window tables +
                                    per-day table + equity chart
"""

import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings('ignore', category=PerformanceWarning)
except ImportError:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_FEATURES_DIR = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'VS_0006_dataFunc', 'VS_6006_FeatureEngineering'))
for _p in (_HERE, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import strategyS4003V1, load_data_from_db, OUTPUT_DIR, DT_FORMAT
from S4003_3_randomforest_glm import (get_db_range, _restore_params, EXIT_COST,
                                      W1_GRID, N1_GRID, E1_GRID)
from S4003_5_randomforest_glm import (simulate_window, window_metrics,
                                      _sharpe_from_eq, _max_dd_from_eq,
                                      MIN_IS_TRADES)

PROGRAM_NAME = 'S4003_8_randomforest_glm.py'
START_DAYS_AGO = 365    # walk-forward start: 1 year ago from today
END_DAYS_AGO = 7        # overall walk-forward end: 7 days before today
IIS_DAYS = 60           # in-sample window per rolling window
OOS_DAYS = 7            # out-of-sample window per rolling window
STEP_DAYS = 1           # window step
WIN_RATE_TARGET = 65.0
GATE_SELECT_DAYS = 5              # mini-OOS days inside calibration for gate selection
GATE_WR_GRID = (0.0, 40.0, 45.0, 50.0, 55.0, 60.0)   # IS win-rate threshold (%)
GATE_MIN_TAKEN = 3                # min taken mini-OOS days for a viable gate
VALUE_PARAMS = ('w1', 'n1', 'e1')
PARAM_GRIDS = {'w1': W1_GRID, 'n1': N1_GRID, 'e1': E1_GRID}


# ===========================================================================
# Main class
# ===========================================================================
class S4003V8RollingWalkForward:
    def __init__(self, ticker='ES', start_days_ago=START_DAYS_AGO,
                 end_days_ago=END_DAYS_AGO):
        self.ticker = ticker
        today = datetime.now().date()
        self.analysis_start = pd.Timestamp(today - timedelta(days=start_days_ago))
        self.overall_end = pd.Timestamp(today - timedelta(days=end_days_ago))
        self.calib_end = self.analysis_start + pd.Timedelta(days=IIS_DAYS)

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, _db_hi = get_db_range(self.ticker)
        print(f"[{PROGRAM_NAME}]")
        print(f"Loading {self.ticker} 1-min bars [{db_lo} .. {self.overall_end}] ...")
        self.df = load_data_from_db(self.ticker, str(db_lo),
                                    str(self.overall_end) + " 23:59:59")
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.sig = strategyS4003V1._prepare(self.df)
        in_period = ((self.sig.index >= self.analysis_start)
                     & (self.sig.index <= self.overall_end + pd.Timedelta(days=1)))
        self.analysis_days = np.unique(np.asarray(self.sig.index[in_period].date))
        self.trading_days = len(self.analysis_days)
        self.grid_days = int(np.sum(self.analysis_days < self.calib_end.date()))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. {self.df.index[-1]}; "
              f"effective analysis period {self.analysis_days[0]} .. "
              f"{self.analysis_days[-1]} ({self.trading_days} trading days); "
              f"calibration (grid) = first IIS window "
              f"{self.analysis_days[0]} .. {self.analysis_days[self.grid_days - 1]} "
              f"({self.grid_days} days), no walk-forward day is inside it")

    # ------------------------------------------------------------------
    def _run_cfg(self, w1, n1, e1):
        trades, eq = simulate_window(self.sig, self.analysis_start,
                                     self.calib_end, w1, n1, e1, None, None)
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
        """Calibration-window (w1, n1, e1) grid = the first IIS window.
        stn1/sln1 daily stops are never set (both stay disabled)."""
        print(f"\nPhase 1: calibration grid on [{self.analysis_start.date()} .. "
              f"{self.calib_end.date()}) over (w1, n1, e1), stn1/sln1 unset "
              f"(daily stops disabled), exit cost ${EXIT_COST:.2f}/trade")
        results = []
        for w1 in W1_GRID:
            for n1 in N1_GRID:
                for e1 in E1_GRID:
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
    def _is_best_cfg(self, day, p1, p2, grid, fv):
        """IS-best (w1/n1/e1) configuration for a mini-OOS day inside the
        calibration window (IS = `day - IIS_DAYS` .. `day - 1`)."""
        is_start = day - timedelta(days=IIS_DAYS)
        is_end = (day - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        best_is, best_pair = None, None
        for v1, v2 in grid:
            kw = dict(fv)
            kw[p1], kw[p2] = v1, v2
            trades, _ = simulate_window(self.sig, is_start, is_end,
                                        kw['w1'], kw['n1'], kw['e1'], None, None)
            m = window_metrics(trades)
            key = (m['trades'] >= MIN_IS_TRADES, m['win_rate'], m['net'])
            if best_is is None or key > best_is:
                best_is, best_pair = key, (v1, v2, m)
        v1, v2, is_m = best_pair
        return v1, v2, is_m

    # ------------------------------------------------------------------
    def select_gate(self):
        """Select the OOS-block gate on calibration data only: mini
        walk-forward over the last GATE_SELECT_DAYS days of the calibration
        window. A block is taken when its IS-selected config has
        IS win rate >= wr_min AND IS net > 0. wr_min = 0.0 is the
        net>0-only baseline."""
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in PARAM_GRIDS[p1] for v2 in PARAM_GRIDS[p2]]
        fv = dict(self.fixed_values)
        calib_days = [d for d in
                      pd.date_range(self.analysis_start,
                                    self.calib_end - timedelta(days=1), freq='D')
                      if d.date() in set(self.analysis_days)]
        mini_days = calib_days[-GATE_SELECT_DAYS:]
        print(f"\nGate selection on calibration mini-OOS days "
              f"{mini_days[0].date()} .. {mini_days[-1].date()} "
              f"(IS {IIS_DAYS}d, never touches real OOS data)")
        rows = []
        for wr_min in GATE_WR_GRID:
            trades_n = wins = 0
            net = 0.0
            taken = 0
            for day in mini_days:
                v1, v2, is_m = self._is_best_cfg(day, p1, p2, grid, fv)
                if is_m['win_rate'] < wr_min or is_m['net'] <= 0:
                    continue
                taken += 1
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                day_end = day.replace(hour=23, minute=59, second=59)
                oos_trades, _ = simulate_window(self.sig, day, day_end,
                                                kw['w1'], kw['n1'], kw['e1'],
                                                None, None)
                m = window_metrics(oos_trades)
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
        viable = gate_tbl[gate_tbl['taken_days'] >= GATE_MIN_TAKEN]
        if len(viable):
            pick = viable.sort_values(['win_rate', 'net', 'taken_days'],
                                      ascending=False).iloc[0]
        else:
            pick = gate_tbl.iloc[0]
            print("  WARNING: no gate met the minimum taken days; "
                  "falling back to net>0-only baseline")
        self.gate_wr = float(pick['gate_wr_min'])
        print(f"-> gate selected: take a block only if IS win rate >= "
              f"{self.gate_wr:.0f}% and IS net > 0")
        return self.gate_wr

    # ------------------------------------------------------------------
    def _build_windows(self):
        """Rolling windows: IIS = [start + i, start + i + IIS_DAYS),
        OOS = [start + i + IIS_DAYS, start + i + IIS_DAYS + OOS_DAYS),
        i += STEP_DAYS, while the OOS block ends on/before overall_end."""
        windows = []
        i = 0
        trading = set(self.analysis_days)
        while True:
            iis_start = self.analysis_start + pd.Timedelta(days=i)
            iis_end = iis_start + pd.Timedelta(days=IIS_DAYS) - pd.Timedelta(seconds=1)
            oos_start = iis_start + pd.Timedelta(days=IIS_DAYS)
            oos_end = oos_start + pd.Timedelta(days=OOS_DAYS) - pd.Timedelta(seconds=1)
            if oos_end > self.overall_end + pd.Timedelta(hours=23, minutes=59, seconds=59):
                break
            oos_days = [d for d in
                        pd.date_range(oos_start, oos_end, freq='D')
                        if d.date() in trading]
            if oos_days:
                windows.append({'window_id': i, 'iis_start': iis_start,
                                'iis_end': iis_end, 'oos_start': oos_start,
                                'oos_end': oos_end, 'oos_days': oos_days})
            i += STEP_DAYS
        return windows

    # ------------------------------------------------------------------
    def walk_forward(self):
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in PARAM_GRIDS[p1] for v2 in PARAM_GRIDS[p2]]
        fv = dict(self.fixed_values)

        windows = self._build_windows()
        print(f"\nPhase 2: rolling walk-forward, {len(windows)} windows "
              f"(IIS {IIS_DAYS}d -> OOS {OOS_DAYS}d, step {STEP_DAYS}d, "
              f"overlapping OOS blocks), gate IS wr >= {self.gate_wr:.0f}% "
              f"& IS net > 0; an OOS day keeps the result of the most "
              f"recent window covering it")

        window_rows = []
        day_results = {}          # oos_date -> latest window's per-day result
        day_trades = {}           # oos_date -> latest window's trades
        day_eq_map = {}           # oos_date -> latest window's equity slice
        taken_days = 0

        print(f"{'win':>5} | {'IIS start':>10} {'OOS start':>10} | "
              f"{p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} {'IS net':>9} | "
              f"{'gate':>5} | {'OOS nTr':>7} {'OOS wr%':>7} {'OOS pnl':>9} "
              f"{'OOS Sharpe':>10} {'OOS MaxDD%':>10}")
        for w in windows:
            iis_start = w['iis_start']
            iis_end = w['iis_end'].replace(hour=23, minute=59, second=59)
            oos_start, oos_end = w['oos_start'], w['oos_end']
            oos_end = oos_end.replace(hour=23, minute=59, second=59)

            best_is, best_pair = None, None
            for v1, v2 in grid:
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                trades, _ = simulate_window(self.sig, iis_start, iis_end,
                                            kw['w1'], kw['n1'], kw['e1'],
                                            None, None)
                m = window_metrics(trades)
                key = (m['trades'] >= MIN_IS_TRADES, m['win_rate'], m['net'])
                if best_is is None or key > best_is:
                    best_is, best_pair = key, (v1, v2, m)
            v1, v2, is_m = best_pair

            gate_pass = (is_m['win_rate'] >= self.gate_wr) and (is_m['net'] > 0)
            o_m = {'trades': 0, 'wins': 0, 'win_rate': 0.0, 'net': 0.0}
            blk_sharpe = blk_maxdd = 0.0
            if gate_pass:
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                oos_trades, oos_eq = simulate_window(self.sig, oos_start, oos_end,
                                                     kw['w1'], kw['n1'], kw['e1'],
                                                     None, None)
                o_m = window_metrics(oos_trades)
                blk_sharpe = _sharpe_from_eq(oos_eq)
                blk_maxdd = _max_dd_from_eq(oos_eq)
                for t in oos_trades:
                    t['oos_day'] = pd.Timestamp(t['entry_dt']).date()
                    t['p1'], t['p2'] = v1, v2

            window_rows.append({
                'window_id': w['window_id'],
                'iis_start': iis_start.date(), 'iis_end': iis_end.date(),
                'oos_start': oos_start.date(), 'oos_end': oos_end.date(),
                p1: v1, p2: v2,
                **{f'fixed_{k}': fv[k] for k in self.fixed_params},
                'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                'is_net': is_m['net'], 'taken': gate_pass,
                'oos_trades': o_m['trades'], 'oos_wins': o_m['wins'],
                'oos_win_rate': o_m['win_rate'], 'oos_pnl': o_m['net'],
                'oos_sharpe': blk_sharpe, 'oos_maxdd': blk_maxdd,
            })
            print(f"{w['window_id']:>5} | {str(iis_start.date()):>10} "
                  f"{str(oos_start.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{'take' if gate_pass else 'skip':>5} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{blk_sharpe:>10.2f} {blk_maxdd:>10.2f}")

            if not gate_pass:
                continue

            # per-day breakdown of this window's OOS block; later windows
            # overwrite earlier ones for shared days (freshest IIS wins)
            for day in w['oos_days']:
                day_ts = pd.Timestamp(day)
                day_tr = [t for t in oos_trades
                          if pd.Timestamp(t['oos_day']) == day_ts]
                day_m = window_metrics(day_tr)
                day_eq_slice = oos_eq[oos_eq.index.date == day_ts.date()]
                day_results[day_ts] = {
                    'oos_date': day.date(), 'window_id': w['window_id'],
                    p1: v1, p2: v2,
                    **{f'fixed_{k}': fv[k] for k in self.fixed_params},
                    'is_start': iis_start.date(),
                    'is_end': (day_ts - timedelta(days=1)).date(),
                    'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                    'is_net': is_m['net'], 'taken': True,
                    'oos_trades': day_m['trades'], 'oos_wins': day_m['wins'],
                    'oos_win_rate': day_m['win_rate'], 'oos_pnl': day_m['net'],
                    'oos_sharpe': _sharpe_from_eq(day_eq_slice),
                    'oos_maxdd': _max_dd_from_eq(day_eq_slice),
                }
                day_trades[day_ts] = day_tr
                day_eq_map[day_ts] = day_eq_slice

        self.window_results = pd.DataFrame(window_rows)
        self.taken_windows = int(self.window_results['taken'].sum())

        # consolidated per-day sequence (chronological, one result per day)
        days_sorted = sorted(day_results)
        cum = 0.0
        eq_parts = []
        for day_ts in days_sorted:
            r = day_results[day_ts]
            r['oos_cum_pnl'] = cum + r['oos_pnl']
            eq = day_eq_map.get(day_ts)
            if eq is not None and len(eq):
                eq_parts.append(eq + cum)
            cum += r['oos_pnl']
        taken_days = sum(1 for d in days_sorted if day_results[d]['taken'])
        self.wf_results = pd.DataFrame([day_results[d] for d in days_sorted])
        self.wf_trades = pd.DataFrame(
            [t for d in days_sorted for t in day_trades.get(d, [])])
        self.taken_days = taken_days
        self.oos_equity = pd.concat(eq_parts) if eq_parts else pd.Series(dtype=float)
        return self.wf_results

    # ------------------------------------------------------------------
    def write_outputs(self, out_dir=OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')

        pa_path = os.path.join(out_dir, 'S4003_8_parameter_analysis.csv')
        self.phase1_results.to_csv(pa_path, index=False)

        pv_path = os.path.join(out_dir, 'S4003_8_parameter_values.csv')
        self.parameter_values.assign(
            chosen=', '.join(self.chosen_params),
            program=PROGRAM_NAME).to_csv(pv_path, index=False)

        gc_path = os.path.join(out_dir, 'S4003_8_gate_comparison.csv')
        self.gate_table.assign(chosen=self.gate_wr,
                               program=PROGRAM_NAME).to_csv(gc_path, index=False)

        wr_path = os.path.join(out_dir, 'S4003_8_window_results.csv')
        self.window_results.to_csv(wr_path, index=False)

        wf_path = os.path.join(out_dir, 'S4003_8_walkforward_results.csv')
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, 'S4003_8_walkforward_trades.csv')
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
        gated_out = len(self.wf_results) - self.taken_days
        eq = self.oos_equity if len(self.oos_equity) else \
            self.wf_results.set_index('oos_date')['oos_cum_pnl']
        chart = strategyS4003V1._equity_chart_html(eq)

        day_rows = ''.join(
            f"<tr><td>{r['oos_date']}</td><td>{r['window_id']}</td>"
            f"<td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td><td>{r['oos_trades']}</td>"
            f"<td>{r['oos_win_rate']:.2f}</td><td>{r['oos_pnl']:.2f}</td>"
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
            f"<td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        win_rows = ''.join(
            f"<tr><td>{r['window_id']}</td><td>{r['iis_start']}</td>"
            f"<td>{r['iis_end']}</td><td>{r['oos_start']}</td>"
            f"<td>{r['oos_end']}</td><td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td>"
            f"<td>{'take' if r['taken'] else 'skip'}</td>"
            f"<td>{r['oos_trades']}</td><td>{r['oos_win_rate']:.2f}</td>"
            f"<td>{r['oos_pnl']:.2f}</td>"
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td></tr>"
            for _, r in self.window_results.iterrows())
        gate_rows = ''.join(
            f"<tr><td>&gt;= {r['gate_wr_min']:.0f}% &amp; IS net &gt; 0</td>"
            f"<td>{r['taken_days']}</td><td>{r['trades']}</td>"
            f"<td>{r['win_rate']:.2f}</td><td>{r['net']:.2f}</td></tr>"
            for _, r in self.gate_table.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        html = f"""<html><head><title>S4003_8 Rolling Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body>
<p style="font-family:monospace;font-size:15px;margin-bottom:4px;"><b>Program: {PROGRAM_NAME}</b></p>
<h2>S4003_8 Rolling Walk-Forward Parameter Test</h2>
<p>Period: {self.analysis_days[0]} .. {self.analysis_days[-1]} |
rolling windows: IIS {IIS_DAYS}d -&gt; OOS {OOS_DAYS}d, step {STEP_DAYS}d
({len(self.window_results)} windows, overlapping OOS blocks) |
stn1/sln1 unset (daily stops disabled) |
exit cost ${EXIT_COST:.2f}/trade<br/>
Parameter selection (grid) window: {self.analysis_days[0]} .. {self.analysis_days[self.grid_days - 1]}
- no walk-forward day is inside it (no lookahead)<br/>
Gate selected on calibration mini-OOS days only (no real OOS data touched)<br/>
An OOS day keeps the result of the most recent window covering it.</p>
<h3>Gate selection (calibration mini walk-forward)</h3>
<table><tr><th>gate</th><th>taken days</th><th>trades</th><th>win%</th>
<th>net</th></tr>
{gate_rows}</table>
<p>Selected gate: take a block only if IS win rate &gt;= {self.gate_wr:.0f}%
and IS net &gt; 0; skipped blocks produce no trades.</p>
<h3>Summary</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
<tr><th>Windows taken / gated out</th><td>{self.taken_windows} / {len(self.window_results) - self.taken_windows}</td></tr>
<tr><th>OOS days traded</th><td>{self.taken_days}</td></tr>
<tr><th>Out-of-sample trades</th><td>{n} ({n / max(self.trading_days, 1):.2f}/day over all OOS days)</td></tr>
<tr><th>Out-of-sample win rate</th><td>{wr:.2f}% (target &gt;= {WIN_RATE_TARGET:.0f}% -> {'PASS' if wr >= WIN_RATE_TARGET else 'MISS'})</td></tr>
<tr><th>Out-of-sample net profit</th><td>{net:,.2f}</td></tr>
<tr><th>Out-of-sample Sharpe ratio</th><td>{overall_sharpe:.2f} (rf 3% annualized)</td></tr>
<tr><th>Out-of-sample max system drawdown</th><td>{overall_maxdd:.2f}%</td></tr>
</table>
<h3>Per window</h3>
<table><tr><th>win</th><th>IIS start</th><th>IIS end</th><th>OOS start</th>
<th>OOS end</th><th>{p1}</th><th>{p2}</th><th>IS trades</th><th>IS win%</th>
<th>IS net</th><th>gate</th><th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th></tr>
{win_rows}</table>
<h3>Per OOS day (consolidated: most recent window per day)</h3>
<p>Per-day Sharpe / max drawdown are computed on that day's bar-level
mark-to-market equity (annualized, 3% risk-free).</p>
<table><tr><th>OOS date</th><th>window</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th><th>cum pnl</th></tr>
{day_rows}</table>
<h3>Equity Curve (cumulative OOS P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f'S4003_8_walkforward_{ts}.html')
        with open(html_path, 'w') as f:
            f.write(html)

        print('\nOutput files:')
        print(f'  parameter analysis: {pa_path}')
        print(f'  parameter values:   {pv_path}')
        print(f'  gate comparison:    {gc_path}')
        print(f'  window results:     {wr_path}')
        print(f'  walkforward results: {wf_path}')
        print(f'  walkforward trades:  {tr_path}')
        print(f'  html:                {html_path}')

        print('\n' + '=' * 60)
        print(f"OUT-OF-SAMPLE AGGREGATE: {n} trades on {self.taken_days} taken "
              f"days ({len(self.window_results) - self.taken_windows} of "
              f"{len(self.window_results)} windows gated out), win rate "
              f"{wr:.2f}% (target >= {WIN_RATE_TARGET:.0f}% -> "
              f"{'PASS' if wr >= WIN_RATE_TARGET else 'MISS'}), "
              f"net {net:,.2f}, freq {n / max(self.trading_days, 1):.2f}/day, "
              f"Sharpe {overall_sharpe:.2f}, max DD {overall_maxdd:.2f}%")
        print('=' * 60)

    # ------------------------------------------------------------------
    def run(self):
        self._load()
        self.grid_search()
        self.select_gate()
        self.walk_forward()
        self.write_outputs()
        _restore_params()


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_8 rolling walk-forward test: IIS 60d -> OOS 7d, '
                    'step 1 day, from 1 year ago to 7 days ago '
                    '(stn1/sln1 daily stops disabled)')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--start-days-ago', type=int, default=START_DAYS_AGO)
    ap.add_argument('--end-days-ago', type=int, default=END_DAYS_AGO)
    args = ap.parse_args()
    S4003V8RollingWalkForward(ticker=args.ticker,
                              start_days_ago=args.start_days_ago,
                              end_days_ago=args.end_days_ago).run()


if __name__ == '__main__':
    main()
