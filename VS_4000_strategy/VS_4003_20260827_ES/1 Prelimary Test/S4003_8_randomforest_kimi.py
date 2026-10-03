"""
S4003_8_randomforest_kimi.py
============================
Walk-forward variant of S4003_7_randomforest_glm.py
(requirement: VS_4000_strategy/VS_4003_20260827_ES/1 Prelimary Test/
S4003_8_kimi_ur_RF V1.txt).

Changes vs S4003_7_randomforest_glm.py (walk-forward test only):
  1. Walk-forward period: from 1 year ago (today) to 7 days before
     today. The first 60 days are the calibration (grid) window - the
     same span as the first in-sample window - so no walk-forward
     out-of-sample day is inside it.
  2. Rolling windows: in-sample (IIS) 60 days -> out-of-sample (OOS)
     7 days, stepped forward 1 day at a time. Example: IIS =
     [1 year ago, 1 year ago + 60 days), OOS = the 7 days right after;
     then both roll forward by 1 day. Consecutive OOS windows therefore
     overlap by 6 days (reported per window; aggregates say so).
  3. Everything else (parameter grid on the calibration window, gate
     selection on calibration mini-OOS days only, stn1/sln1 daily stops
     disabled, win-rate target 65%) is unchanged from S4003_7.

Outputs (backTestResult folder, structure similar to S4003_7):
  S4003_8_parameter_analysis.csv   calibration (w1, n1, e1) grid results
  S4003_8_parameter_values.csv     win-rate spread per parameter
  S4003_8_gate_comparison.csv      gate-candidate mini walk-forward results
  S4003_8_walkforward_results.csv  per-OOS-window metrics incl. Sharpe/MaxDD
  S4003_8_walkforward_trades.csv   every out-of-sample trade (taken windows)
  S4003_8_walkforward_<ts>.html    program name on top + gate info +
                                   per-OOS-window table + equity chart
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

PROGRAM_NAME = 'S4003_8_randomforest_kimi.py'
TODAY = pd.Timestamp.today().normalize()
ANALYSIS_START = TODAY - pd.DateOffset(years=1)                 # 1 year ago
ANALYSIS_END = (TODAY - pd.Timedelta(days=7)).replace(
    hour=23, minute=59, second=59)                              # 7 days before today
WIN_RATE_TARGET = 65.0
CALIB_DAYS = 60   # calibration window = span of the first in-sample window
GRID_END = ANALYSIS_START + pd.Timedelta(days=CALIB_DAYS)
IS_DAYS = 60                    # in-sample (IIS) window length
OOS_DAYS = 7                    # out-of-sample (OOS) window length
STEP_DAYS = 1                   # rolling step
GATE_SELECT_DAYS = 5              # mini-OOS days inside calibration for gate selection
GATE_WR_GRID = (0.0, 40.0, 45.0, 50.0, 55.0, 60.0)   # IS win-rate threshold (%)
GATE_MIN_TAKEN = 3                # min taken mini-OOS days for a viable gate
VALUE_PARAMS = ('w1', 'n1', 'e1')
PARAM_GRIDS = {'w1': W1_GRID, 'n1': N1_GRID, 'e1': E1_GRID}


# ===========================================================================
# Main class
# ===========================================================================
class S4003V8WalkForward:
    def __init__(self, ticker='ES'):
        self.ticker = ticker

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, _db_hi = get_db_range(self.ticker)
        print(f"[{PROGRAM_NAME}]")
        print(f"Loading {self.ticker} 1-min bars [{db_lo} .. {ANALYSIS_END}] ...")
        self.df = load_data_from_db(self.ticker, str(db_lo), str(ANALYSIS_END))
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.sig = strategyS4003V1._prepare(self.df)
        self.data_end = self.df.index[-1]
        self.effective_end = min(ANALYSIS_END, self.data_end)
        in_period = (self.sig.index >= ANALYSIS_START) & \
                    (self.sig.index <= self.effective_end)
        self.analysis_days = np.unique(np.asarray(self.sig.index[in_period].date))
        self.trading_days = len(self.analysis_days)
        self.grid_days = int(np.sum(self.analysis_days < GRID_END.date()))
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. {self.df.index[-1]}; "
              f"walk-forward period {ANALYSIS_START.date()} .. "
              f"{ANALYSIS_END.date()} (data ends {self.data_end.date()}); "
              f"calibration (grid) window {self.analysis_days[0]} .. "
              f"{self.analysis_days[self.grid_days - 1]} ({self.grid_days} days), "
              f"no walk-forward OOS window is inside it")

    # ------------------------------------------------------------------
    def _run_cfg(self, w1, n1, e1):
        trades, eq = simulate_window(self.sig, ANALYSIS_START, GRID_END,
                                     w1, n1, e1, None, None)
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
        data before the first walk-forward out-of-sample window
        (no lookahead)."""
        print(f"\nPhase 1: calibration grid on [{ANALYSIS_START.date()} .. "
              f"{GRID_END.date()}) over (w1, n1, e1), stn1/sln1 unset (daily "
              f"stops disabled), exit cost ${EXIT_COST:.2f}/trade")
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
        """IS-best (w1/n1/e1) configuration for an OOS window starting at
        `day`: full grid walk over the rolling IS window of IS_DAYS days
        ending the day before; selection key is (enough trades, win rate,
        net). Returns (v1, v2, is_metrics)."""
        is_start = day - timedelta(days=IS_DAYS)
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
        """Select the OOS-window gate on calibration data only: mini
        walk-forward over the last GATE_SELECT_DAYS trading days of the
        calibration window (1-day mini-OOS with rolling IS_DAYS in-sample,
        as in S4003_7). A window is taken when its IS-selected config has
        IS win rate >= wr_min AND IS net > 0. wr_min = 0.0 is the
        net>0-only baseline."""
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in PARAM_GRIDS[p1] for v2 in PARAM_GRIDS[p2]]
        fv = dict(self.fixed_values)
        calib_days = [d for d in
                      pd.date_range(ANALYSIS_START, GRID_END - timedelta(days=1),
                                    freq='D')
                      if d.date() in set(self.analysis_days)]
        mini_days = calib_days[-GATE_SELECT_DAYS:]
        print(f"\nGate selection on calibration mini-OOS days "
              f"{mini_days[0].date()} .. {mini_days[-1].date()} "
              f"(rolling IS {IS_DAYS}d, never touches real OOS data)")
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
        print(f"-> gate selected: take a window only if IS win rate >= "
              f"{self.gate_wr:.0f}% and IS net > 0 "
              f"(maximizes calibration mini-OOS win rate)")
        return self.gate_wr

    # ------------------------------------------------------------------
    def walk_forward(self):
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in PARAM_GRIDS[p1] for v2 in PARAM_GRIDS[p2]]
        fv = dict(self.fixed_values)

        # OOS window starts on trading days, stepped 1 calendar day; the
        # full OOS_DAYS window must end by the walk-forward end date.
        last_start = self.effective_end - timedelta(days=OOS_DAYS - 1)
        oos_starts = [d for d in pd.date_range(GRID_END, last_start, freq='D')
                      if d.date() in set(self.analysis_days)]
        print(f"\nPhase 2: rolling walk-forward from {oos_starts[0].date()} "
              f"(after the calibration window), in-sample {IS_DAYS} days -> "
              f"out-of-sample {OOS_DAYS} days, step {STEP_DAYS} day, "
              f"{len(grid)} combinations per window, "
              f"{len(oos_starts)} overlapping OOS windows, gate IS wr >= "
              f"{self.gate_wr:.0f}% & IS net > 0")

        rows, trades_all = [], []
        cum = 0.0
        cum_points = []        # (oos_end timestamp, cumulative window net)
        taken_windows = 0
        print(f"{'OOS start':>10} {'OOS end':>10} | {p1:>6} {p2:>6} | "
              f"{'IS nTr':>6} {'IS wr%':>7} {'IS net':>9} | {'gate':>5} | "
              f"{'OOS nTr':>7} {'OOS wr%':>7} {'OOS pnl':>9} "
              f"{'OOS Sharpe':>10} {'OOS MaxDD%':>10} {'cum':>9}")
        for day in oos_starts:
            v1, v2, is_m = self._is_best_cfg(day, p1, p2, grid, fv)
            gate_pass = (is_m['win_rate'] >= self.gate_wr) and (is_m['net'] > 0)
            is_start = day - timedelta(days=IS_DAYS)
            oos_end = (day + timedelta(days=OOS_DAYS - 1)).replace(
                hour=23, minute=59, second=59)
            o_m = {'trades': 0, 'wins': 0, 'win_rate': 0.0, 'net': 0.0}
            win_sharpe = win_maxdd = 0.0
            if gate_pass:
                taken_windows += 1
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                oos_trades, oos_eq = simulate_window(self.sig, day, oos_end,
                                                     kw['w1'], kw['n1'],
                                                     kw['e1'], None, None)
                o_m = window_metrics(oos_trades)
                win_sharpe = _sharpe_from_eq(oos_eq)
                win_maxdd = _max_dd_from_eq(oos_eq)
                cum += o_m['net']
                for t in oos_trades:
                    t['oos_start'] = day.date()
                    t['oos_end'] = oos_end.date()
                    t['p1'], t['p2'] = v1, v2
                trades_all.extend(oos_trades)
            cum_points.append((oos_end, cum))
            rows.append({
                'oos_start': day.date(), 'oos_end': oos_end.date(),
                p1: v1, p2: v2,
                **{f'fixed_{k}': fv[k] for k in self.fixed_params},
                'is_start': is_start.date(), 'is_end': (day - timedelta(days=1)).date(),
                'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                'is_net': is_m['net'], 'taken': gate_pass,
                'oos_trades': o_m['trades'], 'oos_wins': o_m['wins'],
                'oos_win_rate': o_m['win_rate'], 'oos_pnl': o_m['net'],
                'oos_sharpe': win_sharpe, 'oos_maxdd': win_maxdd,
                'oos_cum_pnl': cum,
            })
            print(f"{str(day.date()):>10} {str(oos_end.date()):>10} | "
                  f"{str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{'take' if gate_pass else 'skip':>5} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{win_sharpe:>10.2f} {win_maxdd:>10.2f} {cum:>9.2f}")
        self.wf_results = pd.DataFrame(rows)
        self.wf_trades = pd.DataFrame(trades_all)
        self.taken_windows = taken_windows
        self.n_windows = len(oos_starts)
        # window-level equity: cumulative sum of per-window net over the
        # overlapping OOS windows (indexed by window end date)
        self.oos_equity = pd.Series(
            [c for _, c in cum_points],
            index=pd.DatetimeIndex([e for e, _ in cum_points]))
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

        wf_path = os.path.join(out_dir, 'S4003_8_walkforward_results.csv')
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, 'S4003_8_walkforward_trades.csv')
        if len(self.wf_trades):
            out = self.wf_trades.copy()
            out['entry_dt'] = out['entry_dt'].dt.strftime(DT_FORMAT)
            out['exit_dt'] = out['exit_dt'].dt.strftime(DT_FORMAT)
            out = out[['oos_start', 'oos_end', 'type', 'entry_dt',
                       'entry_price', 'exit_dt', 'exit_price', 'pnl',
                       'reason', 'p1', 'p2']]
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
        gated_out = self.n_windows - self.taken_windows
        eq = self.oos_equity if len(self.oos_equity) else \
            self.wf_results.set_index('oos_end')['oos_cum_pnl']
        chart = strategyS4003V1._equity_chart_html(eq)

        rows_html = ''.join(
            f"<tr><td>{r['oos_start']}</td><td>{r['oos_end']}</td>"
            f"<td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td>"
            f"<td>{'take' if r['taken'] else 'skip'}</td>"
            f"<td>{r['oos_trades']}</td><td>{r['oos_win_rate']:.2f}</td>"
            f"<td>{r['oos_pnl']:.2f}</td>"
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
            f"<td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        gate_rows = ''.join(
            f"<tr><td>&gt;= {r['gate_wr_min']:.0f}% &amp; IS net &gt; 0</td>"
            f"<td>{r['taken_days']}</td><td>{r['trades']}</td>"
            f"<td>{r['win_rate']:.2f}</td><td>{r['net']:.2f}</td></tr>"
            for _, r in self.gate_table.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        html = f"""<html><head><title>S4003_8 Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body>
<p style="font-family:monospace;font-size:15px;margin-bottom:4px;"><b>Program: {PROGRAM_NAME}</b></p>
<h2>S4003_8 Walk-Forward Parameter Test (rolling IIS {IS_DAYS}d / OOS {OOS_DAYS}d, step {STEP_DAYS}d)</h2>
<p>Walk-forward period: {ANALYSIS_START.date()} .. {ANALYSIS_END.date()}
(1 year ago .. 7 days before today; data ends {self.data_end.date()}) |
rolling in-sample {IS_DAYS}d -&gt; out-of-sample {OOS_DAYS}d, step {STEP_DAYS} day
(consecutive OOS windows overlap by {OOS_DAYS - STEP_DAYS} days) |
stn1/sln1 unset (daily stops disabled) |
exit cost ${EXIT_COST:.2f}/trade<br/>
Parameter selection (grid) window: {self.analysis_days[0]} .. {self.analysis_days[self.grid_days - 1]}
- no walk-forward OOS window is inside it (no lookahead)<br/>
Gate selected on calibration mini-OOS days only (no real OOS data touched)</p>
<h3>Gate selection (calibration mini walk-forward)</h3>
<table><tr><th>gate</th><th>taken days</th><th>trades</th><th>win%</th>
<th>net</th></tr>
{gate_rows}</table>
<p>Selected gate: take a window only if IS win rate &gt;= {self.gate_wr:.0f}%
and IS net &gt; 0; skipped windows produce no trades.</p>
<h3>Summary (aggregated over overlapping OOS windows)</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
<tr><th>OOS windows taken / gated out</th><td>{self.taken_windows} / {gated_out} (of {self.n_windows})</td></tr>
<tr><th>Out-of-sample trades</th><td>{n} ({n / max(self.taken_windows, 1):.2f}/taken window; windows overlap, so trades are counted per window)</td></tr>
<tr><th>Out-of-sample win rate</th><td>{wr:.2f}% (target &gt;= {WIN_RATE_TARGET:.0f}% -> {'PASS' if wr >= WIN_RATE_TARGET else 'MISS'})</td></tr>
<tr><th>Out-of-sample net profit (sum of window net)</th><td>{net:,.2f}</td></tr>
<tr><th>Out-of-sample Sharpe ratio</th><td>{overall_sharpe:.2f} (window-level equity, rf 3% annualized)</td></tr>
<tr><th>Out-of-sample max system drawdown</th><td>{overall_maxdd:.2f}% (window-level equity)</td></tr>
</table>
<h3>Per out-of-sample window</h3>
<p>Per-window Sharpe / max drawdown are computed on that window's bar-level
mark-to-market equity (annualized, 3% risk-free). Skipped windows show no
OOS activity; cum pnl (sum of taken-window net) is unchanged.</p>
<table><tr><th>OOS start</th><th>OOS end</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th><th>gate</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th><th>cum pnl</th></tr>
{rows_html}</table>
<h3>Equity Curve (cumulative OOS window P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f'S4003_8_walkforward_{ts}.html')
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
        print(f"OUT-OF-SAMPLE AGGREGATE (overlapping windows): {n} trades on "
              f"{self.taken_windows} taken windows ({gated_out} gated out), "
              f"win rate {wr:.2f}% (target >= {WIN_RATE_TARGET:.0f}% -> "
              f"{'PASS' if wr >= WIN_RATE_TARGET else 'MISS'}), "
              f"net {net:,.2f}, {n / max(self.taken_windows, 1):.2f} trades/"
              f"taken window, Sharpe {overall_sharpe:.2f}, "
              f"max DD {overall_maxdd:.2f}%")
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
        description='S4003_8 rolling walk-forward test (IIS 60d -> OOS 7d, '
                    'step 1d, from 1 year ago to 7 days before today; '
                    'stn1/sln1 daily stops disabled)')
    ap.add_argument('--ticker', default='ES')
    args = ap.parse_args()
    S4003V8WalkForward(ticker=args.ticker).run()


if __name__ == '__main__':
    main()
