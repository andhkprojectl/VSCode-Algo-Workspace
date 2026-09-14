"""
S4003_5_linearRegression_glm.py
===============================
Linear-regression variant of S4003_5_randomforest_glm.py
(requirement: VS_8000_Strategy/VS_4003_20260827_ES/2 Strategy/
S4003_5_glm_ur_OLS V1.txt).

Target: out-of-sample win rate >= 1.15 x the reference backtest win rate
(the phase-1 best engine configuration over the analysis period).

Differences vs S4003_5_randomforest_glm.py:
  - An OLS entry gate (StandardScaler + LinearRegression, i.e. the
    S4003_4_linearRegression model) overlays the walk-forward: before each
    out-of-sample day, the model is trained on all labelled trades that
    CLOSED before that day (expanding, past-only, honest) and the day's
    entry signals with a predicted win score below the gate threshold are
    vetoed; the day is then re-simulated. Where "random forest" appeared
    in the flow, the linear regression takes its place.
  - Label pool for the gate: the fixed phase-1 best configuration is
    simulated over the whole loaded history; each trade's features (the
    S0001 feature set at its signal bar) and win label form the training
    pool. Labels therefore depend on the fixed parameters, never on the
    walk-forward choices.
  - Output files use the SAME names as S4003_5_randomforest_glm.py
    (running this program replaces the RF version's outputs).
  - The HTML report carries the program name at the top.

Everything else is identical to S4003_5_randomforest_glm.py: analysis
period 2026-06-01 .. 2026-08-31 (DB data ends 2026-08-21), 2 most valuable
parameters chosen by win-rate spread (w1, n1), the other 3 fixed at the
backtest best, rolling 1-week in-sample -> 1-day out-of-sample windows.

Outputs (backTestResult folder, same names as the RF version):
  S4003_5_parameter_analysis.csv
  S4003_5_walkforward_results.csv     (+ gate_kept / gate_vetoed columns)
  S4003_5_walkforward_trades.csv
  S4003_5_walkforward_<ts>.html       (program name at the top)
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
    _HERE, '..', '..', 'VS_0001_GeneralStrategy'))
for _p in (_HERE, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import strategyS4003V1, load_data_from_db, OUTPUT_DIR, DT_FORMAT
from S4003_3_randomforest_glm import (get_db_range, _set_params, _restore_params,
                                      EXIT_COST, W1_GRID, N1_GRID, E1_GRID,
                                      build_samples)
from S4003_4_randomforest_glm import simulate_with_daily_stops, STN1_GRID, SLN1_GRID
from S4003_4_linearRegression_glm import _make_model as _make_ols
from S4003_5_randomforest_glm import (simulate_window, window_metrics,
                                      _sharpe_from_eq, _max_dd_from_eq,
                                      ANALYSIS_START, ANALYSIS_END, IS_DAYS,
                                      MIN_IS_TRADES, PARAM_GRIDS)
from S0001_GeneralFeature_2_glm import ESFeatureEngineer

PROGRAM_NAME = 'S4003_5_linearRegression_glm.py'
WIN_RATE_TARGET_INCREASE = 0.15
MIN_GATE_TRAIN = 20              # minimum closed trades before the gate fires
THR_LONG = 0.50                  # OLS gate thresholds
THR_SHORT = 0.50


# ===========================================================================
# Main class
# ===========================================================================
class S4003V5LinearRegression:
    def __init__(self, ticker='ES', top_k=3, thr_long=THR_LONG,
                 thr_short=THR_SHORT):
        self.ticker = ticker
        self.top_k = top_k
        self.thr_long = thr_long
        self.thr_short = thr_short

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, _db_hi = get_db_range(self.ticker)
        print(f"[{PROGRAM_NAME}]")
        print(f"Loading {self.ticker} 1-min bars "
              f"[{db_lo} .. {ANALYSIS_END}] (warm-up from DB start) ...")
        self.df = load_data_from_db(self.ticker, str(db_lo), str(ANALYSIS_END))
        if self.df.empty:
            raise RuntimeError('no bars returned from database')
        self.sig = strategyS4003V1._prepare(self.df)
        in_period = (self.sig.index >= ANALYSIS_START) & (self.sig.index <= ANALYSIS_END)
        self.analysis_days = np.unique(np.asarray(self.sig.index[in_period].date))
        self.trading_days = len(self.analysis_days)
        print(f"Loaded {len(self.df)} bars: {self.df.index[0]} .. {self.df.index[-1]}; "
              f"analysis period {ANALYSIS_START.date()} .. {self.df.index[-1].date()} "
              f"({self.trading_days} trading days)")

    # ------------------------------------------------------------------
    def _run_cfg(self, w1, n1, e1, stn1, sln1):
        trades, _eq = simulate_window(self.sig, ANALYSIS_START, ANALYSIS_END,
                                      w1, n1, e1, stn1, sln1)
        m = window_metrics(trades)
        return {
            'w1': w1, 'n1': n1, 'e1': e1,
            'stn1': stn1 if stn1 is not None else 'none',
            'sln1': sln1 if sln1 is not None else 'none',
            'trades': m['trades'], 'wins': m['wins'],
            'win_rate': m['win_rate'], 'net': m['net'],
            'freq': m['trades'] / max(self.trading_days, 1),
        }

    @staticmethod
    def _rank_key(r):
        viable = (r['freq'] >= 0.5 and r['net'] > 0)
        return (viable, r['win_rate'], r['net'])

    # ------------------------------------------------------------------
    def phase1(self):
        print(f"\nPhase 1: backtest over the analysis period "
              f"(exit cost ${EXIT_COST:.2f}/trade)")
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        results = []
        for (w1, n1, e1) in combos:
            results.append(self._run_cfg(w1, n1, e1, None, None))
        top = sorted(results, key=self._rank_key, reverse=True)[:self.top_k]
        seen = {(r['w1'], r['n1'], r['e1'], r['stn1'], r['sln1']) for r in results}
        for cfg in top:
            for stn1 in STN1_GRID:
                for sln1 in SLN1_GRID:
                    key = (cfg['w1'], cfg['n1'], cfg['e1'],
                           stn1 if stn1 is not None else 'none',
                           sln1 if sln1 is not None else 'none')
                    if key in seen:
                        continue
                    results.append(self._run_cfg(cfg['w1'], cfg['n1'], cfg['e1'],
                                                 stn1, sln1))
                    seen.add(key)
        self.phase1_results = pd.DataFrame(results)

        value_rows = []
        for p in ('w1', 'n1', 'e1', 'stn1', 'sln1'):
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
        self.fixed_params = [p for p in ('w1', 'n1', 'e1', 'stn1', 'sln1')
                             if p not in self.chosen_params]

        best = sorted(results, key=self._rank_key, reverse=True)[0]
        self.best_cfg = best
        self.fixed_values = {p: best[p] for p in self.fixed_params}
        self.target_win_rate = best['win_rate'] * (1.0 + WIN_RATE_TARGET_INCREASE)

        print('\nParameter value analysis (win-rate spread across values):')
        for _, row in value_tbl.iterrows():
            print(f"  {row['parameter']:>5}: spread {row['value_spread']:5.2f} pp  "
                  f"[{row['best_win_rate_by_value']}]")
        print(f"\n2 most valuable parameters for walk-forward: "
              f"{self.chosen_params[0]}, {self.chosen_params[1]}")
        print(f"Other 3 fixed at backtest best "
              f"({', '.join(f'{p}={self.fixed_values[p]}' for p in self.fixed_params)}):")
        print(f"  w1={best['w1']}, n1={best['n1']}, e1={best['e1']}, "
              f"stn1={best['stn1']}, sln1={best['sln1']} -> "
              f"{best['trades']} trades, win rate {best['win_rate']:.2f}%, "
              f"net {best['net']:.2f}")
        print(f"Target: OOS win rate >= {best['win_rate']:.2f}% x 1.15 = "
              f"{self.target_win_rate:.2f}%")
        return best

    # ------------------------------------------------------------------
    def build_gate_pool(self):
        """Label pool for the OLS gate: fixed best configuration simulated
        over the whole loaded history -> (features, labels, meta)."""
        print("\nBuilding OLS gate label pool (fixed best configuration, "
              "whole loaded history) ...")
        b = self.best_cfg
        stn1 = None if b['stn1'] == 'none' else b['stn1']
        sln1 = None if b['sln1'] == 'none' else b['sln1']
        pool_trades, _eq = simulate_window(self.sig, self.sig.index[0],
                                           ANALYSIS_END, b['w1'], b['n1'],
                                           b['e1'], stn1, sln1)
        print("Computing S0001 feature set (ESFeatureEngineer) ...")
        df_lower = self.df.rename(columns={'Open': 'open', 'High': 'high',
                                           'Low': 'low', 'Close': 'close',
                                           'Volume': 'volume'})
        fe = ESFeatureEngineer(df_lower)
        fe.build_all()
        self.feats = fe.features
        X, y, meta, drop_cols = build_samples(pool_trades, self.feats)
        self.pool_X, self.pool_y, self.pool_meta = X, y, meta
        self.pool_by_entry = {(row['entry_dt'], row['direction']): j
                              for j, row in meta.iterrows()}
        wins = int(y.sum())
        print(f"Gate label pool: {len(X)} labelled trades "
              f"(win rate {wins / len(y) * 100 if len(y) else 0:.1f}%, "
              f"{len(drop_cols)} sparse features dropped)")

    # ------------------------------------------------------------------
    def _gate_day(self, day_start_ts, first_pass_trades, kw):
        """Veto entry signals of first-pass day trades whose OLS win score
        is below the gate; re-simulate the day. Returns (trades, equity,
        kept, vetoed)."""
        train_mask = self.pool_meta['exit_dt'] < day_start_ts
        n_train = int(train_mask.sum())
        model = None
        if n_train >= MIN_GATE_TRAIN:
            model = _make_ols().fit(self.pool_X[train_mask.to_numpy()],
                                    self.pool_y[train_mask.to_numpy()])
        veto = []
        scored = 0
        if model is not None:
            for t in first_pass_trades:
                key = (t['entry_dt'], t['type'])
                j = self.pool_by_entry.get(key)
                if j is None:
                    continue
                p = float(model.predict(self.pool_X.iloc[[j]])[0])
                scored += 1
                thr = self.thr_long if t['type'] == 'Long' else self.thr_short
                if p < thr:
                    pos = self.sig.index.searchsorted(t['entry_dt'])
                    veto.append((pos, t['type']))

        bcol = self.sig.columns.get_loc('buySig')
        scol = self.sig.columns.get_loc('shortSig')
        saved = []
        for pos, direction in veto:
            col = bcol if direction == 'Long' else scol
            saved.append((col, pos, self.sig.iat[pos, col]))
            self.sig.iat[pos, col] = 0
        try:
            trades, eq = simulate_window(self.sig, day_start_ts,
                                         day_start_ts.replace(hour=23, minute=59,
                                                              second=59),
                                         kw['w1'], kw['n1'], kw['e1'],
                                         None if kw['stn1'] == 'none' else kw['stn1'],
                                         None if kw['sln1'] == 'none' else kw['sln1'])
        finally:
            for col, pos, old in saved:
                self.sig.iat[pos, col] = old
        return trades, eq, scored - len(veto), len(veto)

    # ------------------------------------------------------------------
    def walk_forward(self):
        p1, p2 = self.chosen_params
        grid = [(v1, v2) for v1 in PARAM_GRIDS[p1] for v2 in PARAM_GRIDS[p2]]
        fv = dict(self.fixed_values)

        def to_val(x):
            return None if x == 'none' else x

        oos_days = [d for d in pd.date_range(ANALYSIS_START + timedelta(days=IS_DAYS),
                                             ANALYSIS_END, freq='D')
                    if d.date() in set(self.analysis_days)]
        print(f"\nPhase 2: walk-forward (OLS gate thresholds "
              f"L {self.thr_long:.2f} / S {self.thr_short:.2f}), in-sample "
              f"{IS_DAYS} days -> out-of-sample 1 day, {len(grid)} combinations "
              f"per window, {len(oos_days)} out-of-sample days")

        rows, trades_all, eq_segments = [], [], []
        cum = 0.0
        print(f"{'OOS day':>10} | {p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} "
              f"{'IS net':>9} | {'OOS nTr':>7} {'OOS wr%':>7} {'OOS pnl':>9} "
              f"{'gateK':>5} {'gateV':>5} | {'Sharpe':>7} {'MaxDD%':>7} {'cum':>9}")
        for day in oos_days:
            is_start = day - timedelta(days=IS_DAYS)
            is_end = (day - timedelta(days=1)).replace(hour=23, minute=59, second=59)
            day_end = day.replace(hour=23, minute=59, second=59)
            best_is, best_pair = None, None
            for v1, v2 in grid:
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                trades, _ = simulate_window(self.sig, is_start, is_end,
                                            kw['w1'], kw['n1'], kw['e1'],
                                            to_val(kw['stn1']), to_val(kw['sln1']))
                m = window_metrics(trades)
                key = (m['trades'] >= MIN_IS_TRADES, m['win_rate'], m['net'])
                if best_is is None or key > best_is:
                    best_is, best_pair = key, (v1, v2, m)
            v1, v2, is_m = best_pair
            kw = dict(fv)
            kw[p1], kw[p2] = v1, v2
            first_trades, _ = simulate_window(self.sig, day, day_end,
                                              kw['w1'], kw['n1'], kw['e1'],
                                              to_val(kw['stn1']), to_val(kw['sln1']))
            oos_trades, oos_eq, kept, vetoed = self._gate_day(day, first_trades, kw)
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
                'is_start': is_start.date(), 'is_end': (day - timedelta(days=1)).date(),
                'is_trades': is_m['trades'], 'is_win_rate': is_m['win_rate'],
                'is_net': is_m['net'],
                'oos_trades': o_m['trades'], 'oos_wins': o_m['wins'],
                'oos_win_rate': o_m['win_rate'], 'oos_pnl': o_m['net'],
                'gate_kept': kept, 'gate_vetoed': vetoed,
                'oos_sharpe': day_sharpe, 'oos_maxdd': day_maxdd,
                'oos_cum_pnl': cum,
            })
            print(f"{str(day.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{kept:>5} {vetoed:>5} | {day_sharpe:>7.2f} {day_maxdd:>7.2f} "
                  f"{cum:>9.2f}")
        self.wf_results = pd.DataFrame(rows)
        self.wf_trades = pd.DataFrame(trades_all)
        self.oos_equity = (pd.concat(eq_segments) if eq_segments
                           else pd.Series(dtype=float))
        return self.wf_results

    # ------------------------------------------------------------------
    def write_outputs(self, out_dir=OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')

        pa_path = os.path.join(out_dir, 'S4003_5_parameter_analysis.csv')
        self.phase1_results.to_csv(pa_path, index=False)
        pv_path = os.path.join(out_dir, 'S4003_5_parameter_values.csv')
        self.parameter_values.assign(
            chosen=', '.join(self.chosen_params),
            program=PROGRAM_NAME).to_csv(pv_path, index=False)

        wf_path = os.path.join(out_dir, 'S4003_5_walkforward_results.csv')
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, 'S4003_5_walkforward_trades.csv')
        if len(self.wf_trades):
            out = self.wf_trades.copy()
            out['entry_dt'] = out['entry_dt'].dt.strftime(DT_FORMAT)
            out['exit_dt'] = out['exit_dt'].dt.strftime(DT_FORMAT)
            out = out[['oos_day', 'type', 'entry_dt', 'entry_price', 'exit_dt',
                       'exit_price', 'pnl', 'reason', 'p1', 'p2']]
        else:
            out = self.wf_trades
        out.to_csv(tr_path, index=False)

        # ---- HTML (program name on top, per requirement) ----
        n = len(self.wf_trades)
        wins = int((self.wf_trades['pnl'] > 0).sum()) if n else 0
        wr = wins / n * 100.0 if n else 0.0
        net = float(self.wf_trades['pnl'].sum()) if n else 0.0
        overall_sharpe = _sharpe_from_eq(self.oos_equity)
        overall_maxdd = _max_dd_from_eq(self.oos_equity)
        p1, p2 = self.chosen_params
        eq = self.oos_equity if len(self.oos_equity) else \
            self.wf_results.set_index('oos_date')['oos_cum_pnl']
        chart = strategyS4003V1._equity_chart_html(eq)

        rows_html = ''.join(
            f"<tr><td>{r['oos_date']}</td><td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td>"
            f"<td>{r['oos_trades']}</td><td>{r['oos_win_rate']:.2f}</td>"
            f"<td>{r['oos_pnl']:.2f}</td>"
            f"<td>{r['gate_kept']}</td><td>{r['gate_vetoed']}</td>"
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
            f"<td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        wr_ok = wr >= self.target_win_rate
        html = f"""<html><head><title>S4003_5 Walk-Forward Results (linear regression)</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body>
<p style="font-family:monospace;font-size:15px;margin-bottom:4px;"><b>Program: {PROGRAM_NAME}</b></p>
<h2>S4003_5 Walk-Forward Parameter Test (linear regression)</h2>
<p>Period: {ANALYSIS_START.date()} .. {self.df.index[-1].date()} |
rolling in-sample {IS_DAYS}d -&gt; out-of-sample 1d |
exit cost ${EXIT_COST:.2f}/trade |
OLS gate: StandardScaler + LinearRegression, trained on all trades closed
before each OOS day, thresholds L {self.thr_long:.2f} / S {self.thr_short:.2f}</p>
<h3>Summary</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
<tr><th>Out-of-sample trades</th><td>{n} ({n / max(self.trading_days, 1):.2f}/day)</td></tr>
<tr><th>Out-of-sample win rate</th><td>{wr:.2f}% (target &gt;= {self.target_win_rate:.2f}% = reference x 1.15 -> {'PASS' if wr_ok else 'MISS'})</td></tr>
<tr><th>Out-of-sample net profit</th><td>{net:,.2f}</td></tr>
<tr><th>Out-of-sample Sharpe ratio</th><td>{overall_sharpe:.2f} (rf 3% annualized)</td></tr>
<tr><th>Out-of-sample max system drawdown</th><td>{overall_maxdd:.2f}%</td></tr>
</table>
<h3>Per out-of-sample period</h3>
<p>gateK/gateV = OLS-gated entry signals kept/vetoed that day; per-day
Sharpe / max drawdown are computed on that day's bar-level mark-to-market
equity (annualized, 3% risk-free).</p>
<table><tr><th>OOS date</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>gateK</th><th>gateV</th><th>OOS Sharpe</th><th>OOS MaxDD%</th>
<th>cum pnl</th></tr>
{rows_html}</table>
<h3>Equity Curve (cumulative OOS P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f'S4003_5_walkforward_{ts}.html')
        with open(html_path, 'w') as f:
            f.write(html)

        print('\nOutput files:')
        print(f'  parameter analysis: {pa_path}')
        print(f'  walkforward results: {wf_path}')
        print(f'  walkforward trades:  {tr_path}')
        print(f'  html:                {html_path}')

        print('\n' + '=' * 60)
        print(f"OUT-OF-SAMPLE AGGREGATE: {n} trades, win rate {wr:.2f}% "
              f"(target >= {self.target_win_rate:.2f}% -> "
              f"{'PASS' if wr_ok else 'MISS'}), "
              f"net {net:,.2f}, freq {n / max(self.trading_days, 1):.2f}/day, "
              f"Sharpe {overall_sharpe:.2f}, max DD {overall_maxdd:.2f}%")
        print('=' * 60)

    # ------------------------------------------------------------------
    def run(self):
        self._load()
        self.phase1()
        self.build_gate_pool()
        self.walk_forward()
        self.write_outputs()
        _restore_params()


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_5 walk-forward + OLS linear-regression gate')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--thr-long', type=float, default=THR_LONG,
                    help='OLS gate threshold for long entries')
    ap.add_argument('--thr-short', type=float, default=THR_SHORT,
                    help='OLS gate threshold for short entries')
    args = ap.parse_args()
    S4003V5LinearRegression(ticker=args.ticker, top_k=args.top_k,
                            thr_long=args.thr_long,
                            thr_short=args.thr_short).run()


if __name__ == '__main__':
    main()
