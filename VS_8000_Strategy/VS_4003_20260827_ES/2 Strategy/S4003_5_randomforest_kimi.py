"""
S4003_5_randomforest_kimi.py
============================
Walk-forward parameter test on top of the S4003_4 (kimi) engine
(requirement: VS_8000_Strategy/VS_4003_20260827_ES/2 Strategy/
S4003_5_kimi_ur_RF V1.txt).

Target: out-of-sample win rate >= 65%.

Steps:
  1. Analysis period 2026-06-01 .. 2026-08-31 (DB data currently ends
     2026-08-21, so the last out-of-sample day is 2026-08-21).
  2. Phase 1 (backtest over the analysis period): two-phase grid over the
     5 engine parameters w1, n1, e1, stn1, sln1 (same grids as S4003_4)
     -> per-parameter VALUE = spread of the best achievable win rate
     across the parameter's values; the two largest spreads are the
     "2 most valuable parameters" for the walk-forward test. The other
     three parameters stay at the best backtest configuration.
     Note: the RF P(Win) gate of S4003_2-4 is not retrained inside weekly
     windows (too few trades per week); this program optimizes engine
     parameters on the S4003_4 engine (signals + exits + daily stops).
  3. Phase 2 (walk-forward, pattern from VS_0003_test/walkForwardTest.py):
     rolling 1-week in-sample window -> grid-search the 2 chosen
     parameters -> apply the winners to the NEXT 1 day (out-of-sample),
     then roll both windows forward by one day until the analysis end.
     In-sample selection: maximize win rate subject to >= MIN_IS_TRADES
     trades (tie-break: net profit).

Outputs (backTestResult folder):
  S4003_5_kimi_parameter_analysis.csv   phase-1 candidates + parameter values
  S4003_5_kimi_walkforward_results.csv  per out-of-sample day: chosen 2
                                        parameter values + IS/OOS metrics
  S4003_5_kimi_walkforward_trades.csv   every out-of-sample trade
  S4003_5_kimi_walkforward_<ts>.html    summary + per-OOS-period table with
                                        the 2 parameter values + equity chart
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from S4003_1_glm import strategyS4003V1, load_data_from_db, OUTPUT_DIR, DT_FORMAT
from S4003_2_randomforest_kimi import get_db_range
from S4003_3_randomforest_kimi import (_set_params, _restore_params,
                                       W1_GRID, N1_GRID, E1_GRID)
from S4003_4_randomforest_kimi import (_simulate_daily_stop,
                                       STN1_GRID, SLN1_GRID)

EXIT_COST = strategyS4003V1.commission_per_trade

ANALYSIS_START = pd.Timestamp('2026-06-01 00:00:00')
ANALYSIS_END = pd.Timestamp('2026-08-31 23:59:59')
WIN_RATE_TARGET = 65.0
IS_DAYS = 7                      # in-sample window length (days)
MIN_IS_TRADES = 5                # minimum in-sample trades for selection
LOOKBACK_BARS = 100              # engine warm-up bars kept before a window

PARAM_GRIDS = {'w1': W1_GRID, 'n1': N1_GRID, 'e1': E1_GRID,
               'stn1': STN1_GRID, 'sln1': SLN1_GRID}


# ===========================================================================
# Window simulation helpers
# ===========================================================================
def simulate_window(sig, start_ts, end_ts, w1, n1, e1, stn1, sln1):
    """Run the S4003_4 (kimi) engine on [start_ts, end_ts] with entries gated
    to the window (LOOKBACK_BARS bars kept before start for engine warm-up).
    Returns the list of trade dicts."""
    _set_params(w1, n1, e1)
    lo = sig.index.searchsorted(start_ts)
    lo = max(0, lo - LOOKBACK_BARS)
    hi = sig.index.searchsorted(end_ts, side='right')
    sl = sig.iloc[lo:hi].copy()
    trades, _eq, _im = _simulate_daily_stop(sl, stn1, sln1,
                                            trade_start_ts=start_ts)
    return trades


def window_metrics(trades):
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    net = float(sum(t['pnl'] for t in trades))
    wr = wins / n * 100.0 if n else 0.0
    return {'trades': n, 'wins': wins, 'win_rate': wr, 'net': net}


# ===========================================================================
# Main class
# ===========================================================================
class S4003V5KimiWalkForward:
    def __init__(self, ticker='ES', top_k=3):
        self.ticker = ticker
        self.top_k = top_k

    # ------------------------------------------------------------------
    def _load(self):
        db_lo, _db_hi = get_db_range(self.ticker)
        db_lo = pd.Timestamp(db_lo)
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
        trades = simulate_window(self.sig, ANALYSIS_START, ANALYSIS_END,
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
        """Frequency- and net-viable candidates first, then win rate, net."""
        viable = (r['freq'] >= 0.5 and r['net'] > 0)
        return (viable, r['win_rate'], r['net'])

    # ------------------------------------------------------------------
    def phase1(self):
        print(f"\nPhase 1: backtest over the analysis period "
              f"(exit cost ${EXIT_COST:.2f}/trade), daily stops off first")
        combos = [(w1, n1, e1) for w1 in W1_GRID for n1 in N1_GRID for e1 in E1_GRID]
        results = []
        for (w1, n1, e1) in combos:
            r = self._run_cfg(w1, n1, e1, None, None)
            results.append(r)
        top = sorted(results, key=self._rank_key, reverse=True)[:self.top_k]
        seen = {(r['w1'], r['n1'], r['e1'], r['stn1'], r['sln1']) for r in results}
        print(f"Phase 1b: daily-stop grid on top {len(top)} configurations")
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

        # ---- parameter value analysis (win-rate sensitivity) ----
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
        return best

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
        print(f"\nPhase 2: walk-forward, in-sample {IS_DAYS} days -> "
              f"out-of-sample 1 day, {len(grid)} combinations per window, "
              f"{len(oos_days)} out-of-sample days")

        rows, trades_all = [], []
        cum = 0.0
        print(f"{'OOS day':>10} | {p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} "
              f"{'IS net':>9} | {'OOS nTr':>7} {'OOS wr%':>7} {'OOS pnl':>9} {'cum':>9}")
        for day in oos_days:
            is_start = day - timedelta(days=IS_DAYS)
            is_end = (day - timedelta(days=1)).replace(hour=23, minute=59, second=59)
            day_end = day.replace(hour=23, minute=59, second=59)
            best_is, best_pair = None, None
            for v1, v2 in grid:
                kw = dict(fv)
                kw[p1], kw[p2] = v1, v2
                trades = simulate_window(self.sig, is_start, is_end,
                                         kw['w1'], kw['n1'], kw['e1'],
                                         to_val(kw['stn1']), to_val(kw['sln1']))
                m = window_metrics(trades)
                key = (m['trades'] >= MIN_IS_TRADES, m['win_rate'], m['net'])
                if best_is is None or key > best_is:
                    best_is, best_pair = key, (v1, v2, m)
            v1, v2, is_m = best_pair
            kw = dict(fv)
            kw[p1], kw[p2] = v1, v2
            oos_trades = simulate_window(self.sig, day, day_end,
                                         kw['w1'], kw['n1'], kw['e1'],
                                         to_val(kw['stn1']), to_val(kw['sln1']))
            o_m = window_metrics(oos_trades)
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
                'oos_cum_pnl': cum,
            })
            print(f"{str(day.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{cum:>9.2f}")
        self.wf_results = pd.DataFrame(rows)
        self.wf_trades = pd.DataFrame(trades_all)
        return self.wf_results

    # ------------------------------------------------------------------
    def write_outputs(self, out_dir=OUTPUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')

        pa_path = os.path.join(out_dir, 'S4003_5_kimi_parameter_analysis.csv')
        self.phase1_results.to_csv(pa_path, index=False)
        self.parameter_values.assign(
            chosen=', '.join(self.chosen_params)).to_csv(pa_path, index=False,
                                                         mode='a')

        wf_path = os.path.join(out_dir, 'S4003_5_kimi_walkforward_results.csv')
        self.wf_results.to_csv(wf_path, index=False)

        tr_path = os.path.join(out_dir, 'S4003_5_kimi_walkforward_trades.csv')
        if len(self.wf_trades):
            out = self.wf_trades.copy()
            out['entry_dt'] = out['entry_dt'].dt.strftime(DT_FORMAT)
            out['exit_dt'] = out['exit_dt'].dt.strftime(DT_FORMAT)
            out = out[['oos_day', 'type', 'entry_dt', 'entry_price', 'exit_dt',
                       'exit_price', 'pnl', 'reason', 'p1', 'p2']]
        else:
            out = self.wf_trades
        out.to_csv(tr_path, index=False)

        # ---- HTML summary (per-OOS-period table incl. the 2 parameters) ----
        n = len(self.wf_trades)
        wins = int((self.wf_trades['pnl'] > 0).sum()) if n else 0
        wr = wins / n * 100.0 if n else 0.0
        net = float(self.wf_trades['pnl'].sum()) if n else 0.0
        p1, p2 = self.chosen_params

        # ---- aggregate OOS walk-forward metrics ----
        init_cap = strategyS4003V1.initial_capital
        eq = (init_cap + self.wf_results.set_index('oos_date')['oos_cum_pnl'])
        eq.index = pd.to_datetime(eq.index)
        eq = eq.sort_index()
        peak = eq.cummax()
        max_dd = float(((peak - eq) / peak).max() * 100.0) if len(eq) else 0.0
        d_ret = eq.pct_change().dropna()
        rf_d = strategyS4003V1.risk_free_annual / 252.0
        sharpe = (float((d_ret - rf_d).mean() / d_ret.std(ddof=1) * np.sqrt(252))
                  if len(d_ret) > 1 and d_ret.std(ddof=1) > 0 else 0.0)
        self.oos_metrics = {'net': net, 'trades': n, 'win_rate': wr,
                            'max_dd': max_dd, 'sharpe': sharpe}
        chart = strategyS4003V1._equity_chart_html(eq)

        rows_html = ''.join(
            f"<tr><td>{r['oos_date']}</td><td>{r[p1]}</td><td>{r[p2]}</td>"
            f"<td>{r['is_trades']}</td><td>{r['is_win_rate']:.2f}</td>"
            f"<td>{r['is_net']:.2f}</td>"
            f"<td>{r['oos_trades']}</td><td>{r['oos_win_rate']:.2f}</td>"
            f"<td>{r['oos_pnl']:.2f}</td><td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        wr_pass = wr >= WIN_RATE_TARGET
        html = f"""<html><head><title>S4003_5 (kimi) Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}
td.pos{{color:green;}} td.neg{{color:red;}}</style></head>
<body><h2>S4003_5 (kimi) Walk-Forward Parameter Test</h2>
<p>Period: {ANALYSIS_START.date()} .. {self.df.index[-1].date()} |
rolling in-sample {IS_DAYS}d -&gt; out-of-sample 1d |
exit cost ${EXIT_COST:.2f}/trade</p>
<h3>Walk-Forward Test Result (out-of-sample aggregate)</h3>
<table>
<tr><th style="text-align:left">Net Profit</th>
    <td style="text-align:right" class="{'pos' if net >= 0 else 'neg'}">{net:,.2f}</td></tr>
<tr><th style="text-align:left">Number of Trades</th>
    <td style="text-align:right">{n} ({n / max(self.trading_days, 1):.2f}/day)</td></tr>
<tr><th style="text-align:left">Win Rate</th>
    <td style="text-align:right" class="{'pos' if wr_pass else 'neg'}">{wr:.2f}%
    (target &gt;= {WIN_RATE_TARGET:.0f}% -&gt; {'PASS' if wr_pass else 'MISS'})</td></tr>
<tr><th style="text-align:left">Max System Drawdown</th>
    <td style="text-align:right">{max_dd:.2f}%</td></tr>
<tr><th style="text-align:left">Sharpe Ratio</th>
    <td style="text-align:right">{sharpe:.4f}</td></tr>
</table>
<h3>Configuration</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
</table>
<h3>Per out-of-sample period</h3>
<table><tr><th>OOS date</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th><th>cum pnl</th></tr>
{rows_html}</table>
<h3>Equity Curve (cumulative OOS P&amp;L)</h3>{chart}
</body></html>"""
        html_path = os.path.join(out_dir, f'S4003_5_kimi_walkforward_{ts}.html')
        with open(html_path, 'w') as f:
            f.write(html)

        print('\nOutput files:')
        print(f'  parameter analysis: {pa_path}')
        print(f'  walkforward results: {wf_path}')
        print(f'  walkforward trades:  {tr_path}')
        print(f'  html:                {html_path}')

        m = self.oos_metrics
        print('\n' + '=' * 60)
        print("WALK-FORWARD TEST RESULT (out-of-sample aggregate):")
        print(f"  Net Profit:            {m['net']:,.2f}")
        print(f"  Number of Trades:      {m['trades']} "
              f"({m['trades'] / max(self.trading_days, 1):.2f}/day)")
        print(f"  Win Rate:              {m['win_rate']:.2f}% "
              f"(target >= {WIN_RATE_TARGET:.0f}% -> "
              f"{'PASS' if m['win_rate'] >= WIN_RATE_TARGET else 'MISS'})")
        print(f"  Max System Drawdown:   {m['max_dd']:.2f}%")
        print(f"  Sharpe Ratio:          {m['sharpe']:.4f}")
        print('=' * 60)

    # ------------------------------------------------------------------
    def run(self):
        self._load()
        self.phase1()
        self.walk_forward()
        self.write_outputs()
        _restore_params()


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='S4003_5 (kimi) walk-forward test of the 2 most valuable parameters')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--top-k', type=int, default=3)
    args = ap.parse_args()
    S4003V5KimiWalkForward(ticker=args.ticker, top_k=args.top_k).run()


if __name__ == '__main__':
    main()
