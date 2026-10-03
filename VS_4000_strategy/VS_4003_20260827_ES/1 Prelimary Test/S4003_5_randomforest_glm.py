"""
S4003_5_randomforest_glm.py
===========================
Walk-forward parameter test on top of the S4003_4 engine
(requirement: VS_4000_strategy/VS_4003_20260827_ES/1 Prelimary Test/
S4003_5_glm_ur_RF V1.txt).

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
  S4003_5_parameter_analysis.csv     phase-1 candidates + parameter values
  S4003_5_walkforward_results.csv    per out-of-sample day: chosen 2
                                     parameter values + IS/OOS metrics
  S4003_5_walkforward_trades.csv     every out-of-sample trade
  S4003_5_walkforward_<ts>.html      summary + per-OOS-period table with
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
_FEATURES_DIR = os.path.normpath(os.path.join(
    _HERE, '..', '..', '..', 'VS_0006_dataFunc', 'VS_6006_FeatureEngineering'))
for _p in (_HERE, _FEATURES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from S4003_1_glm import strategyS4003V1, load_data_from_db, OUTPUT_DIR, DT_FORMAT
from S4003_3_randomforest_glm import (get_db_range, _set_params, _restore_params,
                                      EXIT_COST, W1_GRID, N1_GRID, E1_GRID)
from S4003_4_randomforest_glm import simulate_with_daily_stops, STN1_GRID, SLN1_GRID

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
    """Run the S4003_4 engine on [start_ts, end_ts] with entries gated to the
    window (LOOKBACK_BARS bars kept before start for engine warm-up).
    Returns (trades list, bar-level equity pd.Series for the window)."""
    _set_params(w1, n1, e1)
    lo = sig.index.searchsorted(start_ts)
    lo = max(0, lo - LOOKBACK_BARS)
    hi = sig.index.searchsorted(end_ts, side='right')
    sl = sig.iloc[lo:hi].copy()
    pre = sl.index < start_ts
    sl.loc[pre, 'buySig'] = 0
    sl.loc[pre, 'shortSig'] = 0
    trades, eq, _im = simulate_with_daily_stops(sl, stn1, sln1)
    keep = np.asarray(sl.index >= start_ts)
    eq_series = pd.Series(eq[keep], index=sl.index[keep])
    return trades, eq_series


def _sharpe_from_eq(eq: pd.Series, rf_annual: float = 0.03) -> float:
    """Annualized Sharpe of a bar-level equity series (3% risk-free),
    same methodology as strategyS4003V1._compute_stats."""
    vals = eq.to_numpy(dtype=float)
    if len(vals) < 3:
        return 0.0
    rets = np.diff(vals) / vals[:-1]
    if len(rets) < 2 or np.std(rets, ddof=1) <= 0:
        return 0.0
    years = max((eq.index[-1] - eq.index[0]).total_seconds()
                / (365.25 * 86400.0), 1e-9)
    bpy = len(rets) / years
    ex = rets - rf_annual / bpy
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(bpy))


def _max_dd_from_eq(eq: pd.Series) -> float:
    """Max system drawdown (%) of a bar-level equity series."""
    vals = eq.to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    peak = np.maximum.accumulate(vals)
    denom = np.where(peak > 0, peak, np.nan)
    dd = (peak - vals) / denom
    return float(np.nanmax(dd) * 100.0)


def window_metrics(trades):
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    net = float(sum(t['pnl'] for t in trades))
    wr = wins / n * 100.0 if n else 0.0
    return {'trades': n, 'wins': wins, 'win_rate': wr, 'net': net}


# ===========================================================================
# Main class
# ===========================================================================
class S4003V5WalkForward:
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

        rows, trades_all, eq_segments = [], [], []
        cum = 0.0
        print(f"{'OOS day':>10} | {p1:>6} {p2:>6} | {'IS nTr':>6} {'IS wr%':>7} "
              f"{'IS net':>9} | {'OOS nTr':>7} {'OOS wr%':>7} {'OOS pnl':>9} "
              f"{'OOS Sharpe':>10} {'OOS MaxDD%':>10} {'cum':>9}")
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
            oos_trades, oos_eq = simulate_window(self.sig, day, day_end,
                                                 kw['w1'], kw['n1'], kw['e1'],
                                                 to_val(kw['stn1']), to_val(kw['sln1']))
            o_m = window_metrics(oos_trades)
            day_sharpe = _sharpe_from_eq(oos_eq)
            day_maxdd = _max_dd_from_eq(oos_eq)
            # rebase the day's equity onto the running OOS equity level
            # (keep the initial-capital base so returns/DD stay well-behaved)
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
                'oos_sharpe': day_sharpe, 'oos_maxdd': day_maxdd,
                'oos_cum_pnl': cum,
            })
            print(f"{str(day.date()):>10} | {str(v1):>6} {str(v2):>6} | "
                  f"{is_m['trades']:>6} {is_m['win_rate']:>7.2f} {is_m['net']:>9.2f} | "
                  f"{o_m['trades']:>7} {o_m['win_rate']:>7.2f} {o_m['net']:>9.2f} "
                  f"{day_sharpe:>10.2f} {day_maxdd:>10.2f} {cum:>9.2f}")
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
            chosen=', '.join(self.chosen_params)).to_csv(pv_path, index=False)

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

        # ---- HTML summary (per-OOS-period table incl. the 2 parameters) ----
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
            f"<td>{r['oos_sharpe']:.2f}</td><td>{r['oos_maxdd']:.2f}</td>"
            f"<td>{r['oos_cum_pnl']:.2f}</td></tr>"
            for _, r in self.wf_results.iterrows())
        fixed_txt = ', '.join(f"{k}={v}" for k, v in self.fixed_values.items())
        html = f"""<html><head><title>S4003_5 Walk-Forward Results</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;margin:12px 0;}}
th,td{{padding:4px 10px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body><h2>S4003_5 Walk-Forward Parameter Test</h2>
<p>Period: {ANALYSIS_START.date()} .. {self.df.index[-1].date()} |
rolling in-sample {IS_DAYS}d -&gt; out-of-sample 1d |
exit cost ${EXIT_COST:.2f}/trade</p>
<h3>Summary</h3>
<table>
<tr><th>Chosen walk-forward parameters</th><td>{p1}, {p2}</td></tr>
<tr><th>Fixed at backtest best</th><td>{fixed_txt}</td></tr>
<tr><th>Out-of-sample trades</th><td>{n} ({n / max(self.trading_days, 1):.2f}/day)</td></tr>
<tr><th>Out-of-sample win rate</th><td>{wr:.2f}% (target &gt;= {WIN_RATE_TARGET:.0f}%)</td></tr>
<tr><th>Out-of-sample net profit</th><td>{net:,.2f}</td></tr>
<tr><th>Out-of-sample Sharpe ratio</th><td>{overall_sharpe:.2f} (rf 3% annualized)</td></tr>
<tr><th>Out-of-sample max system drawdown</th><td>{overall_maxdd:.2f}%</td></tr>
</table>
<h3>Per out-of-sample period</h3>
<p>Per-day Sharpe / max drawdown are computed on that day's bar-level
mark-to-market equity (annualized, 3% risk-free).</p>
<table><tr><th>OOS date</th><th>{p1}</th><th>{p2}</th>
<th>IS trades</th><th>IS win%</th><th>IS net</th>
<th>OOS trades</th><th>OOS win%</th><th>OOS pnl</th>
<th>OOS Sharpe</th><th>OOS MaxDD%</th><th>cum pnl</th></tr>
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
              f"(target >= {WIN_RATE_TARGET:.0f}% -> "
              f"{'PASS' if wr >= WIN_RATE_TARGET else 'MISS'}), "
              f"net {net:,.2f}, freq {n / max(self.trading_days, 1):.2f}/day, "
              f"Sharpe {overall_sharpe:.2f}, max DD {overall_maxdd:.2f}%")
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
        description='S4003_5 walk-forward test of the 2 most valuable parameters')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--top-k', type=int, default=3)
    args = ap.parse_args()
    S4003V5WalkForward(ticker=args.ticker, top_k=args.top_k).run()


if __name__ == '__main__':
    main()
