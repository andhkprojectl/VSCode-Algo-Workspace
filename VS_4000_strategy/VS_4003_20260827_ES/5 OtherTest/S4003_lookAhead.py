"""
S4003_lookAhead.py
==================
Look-ahead (future-data leakage) test for strategy class strategyS4003
(`VS_4000_strategy/VS_4003_20260827_ES/2 Strategy/S4003_strategy.py`).
TASKS.md T3.1. No imports from `1 Prelimary Test`.

Approach follows `VS_0003_test/otherTest.py` (truncation comparison, adapted to
DataFrames from MariaDB instead of a CSV file) and
`VS_4000_strategy/VS_4004_20260204_NVDA/5 OtherTest/lookAheadTest_S8004_v1.py`
(HTML report style), plus execution-timing and previous-bar checks.

Checks performed:
  1. Signal causality (truncation test): for sampled bars t, re-run
     strategyS4003.generate_signals() on data truncated to bar t and compare the
     signal columns (signal / buy00 / short00) of the last `truncationCompareBars`
     rows against the baseline full-series run. Any difference at bars <= t means
     the signal used future data.
  2. Indicator causality: same truncated runs - compare the computed indicator
     columns at bar t against the baseline. Differences mean an indicator reads
     future bars.
  3. Next-bar execution: for every trade from strategyS4003._simulate(), verify
     the entry bar is the bar AFTER the raw signal bar (buy00/short00 at t-1,
     execution signal at t, fill at bar t Open-derived limit price inside bar t's
     Low/High range - never a same-bar fill on the signal bar's Close).
  4. Previous-bar references: the strategy intentionally uses previous-bar values
     (buySig = buy00 shifted 1 bar; entry limit offset = ATR14[t-1]/e1; stop anchor
     = max ATR14 of the 3 bars before the entry bar). Verify with vectorized
     identities that these really use t-1 (or earlier) data, and prove by
     truncation that the stop anchor of sampled trades is computable from data
     strictly before the entry bar.

Parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml` section
`LookAhead Test` (yaml.safe_load). Data from MariaDB table `tableName`
(default ticker1Min); credentials from `VS_0002_config/.env` (never hardcoded).

Output: HTML report `S4003_LookAheadTest_<YYYYMMDDHHMMSS>.html` in the config
`outputFolder`, stating whether the strategy contains a look-ahead issue, with
details of any issue found.

Usage:
    python S4003_lookAhead.py [--config PATH_TO_CONFIG_YAML]
"""

import argparse
import html as html_mod
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import mysql.connector
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = WORKSPACE_ROOT / 'VS_4000_strategy' / 'VS_4003_20260827_ES' / 'config.yaml'
ENV_PATH = WORKSPACE_ROOT / 'VS_0002_config' / '.env'

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '2 Strategy'))
from S4003_strategy import strategyS4003, loadStrategyConfig  # noqa: E402

DT_FORMAT = '%d/%m/%Y %H:%M'

# columns compared in the truncation test
SIGNAL_COLS = ['signal', 'buy00', 'short00']
INDICATOR_COLS = ['atr14', 'max3BarAtr14_1', 'vwap50', 'regSlop3Vwap50Bar',
                  'regSlop5vwap50Bar', 'regSlop5vwap50BarL0Sum15',
                  'llvBar5_1', 'llvBar5_2', 'crossUpVWap50', 'crossDownVWap50',
                  'largeBullishBody', 'largeBearishBody']

MAX_REPORT_ROWS = 300  # cap detail rows per check in the HTML report


# ---------------------------------------------------------------------------
# Config / data loading
# ---------------------------------------------------------------------------
def loadLookAheadConfig(config_path=None) -> dict:
    """Return the `LookAhead Test` section of the local strategy config."""
    config_path = Path(config_path) if config_path else CONFIG_PATH
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('LookAhead Test') or {}


def load1MinDataFromDb(symbol, table, start_date=None, end_date=None) -> pd.DataFrame:
    """Load 1-min bars from MariaDB (adapted from
    VS_0006_dataFunc/VS_6004_GetDataFromMariaDb/VS_6004_GetDataFromMariaDb.py).
    Returns DataFrame [open, high, low, close, volume] with DatetimeIndex asc."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=str(ENV_PATH), override=False)
    if not re.fullmatch(r'[A-Za-z0-9_]+', str(table)):
        raise ValueError(f"Invalid table name in config: {table!r}")
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )
    try:
        q = (f"SELECT datetime1, open, high, low, close, volume FROM {table} "
             "WHERE ticker = %s")
        params = [symbol]
        if start_date:
            q += " AND datetime1 >= %s"
            params.append(start_date)
        if end_date:
            q += " AND datetime1 <= %s"
            params.append(end_date)
        q += " ORDER BY datetime1 ASC"
        cur = conn.cursor()
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=['datetime1', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty:
        raise ValueError(f"No rows returned from {table} for ticker {symbol!r} "
                         f"in range [{start_date}, {end_date}]")
    df['datetime1'] = pd.to_datetime(df['datetime1'])
    df = df.set_index('datetime1')
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vals_equal(a, b) -> bool:
    """NaN-safe scalar comparison (booleans exact, floats with tight tolerance)."""
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        try:
            return bool(a) == bool(b)
        except (TypeError, ValueError):
            return False
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-12,
                               equal_nan=True))
    except (TypeError, ValueError):
        return a == b


def _select_truncation_points(n, signal_bars, min_idx, max_tests):
    """Evenly spaced truncation points across [min_idx, n), plus all baseline
    signal bars (sub-sampled evenly if there are more than max_tests)."""
    even = set(np.linspace(min_idx, n - 1, max_tests).astype(int).tolist())
    sig = [int(i) for i in signal_bars if i >= min_idx]
    if len(sig) > max_tests:
        keep = np.linspace(0, len(sig) - 1, max_tests).astype(int)
        sig = [sig[k] for k in keep]
    return sorted(even | set(sig))


# ---------------------------------------------------------------------------
# Check 1 + 2: truncation test (signal and indicator causality)
# ---------------------------------------------------------------------------
def check_truncation(df, baseline, max_tests, compare_bars, min_idx):
    """Re-run generate_signals on series truncated at sampled bar t; any value at
    bars <= t that differs from the full-series baseline indicates look-ahead."""
    n = len(df)
    min_idx = min(min_idx, n - 1)
    signal_bars = np.where(baseline['signal'].to_numpy() != 0)[0]
    points = _select_truncation_points(n, signal_bars, min_idx, max_tests)
    print(f"  Truncation points to test: {len(points)} "
          f"(incl. {min(len(signal_bars), max_tests)} signal bars)")

    issues = []
    mismatch_count = 0
    for k, t in enumerate(points, 1):
        res = strategyS4003.generate_signals(df.iloc[:t + 1])
        lo = max(0, t - compare_bars + 1)

        # signal columns over the last `compare_bars` rows of the truncated run
        for col in SIGNAL_COLS:
            base_vals = baseline[col].iloc[lo:t + 1].to_numpy()
            trunc_vals = res[col].iloc[lo:].to_numpy()
            for off in np.nonzero(base_vals != trunc_vals)[0]:
                bar_i = lo + int(off)
                mismatch_count += 1
                if len(issues) < MAX_REPORT_ROWS:
                    issues.append({
                        'Truncated At': str(df.index[t]),
                        'Mismatched Bar': str(df.index[bar_i]),
                        'Column': col,
                        'Baseline (full series)': str(base_vals[off]),
                        'Truncated Run': str(trunc_vals[off]),
                    })

        # indicator columns at bar t only
        for col in INDICATOR_COLS:
            bv = baseline[col].iloc[t]
            tv = res[col].iloc[-1]
            if not _vals_equal(bv, tv):
                mismatch_count += 1
                if len(issues) < MAX_REPORT_ROWS:
                    issues.append({
                        'Truncated At': str(df.index[t]),
                        'Mismatched Bar': str(df.index[t]),
                        'Column': col,
                        'Baseline (full series)': str(bv),
                        'Truncated Run': str(tv),
                    })

        if k % 25 == 0 or k == len(points):
            print(f"  Progress: {k}/{len(points)} truncation runs, "
                  f"{mismatch_count} mismatches so far")

    print(f"  Truncation test done: {len(points)} runs, {mismatch_count} mismatches")
    return issues, mismatch_count, len(points)


# ---------------------------------------------------------------------------
# Check 3: next-bar execution timing
# ---------------------------------------------------------------------------
def check_execution_timing(sig, trades):
    """Every entry must be on the bar AFTER the raw signal bar, at an
    Open-derived price inside the entry bar's Low/High range (no same-bar fill
    on the signal bar's Close)."""
    issues = []
    O = sig['Open'].to_numpy(float)
    H = sig['High'].to_numpy(float)
    L = sig['Low'].to_numpy(float)
    atr = sig['atr14'].to_numpy(float)
    frac = strategyS4003.entry_limit_atr_frac
    limit_mode = (strategyS4003.entry_price_mode == 'limit')

    for tr in trades:
        e = int(tr['entry_bar'])
        is_long = (tr['type'] == 'Long')
        raw_col = 'buy00' if is_long else 'short00'
        sig_col = 'buySig' if is_long else 'shortSig'

        def _issue(rule, detail):
            issues.append({
                'Trade': f"{tr['type']} entry {tr['entry_dt']}",
                'Rule': rule,
                'Detail': detail,
            })

        if e < 1:
            _issue('entry bar after signal bar', f'entry_bar={e} has no previous bar')
            continue

        signal_dt = sig.index[e - 1]
        # 1) raw signal must be on bar e-1, execution signal on bar e
        if not bool(sig[raw_col].iloc[e - 1]):
            _issue('raw signal on previous bar',
                   f'{raw_col}[{signal_dt}] is False but trade entered at bar {e}')
        if int(sig[sig_col].iloc[e]) != 1:
            _issue('execution signal on entry bar',
                   f'{sig_col}[{sig.index[e]}] != 1')
        if not (tr['entry_dt'] > signal_dt):
            _issue('entry after signal bar',
                   f'entry_dt {tr["entry_dt"]} not after signal bar {signal_dt}')

        # 2) fill price must be the bar-e Open-derived limit price (limit mode)
        #    or the plain bar-e Open (open mode) - never the signal bar Close
        if limit_mode and not np.isnan(atr[e - 1]):
            off = atr[e - 1] / frac
            expected = max(O[e] - off, L[e]) if is_long else min(O[e] + off, H[e])
        else:
            expected = O[e]
        if not np.isclose(tr['entry_price'], expected, rtol=1e-9, atol=1e-9):
            _issue('next-bar Open-derived fill',
                   f'entry_price {tr["entry_price"]} != expected {expected} '
                   f'(mode={strategyS4003.entry_price_mode}, ATR14[prev bar]='
                   f'{atr[e - 1]:.4f})')
        if not (L[e] - 1e-9 <= tr['entry_price'] <= H[e] + 1e-9):
            _issue('fill inside entry bar range',
                   f'entry_price {tr["entry_price"]} outside '
                   f'[{L[e]}, {H[e]}] of entry bar')

    print(f"  Execution timing check done: {len(trades)} trades, {len(issues)} issues")
    return issues


# ---------------------------------------------------------------------------
# Check 4: intentional previous-bar references really use <= t-1 data
# ---------------------------------------------------------------------------
def check_previous_bar_usage(df, sig, trades, max_trades):
    issues = []

    # (a) vectorized identities over the whole series -----------------------
    exp_buy = sig['buy00'].shift(1).fillna(False).astype(int)
    bad = np.nonzero(sig['buySig'].to_numpy() != exp_buy.to_numpy())[0]
    for i in bad[:MAX_REPORT_ROWS]:
        issues.append({'Check': 'buySig == buy00 shifted 1 bar',
                       'Bar': str(sig.index[i]),
                       'Detail': f"buySig={sig['buySig'].iloc[i]} but "
                                 f"buy00[prev]={bool(sig['buy00'].iloc[i - 1]) if i > 0 else 'n/a'}"})

    exp_short = sig['short00'].shift(1).fillna(False).astype(int)
    bad = np.nonzero(sig['shortSig'].to_numpy() != exp_short.to_numpy())[0]
    for i in bad[:MAX_REPORT_ROWS]:
        issues.append({'Check': 'shortSig == short00 shifted 1 bar',
                       'Bar': str(sig.index[i]),
                       'Detail': f"shortSig={sig['shortSig'].iloc[i]} but "
                                 f"short00[prev]={bool(sig['short00'].iloc[i - 1]) if i > 0 else 'n/a'}"})

    # stop anchor max3BarAtr14_1[t] must equal max(atr14[t-3 .. t-1]) (<= t-1 data)
    exp_max3 = sig['atr14'].shift(1).rolling(3, min_periods=1).max()
    neq = ~np.isclose(sig['max3BarAtr14_1'].to_numpy(float), exp_max3.to_numpy(float),
                      rtol=1e-9, atol=1e-12, equal_nan=True)
    for i in np.nonzero(neq)[0][:MAX_REPORT_ROWS]:
        issues.append({'Check': 'stop anchor = max ATR14 of 3 bars BEFORE bar t',
                       'Bar': str(sig.index[i]),
                       'Detail': f"max3BarAtr14_1={sig['max3BarAtr14_1'].iloc[i]} but "
                                 f"max(atr14[t-3..t-1])={exp_max3.iloc[i]}"})

    # (b) truncation proof for sampled trade entry bars ----------------------
    # Recompute on data ending at bar e-1: the entry limit offset (ATR14[e-1])
    # and the stop anchor (max of last 3 ATR14 before e) must be reproducible
    # without any data from bar e onward.
    entry_bars = sorted({int(tr['entry_bar']) for tr in trades if int(tr['entry_bar']) >= 1})
    if len(entry_bars) > max_trades:
        keep = np.linspace(0, len(entry_bars) - 1, max_trades).astype(int)
        entry_bars = [entry_bars[k] for k in keep]
    for e in entry_bars:
        res = strategyS4003.generate_signals(df.iloc[:e])  # data <= e-1 only
        atr_prev = res['atr14'].iloc[-1]
        if not _vals_equal(atr_prev, sig['atr14'].iloc[e - 1]):
            issues.append({'Check': 'entry ATR14[prev bar] computable without bar t',
                           'Bar': str(sig.index[e]),
                           'Detail': f'ATR14 from data <= t-1 = {atr_prev}, '
                                     f'baseline = {sig["atr14"].iloc[e - 1]}'})
        max3_prev = res['atr14'].iloc[-3:].max()
        if not _vals_equal(max3_prev, sig['max3BarAtr14_1'].iloc[e]):
            issues.append({'Check': 'stop anchor computable without entry-bar data',
                           'Bar': str(sig.index[e]),
                           'Detail': f'max3 ATR14 from data <= t-1 = {max3_prev}, '
                                     f'baseline max3BarAtr14_1[t] = '
                                     f'{sig["max3BarAtr14_1"].iloc[e]}'})

    print(f"  Previous-bar usage check done: {len(entry_bars)} trade entry bars "
          f"re-verified by truncation, {len(issues)} issues total")
    return issues


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _issues_table(issues, empty_text):
    if not issues:
        return f'<p>{empty_text}</p>'
    cols = list(issues[0].keys())
    head = ''.join(f'<th>{html_mod.escape(c)}</th>' for c in cols)
    rows = ''
    for i, iss in enumerate(issues, 1):
        tds = ''.join(f'<td>{html_mod.escape(str(iss.get(c, "")))}</td>' for c in cols)
        rows += f'<tr><td>{i}</td>{tds}</tr>'
    return (f'<table><thead><tr><th>#</th>{head}</tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def generate_html_report(context, results, output_path):
    has_issue = results['total_issues'] > 0
    verdict = ('YES - LOOK-AHEAD ISSUE DETECTED' if has_issue
               else 'NO - no look-ahead issue detected')
    verdict_color = '#c0392b' if has_issue else '#1e8449'

    def stat_box(num, label, ok=True):
        color = '#1e8449' if ok else '#c0392b'
        return (f'<div class="stat-box"><div class="num" style="color:{color}">'
                f'{num}</div><div class="label">{label}</div></div>')

    boxes = ''.join([
        stat_box(context['n_bars'], 'Bars loaded'),
        stat_box(results['n_trunc_runs'], 'Truncation runs'),
        stat_box(results['n_trunc_mismatch'], 'Truncation mismatches',
                 ok=results['n_trunc_mismatch'] == 0),
        stat_box(results['n_trades'], 'Trades checked'),
        stat_box(len(results['exec_issues']), 'Execution timing issues',
                 ok=len(results['exec_issues']) == 0),
        stat_box(len(results['prevbar_issues']), 'Previous-bar issues',
                 ok=len(results['prevbar_issues']) == 0),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Look-Ahead Test Report - S4003</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f5f5f5; }}
 h1 {{ color: #333; }}
 .summary {{ background: #fff; padding: 20px; border-radius: 8px;
             box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
 .verdict {{ font-size: 1.4em; font-weight: bold; color: {verdict_color}; margin: 10px 0; }}
 table {{ border-collapse: collapse; width: 100%; background: #fff; margin-bottom: 25px; }}
 th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 0.9em; }}
 th {{ background: #4a90d9; color: #fff; }}
 tr:nth-child(even) {{ background: #f9f9f9; }}
 .stat-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 15px 0; }}
 .stat-box {{ background: #eef; padding: 12px; border-radius: 6px; text-align: center; }}
 .stat-box .num {{ font-size: 2em; font-weight: bold; }}
 .stat-box .label {{ font-size: 0.9em; color: #666; }}
 .note {{ background:#fff; padding:15px 20px; border-radius:8px; color:#444; }}
</style>
</head>
<body>
<h1>Look-Ahead Test Report</h1>
<p><strong>Strategy:</strong> S4003 - strategyS4003 (ES 1-min price action) &nbsp;|&nbsp;
   <strong>Symbol:</strong> {html_mod.escape(str(context['symbol']))} &nbsp;|&nbsp;
   <strong>Data:</strong> MariaDB {html_mod.escape(str(context['table']))} &nbsp;|&nbsp;
   <strong>Range:</strong> {context['data_start']} - {context['data_end']} &nbsp;|&nbsp;
   <strong>Generated:</strong> {datetime.now().strftime(DT_FORMAT)}</p>

<div class="summary">
 <h2>Summary</h2>
 <div class="verdict">Verdict: {verdict}</div>
 <div class="stat-grid">{boxes}</div>
</div>

<h2>1. Signal Causality (truncation test)</h2>
<p>Signals re-computed on data truncated at bar t must equal the full-series
baseline at all bars &lt;= t. Compared columns: signal, buy00, short00
(last {context['compare_bars']} bars per run) and indicator columns
({', '.join(INDICATOR_COLS)}) at bar t.</p>
{_issues_table(results['trunc_issues'], 'No mismatches - signals and indicators depend only on past data.')}

<h2>2. Next-Bar Execution Timing</h2>
<p>Every trade must enter on the bar AFTER the raw signal bar (buy00/short00 at
t-1, execution at bar t), at an Open-derived limit price inside the entry bar's
Low/High range - never a same-bar fill on the signal bar's Close.</p>
{_issues_table(results['exec_issues'], 'All trades enter on the bar after the signal bar at an Open-derived price.')}

<h2>3. Intentional Previous-Bar References</h2>
<p>The strategy deliberately uses previous-bar values: buySig/shortSig = raw
signal shifted 1 bar; entry limit offset = ATR14[t-1]/{context['atr_frac']};
stop/profit anchor = max ATR14 of the 3 bars before the entry bar. These were
verified to really use only &lt;= t-1 data (vectorized identities plus truncation
proof on sampled trade entry bars).</p>
{_issues_table(results['prevbar_issues'], 'All previous-bar references verified to use only data before bar t.')}

<div class="note">
<strong>Methodology:</strong> baseline = strategyS4003.generate_signals() on the
full loaded series; truncation runs re-compute signals on df.iloc[:t+1] for
{results['n_trunc_runs']} sampled bars (evenly spaced + all baseline signal bars).
Execution-timing and previous-bar checks use strategyS4003._simulate() trades
(entries gated to &gt;= strategy startDate {context['strategy_start']}).
A single mismatch in any check flips the verdict to look-ahead issue detected.
</div>
</body>
</html>"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path=None):
    cfg = loadLookAheadConfig(config_path)
    strategyS4003.applyStrategyConfig(loadStrategyConfig(config_path))

    symbol = cfg.get('symbolName', 'ES')
    table = cfg.get('tableName', 'ticker1Min')
    start_date = cfg.get('startDate') or None
    end_date = cfg.get('endDate') or None
    output_folder = cfg.get('outputFolder')
    if not output_folder:
        raise KeyError("Config section 'LookAhead Test' is missing key 'outputFolder'")
    max_tests = int(cfg.get('maxTruncationTests', 200))
    compare_bars = int(cfg.get('truncationCompareBars', 50))
    min_idx = int(cfg.get('minTestBarIndex', 200))
    max_prev_trades = int(cfg.get('maxPrevBarTradesChecked', 25))

    print("=" * 70)
    print("  LOOK-AHEAD TEST - strategyS4003 (ES 1-min)")
    print("=" * 70)
    print(f"Config      : {config_path or CONFIG_PATH}")
    print(f"Symbol      : {symbol} | Table: {table} | Range: {start_date} - {end_date}")

    print("\nLoading data from MariaDB ...")
    df = load1MinDataFromDb(symbol, table, start_date, end_date)
    print(f"  Loaded {len(df)} bars: {df.index[0]} - {df.index[-1]}")

    print("\nRunning baseline generate_signals on full series ...")
    baseline = strategyS4003.generate_signals(df)
    n_sig = int((baseline['signal'] != 0).sum())
    print(f"  Baseline execution signals: {n_sig} "
          f"(buy={int((baseline['signal'] == 1).sum())}, "
          f"short={int((baseline['signal'] == -1).sum())})")

    print("\n[1/3] Truncation test (signal + indicator causality) ...")
    trunc_issues, n_mismatch, n_runs = check_truncation(
        df, baseline, max_tests, compare_bars, min_idx)

    print("\n[2/3] Trade simulation for execution-timing check ...")
    sig = baseline  # generate_signals returns _prepare output + signal column
    trade_start = pd.Timestamp(strategyS4003.start_date)
    gate = trade_start if sig.index[0] < trade_start else None
    trades, _equity, _in_market = strategyS4003._simulate(sig, trade_start_ts=gate)
    print(f"  Simulated trades: {len(trades)}")
    exec_issues = check_execution_timing(sig, trades)

    print("\n[3/3] Previous-bar reference checks ...")
    prevbar_issues = check_previous_bar_usage(df, sig, trades, max_prev_trades)

    total_issues = n_mismatch + len(exec_issues) + len(prevbar_issues)
    results = {
        'trunc_issues': trunc_issues,
        'n_trunc_mismatch': n_mismatch,
        'n_trunc_runs': n_runs,
        'exec_issues': exec_issues,
        'prevbar_issues': prevbar_issues,
        'n_trades': len(trades),
        'total_issues': total_issues,
    }
    context = {
        'symbol': symbol, 'table': table, 'n_bars': len(df),
        'data_start': df.index[0], 'data_end': df.index[-1],
        'compare_bars': compare_bars, 'atr_frac': strategyS4003.entry_limit_atr_frac,
        'strategy_start': strategyS4003.start_date,
    }

    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    out_path = Path(output_folder) / f'S4003_LookAheadTest_{ts}.html'
    generate_html_report(context, results, out_path)

    print("\n" + "=" * 70)
    if total_issues:
        print(f"  VERDICT: LOOK-AHEAD ISSUE DETECTED - {total_issues} issue(s)")
        print(f"    truncation mismatches : {n_mismatch}")
        print(f"    execution timing      : {len(exec_issues)}")
        print(f"    previous-bar checks   : {len(prevbar_issues)}")
    else:
        print("  VERDICT: NO look-ahead issue detected")
    print(f"  Report: {out_path}")
    print("=" * 70)
    return total_issues == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S4003 look-ahead test (T3.1)')
    parser.add_argument('--config', default=None,
                        help='Path to config.yaml (default: strategy folder config)')
    args = parser.parse_args()
    main(config_path=args.config)
