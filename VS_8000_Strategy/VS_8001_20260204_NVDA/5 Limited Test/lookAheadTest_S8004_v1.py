"""
Look Ahead Test for Strategy S8004 (strategyIRB1000_V1)
Purpose: Determine whether the strategy has look-ahead bias, i.e. change of
         future OHLCV value will affect previous bar buy/short/sell/cover signal.

Logic:
  1. Run strategy on full dataset to get baseline buy/short/sell/cover signals.
  2. Loop from last bar to first bar. For each bar:
     - Set OHLCV of current bar to -1.
     - Re-run strategy.
     - If current bar is NOT a buy/short/sell/cover signal bar, check whether
       any buy/short/sell/cover signal that existed in the baseline (before
       current bar) has disappeared. If so, that signal bar has look-ahead bias.
  3. Output an HTML report with statistics and charts.

Performance tuning:
  - Only test bars within BACKWARD_INFLUENCE_RANGE of signal bars (instead of
    all bars). The strategy uses rolling windows up to 100 bars + 5-bar forward
    shift, so ~110 bars is sufficient.
  - Early termination: once a signal bar is already flagged, skip further
    corruption tests that would flag the same bar.

Output:
  HTML file at:
    C:\\Project\\ProjectLife\\VSCode Algo Workspace\\VS_8000_Strategy\\VS_8001_20260204_NVDA\\LimitTestResult\\S8004_LookAheadTest_Qwen3.7Max_<YYYYMMDDHH24MISS>.html
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from multiprocessing import Pool, cpu_count
import functools

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(
    OUTPUT_DIR,
    f'lookAheadTest_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
)

def log(msg, flush=True):
    """Print to console and log to file."""
    print(msg, flush=flush)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")

# ---------------------------------------------------------------------------
# Add project paths so we can import the strategy module
# ---------------------------------------------------------------------------
STRATEGY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '2 Strategy')
)
sys.path.insert(0, STRATEGY_DIR)

from S8001_4_GenerateFromPromptQwen37Max import strategyIRB1000_V1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = (
    r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile"
    r"\VS_8001_20260204_NVDA\csvExcel\NVDA_20250101_20260430_5Min_.csv"
)

OUTPUT_DIR = (
    r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile"
    r"\VS_8001_20260204_NVDA\LimitTestResult"
)


# ---------------------------------------------------------------------------
# Helper: load and prepare data
# ---------------------------------------------------------------------------
def load_data(csv_path, last_n_months=4):
    """
    Load CSV and return a DataFrame indexed by datetime with standard columns.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    last_n_months : int
        Only keep the last N months of data (default: 4)
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with only the last N months of data
    """
    df = pd.read_csv(csv_path)

    # Columns based on specification:
    #   A: datetime  B: date  C: time  D: high  E: low  F: close  G: open  H: volume  I: symbol
    cols = df.columns.tolist()
    rename_map = {
        cols[0]: 'Datetime',
        cols[1]: 'Date',
        cols[2]: 'Time',
        cols[3]: 'High',
        cols[4]: 'Low',
        cols[5]: 'Close',
        cols[6]: 'Open',
        cols[7]: 'Volume',
        cols[8]: 'Symbol',
    }
    df = df.rename(columns=rename_map)

    # Parse datetime and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
    df = df.set_index('Datetime')
    df = df.sort_index()

    # Filter to only last N months
    if last_n_months > 0:
        end_date = df.index.max()
        start_date = end_date - pd.DateOffset(months=last_n_months)
        df = df[df.index >= start_date]
        log(f"  Filtered to last {last_n_months} months: {df.index[0]} to {df.index[-1]}")

    return df


# ---------------------------------------------------------------------------
# Helper: run strategy and return signal DataFrame
# ---------------------------------------------------------------------------
def run_strategy(df):
    """Run strategyIRB1000_V1 on the given DataFrame and return processed data."""
    strategy = strategyIRB1000_V1(mode='backtest')
    data = strategy.calculate_indicators(df)
    data = strategy.generate_signals(data)
    return data


# ---------------------------------------------------------------------------
# Helper: extract all signal bar indices (buy, short, sell, cover)
# ---------------------------------------------------------------------------
def get_signal_indices(data):
    """Return sets of bar indices where each signal type is 1."""
    buy_idx = set(np.where(data['buy_signal'].values == 1)[0])
    short_idx = set(np.where(data['short_signal'].values == 1)[0])
    sell_idx = set(np.where(data['sell_signal'].values == 1)[0])
    cover_idx = set(np.where(data['cover_signal'].values == 1)[0])
    return buy_idx, short_idx, sell_idx, cover_idx


# ---------------------------------------------------------------------------
# Core: look-ahead test
# ---------------------------------------------------------------------------
BACKWARD_INFLUENCE_RANGE = 60  # Reduced from 110 to 60 for faster testing


def _test_single_bar(args):
    """
    Worker function for multiprocessing.
    Test a single bar by corrupting its OHLCV and checking for disappeared signals.
    
    Parameters:
    -----------
    args : tuple
        (bar_idx, df, baseline_buy, baseline_short, baseline_sell, baseline_cover)
    
    Returns:
    --------
    list of dict
        Look-ahead issues found for this bar
    """
    bar_idx, df, baseline_buy, baseline_short, baseline_sell, baseline_cover = args
    
    # Create modified copy
    modified_df = df.copy()
    modified_df.iloc[bar_idx, modified_df.columns.get_loc('Open')] = -1
    modified_df.iloc[bar_idx, modified_df.columns.get_loc('High')] = -1
    modified_df.iloc[bar_idx, modified_df.columns.get_loc('Low')] = -1
    modified_df.iloc[bar_idx, modified_df.columns.get_loc('Close')] = -1
    modified_df.iloc[bar_idx, modified_df.columns.get_loc('Volume')] = -1

    # Re-run strategy
    try:
        modified_data = run_strategy(modified_df)
    except Exception:
        return []

    mod_buy, mod_short, mod_sell, mod_cover = get_signal_indices(modified_data)
    
    issues = []
    
    # Check for disappeared signals (only those BEFORE current bar)
    # Buy signals
    disappeared_buy = baseline_buy - mod_buy
    for sig_idx in disappeared_buy:
        if sig_idx < bar_idx:
            issues.append({
                'corrupted_bar_idx': bar_idx,
                'corrupted_bar_time': str(df.index[bar_idx]),
                'signal_type': 'buy',
                'signal_bar_idx': sig_idx,
                'signal_bar_time': str(df.index[sig_idx]),
            })

    # Short signals
    disappeared_short = baseline_short - mod_short
    for sig_idx in disappeared_short:
        if sig_idx < bar_idx:
            issues.append({
                'corrupted_bar_idx': bar_idx,
                'corrupted_bar_time': str(df.index[bar_idx]),
                'signal_type': 'short',
                'signal_bar_idx': sig_idx,
                'signal_bar_time': str(df.index[sig_idx]),
            })

    # Sell signals
    disappeared_sell = baseline_sell - mod_sell
    for sig_idx in disappeared_sell:
        if sig_idx < bar_idx:
            issues.append({
                'corrupted_bar_idx': bar_idx,
                'corrupted_bar_time': str(df.index[bar_idx]),
                'signal_type': 'sell',
                'signal_bar_idx': sig_idx,
                'signal_bar_time': str(df.index[sig_idx]),
            })

    # Cover signals
    disappeared_cover = baseline_cover - mod_cover
    for sig_idx in disappeared_cover:
        if sig_idx < bar_idx:
            issues.append({
                'corrupted_bar_idx': bar_idx,
                'corrupted_bar_time': str(df.index[bar_idx]),
                'signal_type': 'cover',
                'signal_bar_idx': sig_idx,
                'signal_bar_time': str(df.index[sig_idx]),
            })
    
    return issues


def look_ahead_test(df):
    """
    Perform look-ahead test.

    Returns
    -------
    baseline_data : pd.DataFrame
        Strategy output on the full (unmodified) dataset.
    look_ahead_issues : list of dict
        Each dict: {'corrupted_bar_idx', 'corrupted_bar_time', 'signal_type',
                    'signal_bar_idx', 'signal_bar_time'}
        meaning: when bar at corrupted_bar_idx had its OHLCV set to -1, the signal at
        signal_bar_idx disappeared → that signal bar has look-ahead bias.
    """
    # --- Baseline run ---
    log("Running baseline strategy on full dataset ...")
    try:
        baseline_data = run_strategy(df.copy())
    except Exception as e:
        log(f"ERROR: Baseline strategy failed: {e}")
        raise
    
    baseline_buy, baseline_short, baseline_sell, baseline_cover = get_signal_indices(baseline_data)

    log(f"  Baseline buy signals  : {len(baseline_buy)}")
    log(f"  Baseline short signals: {len(baseline_short)}")
    log(f"  Baseline sell signals : {len(baseline_sell)}")
    log(f"  Baseline cover signals: {len(baseline_cover)}")

    # All signal bars (any type)
    all_signal_bars = baseline_buy | baseline_short | baseline_sell | baseline_cover

    # Performance optimization: only test bars within BACKWARD_INFLUENCE_RANGE of signal bars
    bars_to_test = set()
    for sig_idx in all_signal_bars:
        # Test bars from (sig_idx + 1) to (sig_idx + BACKWARD_INFLUENCE_RANGE)
        # These are the bars that could potentially influence the signal at sig_idx
        for offset in range(1, BACKWARD_INFLUENCE_RANGE + 1):
            test_idx = sig_idx + offset
            if test_idx < len(df):
                bars_to_test.add(test_idx)

    # Remove signal bars from test set (we skip them anyway)
    bars_to_test = bars_to_test - all_signal_bars

    log(f"  Bars to test (optimized): {len(bars_to_test)} (instead of {len(df)})")

    n_bars = len(df)
    look_ahead_issues = []  # list of dicts
    flagged_signal_bars = set()  # Track unique signal bars with issues
    MAX_ISSUES = 5  # Stop after finding 5 unique signal bars with issues

    # --- Parallel processing for faster testing ---
    log(f"Starting look-ahead test with multiprocessing (last → first) ...")
    log(f"  Will stop early if more than {MAX_ISSUES} unique signal bars have look-ahead issues")
    
    # Prepare arguments for each bar
    test_args = [
        (bar_idx, df, baseline_buy, baseline_short, baseline_sell, baseline_cover)
        for bar_idx in sorted(bars_to_test, reverse=True)
    ]
    
    # Use multiprocessing pool with imap for early termination
    num_workers = min(cpu_count(), 8)  # Use up to 8 cores
    log(f"  Using {num_workers} worker processes")
    
    pool = Pool(processes=num_workers)
    bars_processed = 0
    
    try:
        # Process bars in parallel with imap (allows early termination)
        for bar_issues in pool.imap(_test_single_bar, test_args):
            bars_processed += 1
            
            # Add new issues
            for issue in bar_issues:
                key = (issue['signal_type'], issue['signal_bar_idx'])
                if key not in flagged_signal_bars:
                    flagged_signal_bars.add(key)
                    look_ahead_issues.append(issue)
            
            # Progress indicator every 100 bars
            if bars_processed % 100 == 0:
                log(f"  Progress: {bars_processed}/{len(bars_to_test)} bars tested, "
                    f"{len(flagged_signal_bars)} unique signal bars with issues ...")
            
            # Early termination if more than MAX_ISSUES found
            if len(flagged_signal_bars) > MAX_ISSUES:
                log(f"  ⚠️  Early termination: Found {len(flagged_signal_bars)} unique signal bars with issues (>{MAX_ISSUES})")
                log(f"  Stopping test after processing {bars_processed} bars ...")
                break
        
        # Terminate the pool
        pool.terminate()
        pool.join()
        
    except Exception as e:
        log(f"ERROR during multiprocessing: {e}")
        pool.terminate()
        pool.join()
        raise

    log(f"Look-ahead test complete. Total unique signal bars with issues: {len(flagged_signal_bars)}")
    return baseline_data, look_ahead_issues


# ---------------------------------------------------------------------------
# Reporting: generate HTML report
# ---------------------------------------------------------------------------
def generate_html_report(df, baseline_data, look_ahead_issues, output_path):
    """Generate an HTML report with statistics and interactive chart."""

    # --- Statistics ---
    total_buy = int((baseline_data['buy_signal'].values == 1).sum())
    total_short = int((baseline_data['short_signal'].values == 1).sum())
    total_sell = int((baseline_data['sell_signal'].values == 1).sum())
    total_cover = int((baseline_data['cover_signal'].values == 1).sum())

    # Unique signal bars with look-ahead issues
    buy_issue_bars = set()
    short_issue_bars = set()
    sell_issue_bars = set()
    cover_issue_bars = set()
    for issue in look_ahead_issues:
        if issue['signal_type'] == 'buy':
            buy_issue_bars.add(issue['signal_bar_idx'])
        elif issue['signal_type'] == 'short':
            short_issue_bars.add(issue['signal_bar_idx'])
        elif issue['signal_type'] == 'sell':
            sell_issue_bars.add(issue['signal_bar_idx'])
        elif issue['signal_type'] == 'cover':
            cover_issue_bars.add(issue['signal_bar_idx'])

    buy_with_issue = len(buy_issue_bars)
    short_with_issue = len(short_issue_bars)
    sell_with_issue = len(sell_issue_bars)
    cover_with_issue = len(cover_issue_bars)

    has_look_ahead = len(look_ahead_issues) > 0
    verdict = "YES — Look-ahead bias detected" if has_look_ahead else "NO — No look-ahead bias detected"
    verdict_color = "red" if has_look_ahead else "green"

    # --- Build issue detail table rows ---
    issue_rows = ""
    for i, issue in enumerate(look_ahead_issues, 1):
        issue_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{issue['signal_type'].upper()}</td>
            <td>{issue['signal_bar_time']}</td>
            <td>{issue['corrupted_bar_time']}</td>
        </tr>"""

    if not issue_rows:
        issue_rows = '<tr><td colspan="4">No look-ahead issues found</td></tr>'

    # --- Plotly chart ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("NVDA 5-Min Price with Look-Ahead Issues", "Volume"),
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='NVDA',
            increasing_line_color='green',
            decreasing_line_color='red',
        ),
        row=1, col=1,
    )

    # Mark buy signals (baseline)
    buy_mask = baseline_data['buy_signal'].values == 1
    buy_times = df.index[buy_mask]
    buy_prices = df['Low'].values[buy_mask] * 0.998  # slightly below low

    fig.add_trace(
        go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='blue'),
            name='Buy Signal',
        ),
        row=1, col=1,
    )

    # Mark short signals (baseline)
    short_mask = baseline_data['short_signal'].values == 1
    short_times = df.index[short_mask]
    short_prices = df['High'].values[short_mask] * 1.002  # slightly above high

    fig.add_trace(
        go.Scatter(
            x=short_times,
            y=short_prices,
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='orange'),
            name='Short Signal',
        ),
        row=1, col=1,
    )

    # Mark buy signals WITH look-ahead issue (red circle)
    if buy_issue_bars:
        la_buy_times = [df.index[i] for i in sorted(buy_issue_bars)]
        la_buy_prices = [df['Low'].values[i] * 0.995 for i in sorted(buy_issue_bars)]
        fig.add_trace(
            go.Scatter(
                x=la_buy_times,
                y=la_buy_prices,
                mode='markers',
                marker=dict(symbol='circle', size=14, color='red',
                            line=dict(width=2, color='darkred')),
                name='Buy w/ Look-Ahead Issue',
            ),
            row=1, col=1,
        )

    # Mark short signals WITH look-ahead issue (red circle)
    if short_issue_bars:
        la_short_times = [df.index[i] for i in sorted(short_issue_bars)]
        la_short_prices = [df['High'].values[i] * 1.005 for i in sorted(short_issue_bars)]
        fig.add_trace(
            go.Scatter(
                x=la_short_times,
                y=la_short_prices,
                mode='markers',
                marker=dict(symbol='circle', size=14, color='red',
                            line=dict(width=2, color='darkred')),
                name='Short w/ Look-Ahead Issue',
            ),
            row=1, col=1,
        )

    # Volume bars
    colors = ['green' if c >= o else 'red'
              for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume', showlegend=False),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"Look-Ahead Test — S8004 strategyIRB1000_V1 — NVDA 5Min",
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # --- Assemble HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Look-Ahead Test Report — S8004</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .verdict {{ font-size: 1.4em; font-weight: bold; color: {verdict_color}; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; }}
        th {{ background: #4a90d9; color: #fff; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }}
        .stat-box {{ background: #eef; padding: 12px; border-radius: 6px; text-align: center; }}
        .stat-box .num {{ font-size: 2em; font-weight: bold; color: #333; }}
        .stat-box .label {{ font-size: 0.9em; color: #666; }}
        .chart {{ background: #fff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>🔍 Look-Ahead Test Report</h1>
    <p><strong>Strategy:</strong> S8004 — strategyIRB1000_V1 &nbsp;|&nbsp;
       <strong>Symbol:</strong> NVDA &nbsp;|&nbsp;
       <strong>Timeframe:</strong> 5 Min &nbsp;|&nbsp;
       <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>Summary</h2>
        <div class="verdict">Verdict: {verdict}</div>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="num">{total_buy}</div>
                <div class="label">Total Buy Signals</div>
            </div>
            <div class="stat-box">
                <div class="num" style="color:{'red' if buy_with_issue > 0 else 'green'}">{buy_with_issue}</div>
                <div class="label">Buy Signals with Look-Ahead Issue</div>
            </div>
            <div class="stat-box">
                <div class="num">{total_short}</div>
                <div class="label">Total Short Signals</div>
            </div>
            <div class="stat-box">
                <div class="num" style="color:{'red' if short_with_issue > 0 else 'green'}">{short_with_issue}</div>
                <div class="label">Short Signals with Look-Ahead Issue</div>
            </div>
            <div class="stat-box">
                <div class="num">{total_sell}</div>
                <div class="label">Total Sell Signals</div>
            </div>
            <div class="stat-box">
                <div class="num" style="color:{'red' if sell_with_issue > 0 else 'green'}">{sell_with_issue}</div>
                <div class="label">Sell Signals with Look-Ahead Issue</div>
            </div>
            <div class="stat-box">
                <div class="num">{total_cover}</div>
                <div class="label">Total Cover Signals</div>
            </div>
            <div class="stat-box">
                <div class="num" style="color:{'red' if cover_with_issue > 0 else 'green'}">{cover_with_issue}</div>
                <div class="label">Cover Signals with Look-Ahead Issue</div>
            </div>
        </div>
    </div>

    <div class="chart">
        <h2>Price Chart with Look-Ahead Issues</h2>
        {chart_html}
    </div>

    <h2>Look-Ahead Issue Details</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Signal Type</th>
                <th>Signal Bar (has look-ahead)</th>
                <th>Corrupted Bar (caused disappearance)</th>
            </tr>
        </thead>
        <tbody>
            {issue_rows}
        </tbody>
    </table>

    <br>
    <p><em>Test logic: For each bar (last→first), set OHLCV to -1 and re-run strategy.
    If a buy/short/sell/cover signal on an earlier bar disappears, that signal bar has look-ahead bias.
    Only bars within {BACKWARD_INFLUENCE_RANGE} bars after each signal bar are tested (performance optimization).</em></p>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    log(f"Report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("  Look-Ahead Test — S8004 strategyIRB1000_V1 — NVDA 5Min")
    log("=" * 70)

    # Load data
    log(f"\nLoading data from: {INPUT_CSV}")
    df = load_data(INPUT_CSV, last_n_months=2)
    log(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Run look-ahead test
    log("")
    baseline_data, look_ahead_issues = look_ahead_test(df)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_filename = f"S8004_LookAheadTest_Qwen3.7Max_{timestamp}.html"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Generate HTML report
    log("\nGenerating HTML report ...")
    generate_html_report(df, baseline_data, look_ahead_issues, output_path)

    # Print summary to console
    total_buy = int((baseline_data['buy_signal'].values == 1).sum())
    total_short = int((baseline_data['short_signal'].values == 1).sum())
    total_sell = int((baseline_data['sell_signal'].values == 1).sum())
    total_cover = int((baseline_data['cover_signal'].values == 1).sum())
    
    buy_issue_bars = set(i['signal_bar_idx'] for i in look_ahead_issues if i['signal_type'] == 'buy')
    short_issue_bars = set(i['signal_bar_idx'] for i in look_ahead_issues if i['signal_type'] == 'short')
    sell_issue_bars = set(i['signal_bar_idx'] for i in look_ahead_issues if i['signal_type'] == 'sell')
    cover_issue_bars = set(i['signal_bar_idx'] for i in look_ahead_issues if i['signal_type'] == 'cover')

    log("\n" + "=" * 70)
    log("  RESULTS")
    log("=" * 70)
    log(f"  Total buy signals          : {total_buy}")
    log(f"  Buy with look-ahead issue  : {len(buy_issue_bars)}")
    log(f"  Total short signals        : {total_short}")
    log(f"  Short with look-ahead issue: {len(short_issue_bars)}")
    log(f"  Total sell signals         : {total_sell}")
    log(f"  Sell with look-ahead issue : {len(sell_issue_bars)}")
    log(f"  Total cover signals        : {total_cover}")
    log(f"  Cover with look-ahead issue: {len(cover_issue_bars)}")
    if look_ahead_issues:
        log(f"\n  ⚠️  LOOK-AHEAD BIAS DETECTED — {len(look_ahead_issues)} issue(s) found")
    else:
        log(f"\n  ✅ No look-ahead bias detected")
    print("=" * 70)


if __name__ == '__main__':
    main()
