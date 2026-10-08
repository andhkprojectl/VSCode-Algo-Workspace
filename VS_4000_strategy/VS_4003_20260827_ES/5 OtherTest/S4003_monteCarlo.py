"""
S4003_monteCarlo.py
===================
Monte Carlo test for strategyS4003 (VS_4003_20260827_ES). TASKS.md T3.2.

Runs the strategy class in `2 Strategy/S4003_strategy.py` on the configured
period (MariaDB ticker1Min data) to obtain the real per-trade P&L series, then
bootstraps `numRuns` simulated equity paths by resampling the trade sequence
with replacement (same approach as VS_0003_test/monteCarloSimulation.py) to
estimate the distribution of final equity, annual return and max drawdown.

Parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml` section
`MonteCarlo Test` (symbolName, tableName, startDate, endDate, numRuns,
randomSeed, initCapital, commissionPerTrade, outputFolder). Strategy signal
parameters come from the `Strategy` section of the same file. Empty
startDate/endDate = full DB range. No imports from `1 Prelimary Test`.

Output: one html file in `outputFolder` with a percentile table (99/95/90/75/50
percent) of final equity, annual return, max drawdown ($) and max drawdown (%),
plus an equity chart and a drawdown chart.

Usage:
    python S4003_monteCarlo.py [--config PATH_TO_CONFIG_YAML]

Import-safe: no DB connections and no config access at import time.
"""

import argparse
import base64
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2 Strategy"))
from S4003_strategy import strategyS4003, loadStrategyConfig

DT_FORMAT = '%d/%m/%Y %H:%M'
PCTILES = [99, 95, 90, 75, 50]
NO_GATE_DATE = '1900-01-01 00:00:00'   # start_date far in the past -> no entry gating


# ===========================================================================
# Config
# ===========================================================================
def loadMonteCarloConfig(config_path=None) -> dict:
    """Return the `MonteCarlo Test` section of the local strategy config."""
    if config_path is None:
        config_path = (WORKSPACE_ROOT / 'VS_4000_strategy' /
                       'VS_4003_20260827_ES' / 'config.yaml')
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('MonteCarlo Test') or {}


def applyMonteCarloConfig(mc: dict, config_path=None) -> dict:
    """Apply strategy params (`Strategy` section) with the Monte Carlo run
    window/cost overrides onto strategyS4003 class attributes. Returns the
    merged config dict used."""
    merged = dict(loadStrategyConfig(config_path) or {})
    merged['startDate'] = str(mc['startDate']) if mc.get('startDate') else NO_GATE_DATE
    for key in ('initCapital', 'commissionPerTrade'):
        if mc.get(key) is not None:
            merged[key] = float(mc[key])
    strategyS4003.applyStrategyConfig(merged)
    return merged


# ===========================================================================
# Data: MariaDB ticker1Min (adapted from
# VS_0006_dataFunc/VS_6004_GetDataFromMariaDb/VS_6004_GetDataFromMariaDb.py)
# ===========================================================================
def _load_db_env():
    from dotenv import load_dotenv
    env_path = WORKSPACE_ROOT / 'VS_0002_config' / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)


def load_1min_data_from_db(ticker='ES', table_name='ticker1Min',
                           start_date=None, end_date=None) -> pd.DataFrame:
    """Load 1-min bars with columns [open, high, low, close, volume] and a
    DatetimeIndex, sorted ascending. start_date/end_date are inclusive bounds
    on datetime1 (None = unbounded)."""
    import mysql.connector
    if not re.fullmatch(r'[A-Za-z0-9_]+', str(table_name)):
        raise ValueError(f'invalid table name: {table_name!r}')
    _load_db_env()
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '3306')),
        user=os.getenv('DB_USER', 'ibUser1'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'IBTradingDb'),
    )
    try:
        cur = conn.cursor()
        q = (f'SELECT datetime1, open, high, low, close, volume FROM {table_name} '
             'WHERE ticker = %s')
        params = [ticker]
        if start_date:
            q += ' AND datetime1 >= %s'
            params.append(str(start_date))
        if end_date:
            q += ' AND datetime1 <= %s'
            params.append(str(end_date))
        q += ' ORDER BY datetime1 ASC'
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=['datetime1', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime1'] = pd.to_datetime(df['datetime1'])
    df = df.set_index('datetime1')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df


# ===========================================================================
# Monte Carlo engine (bootstrap resampling of the real trade P&L sequence,
# same approach as VS_0003_test/monteCarloSimulation.py monteCarloSimulation1)
# ===========================================================================
def run_monte_carlo(trade_pnl, num_runs, init_balance, years, seed=None) -> dict:
    """Resample the per-trade P&L series with replacement `num_runs` times,
    build an equity path per run, and collect final equity / annual return /
    max drawdown distributions. `years` is the length of the real test window
    (for annualizing). Returns a dict of numpy arrays."""
    pnl = np.asarray(trade_pnl, dtype=float)
    n = pnl.size
    rng = np.random.default_rng(seed)
    sim = pnl[rng.integers(0, n, size=(int(num_runs), n))]
    eq = init_balance + np.cumsum(sim, axis=1)
    eq0 = np.concatenate([np.full((int(num_runs), 1), init_balance), eq], axis=1)

    peak = np.maximum.accumulate(eq0, axis=1)
    dd_usd = peak - eq0
    dd_pct = np.where(peak > 0, dd_usd / np.where(peak > 0, peak, 1.0), 0.0)

    final = eq0[:, -1]
    annual = np.where(final > 0, (final / init_balance) ** (1.0 / years) - 1.0, -1.0) * 100.0
    return {
        'equity_paths': eq0,                       # (num_runs, n_trades + 1)
        'final_equity': final,
        'annual_return_pct': annual,
        'max_dd_usd': dd_usd.max(axis=1),
        'max_dd_pct': dd_pct.max(axis=1) * 100.0,
        'prob_ruin_pct': float(np.mean(final <= 0) * 100.0),
    }


def percentile_table(mc: dict) -> pd.DataFrame:
    """Percentile rows (99/95/90/75/50) x metrics columns."""
    data = {
        'Final Equity ($)': np.percentile(mc['final_equity'], PCTILES),
        'Annual Return (%)': np.percentile(mc['annual_return_pct'], PCTILES),
        'Max Drawdown ($)': np.percentile(mc['max_dd_usd'], PCTILES),
        'Max Drawdown (%)': np.percentile(mc['max_dd_pct'], PCTILES),
    }
    return pd.DataFrame(data, index=[f'{p}%' for p in PCTILES])


# ===========================================================================
# Charts (matplotlib PNG embedded as base64)
# ===========================================================================
def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    return base64.b64encode(buf.getvalue()).decode()


def _equity_chart_b64(mc: dict, real_eq: np.ndarray, init_balance: float) -> str:
    """Simulated equity paths (sample + 5/50/95 percentile band) vs the real
    backtest equity, x-axis = trade number."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eq0 = mc['equity_paths']
    n_runs = eq0.shape[0]
    x = np.arange(eq0.shape[1])
    sample = eq0[np.linspace(0, n_runs - 1, min(200, n_runs)).astype(int)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, sample.T, color='#1f77b4', alpha=0.05, linewidth=0.6)
    ax.plot(x, np.percentile(eq0, 95, axis=0), color='#2ca02c', linewidth=1.2,
            label='simulated 95th pct')
    ax.plot(x, np.percentile(eq0, 50, axis=0), color='#ff7f0e', linewidth=1.4,
            label='simulated median')
    ax.plot(x, np.percentile(eq0, 5, axis=0), color='#d62728', linewidth=1.2,
            label='simulated 5th pct')
    ax.plot(np.arange(len(real_eq)), real_eq, color='#000000', linewidth=1.8,
            label='real backtest equity')
    ax.axhline(init_balance, color='#888888', linewidth=0.8, linestyle='--')
    ax.set_title('S4003 Monte Carlo Equity Paths (trade sequence resampled with replacement)')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Equity ($)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


def _drawdown_chart_b64(mc: dict, real_dd_pct: np.ndarray) -> str:
    """Underwater (drawdown %) curves of simulated paths vs the real backtest,
    x-axis = trade number."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eq0 = mc['equity_paths']
    n_runs = eq0.shape[0]
    peak = np.maximum.accumulate(eq0, axis=1)
    dd = np.where(peak > 0, (eq0 - peak) / np.where(peak > 0, peak, 1.0), 0.0) * 100.0
    x = np.arange(eq0.shape[1])
    sample = dd[np.linspace(0, n_runs - 1, min(200, n_runs)).astype(int)]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(x, sample.T, color='#1f77b4', alpha=0.05, linewidth=0.6)
    ax.plot(x, np.percentile(dd, 50, axis=0), color='#ff7f0e', linewidth=1.4,
            label='simulated median')
    ax.plot(x, np.percentile(dd, 5, axis=0), color='#d62728', linewidth=1.2,
            label='simulated 5th pct (deep DD)')
    ax.plot(np.arange(len(real_dd_pct)), real_dd_pct, color='#000000', linewidth=1.8,
            label='real backtest drawdown')
    ax.set_title('S4003 Monte Carlo Drawdown (underwater, % from running peak)')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ===========================================================================
# HTML output
# ===========================================================================
def write_html_report(mc: dict, pctl: pd.DataFrame, results: dict,
                      mc_cfg: dict, real_pnl: np.ndarray, out_dir: str) -> str:
    """Write the Monte Carlo html report (percentile table + equity chart +
    drawdown chart) into `out_dir`. Returns the html file path."""
    os.makedirs(out_dir, exist_ok=True)
    init_balance = float(mc_cfg['initCapital'])
    real_eq0 = np.concatenate([[init_balance], init_balance + np.cumsum(real_pnl)])
    real_peak = np.maximum.accumulate(real_eq0)
    real_dd_pct = (real_eq0 - real_peak) / np.where(real_peak > 0, real_peak, 1.0) * 100.0

    eq_b64 = _equity_chart_b64(mc, real_eq0, init_balance)
    dd_b64 = _drawdown_chart_b64(mc, real_dd_pct)

    stats = results['stats']
    rows = ''.join(
        f'<tr><th style="text-align:left">{idx}</th>'
        + ''.join(f'<td style="text-align:right">{v:,.2f}</td>' for v in row)
        + '</tr>'
        for idx, row in pctl.iterrows())
    head = ''.join(f'<th>{c}</th>' for c in pctl.columns)

    html = f"""<html><head><title>S4003 Monte Carlo Test</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;}}
th,td{{padding:4px 12px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body><h2>S4003 Monte Carlo Test Results</h2>
<p>Symbol: {mc_cfg.get('symbolName')} | Table: {mc_cfg.get('tableName')} |
Period: {stats['Start Date']} - {stats['End Date']} ({stats['Bars']} bars) |
Runs: {int(mc_cfg['numRuns'])} | Init capital: ${init_balance:,.2f} |
Commission/trade: ${float(mc_cfg['commissionPerTrade']):,.2f} |
Generated: {datetime.now().strftime(DT_FORMAT)}</p>
<p>Real backtest: {stats['Number of trades']} trades |
Net profit: ${stats['Net Profit']:,.2f} ({stats['Net Profit %']:.2f}%) |
Win rate: {stats['Number of wins %']:.1f}% |
Max system drawdown: {stats['Maximum system drawdown']:.2f}% |
Annual return: {stats['Annual Return %']:.2f}%<br/>
Monte Carlo: probability of ruin (final equity &le; 0) = {mc['prob_ruin_pct']:.2f}% |
Median final equity: ${np.percentile(mc['final_equity'], 50):,.2f} |
95th-pct max drawdown: ${np.percentile(mc['max_dd_usd'], 95):,.2f}
({np.percentile(mc['max_dd_pct'], 95):.2f}%)</p>

<h3>Percentile Table (over {int(mc_cfg['numRuns'])} simulated paths)</h3>
<table><tr><th>Percentile</th>{head}</tr>{rows}</table>
<p style="font-size:12px;color:#555">Note: for Final Equity / Annual Return a higher
percentile is a better outcome; for Max Drawdown a higher percentile is a deeper
(worse) drawdown. Drawdown percentiles are read as "p% of runs had a max drawdown
no worse than this".</p>

<h3>Equity Chart</h3><img src="data:image/png;base64,{eq_b64}" alt="equity chart"/>
<h3>Drawdown Chart</h3><img src="data:image/png;base64,{dd_b64}" alt="drawdown chart"/>
</body></html>"""

    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    html_path = os.path.join(out_dir, f'S4003_monteCarlo_{ts}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html_path


# ===========================================================================
# Main
# ===========================================================================
def main(config_path=None) -> dict:
    mc_cfg = loadMonteCarloConfig(config_path)
    applyMonteCarloConfig(mc_cfg, config_path)

    symbol = str(mc_cfg.get('symbolName', 'ES'))
    table = str(mc_cfg.get('tableName', 'ticker1Min'))
    init_balance = float(mc_cfg.get('initCapital', 80000.0))
    num_runs = int(mc_cfg.get('numRuns', 5000))
    seed = mc_cfg.get('randomSeed')

    # load data: full DB range, or [startDate - warmupDays, endDate]
    db_start = db_end = None
    if mc_cfg.get('startDate'):
        db_start = (pd.Timestamp(mc_cfg['startDate'])
                    - pd.Timedelta(days=int(strategyS4003.warmup_days))).strftime('%Y-%m-%d %H:%M:%S')
    if mc_cfg.get('endDate'):
        db_end = str(pd.Timestamp(mc_cfg['endDate']))
    print(f'--- Loading {symbol} 1-min bars from {table} '
          f'({db_start or "DB min"} .. {db_end or "DB max"}) ---')
    df = load_1min_data_from_db(symbol, table, db_start, db_end)
    if df.empty:
        raise RuntimeError(f'no data loaded for {symbol} from {table}')
    print(f'Loaded {len(df)} bars ({df.index[0]} .. {df.index[-1]})')

    # real backtest on the configured period -> real per-trade P&L series
    results = strategyS4003.run_full_backtest(df)
    real_pnl = results['trades']['pnl'].to_numpy(dtype=float) \
        if len(results['trades']) else np.array([])
    if real_pnl.size == 0:
        raise RuntimeError('no trades generated during backtest; cannot run Monte Carlo')
    print(f"Backtest complete. {real_pnl.size} trades, "
          f"net P&L ${real_pnl.sum():,.2f}")

    eq_index = results['equity'].index
    years = max((eq_index[-1] - eq_index[0]).total_seconds() / (365.25 * 86400.0), 1e-9)

    print(f'--- Starting Monte Carlo Simulation ({num_runs} runs) ---')
    mc = run_monte_carlo(real_pnl, num_runs, init_balance, years, seed=seed)
    pctl = percentile_table(mc)

    print('\n==========================================================')
    print('              MONTE CARLO SIMULATION RESULTS              ')
    print('==========================================================')
    print(pctl.to_string(float_format=lambda v: f'{v:,.2f}'))
    print(f"\nProbability of ruin: {mc['prob_ruin_pct']:.2f}%")
    print('==========================================================\n')

    html_path = write_html_report(mc, pctl, results, mc_cfg, real_pnl,
                                  str(mc_cfg['outputFolder']))
    print(f'HTML report written: {html_path}')
    return {'mc': mc, 'percentiles': pctl, 'stats': results['stats'], 'html': html_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S4003 Monte Carlo test (TASKS.md T3.2)')
    parser.add_argument('--config', default=None,
                        help='path to config.yaml (default: strategy folder config.yaml)')
    args = parser.parse_args()
    main(args.config)
