"""
S6002_1_GenStatisticsRelationCsvFile.py
=======================================
Find alpha in NVDA 1-minute data by computing ~112 statistics columns and
4 forward revenue targets, then finding single & 2-feature combinations with
strong relation to revenue via correlation (|r|>=0.5) and R-squared (>=0.25).

Plan: .github/prompts/plan-nvda1MinAlphaFinder.prompt.md

Input : C:\\Project\\ProjectLife\\VSCode Algo Workspace DataFile\\VS_6002_findAlpha\\csvExcel\\NVDA_20260101_20260615_1Min.csv
Output: C:\...\VS_6002_findAlpha\Output (files suffixed with YYYYMMDDHH24MISS)
    - S6002_1_statistics_revenue_*.csv   (input cols + all stats + rt cols)
    - S6002_1_relation_summary_*.csv     (strong relations table)
    - S6002_1_scatter_plots_*.html       (plotly scatter for strong relations)
"""

import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Plotly is optional at import time; only needed for HTML output
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_6002_findAlpha\csvExcel\NVDA_20260101_20260615_1Min.csv"
OUTPUT_DIR = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_6002_findAlpha\Output"

# Relation thresholds (relaxed per user request)
CORR_THRESHOLD = 0.5      # |r| >= 0.5
R2_THRESHOLD = 0.25       # R^2 >= 0.25

# Rolling percentile window
PCT_WINDOW = 100

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD = 2

# Cap scatter plots in HTML to keep file manageable
MAX_SCATTER_PLOTS = 20


# ===========================================================================
# Phase A - Scaffolding & I/O
# ===========================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    """Read NVDA 1-min CSV, rename columns A-I, parse datetime, set index."""
    df = pd.read_csv(csv_path)
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
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')
    df = df.set_index('Datetime')
    df = df.sort_index()
    return df


def filter_regular(df: pd.DataFrame, only_regular: bool = True) -> pd.DataFrame:
    """If only_regular=True, keep rows with 09:30 <= time < 16:00."""
    if not only_regular:
        return df
    t = df.index.time
    mask = (t >= pd.Timestamp("09:30").time()) & (t < pd.Timestamp("16:00").time())
    return df[mask]


# ===========================================================================
# Phase B - Statistics helpers
# ===========================================================================
def rolling_pct(s: pd.Series, window: int = PCT_WINDOW, q: float = 90) -> pd.Series:
    """Rolling percentile using nanpercentile (handles sparse NaN)."""
    return s.rolling(window=window, min_periods=1).apply(
        lambda x: np.nanpercentile(x, q), raw=True
    )


def add_pct_cols(df: pd.DataFrame, col: str) -> None:
    """Add {col}_90 and {col}_10 rolling percentile columns in-place."""
    df[f"{col}_90"] = rolling_pct(df[col], PCT_WINDOW, 90)
    df[f"{col}_10"] = rolling_pct(df[col], PCT_WINDOW, 10)


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """True Range ATR: TR=max(H-L, |H-prevC|, |L-prevC|); atr=TR.rolling(n).mean()."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """Manual RSI via delta/gain/loss rolling mean."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calc_bollinger(df: pd.DataFrame, period: int = BB_PERIOD, std_mult: int = BB_STD):
    """Return (top_band, bottom_band) Series."""
    rolling_mean = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    top = rolling_mean + (rolling_std * std_mult)
    bot = rolling_mean - (rolling_std * std_mult)
    return top, bot


def calc_irb(df: pd.DataFrame) -> pd.Series:
    """Inside Range Bar flag (bullish OR bearish) per S8001_2 0.45 threshold."""
    hl_range = df['High'] - df['Low']
    upper_th = df['High'] - hl_range * 0.45
    lower_th = df['Low'] + hl_range * 0.45
    bullish = (
        (df['Close'] > df['Low'])
        & (df['Open'] > df['Low'])
        & (df['Close'] < lower_th)
        & (df['Open'] < lower_th)
    )
    bearish = (
        (df['Close'] < df['High'])
        & (df['Open'] < df['High'])
        & (df['Close'] > upper_th)
        & (df['Open'] > upper_th)
    )
    return (bullish | bearish).astype(int)


# ===========================================================================
# Phase B - Statistics (4c)
# ===========================================================================
def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all ~112 statistics columns. Returns a copy with new columns."""
    out = df.copy()

    # --- ATR group ---
    for n in [3, 5, 10, 15]:
        out[f'atr{n}'] = calc_atr(out, n)
        out[f'atr{n}Diff1'] = out[f'atr{n}'].diff(1)
        out[f'atr{n}Diff3'] = out[f'atr{n}'].diff(3)
        add_pct_cols(out, f'atr{n}')
        add_pct_cols(out, f'atr{n}Diff1')
        add_pct_cols(out, f'atr{n}Diff3')

    # --- Close-diff group ---
    for n in [1, 2, 3, 5]:
        out[f'cDiff{n}'] = out['Close'].diff(n)
        add_pct_cols(out, f'cDiff{n}')

    # --- EMA group ---
    for n in [5, 10, 20]:
        ema = calc_ema(out['Close'], n)
        out[f'cDiffEma{n}'] = out['Close'] - ema
        add_pct_cols(out, f'cDiffEma{n}')

    # --- Bollinger Band group ---
    bb_top, bb_bot = calc_bollinger(out, BB_PERIOD, BB_STD)
    close = out['Close']
    prev_close = close.shift(1)

    # Cross up BB top: prev close <= top & cur close > top -> save close at cross
    cross_up = (prev_close <= bb_top) & (close > bb_top)
    out['cCrossUpBBTop'] = close.where(cross_up, np.nan)
    out['cCrossUpBBTopDiff1'] = close.shift(-1) - out['cCrossUpBBTop']
    out['cCrossUpBBTopDiff3'] = close.shift(-3) - out['cCrossUpBBTop']
    out['cCrossUpBBTopDiff5'] = close.shift(-5) - out['cCrossUpBBTop']

    # Cross down BB bottom: prev close >= bot & cur close < bot -> save close at cross
    cross_down = (prev_close >= bb_bot) & (close < bb_bot)
    out['cCrossDownBBBottom'] = close.where(cross_down, np.nan)
    out['cCrossDownBBBottomDiff1'] = close.shift(-1) - out['cCrossDownBBBottom']
    out['cCrossDownBBBottomDiff3'] = close.shift(-3) - out['cCrossDownBBBottom']
    out['cCrossDownBBBottomDiff5'] = close.shift(-5) - out['cCrossDownBBBottom']

    for bb_col in [
        'cCrossUpBBTop', 'cCrossUpBBTopDiff1', 'cCrossUpBBTopDiff3', 'cCrossUpBBTopDiff5',
        'cCrossDownBBBottom', 'cCrossDownBBBottomDiff1', 'cCrossDownBBBottomDiff3', 'cCrossDownBBBottomDiff5',
    ]:
        add_pct_cols(out, bb_col)

    # --- RSI group ---
    for n in [2, 6, 14]:
        out[f'rsi{n}'] = calc_rsi(out['Close'], n)
        out[f'rsi{n}Diff1'] = out[f'rsi{n}'].diff(1)
        out[f'rsi{n}Diff5'] = out[f'rsi{n}'].diff(5)
        add_pct_cols(out, f'rsi{n}')
        add_pct_cols(out, f'rsi{n}Diff1')
        add_pct_cols(out, f'rsi{n}Diff5')

    # --- Volume group ---
    out['vDiff1'] = (out['Volume'] - out['Volume'].shift(1)) / out['Volume'].shift(1)
    add_pct_cols(out, 'vDiff1')

    # --- IRB ---
    out['irb1'] = calc_irb(out)

    return out


# ===========================================================================
# Phase C - Revenue (4d)
# ===========================================================================
def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward revenue targets rt1/3/5/8 from open price."""
    out = df.copy()
    open_ = out['Open']
    out['rt1'] = open_.shift(-2) - open_.shift(-1)
    out['rt3'] = open_.shift(-4) - open_.shift(-1)
    out['rt5'] = open_.shift(-6) - open_.shift(-1)
    out['rt8'] = open_.shift(-9) - open_.shift(-1)
    return out


# ===========================================================================
# Phase D - Relation analysis (4f)
# ===========================================================================
def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of statistics columns (exclude input + rt columns)."""
    input_cols = {'Date', 'Time', 'High', 'Low', 'Close', 'Open', 'Volume', 'Symbol'}
    rt_cols = {'rt1', 'rt3', 'rt5', 'rt8'}
    return [c for c in df.columns if c not in input_cols and c not in rt_cols]


def find_strong_relations(df: pd.DataFrame, features: list, rt_cols: list):
    """
    Find single & 2-feature pair relations with strong relation to revenue.
    Returns list of dicts: {type, features, rt, corr, r2}.

    Uses pairwise correlation (pandas corr() handles NaN per-pair), so sparse
    columns like BB cross events don't eliminate all rows.
    """
    relations = []

    # Use pairwise correlation on the full dataframe (no global dropna).
    # pandas .corr() drops NaN pairwise, so each (feature, rt) pair uses
    # only rows where both are non-NaN.
    corr_matrix = df[features + rt_cols].corr()
    # Count usable rows for reporting (use rt1 as reference)
    n_usable = df[rt_cols].dropna().shape[0]
    print(f"  Usable rows (all rt non-NaN): {n_usable}")
    if n_usable == 0:
        print("WARNING: no usable rows; skipping relation analysis.")
        return relations, df[features + rt_cols].dropna()

    # --- Singles ---
    for feat in features:
        for rt in rt_cols:
            r = corr_matrix.loc[feat, rt]
            if np.isnan(r):
                continue
            r2 = r * r
            if abs(r) >= CORR_THRESHOLD:
                relations.append({
                    'type': 'single',
                    'features': feat,
                    'rt': rt,
                    'corr': r,
                    'r2': r2,
                })

    # --- Pairs (closed-form 2-feature R^2 from correlations) ---
    n_feats = len(features)
    n_pairs = n_feats * (n_feats - 1) // 2
    print(f"  Evaluating {n_pairs} pairs x {len(rt_cols)} rt targets...")
    for f1, f2 in combinations(features, 2):
        r_12 = corr_matrix.loc[f1, f2]
        if np.isnan(r_12) or abs(r_12) >= 0.9999:
            # near-perfect collinearity -> skip (denominator ~0)
            continue
        denom = 1 - r_12 * r_12
        for rt in rt_cols:
            r_y1 = corr_matrix.loc[f1, rt]
            r_y2 = corr_matrix.loc[f2, rt]
            if np.isnan(r_y1) or np.isnan(r_y2):
                continue
            r2 = (r_y1 * r_y1 + r_y2 * r_y2 - 2 * r_y1 * r_y2 * r_12) / denom
            if r2 >= R2_THRESHOLD:
                relations.append({
                    'type': 'pair',
                    'features': f"{f1} + {f2}",
                    'rt': rt,
                    'corr': np.nan,  # not defined for pairs
                    'r2': r2,
                })

    # For scatter plots, build a clean df using only non-sparse features + rt
    # (drop rows where any rt is NaN; features with NaN handled by plotly)
    clean = df[rt_cols].dropna().join(df[features])
    return relations, clean


# ===========================================================================
# Phase E - Output (5)
# ===========================================================================
def write_statistics_csv(df: pd.DataFrame, out_dir: str, ts: str) -> str:
    path = os.path.join(out_dir, f"S6002_1_statistics_revenue_{ts}.csv")
    df.to_csv(path)
    print(f"  Wrote {path} ({len(df)} rows, {len(df.columns)} cols)")
    return path


def write_relation_summary(relations: list, out_dir: str, ts: str) -> str:
    path = os.path.join(out_dir, f"S6002_1_relation_summary_{ts}.csv")
    if not relations:
        pd.DataFrame(columns=['type', 'features', 'rt', 'corr', 'r2']).to_csv(path, index=False)
        print(f"  Wrote {path} (0 strong relations)")
        return path
    summary = pd.DataFrame(relations)
    summary = summary.sort_values('r2', ascending=False)
    summary.to_csv(path, index=False)
    print(f"  Wrote {path} ({len(summary)} strong relations)")
    return path


def write_scatter_html(relations: list, clean: pd.DataFrame, out_dir: str, ts: str) -> str:
    path = os.path.join(out_dir, f"S6002_1_scatter_plots_{ts}.html")
    if not _HAS_PLOTLY:
        with open(path, 'w') as f:
            f.write("<html><body><h2>plotly not installed; no scatter plots.</h2></body></html>")
        print(f"  Wrote {path} (plotly not installed)")
        return path

    # Sort by r2 desc, cap to MAX_SCATTER_PLOTS
    sorted_rels = sorted(relations, key=lambda r: r['r2'], reverse=True)
    top_rels = sorted_rels[:MAX_SCATTER_PLOTS]

    if not top_rels:
        with open(path, 'w') as f:
            f.write(
                "<html><head><meta charset='utf-8'></head><body>"
                "<h2>No strong relation found</h2>"
                "<p>No single or pair feature met the thresholds "
                f"|r|>={CORR_THRESHOLD} or R^2>={R2_THRESHOLD}.</p>"
                "</body></html>"
            )
        print(f"  Wrote {path} (no strong relations)")
        return path

    fig = make_subplots(
        rows=len(top_rels), cols=1,
        subplot_titles=[
            f"{r['type']}: {r['features']} vs {r['rt']} (r2={r['r2']:.4f})"
            for r in top_rels
        ],
        vertical_spacing=0.04,
    )

    for i, rel in enumerate(top_rels, start=1):
        rt = rel['rt']
        if rel['type'] == 'single':
            feat = rel['features']
            # Drop NaN rows for this feature+rt pair only
            plot_df = clean[[feat, rt]].dropna()
            fig.add_trace(
                go.Scattergl(
                    x=plot_df[feat], y=plot_df[rt], mode='markers', marker=dict(size=3, opacity=0.4),
                    name=f"{feat} vs {rt}",
                ),
                row=i, col=1,
            )
            fig.update_xaxes(title_text=feat, row=i, col=1)
            fig.update_yaxes(title_text=rt, row=i, col=1)
        else:
            # pair: 2 features -> use first feature as x, color by second, y=rt
            f1, f2 = rel['features'].split(' + ')
            plot_df = clean[[f1, f2, rt]].dropna()
            fig.add_trace(
                go.Scattergl(
                    x=plot_df[f1], y=plot_df[rt], mode='markers',
                    marker=dict(size=3, opacity=0.4, color=plot_df[f2], colorscale='Viridis', showscale=True),
                    name=f"{f1}+{f2} vs {rt}",
                ),
                row=i, col=1,
            )
            fig.update_xaxes(title_text=f"{f1} (color={f2})", row=i, col=1)
            fig.update_yaxes(title_text=rt, row=i, col=1)

    fig.update_layout(height=400 * len(top_rels), showlegend=False, title_text="S6002_1 Strong Relations")
    fig.write_html(path)
    print(f"  Wrote {path} ({len(top_rels)} scatter plots)")
    return path


# ===========================================================================
# Main
# ===========================================================================
def main(only_regular: bool = True):
    print("=" * 70)
    print("S6002_1_GenStatisticsRelationCsvFile - NVDA 1-min Alpha Finder")
    print("=" * 70)

    # Phase A
    print("\n[Phase A] Loading data...")
    df = load_data(INPUT_CSV)
    print(f"  Loaded {len(df)} rows from {INPUT_CSV}")
    df = filter_regular(df, only_regular=only_regular)
    print(f"  After regular-session filter: {len(df)} rows")

    # Phase B
    print("\n[Phase B] Computing statistics...")
    df = compute_statistics(df)
    n_stat_cols = len(get_feature_columns(df))
    print(f"  Computed {n_stat_cols} statistics columns")

    # Phase C
    print("\n[Phase C] Computing revenue targets...")
    df = compute_revenue(df)
    print(f"  Computed rt1/rt3/rt5/rt8")

    # Phase D
    print("\n[Phase D] Finding strong relations...")
    features = get_feature_columns(df)
    rt_cols = ['rt1', 'rt3', 'rt5', 'rt8']
    relations, clean = find_strong_relations(df, features, rt_cols)
    n_single = sum(1 for r in relations if r['type'] == 'single')
    n_pair = sum(1 for r in relations if r['type'] == 'pair')
    print(f"  Found {n_single} strong singles, {n_pair} strong pairs")

    # Phase E
    print("\n[Phase E] Writing outputs...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    write_statistics_csv(df, OUTPUT_DIR, ts)
    write_relation_summary(relations, OUTPUT_DIR, ts)
    write_scatter_html(relations, clean, OUTPUT_DIR, ts)

    print("\nDone.")


if __name__ == "__main__":
    only_regular_arg = True
    if len(sys.argv) > 1 and sys.argv[1].upper() == 'N':
        only_regular_arg = False
    main(only_regular=only_regular_arg)
