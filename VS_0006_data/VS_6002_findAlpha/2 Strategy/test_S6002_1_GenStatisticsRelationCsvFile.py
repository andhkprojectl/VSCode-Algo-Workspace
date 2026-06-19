"""
test_S6002_1_GenStatisticsRelationCsvFile.py
============================================
Pytest suite for S6002_1_GenStatisticsRelationCsvFile.py.
Uses synthetic small dataframes with known values.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the strategy module importable
STRATEGY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRATEGY_DIR))

import S6002_1_GenStatisticsRelationCsvFile as s6002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_df(n=50, start_price=100.0, seed=42):
    """Build a synthetic OHLCV dataframe with Datetime index."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2026-01-05 09:30", periods=n, freq="1min")
    close = start_price + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.rand(n) * 0.3
    low = close - rng.rand(n) * 0.3
    open_ = close + rng.randn(n) * 0.1
    vol = rng.randint(1000, 10000, n).astype(float)
    df = pd.DataFrame({
        'Date': dates.strftime('%m/%d/%Y'),
        'Time': dates.strftime('%H:%M:%S'),
        'High': high,
        'Low': low,
        'Close': close,
        'Open': open_,
        'Volume': vol,
        'Symbol': 'NVDA',
    }, index=dates)
    df.index.name = 'Datetime'
    return df


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------
def test_atr_true_range():
    """Known H/L/C -> expected ATR via True Range rolling mean."""
    df = pd.DataFrame({
        'High': [10.0, 12.0, 11.0],
        'Low':  [8.0,  9.0, 10.0],
        'Close':[9.0, 11.0, 10.5],
    })
    # TR[0] = 10-8 = 2 (no prev close)
    # TR[1] = max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3
    # TR[2] = max(11-10, |11-11|, |10-11|) = max(1, 0, 1) = 1
    atr2 = s6002.calc_atr(df, 2)
    # atr2[1] = mean(TR[0], TR[1]) = (2+3)/2 = 2.5
    assert atr2.iloc[1] == pytest.approx(2.5)
    # atr2[2] = mean(TR[1], TR[2]) = (3+1)/2 = 2.0
    assert atr2.iloc[2] == pytest.approx(2.0)
    # atr2[0] is NaN (need 2 bars)
    assert np.isnan(atr2.iloc[0])


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------
def test_rsi_monotonic_up():
    """Monotonic up series -> RSI ~ 100 (avg_loss ~ 0)."""
    close = pd.Series(np.arange(1.0, 21.0))  # strictly increasing
    r = s6002.calc_rsi(close, period=5)
    # After warmup, RSI should be ~100 (losses are 0 -> rs=inf)
    # Use a large number comparison; pandas may produce inf or 100
    last_val = r.iloc[-1]
    assert last_val >= 99.0 or np.isinf(last_val)


def test_rsi_range():
    """RSI values should be in [0, 100] (excluding NaN warmup)."""
    df = make_df(50)
    r = s6002.calc_rsi(df['Close'], period=14)
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


# ---------------------------------------------------------------------------
# Rolling percentile
# ---------------------------------------------------------------------------
def test_rolling_pct():
    """Known series -> 90/10 percentile over last 100 bars."""
    s = pd.Series(np.arange(1.0, 101.0))  # 1..100
    p90 = s6002.rolling_pct(s, window=100, q=90)
    p10 = s6002.rolling_pct(s, window=100, q=10)
    # At last bar, window = 1..100; compare against numpy's own percentile
    expected_90 = np.percentile(np.arange(1.0, 101.0), 90)
    expected_10 = np.percentile(np.arange(1.0, 101.0), 10)
    assert p90.iloc[-1] == pytest.approx(expected_90)
    assert p10.iloc[-1] == pytest.approx(expected_10)


def test_rolling_pct_handles_nan():
    """rolling_pct should handle sparse NaN (BB cross cols)."""
    s = pd.Series([np.nan, np.nan, 5.0, np.nan, 10.0, np.nan, 15.0])
    p90 = s6002.rolling_pct(s, window=10, q=90)
    # Should not raise; last value finite
    assert not np.isnan(p90.iloc[-1])


# ---------------------------------------------------------------------------
# Bollinger Band cross
# ---------------------------------------------------------------------------
def test_bb_cross_up():
    """Constructed cross-up -> cCrossUpBBTop = close at cross bar."""
    # Build a series where close crosses above BB top at a known bar
    n = 40
    close = pd.Series([100.0] * 25 + [101.0, 102.0, 103.0, 104.0, 105.0] + [106.0] * 10)
    df = pd.DataFrame({
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Open': close - 0.1,
        'Volume': 1000.0,
    })
    out = s6002.compute_statistics(df)
    # There should be at least one non-NaN cCrossUpBBTop
    assert out['cCrossUpBBTop'].notna().any()
    # Diff1/3/5 at a cross bar = future close - cross close
    cross_idx = out['cCrossUpBBTop'].first_valid_index()
    cross_close = out.loc[cross_idx, 'cCrossUpBBTop']
    # Diff1 at cross bar = close[t+1] - cross_close
    pos = out.index.get_loc(cross_idx)
    if pos + 1 < len(out):
        expected_diff1 = out['Close'].iloc[pos + 1] - cross_close
        assert out['cCrossUpBBTopDiff1'].iloc[pos] == pytest.approx(expected_diff1)


def test_bb_cross_down():
    """Constructed cross-down -> cCrossDownBBBottom = close at cross bar."""
    n = 40
    close = pd.Series([100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 95.0] + [94.0] * 10)
    df = pd.DataFrame({
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Open': close - 0.1,
        'Volume': 1000.0,
    })
    out = s6002.compute_statistics(df)
    assert out['cCrossDownBBBottom'].notna().any()


# ---------------------------------------------------------------------------
# Revenue targets
# ---------------------------------------------------------------------------
def test_revenue_rt():
    """Known future opens -> rt1/3/5/8."""
    df = pd.DataFrame({
        'Open': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
    })
    out = s6002.compute_revenue(df)
    # rt1 at bar 0 = open[2] - open[1] = 12 - 11 = 1
    assert out['rt1'].iloc[0] == pytest.approx(1.0)
    # rt3 at bar 0 = open[4] - open[1] = 14 - 11 = 3
    assert out['rt3'].iloc[0] == pytest.approx(3.0)
    # rt5 at bar 0 = open[6] - open[1] = 16 - 11 = 5
    assert out['rt5'].iloc[0] == pytest.approx(5.0)
    # rt8 at bar 0 = open[9] - open[1] = 19 - 11 = 8
    assert out['rt8'].iloc[0] == pytest.approx(8.0)
    # Last 9 rows of rt8 should be NaN
    assert out['rt8'].iloc[-9:].isna().all()


# ---------------------------------------------------------------------------
# Filter regular session
# ---------------------------------------------------------------------------
def test_filter_regular():
    """Rows outside 9:30-16:00 dropped when only_regular=True."""
    idx = pd.to_datetime([
        "2026-01-05 09:00",
        "2026-01-05 09:30",
        "2026-01-05 12:00",
        "2026-01-05 16:00",
        "2026-01-05 17:00",
    ])
    df = pd.DataFrame({'Close': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    out = s6002.filter_regular(df, only_regular=True)
    # 09:00 dropped, 16:00 dropped (time < 16:00), keep 09:30 and 12:00
    assert len(out) == 2
    assert pd.Timestamp("2026-01-05 09:30") in out.index
    assert pd.Timestamp("2026-01-05 12:00") in out.index


def test_filter_regular_disabled():
    """All rows kept when only_regular=False."""
    idx = pd.to_datetime(["2026-01-05 09:00", "2026-01-05 12:00", "2026-01-05 17:00"])
    df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]}, index=idx)
    out = s6002.filter_regular(df, only_regular=False)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Closed-form pair R^2 vs sklearn
# ---------------------------------------------------------------------------
def test_pair_r2_closed_form():
    """Closed-form 2-feature R^2 matches sklearn LinearRegression."""
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    rng = np.random.RandomState(123)
    n = 500
    f1 = rng.randn(n)
    f2 = rng.randn(n) * 0.5 + f1 * 0.3
    y = 2.0 * f1 - 1.5 * f2 + rng.randn(n) * 0.2

    df = pd.DataFrame({'f1': f1, 'f2': f2, 'y': y})
    corr = df.corr()
    r_y1 = corr.loc['f1', 'y']
    r_y2 = corr.loc['f2', 'y']
    r_12 = corr.loc['f1', 'f2']
    denom = 1 - r_12 * r_12
    r2_closed = (r_y1**2 + r_y2**2 - 2 * r_y1 * r_y2 * r_12) / denom

    # sklearn R^2
    X = df[['f1', 'f2']].values
    model = LinearRegression().fit(X, y)
    r2_sklearn = model.score(X, y)

    assert r2_closed == pytest.approx(r2_sklearn, abs=1e-9)


# ---------------------------------------------------------------------------
# IRB
# ---------------------------------------------------------------------------
def test_irb_flag():
    """Constructed inside-range bar -> irb1=1."""
    # Bar where open and close are both in lower half of [Low, High] (bullish IRB)
    df = pd.DataFrame({
        'High': [10.0, 10.0],
        'Low':  [5.0,  5.0],
        'Close':[5.5,  9.5],   # bar0 close near low (bullish), bar1 close near high (bearish)
        'Open': [5.6,  9.4],
    })
    irb = s6002.calc_irb(df)
    # hl_range = 5; lower_th = 5 + 5*0.45 = 7.25; upper_th = 10 - 5*0.45 = 7.75
    # bar0: close=5.5 < 7.25 and open=5.6 < 7.25 -> bullish -> irb=1
    assert irb.iloc[0] == 1
    # bar1: close=9.5 > 7.75 and open=9.4 > 7.75 -> bearish -> irb=1
    assert irb.iloc[1] == 1


def test_irb_no_flag():
    """Bar where open/close span the range -> irb1=0."""
    df = pd.DataFrame({
        'High': [10.0],
        'Low':  [5.0],
        'Close':[7.5],   # mid-range, not in upper or lower threshold
        'Open': [7.5],
    })
    irb = s6002.calc_irb(df)
    assert irb.iloc[0] == 0


# ---------------------------------------------------------------------------
# Integration: compute_statistics produces expected columns
# ---------------------------------------------------------------------------
def test_compute_statistics_columns():
    """compute_statistics adds expected statistics columns."""
    df = make_df(60)
    out = s6002.compute_statistics(df)
    # ATR
    for n in [3, 5, 10, 15]:
        assert f'atr{n}' in out.columns
        assert f'atr{n}Diff1' in out.columns
        assert f'atr{n}Diff3' in out.columns
        assert f'atr{n}_90' in out.columns
        assert f'atr{n}_10' in out.columns
    # Close diff
    for n in [1, 2, 3, 5]:
        assert f'cDiff{n}' in out.columns
        assert f'cDiff{n}_90' in out.columns
    # EMA
    for n in [5, 10, 20]:
        assert f'cDiffEma{n}' in out.columns
        assert f'cDiffEma{n}_90' in out.columns
    # BB
    assert 'cCrossUpBBTop' in out.columns
    assert 'cCrossUpBBTopDiff1' in out.columns
    assert 'cCrossDownBBBottom' in out.columns
    assert 'cCrossDownBBBottomDiff5' in out.columns
    # RSI
    for n in [2, 6, 14]:
        assert f'rsi{n}' in out.columns
        assert f'rsi{n}Diff1' in out.columns
        assert f'rsi{n}Diff5' in out.columns
    # Volume
    assert 'vDiff1' in out.columns
    assert 'vDiff1_90' in out.columns
    # IRB
    assert 'irb1' in out.columns


def test_load_data_and_rename():
    """load_data renames columns A-I correctly."""
    # Write a tiny CSV
    tmp = pd.DataFrame({
        'datetime': ['01/05/2026 09:30'],
        'date': ['01/05/2026'],
        'time': ['09:30:00'],
        'high': [100.0],
        'low': [99.0],
        'close': [99.5],
        'open': [99.8],
        'volume': [1000],
        'symbol': ['NVDA'],
    })
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
        tmp.to_csv(f, index=False)
        path = f.name
    try:
        df = s6002.load_data(path)
        assert 'High' in df.columns
        assert 'Low' in df.columns
        assert 'Close' in df.columns
        assert 'Open' in df.columns
        assert 'Volume' in df.columns
        assert df.index.name == 'Datetime'
    finally:
        os.remove(path)
