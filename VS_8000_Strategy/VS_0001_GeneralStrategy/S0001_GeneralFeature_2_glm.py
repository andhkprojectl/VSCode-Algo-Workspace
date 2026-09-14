"""
S0001_GeneralFeature_2_glm.py
=============================
Feature engineering for 1-minute ES futures. Derived from
S0001_GeneralFeature_1_kimi.py per requirement
VS_8000_Strategy/VS_0001_GeneralStrategy/S0001_GeneralFeature_2_ur.txt.

Changes vs S0001_GeneralFeature_1_kimi.py:

1. Look-ahead fixes
   - is_swing_high / is_swing_low used future bars (h.shift(-1), h.shift(-2));
     replaced with trailing 3-bar swings (current high/low beyond previous 2).
   - Opening-range breakout features used the completed 09:30-10:00 range at
     every bar of the day; now the range expands causally within the first
     30 minutes and is held constant afterwards (NaN before 09:30).
   - slope_10 / slope_20 excluded the current bar and used a slow per-bar
     linregress loop; now include the current bar (vectorized).

2. Normalization
   - General features: standard scaler (z-score). Default fits on the whole
     sample; method='expanding' fits causally to avoid statistic leakage.
   - Cyclical date/time features (hour, minute, minute_of_day, day_of_week,
     month): sine/cosine transforms (raw values removed). Binary session
     flags are treated as general features.

3. New features
   - slope15_close / slope15_ema6 / slope15_ema9 / slope15_ema20 /
     slope15_ema50: 15-bar linear regression slope, current bar included.
   - rsi{6,9} divergence vs close (signed: +1 bullish, -1 bearish, 0 none),
     rsi{6,9} bullish divergence vs high price, rsi{6,9} bearish divergence
     vs low price (slope-comparison based, causal).

4. MariaDB loader: IBTradingDb.ticker1Min, ticker 'ES'
   (VS_0007_dbAndFile/mariaDB/createIBDbUser.sql; credentials via
   VS_0002_config/.env DB_* variables, repo convention).

5. Performance: OBV, VWAP distance, autocorrelation and slopes vectorized.

V2 additions (requirement: S0001_GeneralFeature_2_ur V2.txt):

6. Market regime features: trend regime (close vs EMA50 with an ATR20
   hysteresis band: +1 up / -1 down / 0 neutral), volatility regime
   (ATR20 percentile within the trailing 500 bars: 0 low / 1 normal /
   2 high) and a combined `market_regime` code (trend*10 + vol).

7. Rolling close-price VWAP, periods 3/6/9/20: volume-weighted mean of
   close (`rvwap_n`), its volume-weighted standard deviation
   (`rvwap_n_std`), band levels `rvwap_n_up1/dn1` (vwap +/- 1 std) and
   `rvwap_n_up2/dn2` (vwap +/- 2 std), plus normalized views
   `rvwap_n_dist` (relative distance) and `rvwap_n_pos` (z-position).
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import mysql.connector

OUTPUT_DIR = os.path.join(
    r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile", "VS_0001_GeneralStrategy")


# ===========================================================================
# Data loader (MariaDB IBTradingDb.ticker1Min)
# ===========================================================================
def _load_db_env():
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        for p in (Path(__file__).resolve().parents[2] / "VS_0002_config" / ".env",
                  Path.cwd() / "VS_0002_config" / ".env"):
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
                return
    except ImportError:
        pass


def load_data_from_db(ticker='ES', start_date=None, end_date=None) -> pd.DataFrame:
    """Load 1-min bars with columns [open, high, low, close, volume] and a
    DatetimeIndex, sorted ascending."""
    _load_db_env()
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )
    try:
        cur = conn.cursor()
        q = ("SELECT datetime1, open, high, low, close, volume FROM ticker1Min "
             "WHERE ticker = %s")
        params = [ticker]
        if start_date:
            q += " AND datetime1 >= %s"
            params.append(start_date)
        if end_date:
            q += " AND datetime1 <= %s"
            params.append(end_date)
        q += " ORDER BY datetime1 ASC"
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
# Indicator helpers
# ===========================================================================
def _rolling_slope_incl(series: pd.Series, window: int) -> pd.Series:
    """OLS slope over the last `window` bars, current bar included (causal)."""
    t = np.arange(window, dtype=float)
    tm = t.mean()
    denom = float(((t - tm) ** 2).sum())

    def _slope(y):
        y = np.asarray(y, dtype=float)
        if np.isnan(y).any():
            return np.nan
        return float(((y - y.mean()) * (t - tm)).sum() / denom)

    return series.rolling(window).apply(_slope, raw=True)


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. avg_loss == 0 -> 100, both zero -> 50."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    return rsi


def _entropy_last(x) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return float(-(p * np.log2(p)).sum())


def _pct_rank_last(x) -> float:
    return float((x <= x[-1]).mean())


# ===========================================================================
# Feature engineer
# ===========================================================================
class ESFeatureEngineer:
    """
    Feature engineering for 1-minute ES futures.

    All features use only information available at or before the current bar
    close (look-ahead issues of the source file fixed, see module docstring).

    Normalization:
      - normalize(method='global')    : standard scaler over the whole sample
      - normalize(method='expanding') : causal expanding mean/std (no leakage)
      - cyclical time features are represented as sine/cosine pairs and are
        not re-scaled (already in [-1, 1])
    """

    #: columns produced as sine/cosine pairs (excluded from standard scaling)
    SIN_COS_SUFFIXES = ('_sin', '_cos')

    def __init__(self, df: pd.DataFrame):
        """
        df columns: ['open','high','low','close','volume'], DatetimeIndex.
        """
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
        self.features_norm = None

    # ============================================================
    # 1. PRICE ACTION & MOMENTUM
    # ============================================================
    def add_returns(self):
        c = self.df['close']
        self.features['ret_1m'] = c.pct_change()
        self.features['ret_5m'] = c.pct_change(5)
        self.features['ret_10m'] = c.pct_change(10)
        self.features['ret_20m'] = c.pct_change(20)
        self.features['log_ret_1m'] = np.log(c / c.shift(1))
        self.features['log_ret_5m'] = np.log(c / c.shift(5))
        return self

    def add_candle_features(self):
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        self.features['body_size'] = abs(c - o)
        self.features['body_pct'] = abs(c - o) / (h - l + 1e-9)
        self.features['body_direction'] = np.sign(c - o)
        self.features['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-9)
        self.features['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-9)
        self.features['wick_ratio'] = self.features['upper_wick'] / (self.features['lower_wick'] + 1e-9)
        self.features['bar_range'] = h - l
        self.features['range_pct'] = (h - l) / c
        self.features['gap'] = (o - c.shift(1)) / c.shift(1)
        return self

    def add_momentum(self):
        c = self.df['close']
        for period in [3, 5, 10, 20]:
            self.features[f'roc_{period}'] = (c - c.shift(period)) / c.shift(period)
        self.features['accel_5'] = self.features['roc_5'] - self.features['roc_5'].shift(3)

        # rolling slopes now INCLUDE the current bar (fixed vs _1_kimi)
        self.features['slope_10'] = _rolling_slope_incl(c, 10)
        self.features['slope_20'] = _rolling_slope_incl(c, 20)

        for ma in [5, 10, 20, 50]:
            self.features[f'dist_sma_{ma}'] = (c - c.rolling(ma).mean()) / c
            self.features[f'dist_ema_{ma}'] = (c - c.ewm(span=ma, adjust=False).mean()) / c
        self.features['sma5_10_cross'] = (c.rolling(5).mean() - c.rolling(10).mean()) / c
        self.features['sma10_20_cross'] = (c.rolling(10).mean() - c.rolling(20).mean()) / c
        return self

    # ============================================================
    # NEW: linear regression slopes (15 bars, current bar included)
    # ============================================================
    def add_linreg_slopes(self, window: int = 15):
        """slope15_* for close, ema6, ema9, ema20, ema50."""
        c = self.df['close']
        self.features[f'slope{window}_close'] = _rolling_slope_incl(c, window)
        for p in (6, 9, 20, 50):
            ema = c.ewm(span=p, adjust=False).mean()
            self.features[f'slope{window}_ema{p}'] = _rolling_slope_incl(ema, window)
        return self

    # ============================================================
    # NEW: RSI divergence features
    # ============================================================
    def add_rsi_divergence(self, window: int = 15):
        """Slope-comparison divergences (causal), using the same inclusive
        `window`-bar linear regression slope as add_linreg_slopes.

        rsi{p}_div_close    : +1 price slope down & RSI slope up (bullish),
                              -1 price slope up & RSI slope down (bearish), else 0
        rsi{p}_bull_div_high: 1 when HIGH-slope < 0 and RSI-slope > 0
        rsi{p}_bear_div_low : 1 when LOW-slope  > 0 and RSI-slope < 0
        """
        c, h, l = self.df['close'], self.df['high'], self.df['low']
        slope_c = _rolling_slope_incl(c, window)
        slope_h = _rolling_slope_incl(h, window)
        slope_l = _rolling_slope_incl(l, window)
        for p in (6, 9):
            slope_r = _rolling_slope_incl(_wilder_rsi(c, p), window)
            rsi_up = slope_r > 0
            rsi_dn = slope_r < 0
            div = np.where((slope_c < 0) & rsi_up, 1,
                           np.where((slope_c > 0) & rsi_dn, -1, 0))
            self.features[f'rsi{p}_div_close'] = div.astype(int)
            self.features[f'rsi{p}_bull_div_high'] = ((slope_h < 0) & rsi_up).astype(int)
            self.features[f'rsi{p}_bear_div_low'] = ((slope_l > 0) & rsi_dn).astype(int)
        return self

    # ============================================================
    # V2: MARKET REGIME
    # ============================================================
    def add_market_regime(self, trend_span: int = 50, atr_period: int = 20,
                          trend_atr_band: float = 0.25, vol_lookback: int = 500,
                          vol_low_pct: float = 0.25, vol_high_pct: float = 0.75):
        """Causal trend/volatility regime classification.

        regime_trend  : +1 close > EMA(trend_span) + trend_atr_band*ATR
                        -1 close < EMA(trend_span) - trend_atr_band*ATR
                         0 otherwise (neutral, ATR hysteresis band)
        regime_vol    : ATR percentile rank within trailing `vol_lookback`
                        bars -> 0 low (<= vol_low_pct), 1 normal,
                        2 high (>= vol_high_pct); NaN during warm-up
        market_regime : combined code = trend*10 + vol
                        (e.g. 12 = uptrend + high vol, -10 = downtrend + low vol)
        """
        c, h, l = self.df['close'], self.df['high'], self.df['low']
        prev = c.shift(1)
        tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()],
                       axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        ema = c.ewm(span=trend_span, adjust=False).mean()
        band = trend_atr_band * atr
        trend = np.where(c > ema + band, 1.0,
                         np.where(c < ema - band, -1.0, 0.0))
        atr_pct = atr.rolling(vol_lookback, min_periods=100).apply(
            _pct_rank_last, raw=True)
        vol_reg = np.where(atr_pct <= vol_low_pct, 0.0,
                           np.where(atr_pct >= vol_high_pct, 2.0, 1.0))
        vol_reg = np.where(np.isnan(atr_pct), np.nan, vol_reg)
        self.features['regime_trend'] = trend
        self.features['regime_vol'] = vol_reg
        self.features['market_regime'] = trend * 10.0 + vol_reg
        return self

    # ============================================================
    # V2: ROLLING CLOSE-PRICE VWAP + BANDS (periods 3/6/9/20)
    # ============================================================
    def add_rolling_vwap(self, periods=(3, 6, 9, 20)):
        """Volume-weighted mean/std of close over rolling windows.

        Per period n:
          rvwap_n       : rolling VWAP of close (volume-weighted mean)
          rvwap_n_std   : volume-weighted standard deviation of close
          rvwap_n_up1   : vwap + 1*std        rvwap_n_dn1 : vwap - 1*std
          rvwap_n_up2   : vwap + 2*std        rvwap_n_dn2 : vwap - 2*std
          rvwap_n_dist  : (close - vwap) / close      (relative distance)
          rvwap_n_pos   : (close - vwap) / std        (z-position in bands)
        Windows with zero total volume yield NaN.
        """
        c, v = self.df['close'], self.df['volume']
        for n in periods:
            V = v.rolling(n).sum()
            V_nz = V.replace(0.0, np.nan)
            m = (c * v).rolling(n).sum() / V_nz
            var = ((c * c * v).rolling(n).sum() / V_nz - m * m).clip(lower=0.0)
            std = np.sqrt(var)
            self.features[f'rvwap_{n}'] = m
            self.features[f'rvwap_{n}_std'] = std
            self.features[f'rvwap_{n}_up1'] = m + 1.0 * std
            self.features[f'rvwap_{n}_dn1'] = m - 1.0 * std
            self.features[f'rvwap_{n}_up2'] = m + 2.0 * std
            self.features[f'rvwap_{n}_dn2'] = m - 2.0 * std
            self.features[f'rvwap_{n}_dist'] = (c - m) / c
            self.features[f'rvwap_{n}_pos'] = (c - m) / (std + 1e-9)
        return self

    # ============================================================
    # 2. VOLATILITY FEATURES
    # ============================================================
    def add_volatility(self):
        c, h, l = self.df['close'], self.df['high'], self.df['low']
        prev_close = c.shift(1)
        tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()],
                       axis=1).max(axis=1)
        for period in [5, 10, 20]:
            self.features[f'atr_{period}'] = tr.rolling(period).mean()
            self.features[f'atr_{period}_pct'] = self.features[f'atr_{period}'] / c
        for period in [5, 10, 20]:
            self.features[f'vol_{period}'] = c.pct_change().rolling(period).std()
            self.features[f'vol_{period}_annual'] = self.features[f'vol_{period}'] * np.sqrt(252 * 390)
        for period in [5, 10]:
            log_hl = np.log(h / l)
            self.features[f'parkinson_{period}'] = np.sqrt(
                (log_hl ** 2).rolling(period).mean() / (4 * np.log(2)))
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        self.features['bb_position'] = (c - sma20) / (2 * std20 + 1e-9)
        self.features['bb_width'] = (4 * std20) / sma20
        self.features['range_vs_5ma'] = (h - l) / (h - l).rolling(5).mean()
        self.features['range_vs_20ma'] = (h - l) / (h - l).rolling(20).mean()
        return self

    # ============================================================
    # 3. VOLUME FEATURES
    # ============================================================
    def add_volume_features(self):
        v = self.df['volume']
        self.features['vol_ratio_5'] = v / v.rolling(5).mean()
        self.features['vol_ratio_20'] = v / v.rolling(20).mean()
        self.features['vol_ratio_hour'] = v / v.rolling(60).mean()
        self.features['vol_change'] = v.pct_change()
        self.features['vol_accel'] = self.features['vol_change'] - self.features['vol_change'].shift(1)
        self.features['vol_sma_5'] = v.rolling(5).mean()
        self.features['vol_sma_20'] = v.rolling(20).mean()
        self.features['volume_price_corr_10'] = self.df['close'].rolling(10).corr(v)

        # vectorized OBV (loop removed)
        sign = np.sign(self.df['close'].diff()).fillna(0.0)
        obv = (sign * v).cumsum()
        self.features['obv'] = obv
        self.features['obv_slope_10'] = obv.diff(10)

        self.features['vwap_dist'] = self._calculate_vwap_distance()
        return self

    def _calculate_vwap_distance(self):
        """Distance from session VWAP (daily reset), vectorized."""
        c, h, l, v = self.df['close'], self.df['high'], self.df['low'], self.df['volume']
        dates = np.asarray(self.df.index.date)
        tp = (h + l + c) / 3.0
        cum_pv = (tp * v).groupby(dates).cumsum()
        cum_v = v.groupby(dates).cumsum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
        return ((c - vwap) / c).fillna(0.0)

    # ============================================================
    # 4. ORDER FLOW & MICROSTRUCTURE
    # ============================================================
    def add_microstructure(self):
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        self.features['delta_proxy'] = (c - o) / (h - l + 1e-9)
        self.features['buying_climax'] = (
            (c > o) & (self.features['upper_wick'] > 0.6) &
            (self.features['vol_ratio_5'] > 2.0)).astype(int)
        self.features['selling_climax'] = (
            (c < o) & (self.features['lower_wick'] > 0.6) &
            (self.features['vol_ratio_5'] > 2.0)).astype(int)
        self.features['effort_result'] = (
            self.features['vol_ratio_5'] / (abs(c.pct_change()) * 100 + 1e-9))
        self.features['absorption_high'] = (
            (h == h.rolling(3).max()) & (self.features['vol_ratio_5'] > 1.5) &
            (abs(c - o) < (h - l) * 0.3)).astype(int)
        self.features['absorption_low'] = (
            (l == l.rolling(3).min()) & (self.features['vol_ratio_5'] > 1.5) &
            (abs(c - o) < (h - l) * 0.3)).astype(int)
        return self

    # ============================================================
    # 5. TIME & SEASONALITY (cyclical -> sine/cosine)
    # ============================================================
    def add_time_features(self):
        """Cyclical date/time features as sine/cosine pairs (raw values
        removed per requirement); session flags kept as binary features."""
        idx = self.df.index
        cyclical = [
            ('hour', np.asarray(idx.hour, dtype=float), 24.0),
            ('minute', np.asarray(idx.minute, dtype=float), 60.0),
            ('minute_of_day', np.asarray(idx.hour * 60 + idx.minute, dtype=float), 1440.0),
            ('day_of_week', np.asarray(idx.dayofweek, dtype=float), 7.0),
            ('month', np.asarray(idx.month, dtype=float), 12.0),
        ]
        for name, values, period in cyclical:
            ang = 2.0 * np.pi * values / period
            self.features[f'{name}_sin'] = np.sin(ang)
            self.features[f'{name}_cos'] = np.cos(ang)

        self.features['is_premarket'] = ((idx.hour < 9) | ((idx.hour == 9) & (idx.minute < 30))).astype(int)
        self.features['is_open_hour'] = ((idx.hour == 9) & (idx.minute >= 30) | (idx.hour == 10)).astype(int)
        self.features['is_midday'] = ((idx.hour >= 11) & (idx.hour <= 13)).astype(int)
        self.features['is_close_hour'] = (idx.hour >= 14).astype(int)
        self.features['is_last_30min'] = ((idx.hour == 15) & (idx.minute >= 30)).astype(int)
        self.features['is_monday'] = (idx.dayofweek == 0).astype(int)
        self.features['is_friday'] = (idx.dayofweek == 4).astype(int)
        return self

    # ============================================================
    # 6. STATISTICAL / DISTRIBUTION FEATURES
    # ============================================================
    def add_statistical_features(self):
        c = self.df['close']
        ret = c.pct_change()
        for window in [10, 20]:
            self.features[f'skew_{window}'] = ret.rolling(window).skew()
            self.features[f'kurt_{window}'] = ret.rolling(window).kurt()
            self.features[f'zscore_{window}'] = (
                (c - c.rolling(window).mean()) / c.rolling(window).std())
            self.features[f'pct_rank_{window}'] = (
                c.rolling(window).apply(_pct_rank_last, raw=True))
        for window in [10, 20]:
            self.features[f'entropy_{window}'] = (
                np.sign(ret).rolling(window).apply(_entropy_last, raw=True))

        # vectorized serial correlation (loop-free)
        self.features['autocorr_1'] = ret.rolling(20).corr(ret.shift(1))
        self.features['autocorr_5'] = ret.rolling(50).corr(ret.shift(5))
        return self

    # ============================================================
    # 7. SUPPORT/RESISTANCE & MARKET STRUCTURE
    # ============================================================
    def add_market_structure(self):
        """Swing highs/lows (trailing only) and causal opening-range breakout
        (look-ahead fixes vs _1_kimi)."""
        h, l, c = self.df['high'], self.df['low'], self.df['close']

        # FIXED: trailing 3-bar swings (no future bars)
        self.features['is_swing_high'] = (
            (h > h.shift(1)) & (h > h.shift(2))).astype(int)
        self.features['is_swing_low'] = (
            (l < l.shift(1)) & (l < l.shift(2))).astype(int)

        self.features['dist_day_high'] = (c - h.expanding().max()) / c
        self.features['dist_day_low'] = (c - l.expanding().min()) / c

        # FIXED: opening range expands causally inside 09:30-10:00 and is
        # held for the rest of the session; NaN before 09:30.
        idx = self.df.index
        minute_of_day = np.asarray(idx.hour * 60 + idx.minute)
        in_or = (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 10 * 60)
        dates = np.asarray(idx.date)
        or_high = h.where(in_or).groupby(dates).cummax().groupby(dates).ffill()
        or_low = l.where(in_or).groupby(dates).cummin().groupby(dates).ffill()
        self.features['or_high_dist'] = (c - or_high) / c
        self.features['or_low_dist'] = (c - or_low) / c

        self.features['consec_up'] = (
            (c > c.shift(1)).astype(int).groupby(
                ((c > c.shift(1)) != (c.shift(1) > c.shift(2))).cumsum()).cumsum())
        self.features['consec_down'] = (
            (c < c.shift(1)).astype(int).groupby(
                ((c < c.shift(1)) != (c.shift(1) < c.shift(2))).cumsum()).cumsum())
        return self

    # ============================================================
    # 8. CROSS-MARKET FEATURES (optional, unchanged semantics)
    # ============================================================
    def add_cross_market(self, spy_df=None, vix_df=None):
        if spy_df is not None:
            self.features['spy_lead_1m'] = spy_df['close'].pct_change().shift(1)
            self.features['spy_corr_10'] = self.df['close'].rolling(10).corr(spy_df['close'])
            self.features['es_spy_ratio'] = self.df['close'] / spy_df['close']
        if vix_df is not None:
            self.features['vix_level'] = vix_df['close']
            self.features['vix_change_1m'] = vix_df['close'].pct_change()
            self.features['vix_change_5m'] = vix_df['close'].pct_change(5)
            self.features['es_vix_corr_20'] = self.df['close'].rolling(20).corr(vix_df['close'])
        return self

    # ============================================================
    # 9. NORMALIZATION
    # ============================================================
    def _sin_cos_columns(self):
        return [c for c in self.features.columns
                if c.endswith(self.SIN_COS_SUFFIXES)]

    def normalize(self, method: str = 'global', expanding_min_periods: int = 1000):
        """Standard-scale all general features (everything except the
        sine/cosine time pairs). method='global' fits mean/std on the whole
        sample (as requested); method='expanding' fits causally."""
        if self.features is None or self.features.empty:
            raise RuntimeError('build features before normalize()')
        sin_cos = self._sin_cos_columns()
        general = [c for c in self.features.columns if c not in sin_cos]
        x = self.features[general]
        if method == 'expanding':
            mean = x.expanding(min_periods=expanding_min_periods).mean()
            std = x.expanding(min_periods=expanding_min_periods).std(ddof=0)
            norm = (x - mean) / std.mask(std == 0, 1.0)
        else:
            mean = x.mean()
            std = x.std(ddof=0)
            # zero-variance columns (e.g. rare-event flags) center to 0
            norm = (x - mean) / std.mask(std == 0, 1.0)
        self.features_norm = pd.concat(
            [norm, self.features[sin_cos]], axis=1)[self.features.columns]
        self.norm_method = method
        return self

    # ============================================================
    # 10. BUILD ALL
    # ============================================================
    def build_all(self):
        (self
         .add_returns()
         .add_candle_features()
         .add_momentum()
         .add_linreg_slopes()
         .add_rsi_divergence()
         .add_market_regime()
         .add_rolling_vwap()
         .add_volatility()
         .add_volume_features()
         .add_microstructure()
         .add_time_features()
         .add_statistical_features()
         .add_market_structure())
        # +/-inf (e.g. volume pct_change from a zero-volume bar) would poison
        # the global standard-scaler statistics -> treat as missing
        self.features = self.features.replace([np.inf, -np.inf], np.nan)
        return self

    def get_features(self, normalized: bool = True) -> pd.DataFrame:
        """Feature DataFrame with NaN rows dropped."""
        frame = self.features_norm if normalized else self.features
        if frame is None:
            raise RuntimeError('run normalize() first (or pass normalized=False)')
        return frame.dropna()

    # ============================================================
    # 11. CAUSALITY SELF-CHECK (fixed-window features)
    # ============================================================
    #: features whose value at bar i depends only on bars <= i AND only on a
    #: fixed lookback window (safe for truncation testing)
    CAUSALITY_TEST_COLS = [
        'ret_5m', 'body_pct', 'upper_wick', 'roc_10', 'slope_10', 'slope_20',
        'slope15_close', 'slope15_ema6', 'slope15_ema9', 'slope15_ema20',
        'slope15_ema50', 'rsi6_div_close', 'rsi6_bull_div_high',
        'rsi6_bear_div_low', 'rsi9_div_close', 'rsi9_bull_div_high',
        'rsi9_bear_div_low', 'atr_10', 'bb_position', 'vol_ratio_20',
        'obv_slope_10', 'vwap_dist', 'is_swing_high', 'is_swing_low',
        'or_high_dist', 'or_low_dist', 'hour_sin', 'minute_of_day_cos',
        'zscore_20', 'pct_rank_20', 'entropy_10', 'autocorr_1', 'consec_up',
        'regime_trend', 'regime_vol', 'market_regime',
        'rvwap_3', 'rvwap_3_std', 'rvwap_3_up1', 'rvwap_3_dn2',
        'rvwap_9', 'rvwap_9_dist', 'rvwap_9_pos',
        'rvwap_20', 'rvwap_20_up2', 'rvwap_20_dn1',
    ]

    @classmethod
    def check_causality(cls, df: pd.DataFrame, tail_bars: int = 3000,
                        n_check: int = 3, tol: float = 1e-8) -> dict:
        """Truncation test on the last bars: features computed on data[:i+1]
        must equal the same row computed on the full slice. Returns a dict of
        mismatched (row, column) -> diff for the tested fixed-window columns."""
        df_slice = df.iloc[-tail_bars:]
        ref = cls(df_slice).build_all().features
        n = len(df_slice)
        bad = {}
        for k in range(1, n_check + 1):
            pos = n - k  # test row at position `pos`
            truncated = cls(df_slice.iloc[:pos + 1]).build_all().features
            for col in cls.CAUSALITY_TEST_COLS:
                if col not in ref.columns or col not in truncated.columns:
                    continue
                a = ref.iloc[pos][col]
                b = truncated.iloc[-1][col]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) != pd.isna(b) or abs(float(a) - float(b)) > tol:
                    bad[(str(df_slice.index[pos]), col)] = (
                        float(a) - float(b) if not (pd.isna(a) or pd.isna(b)) else np.nan)
        return bad


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description='S0001 General Feature engineering (v2)')
    ap.add_argument('--start', default=None, help='start date YYYY-MM-DD')
    ap.add_argument('--end', default=None, help='end date YYYY-MM-DD')
    ap.add_argument('--ticker', default='ES')
    ap.add_argument('--normalize-method', choices=['global', 'expanding'],
                    default='global')
    ap.add_argument('--no-output', action='store_true')
    ap.add_argument('--check-causality', action='store_true',
                    help='run truncation self-test on fixed-window features')
    args = ap.parse_args()

    print(f"Loading {args.ticker} 1-min bars from IBTradingDb.ticker1Min ...")
    df = load_data_from_db(args.ticker, args.start, args.end)
    if df.empty:
        print('ERROR: no rows returned')
        return
    print(f"Loaded {len(df)} bars: {df.index[0]} .. {df.index[-1]}")

    fe = ESFeatureEngineer(df)
    fe.build_all()
    print(f"Raw features: {fe.features.shape[1]} columns")

    if args.check_causality:
        bad = ESFeatureEngineer.check_causality(df)
        if bad:
            print(f"[FAILED] look-ahead detected in {len(bad)} values:")
            for (ts, col), diff in list(bad.items())[:20]:
                print(f"  {ts}  {col}  diff={diff}")
        else:
            print('[PASSED] no look-ahead in tested fixed-window features')

    fe.normalize(method=args.normalize_method)
    feats = fe.get_features(normalized=True)
    print(f"Normalized features ({args.normalize_method}): "
          f"{feats.shape[0]} rows x {feats.shape[1]} columns "
          f"(after dropna; sin/cos columns kept as-is)")

    if not args.no_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        out = os.path.join(OUTPUT_DIR, f'S0001_GeneralFeature_2_{ts}.csv')
        feats.to_csv(out, index=True, index_label='datetime')
        print(f'Output: {out}')


if __name__ == '__main__':
    main()
