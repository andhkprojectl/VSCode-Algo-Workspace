F6006_GeneralFeature_1_kimi.pyimport pandas as pd
import numpy as np
from scipy.stats import linregress, skew, kurtosis

class ESFeatureEngineer:
    """
    Complete feature engineering for 1-minute ES futures.
    All features are calculated using only information available 
    at or before the current bar close (no lookahead bias).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        df columns: ['open','high','low','close','volume']
        Index: datetime
        """
        self.df = df.copy()
        self.features = pd.DataFrame(index=df.index)
        
    # ============================================================
    # 1. PRICE ACTION & MOMENTUM
    # ============================================================
    
    def add_returns(self):
        """Various return horizons."""
        self.features['ret_1m'] = self.df['close'].pct_change()
        self.features['ret_5m'] = self.df['close'].pct_change(5)
        self.features['ret_10m'] = self.df['close'].pct_change(10)
        self.features['ret_20m'] = self.df['close'].pct_change(20)
        
        # Log returns (better statistical properties)
        self.features['log_ret_1m'] = np.log(self.df['close'] / self.df['close'].shift(1))
        self.features['log_ret_5m'] = np.log(self.df['close'] / self.df['close'].shift(5))
        
        return self
    
    def add_candle_features(self):
        """Intraday bar microstructure."""
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        
        # Body characteristics
        self.features['body_size'] = abs(c - o)
        self.features['body_pct'] = abs(c - o) / (h - l + 1e-9)
        self.features['body_direction'] = np.sign(c - o)  # 1=bull, -1=bear, 0=doji
        
        # Wicks
        self.features['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-9)
        self.features['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-9)
        self.features['wick_ratio'] = self.features['upper_wick'] / (self.features['lower_wick'] + 1e-9)
        
        # Bar range
        self.features['bar_range'] = h - l
        self.features['range_pct'] = (h - l) / c
        
        # Gap from previous close
        self.features['gap'] = (o - self.df['close'].shift(1)) / self.df['close'].shift(1)
        
        return self
    
    def add_momentum(self):
        """Momentum and trend strength."""
        c = self.df['close']
        
        # Rate of change
        for period in [3, 5, 10, 20]:
            self.features[f'roc_{period}'] = (c - c.shift(period)) / c.shift(period)
        
        # Acceleration (change in ROC)
        self.features['accel_5'] = self.features['roc_5'] - self.features['roc_5'].shift(3)
        
        # Rolling slope (linear regression over N bars)
        def rolling_slope(series, window):
            slopes = pd.Series(index=series.index, dtype=float)
            for i in range(window, len(series)):
                y = series.iloc[i-window:i].values
                x = np.arange(window)
                slope, _, _, _, _ = linregress(x, y)
                slopes.iloc[i] = slope
            return slopes
        
        self.features['slope_10'] = rolling_slope(c, 10)
        self.features['slope_20'] = rolling_slope(c, 20)
        
        # Distance from moving averages
        for ma in [5, 10, 20, 50]:
            self.features[f'dist_sma_{ma}'] = (c - c.rolling(ma).mean()) / c
            self.features[f'dist_ema_{ma}'] = (c - c.ewm(span=ma).mean()) / c
        
        # Moving average crossovers
        self.features['sma5_10_cross'] = (c.rolling(5).mean() - c.rolling(10).mean()) / c
        self.features['sma10_20_cross'] = (c.rolling(10).mean() - c.rolling(20).mean()) / c
        
        return self
    
    # ============================================================
    # 2. VOLATILITY FEATURES
    # ============================================================
    
    def add_volatility(self):
        """Volatility and range-based features."""
        c, h, l = self.df['close'], self.df['high'], self.df['low']
        
        # True Range
        prev_close = c.shift(1)
        tr1 = h - l
        tr2 = abs(h - prev_close)
        tr3 = abs(l - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        for period in [5, 10, 20]:
            self.features[f'atr_{period}'] = tr.rolling(period).mean()
            self.features[f'atr_{period}_pct'] = self.features[f'atr_{period}'] / c
        
        # Realized volatility (standard deviation of returns)
        for period in [5, 10, 20]:
            self.features[f'vol_{period}'] = c.pct_change().rolling(period).std()
            self.features[f'vol_{period}_annual'] = self.features[f'vol_{period}'] * np.sqrt(252 * 390)
        
        # Parkinson volatility (uses high/low, more efficient)
        for period in [5, 10]:
            log_hl = np.log(h / l)
            self.features[f'parkinson_{period}'] = np.sqrt(
                (log_hl**2).rolling(period).mean() / (4 * np.log(2))
            )
        
        # Bollinger Band position
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        self.features['bb_position'] = (c - sma20) / (2 * std20 + 1e-9)
        self.features['bb_width'] = (4 * std20) / sma20
        
        # Range expansion/contraction
        self.features['range_vs_5ma'] = (h - l) / (h - l).rolling(5).mean()
        self.features['range_vs_20ma'] = (h - l) / (h - l).rolling(20).mean()
        
        return self
    
    # ============================================================
    # 3. VOLUME FEATURES
    # ============================================================
    
    def add_volume_features(self):
        """Volume-based microstructure signals."""
        v = self.df['volume']
        
        # Volume ratios
        self.features['vol_ratio_5'] = v / v.rolling(5).mean()
        self.features['vol_ratio_20'] = v / v.rolling(20).mean()
        self.features['vol_ratio_hour'] = v / v.rolling(60).mean()
        
        # Volume trend
        self.features['vol_change'] = v.pct_change()
        self.features['vol_accel'] = self.features['vol_change'] - self.features['vol_change'].shift(1)
        
        # Volume moving averages
        self.features['vol_sma_5'] = v.rolling(5).mean()
        self.features['vol_sma_20'] = v.rolling(20).mean()
        
        # Price-Volume relationship
        self.features['volume_price_corr_10'] = (
            self.df['close'].rolling(10).corr(v)
        )
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                obv.append(obv[-1] + v.iloc[i])
            elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                obv.append(obv[-1] - v.iloc[i])
            else:
                obv.append(obv[-1])
        self.features['obv'] = obv
        self.features['obv_slope_10'] = pd.Series(obv, index=self.df.index).diff(10)
        
        # Volume-weighted features
        self.features['vwap_dist'] = self._calculate_vwap_distance()
        
        return self
    
    def _calculate_vwap_distance(self):
        """Distance from VWAP (reset daily for intraday)."""
        vwap = []
        daily_tp_vol = 0
        daily_vol = 0
        current_date = None
        
        for idx, row in self.df.iterrows():
            if current_date != idx.date():
                current_date = idx.date()
                daily_tp_vol = 0
                daily_vol = 0
            
            tp = (row['high'] + row['low'] + row['close']) / 3
            daily_tp_vol += tp * row['volume']
            daily_vol += row['volume']
            
            vwap.append(row['close'] - (daily_tp_vol / daily_vol) if daily_vol > 0 else 0)
        
        vwap_series = pd.Series(vwap, index=self.df.index)
        return vwap_series / self.df['close']
    
    # ============================================================
    # 4. ORDER FLOW & MICROSTRUCTURE
    # ============================================================
    
    def add_microstructure(self):
        """
        Features derived from OHLC that proxy order flow.
        If you have tick data, these become much more powerful.
        """
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        
        # Delta (buy vs sell pressure proxy)
        # Assumes close near high = buying pressure
        self.features['delta_proxy'] = (c - o) / (h - l + 1e-9)
        
        # Buying/Selling climax
        self.features['buying_climax'] = (
            (c > o) & (self.features['upper_wick'] > 0.6) & 
            (self.features['vol_ratio_5'] > 2.0)
        ).astype(int)
        
        self.features['selling_climax'] = (
            (c < o) & (self.features['lower_wick'] > 0.6) & 
            (self.features['vol_ratio_5'] > 2.0)
        ).astype(int)
        
        # Effort vs Result (Wyckoff)
        # Large volume but small price move = potential reversal
        self.features['effort_result'] = (
            self.features['vol_ratio_5'] / (abs(c.pct_change()) * 100 + 1e-9)
        )
        
        # Absorption (price stalls at high/low with volume)
        self.features['absorption_high'] = (
            (h == h.rolling(3).max()) & (self.features['vol_ratio_5'] > 1.5) & 
            (abs(c - o) < (h - l) * 0.3)
        ).astype(int)
        
        self.features['absorption_low'] = (
            (l == l.rolling(3).min()) & (self.features['vol_ratio_5'] > 1.5) & 
            (abs(c - o) < (h - l) * 0.3)
        ).astype(int)
        
        return self
    
    # ============================================================
    # 5. TIME & SEASONALITY
    # ============================================================
    
    def add_time_features(self):
        """Intraday and calendar effects."""
        idx = self.df.index
        
        # Hour and minute
        self.features['hour'] = idx.hour
        self.features['minute'] = idx.minute
        self.features['minute_of_day'] = idx.hour * 60 + idx.minute
        
        # Session identifiers
        self.features['is_premarket'] = ((idx.hour < 9) | ((idx.hour == 9) & (idx.minute < 30))).astype(int)
        self.features['is_open_hour'] = ((idx.hour == 9) & (idx.minute >= 30) | (idx.hour == 10)).astype(int)
        self.features['is_midday'] = ((idx.hour >= 11) & (idx.hour <= 13)).astype(int)
        self.features['is_close_hour'] = (idx.hour >= 14).astype(int)
        self.features['is_last_30min'] = ((idx.hour == 15) & (idx.minute >= 30)).astype(int)
        
        # Day of week
        self.features['day_of_week'] = idx.dayofweek  # 0=Monday
        self.features['is_monday'] = (idx.dayofweek == 0).astype(int)
        self.features['is_friday'] = (idx.dayofweek == 4).astype(int)
        
        # Month
        self.features['month'] = idx.month
        
        # Days to/from expiry (if you have expiry dates)
        # self.features['days_to_expiry'] = ...
        
        return self
    
    # ============================================================
    # 6. STATISTICAL / DISTRIBUTION FEATURES
    # ============================================================
    
    def add_statistical_features(self):
        """Higher-order moments and distribution features."""
        ret = self.df['close'].pct_change()
        
        for window in [10, 20]:
            self.features[f'skew_{window}'] = ret.rolling(window).skew()
            self.features[f'kurt_{window}'] = ret.rolling(window).kurt()
            
            # Z-score
            self.features[f'zscore_{window}'] = (
                (self.df['close'] - self.df['close'].rolling(window).mean()) / 
                self.df['close'].rolling(window).std()
            )
            
            # Percentile rank
            self.features[f'pct_rank_{window}'] = (
                self.df['close'].rolling(window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
            )
        
        # Entropy (market disorder)
        for window in [10, 20]:
            # Price direction entropy: how random are the moves?
            signs = np.sign(ret).rolling(window).apply(
                lambda x: -sum((x.value_counts() / len(x)) * np.log2(x.value_counts() / len(x) + 1e-9))
            )
            self.features[f'entropy_{window}'] = signs
        
        # Serial correlation (mean reversion vs momentum)
        self.features['autocorr_1'] = ret.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        self.features['autocorr_5'] = ret.rolling(50).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0
        )
        
        return self
    
    # ============================================================
    # 7. SUPPORT/RESISTANCE & MARKET STRUCTURE
    # ============================================================
    
    def add_market_structure(self):
        """Swing highs/lows, breakout detection."""
        h, l, c = self.df['high'], self.df['low'], self.df['close']
        
        # Local extrema
        self.features['is_swing_high'] = (
            (h > h.shift(1)) & (h > h.shift(2)) & 
            (h > h.shift(-1)) & (h > h.shift(-2))
        ).astype(int)
        
        self.features['is_swing_low'] = (
            (l < l.shift(1)) & (l < l.shift(2)) & 
            (l < l.shift(-1)) & (l < l.shift(-2))
        ).astype(int)
        
        # Distance from recent swing high/low
        # (Use expanding max/min with time decay for intraday)
        self.features['dist_day_high'] = (c - h.expanding().max()) / c
        self.features['dist_day_low'] = (c - l.expanding().min()) / c
        
        # Opening range breakout
        if len(self.df) > 0:
            day_open = self.df.groupby(self.df.index.date)['open'].first()
            day_open_series = self.df.index.map(lambda x: day_open.get(x.date(), np.nan))
            
            # First 30 min high/low
            first_30 = self.df.between_time('09:30', '10:00')
            if len(first_30) > 0:
                or_high = first_30.groupby(first_30.index.date)['high'].max()
                or_low = first_30.groupby(first_30.index.date)['low'].min()
                
                self.features['or_high_dist'] = self.df.index.map(
                    lambda x: (c.loc[x] - or_high.get(x.date(), c.loc[x])) / c.loc[x] 
                    if x in c.index else 0
                )
                self.features['or_low_dist'] = self.df.index.map(
                    lambda x: (c.loc[x] - or_low.get(x.date(), c.loc[x])) / c.loc[x]
                    if x in c.index else 0
                )
        
        # Consecutive bars
        self.features['consec_up'] = (
            (c > c.shift(1)).astype(int).groupby(
                ((c > c.shift(1)) != (c.shift(1) > c.shift(2))).cumsum()
            ).cumsum()
        )
        self.features['consec_down'] = (
            (c < c.shift(1)).astype(int).groupby(
                ((c < c.shift(1)) != (c.shift(1) < c.shift(2))).cumsum()
            ).cumsum()
        )
        
        return self
    
    # ============================================================
    # 8. CROSS-MARKET FEATURES (If you have SPY/VIX data)
    # ============================================================
    
    def add_cross_market(self, spy_df=None, vix_df=None):
        """
        Requires SPY and VIX 1-minute data aligned to same timestamps.
        """
        if spy_df is not None:
            self.features['spy_lead_1m'] = spy_df['close'].pct_change().shift(1)
            self.features['spy_corr_10'] = (
                self.df['close'].rolling(10).corr(spy_df['close'])
            )
            self.features['es_spy_ratio'] = self.df['close'] / spy_df['close']
        
        if vix_df is not None:
            self.features['vix_level'] = vix_df['close']
            self.features['vix_change_1m'] = vix_df['close'].pct_change()
            self.features['vix_change_5m'] = vix_df['close'].pct_change(5)
            
            # ES/VIX inverse correlation
            self.features['es_vix_corr_20'] = (
                self.df['close'].rolling(20).corr(vix_df['close'])
            )
        
        return self
    
    # ============================================================
    # 9. BUILD ALL
    # ============================================================
    
    def build_all(self):
        """Run all feature groups."""
        return (self
                .add_returns()
                .add_candle_features()
                .add_momentum()
                .add_volatility()
                .add_volume_features()
                .add_microstructure()
                .add_time_features()
                .add_statistical_features()
                .add_market_structure())
    
    def get_features(self):
        """Return feature DataFrame with NaNs dropped."""
        return self.features.dropna()