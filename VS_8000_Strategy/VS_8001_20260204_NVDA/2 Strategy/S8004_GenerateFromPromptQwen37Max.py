"""
Strategy IRB1000 V1 for NVDA
Time frame: 5 minute
Period: 09:30 to 16:00 regular NASDAQ trading hour US NY time zone
Symbol: NVDA
"""

import pandas as pd
import numpy as np
from datetime import datetime, time


class strategyIRB1000_V1:
    """
    IRB1000 V1 Strategy class for NVDA 5-minute trading.
    
    This class implements the IRB (Inside Range Breakout) strategy with EMA analysis
    and can be used with backtesting, walk-forward testing, Monte Carlo simulation,
    and live trading systems.
    """
    
    def __init__(self, mode='backtest'):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        mode : str
            'backtest' for backtesting mode, 'live' for live trading mode
        """
        self.mode = mode
        self.position_size = 110  # Number of NVDA stocks per trade
        self.slippage = 0.03
        self.stop_limit_offset = 0.06
        
    def calculate_indicators(self, df):
        """
        Calculate all technical indicators required for the strategy.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all calculated indicators
        """
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Basic price variables
        data['H'] = data['High']
        data['L'] = data['Low']
        data['O'] = data['Open']
        data['C'] = data['Close']
        data['V'] = data['Volume']
        
        # IRB Range calculations
        data['iRbhLRange'] = data['H'] - data['L']
        data['iRbUpperTh'] = data['H'] - data['iRbhLRange'] * 0.45
        data['iRbLowerTh'] = data['L'] + data['iRbhLRange'] * 0.45
        
        # Bullish and Bearish conditions
        data['iRbBullish'] = np.where(
            (data['C'] > data['L']) & 
            (data['O'] > data['L']) & 
            (data['C'] < data['iRbLowerTh']) & 
            (data['O'] < data['iRbLowerTh']),
            1, 0
        )
        
        data['iRbBearish'] = np.where(
            (data['C'] < data['H']) & 
            (data['O'] < data['H']) & 
            (data['C'] > data['iRbUpperTh']) & 
            (data['O'] > data['iRbUpperTh']),
            1, 0
        )
        
        # EMA calculations
        data['ema10'] = data['C'].ewm(span=10, adjust=False).mean()
        data['ema20'] = data['C'].ewm(span=20, adjust=False).mean()
        
        # Linear regression slope of ema10 over 3 bars
        data['regSlopema10'] = self._calculate_regression_slope(data['ema10'], 3)
        
        # EMA20 difference from close
        data['ema20DiffCAbsRound0'] = np.floor(np.abs(data['ema20'] - data['C']) * 100)
        
        # Rank based on ema20DiffCAbsRound0
        data['ema20DiffCAbsRank'] = self._calculate_rank(data['ema20DiffCAbsRound0'])
        
        # Calculate rt10 (difference between next 5 bar close and current close)
        data['rt10'] = data['C'].shift(-5) - data['C']
        
        # Calculate rank statistics over last 100 bars
        rank_stats = self._calculate_rank_statistics(data['ema20DiffCAbsRank'], data['rt10'], 100)
        data['sumEma20DiffCAbsRank'] = rank_stats['sum_rank']
        data['sumEma20DiffCAbsUpRankper'] = rank_stats['sum_up_rank']
        data['pRtEma20Rankper'] = rank_stats['percentage']
        
        # ATR calculation (7 periods)
        data['ATR7'] = self._calculate_atr(data, 7)
        
        return data
    
    def _calculate_regression_slope(self, series, period):
        """
        Calculate linear regression slope over a rolling window.
        
        Parameters:
        -----------
        series : pandas.Series
            Input series
        period : int
            Window size
            
        Returns:
        --------
        pandas.Series
            Regression slope values
        """
        def slope_calc(y):
            if len(y) < period:
                return np.nan
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        return series.rolling(window=period).apply(slope_calc, raw=True)
    
    def _calculate_rank(self, values):
        """
        Calculate rank based on predefined thresholds.
        
        Parameters:
        -----------
        values : pandas.Series
            Input values
            
        Returns:
        --------
        pandas.Series
            Rank values (1-5)
        """
        rank = pd.Series(np.zeros(len(values)), index=values.index)
        rank[(values >= 0) & (values < 10)] = 1
        rank[(values >= 10) & (values < 21)] = 2
        rank[(values >= 21) & (values < 35)] = 3
        rank[(values >= 35) & (values < 66)] = 4
        rank[values >= 66] = 5
        return rank
    
    def _calculate_rank_statistics(self, ranks, rt10, window):
        """
        Calculate rank statistics over a rolling window.
        
        Parameters:
        -----------
        ranks : pandas.Series
            Rank values
        rt10 : pandas.Series
            Future returns (next 5 bar close - current close)
        window : int
            Rolling window size
            
        Returns:
        --------
        dict
            Dictionary with sum_rank, sum_up_rank, and percentage
        """
        sum_rank = pd.Series(np.zeros(len(ranks)), index=ranks.index)
        sum_up_rank = pd.Series(np.zeros(len(ranks)), index=ranks.index)
        percentage = pd.Series(np.zeros(len(ranks)), index=ranks.index)
        
        current_rank = ranks.iloc[-1] if len(ranks) > 0 else 0
        
        for i in range(window, len(ranks)):
            window_ranks = ranks.iloc[i-window:i]
            window_rt10 = rt10.iloc[i-window:i]
            
            # Count occurrences of current rank
            sum_rank.iloc[i] = (window_ranks == current_rank).sum()
            
            # Count occurrences where current rank and rt10 > 0
            sum_up_rank.iloc[i] = ((window_ranks == current_rank) & (window_rt10 > 0)).sum()
            
            # Calculate percentage
            if sum_rank.iloc[i] > 0:
                percentage.iloc[i] = (sum_up_rank.iloc[i] * 100) / sum_rank.iloc[i]
        
        return {
            'sum_rank': sum_rank,
            'sum_up_rank': sum_up_rank,
            'percentage': percentage
        }
    
    def _calculate_atr(self, data, period):
        """
        Calculate Average True Range.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with High, Low, Close columns
        period : int
            ATR period
            
        Returns:
        --------
        pandas.Series
            ATR values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, df):
        """
        Generate buy/sell/short/cover signals based on the strategy logic.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with all indicators calculated
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with signal columns
        """
        data = df.copy()
        
        # Initialize signal columns
        data['buy00'] = 0
        data['short00'] = 0
        data['buy_signal'] = 0
        data['short_signal'] = 0
        data['sell_signal'] = 0
        data['cover_signal'] = 0
        
        # Buy signal conditions
        data['buy00'] = np.where(
            (data['iRbBullish'] == 1) &
            (data['regSlopema10'] > data['regSlopema10'].shift(1)) &
            (data['pRtEma20Rankper'] > 50) &
            (data['ema20DiffCAbsRank'] >= 3) &
            (data['pRtEma20Rankper'] >= data['pRtEma20Rankper'].shift(1)) &
            (data['pRtEma20Rankper'] >= data['pRtEma20Rankper'].shift(2)),
            1, 0
        )
        
        # Short signal conditions
        data['short00'] = np.where(
            (data['iRbBearish'] == 1) &
            (data['regSlopema10'] < data['regSlopema10'].shift(1)) &
            (data['pRtEma20Rankper'] < 50) &
            (data['ema20DiffCAbsRank'] >= 3) &
            (data['pRtEma20Rankper'] <= data['pRtEma20Rankper'].shift(1)) &
            (data['pRtEma20Rankper'] <= data['pRtEma20Rankper'].shift(2)),
            1, 0
        )
        
        # Buy signal triggered when previous bar buy00 is true
        data['buy_signal'] = data['buy00'].shift(1).fillna(0).astype(int)
        
        # Short signal triggered when previous bar short00 is true
        data['short_signal'] = data['short00'].shift(1).fillna(0).astype(int)
        
        # Sell signal: 5 bars after buy
        data['sell_signal'] = data['buy_signal'].shift(5).fillna(0).astype(int)
        
        # Cover signal: 5 bars after short
        data['cover_signal'] = data['short_signal'].shift(5).fillna(0).astype(int)
        
        # Calculate entry and exit prices
        data = self._calculate_trade_prices(data)
        
        return data
    
    def _calculate_trade_prices(self, data):
        """
        Calculate trade entry and exit prices based on mode.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with signals
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trade prices
        """
        # Initialize price columns
        data['buy_price'] = np.nan
        data['sell_price'] = np.nan
        data['short_price'] = np.nan
        data['cover_price'] = np.nan
        data['stop_loss_buy'] = np.nan
        data['stop_limit_buy'] = np.nan
        data['stop_loss_short'] = np.nan
        data['stop_limit_short'] = np.nan
        
        if self.mode == 'live':
            # Live trading: use close price with slippage
            data.loc[data['buy_signal'] == 1, 'buy_price'] = data['C'] + self.slippage
            data.loc[data['sell_signal'] == 1, 'sell_price'] = data['C'] - self.slippage
            data.loc[data['short_signal'] == 1, 'short_price'] = data['C'] - self.slippage
            data.loc[data['cover_signal'] == 1, 'cover_price'] = data['C'] + self.slippage
        else:
            # Backtest: use open price
            data.loc[data['buy_signal'] == 1, 'buy_price'] = data['O']
            data.loc[data['sell_signal'] == 1, 'sell_price'] = data['O']
            data.loc[data['short_signal'] == 1, 'short_price'] = data['O']
            data.loc[data['cover_signal'] == 1, 'cover_price'] = data['O']
        
        # Stop loss calculations (same for both modes)
        # For buy positions
        data.loc[data['buy_signal'] == 1, 'stop_loss_buy'] = (
            data['C'] - data['ATR7'].shift(1) * 2.5
        )
        data.loc[data['buy_signal'] == 1, 'stop_limit_buy'] = (
            data['stop_loss_buy'] - self.stop_limit_offset
        )
        
        # For short positions
        data.loc[data['short_signal'] == 1, 'stop_loss_short'] = (
            data['C'] + data['ATR7'].shift(1) * 2.5
        )
        data.loc[data['short_signal'] == 1, 'stop_limit_short'] = (
            data['stop_loss_short'] + self.stop_limit_offset
        )
        
        return data
    
    def get_trades(self, df):
        """
        Extract trade list from signals.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with signals and prices
            
        Returns:
        --------
        list
            List of trade dictionaries
        """
        trades = []
        
        # Extract buy trades
        buy_signals = df[df['buy_signal'] == 1]
        for idx in buy_signals.index:
            trade = {
                'type': 'long',
                'entry_time': idx,
                'entry_price': df.loc[idx, 'buy_price'],
                'position_size': self.position_size,
                'stop_loss': df.loc[idx, 'stop_loss_buy'],
                'stop_limit': df.loc[idx, 'stop_limit_buy']
            }
            
            # Find corresponding sell signal
            sell_idx = df.index[df.index.get_loc(idx) + 5] if df.index.get_loc(idx) + 5 < len(df) else None
            if sell_idx is not None:
                trade['exit_time'] = sell_idx
                trade['exit_price'] = df.loc[sell_idx, 'sell_price']
            
            trades.append(trade)
        
        # Extract short trades
        short_signals = df[df['short_signal'] == 1]
        for idx in short_signals.index:
            trade = {
                'type': 'short',
                'entry_time': idx,
                'entry_price': df.loc[idx, 'short_price'],
                'position_size': self.position_size,
                'stop_loss': df.loc[idx, 'stop_loss_short'],
                'stop_limit': df.loc[idx, 'stop_limit_short']
            }
            
            # Find corresponding cover signal
            cover_idx = df.index[df.index.get_loc(idx) + 5] if df.index.get_loc(idx) + 5 < len(df) else None
            if cover_idx is not None:
                trade['exit_time'] = cover_idx
                trade['exit_price'] = df.loc[cover_idx, 'cover_price']
            
            trades.append(trade)
        
        return trades
    
    def run(self, df):
        """
        Run the complete strategy on the given data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw market data with Open, High, Low, Close, Volume columns
            
        Returns:
        --------
        tuple
            (processed_data, trades_list)
        """
        # Calculate indicators
        data = self.calculate_indicators(df)
        
        # Generate signals
        data = self.generate_signals(data)
        
        # Extract trades
        trades = self.get_trades(data)
        
        return data, trades


# Example usage
if __name__ == "__main__":
    # Example: Load data and run strategy
    print("Strategy IRB1000 V1 for NVDA")
    print("=" * 50)
    print("\nThis strategy class can be used with:")
    print("- VS_0003_test/backtest.py")
    print("- VS_0003_test/otherTest.py")
    print("- VS_0003_test/monteCarloSimulation.py")
    print("- VS_0003_test/walkForwardTest.py")
    print("- VS_9000_LiveTrade/5min/inCubation_intraday1.py (TBD)")
    print("\nExample usage:")
    print("-" * 50)
    print("""
    from strategyIRB1000_V1 import strategyIRB1000_V1
    
    # Initialize strategy
    strategy = strategyIRB1000_V1(mode='backtest')  # or mode='live'
    
    # Load your market data (5-minute bars)
    # df = pd.read_csv('your_data.csv')
    
    # Run strategy
    data, trades = strategy.run(df)
    
    # Analyze results
    print(f"Total trades: {len(trades)}")
    """)
