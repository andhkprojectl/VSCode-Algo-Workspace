import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

def backtestStrategy(strategy1, inFileName):
    """
    Reads historical data, executes the defined strategy, calculates key performance metrics,
    and displays an interactive HTML dashboard.
    """
    
    # 1. Read Data and Format for the Backtester
    try:
        df = pd.read_csv(inFileName)
    except FileNotFoundError:
        print(f"Error: The file {inFileName} was not found.")
        return

    # Parse the specific datetime format: mm/dd/yyyy hh24:mi
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
    df.set_index('datetime', inplace=True)
    
    # The backtesting library expects capitalized column names
    df.rename(columns={
        'open': 'Open', 
        'high': 'High', 
        'low': 'Low', 
        'close': 'Close', 
        'volume': 'Volume'
    }, inplace=True)
    
    # Ensure data is sorted chronologically
    df.sort_index(ascending=True, inplace=True)

    # 2. Extract configuration from the strategy class (if defined), otherwise use defaults
    commission_rate = getattr(strategy1, 'commission', 0.000)
    # Note: Slippage can be handled by adjusting fill prices or using margin. 
    # For standard equity backtesting, commission encompasses basic costs.
    initial_capital = 100000

    # 3. Initialize the Backtester
    bt = Backtest(
        df, 
        strategy1, 
        cash=initial_capital, 
        commission=commission_rate,
        trade_on_close=False # False means trades execute on the next bar's open
    )

    # 4. Run the Backtest
    print("Running Backtest...")
    stats = bt.run()
    
    # 5. Calculate and Map the Custom Requested Statistics
    trades = stats['_trades']
    
    # Safely handle metrics if no trades were executed
    if len(trades) > 0:
        winning_trades = trades[trades['ReturnPct'] > 0]
        losing_trades = trades[trades['ReturnPct'] <= 0]
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        max_trade_dd = trades['ReturnPct'].min() * 100 # Worst single trade percentage
        net_profit_val = stats['Equity Final [$]'] - initial_capital
    else:
        num_wins = 0
        num_losses = 0
        max_trade_dd = 0.0
        net_profit_val = 0.0

    percentile_days = stats['Exposure Time [%]']
    
    # 6. Display the Results
    print("\n" + "="*40)
    print("      BACKTEST RESULTS SUMMARY")
    print("="*40)
    print(f"Percentile Days (Exposure):   {percentile_days:.2f}%")
    print(f"Net Profit:                   ${net_profit_val:.2f}")
    print(f"Annual Return:                ${stats['Return (Ann.) [%]'] / 100 * initial_capital:.2f} (Est)")
    print(f"% Annual Return:              {stats['Return (Ann.) [%]']:.2f}%")
    print(f"Sharpe Ratio:                 {stats['Sharpe Ratio']:.3f}")
    print(f"Number of Trade:              {stats['# Trades']}")
    print(f"Max System Drawdown:          {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Max Trade Drawdown:           {max_trade_dd:.2f}%")
    print(f"Number of Wins:               {num_wins}")
    print(f"Number of Losses:             {num_losses}")
    print(f"Win Ratio:                    {stats['Win Rate [%]']:.2f}%")
    print("="*40)

    # 7. Output Result to Interactive Dashboard
    # This automatically opens a browser tab with a candlestick chart, equity curve, and trade markers
    bt.plot(open_browser=True)


# =====================================================================
# EXAMPLE USAGE & STRATEGY DEFINITION
# =====================================================================

def SMA(values, n):
    """Simple moving average helper function"""
    return pd.Series(values).rolling(n).mean()

class MACrossoverWithStops(Strategy):
    """
    Example Strategy fulfilling the requirements:
    - Buy on MA cross up, Sell (close long) on MA cross down.
    - Includes Stop Loss and Take Profit logic.
    - Defines commission.
    """
    # Strategy Parameters
    ma_fast = 5
    ma_slow = 50
    commission = 0.001 # 0.1% commission per trade
    
    # Parameters for risk management (percentages)
    stop_loss_pct = 0.02   # 2% stop loss
    take_profit_pct = 0.05 # 5% take profit
    
    def init(self):
        # Precompute indicators
        self.fast_ma = self.I(SMA, self.data.Close, self.ma_fast)
        self.slow_ma = self.I(SMA, self.data.Close, self.ma_slow)

    def next(self):
        current_price = self.data.Close[-1]
        
        # 1. CHECK FOR BUY SIGNAL (Fast MA crosses above Slow MA)
        if crossover(self.fast_ma, self.slow_ma):
            # Close any existing short positions before going long
            if self.position.is_short:
                self.position.close()
                
            if not self.position.is_long:
                # Calculate stop market and limit targets
                sl_price = current_price * (1 - self.stop_loss_pct)
                tp_price = current_price * (1 + self.take_profit_pct)
                
                # Execute Buy with attached Stop Loss and Take profit
                self.buy(sl=sl_price, tp=tp_price)
                
        # 2. CHECK FOR SELL/SHORT SIGNAL (Fast MA crosses below Slow MA)
        elif crossover(self.slow_ma, self.fast_ma):
            # Close existing long position
            if self.position.is_long:
                self.position.close()
                
            if not self.position.is_short:
                # Calculate stop market and limit targets for shorts
                sl_price = current_price * (1 + self.stop_loss_pct)
                tp_price = current_price * (1 - self.take_profit_pct)
                
                # Execute Short with attached Stop Loss and Take profit
                self.sell(sl=sl_price, tp=tp_price)


if __name__ == "__main__":
    # To run this, you need a sample CSV file matching your formatting.
    # Replace 'sample_data.csv' with your actual file path.
    # 
    # Example CSV Structure required in 'sample_data.csv':
    # datetime,date,time,high,low,close,open,volume,symbolName
    # 04/25/2024 09:30,04/25/2024,09:30:00,800.5,798.2,800.0,799.0,15000,NVDA
    
    data_file = "C:\\path\\to\\your\\sample_data.csv" # UPDATE THIS PATH
    
    # Run the backtest using the defined strategy class
    # backtestStrategy(MACrossoverWithStops, data_file)