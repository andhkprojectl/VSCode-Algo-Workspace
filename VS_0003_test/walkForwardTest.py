import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ==========================================
# 1. SAMPLE STRATEGY IMPLEMENTATION
# ==========================================
class MovingAverageCrossStrategy(Strategy):
    """
    Sample Strategy obeying all constraints:
    - Contains Buy/Sell signals (MA Cross)
    - Max 2 parameters for optimization (n1, n2)
    - Contains stop loss condition (using fixed percentage for demonstration)
    - No bare naked entries (every buy has a sell condition or stop loss)
    """
    # Max 2 parameters to optimize
    n1 = 5
    n2 = 50
    stop_loss_pct = 0.05 # 5% stop loss

    def init(self):
        # Calculate moving averages using close prices
        close_prices = self.data.Close
        self.ma1 = self.I(self.calculate_sma, close_prices, self.n1)
        self.ma2 = self.I(self.calculate_sma, close_prices, self.n2)

    def calculate_sma(self, data, window):
        return pd.Series(data).rolling(window).mean()

    def next(self):
        # Sell Signal: If MA1 crosses down MA2, close long position
        if crossover(self.ma2, self.ma1):
            if self.position.is_long:
                self.position.close()

        # Buy Signal: If MA1 crosses up MA2, buy with a stop loss
        elif crossover(self.ma1, self.ma2):
            if not self.position:
                # Execute Market Buy Order with a Stop Market Order for stop loss
                stop_price = self.data.Close[-1] * (1 - self.stop_loss_pct)
                self.buy(sl=stop_price)


# ==========================================
# 2. WALK-FORWARD BACKTEST FUNCTION
# ==========================================
def walkTestStrategy1(strategy1, inFileName, periodUnit, 
                      inSampleWindowStartDate, inSampleWindowEndDate, 
                      inSampleEndDate, outSampleStep, outSampleStepUnit,
                      optimize_params=None):
    """
    Walk-forward backtest a strategy and output results to a dashboard.
    """
    print(f"Loading data from {inFileName}...")
    
    # 1. Read Content
    df = pd.read_csv(inFileName)
    
    # 2. Format Dataframe for Backtesting.py
    # Requires standard naming: Open, High, Low, Close, Volume and a DatetimeIndex
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # Helper function to handle date steps
    def get_time_delta(step, unit):
        if unit == 'M': return relativedelta(months=step)
        elif unit == 'W': return relativedelta(weeks=step)
        elif unit == 'D': return relativedelta(days=step)
        else: raise ValueError("Invalid period unit. Use M, W, or D.")

    # Initialize Walk-Forward Variables
    current_is_start = pd.to_datetime(inSampleWindowStartDate)
    current_is_end = pd.to_datetime(inSampleWindowEndDate)
    final_is_end = pd.to_datetime(inSampleEndDate)
    
    step_delta = get_time_delta(outSampleStep, outSampleStepUnit)
    
    # Storage for Out-of-Sample Results
    oos_trades = []
    oos_equity_curves = []
    current_cash = 100000  # Initial starting capital
    
    if optimize_params is None:
        # Default optimization ranges if none provided
        optimize_params = {'n1': range(5, 15, 5), 'n2': range(20, 60, 10)}

    print("\nStarting Walk-Forward Optimization...")
    print("-" * 50)

    # 3. Walk-Forward Loop
    while current_is_end <= final_is_end:
        oos_start = current_is_end
        oos_end = oos_start + step_delta
        
        print(f"IS Window:  {current_is_start.date()} to {current_is_end.date()}")
        print(f"OOS Window: {oos_start.date()} to {oos_end.date()}")
        
        # Slice Data
        is_data = df[(df.index >= current_is_start) & (df.index < current_is_end)]
        oos_data = df[(df.index >= oos_start) & (df.index < oos_end)]
        
        if is_data.empty or oos_data.empty:
            print("  -> Insufficient data for this window. Stopping walk-forward.")
            break

        # In-Sample Optimization
        # commission=0.001 represents 0.1% slippage/commission charge
        bt_is = Backtest(is_data, strategy1, cash=current_cash, commission=0.001, exclusive_orders=True)
        stats_is = bt_is.optimize(**optimize_params, maximize='Equity Final [$]')
        
        best_params = {}
        for key in optimize_params.keys():
            best_params[key] = getattr(stats_is._strategy, key)
            
        print(f"  -> Best Params: {best_params}")

        # Out-Of-Sample Execution
        bt_oos = Backtest(oos_data, strategy1, cash=current_cash, commission=0.001, exclusive_orders=True)
        stats_oos = bt_oos.run(**best_params)
        
        # Record keeping
        if not stats_oos['_trades'].empty:
            oos_trades.append(stats_oos['_trades'])
            
        # Append equity curve, adjusted for continuity
        eq_curve = stats_oos['_equity_curve']['Equity']
        oos_equity_curves.append(eq_curve)
        
        # Update cash for next OOS period
        current_cash = stats_oos['Equity Final [$]']
        
        # Shift Windows Forward
        current_is_start += step_delta
        current_is_end += step_delta
        print("-" * 50)

    # 4. Consolidate Results & Dashboard Calculation
    if not oos_equity_curves:
        print("No trades executed during Out-of-Sample periods.")
        return

    # Combine all Out-of-Sample Equity Curves
    combined_equity = pd.concat(oos_equity_curves)
    combined_equity = combined_equity[~combined_equity.index.duplicated(keep='last')]
    
    if oos_trades:
        all_trades = pd.concat(oos_trades, ignore_index=True)
    else:
        all_trades = pd.DataFrame()

    # Calculate Requested Metrics
    initial_equity = 100000
    final_equity = combined_equity.iloc[-1]
    
    net_profit = final_equity - initial_equity
    percent_return = (net_profit / initial_equity) * 100
    
    # Days and Annualization
    total_days = (combined_equity.index[-1] - combined_equity.index[0]).days
    annual_return_pct = ((final_equity / initial_equity) ** (365.25 / total_days) - 1) * 100 if total_days > 0 else 0
    annual_return_val = initial_equity * (annual_return_pct / 100)
    
    # Drawdown
    rolling_max = combined_equity.cummax()
    drawdowns = (combined_equity - rolling_max) / rolling_max
    max_sys_dd = drawdowns.min() * 100
    
    if not all_trades.empty:
        num_trades = len(all_trades)
        wins = all_trades[all_trades['ReturnPct'] > 0]
        losses = all_trades[all_trades['ReturnPct'] <= 0]
        num_wins = len(wins)
        num_losses = len(losses)
        win_ratio = (num_wins / num_trades) * 100
        max_trade_dd = all_trades['ReturnPct'].min() * 100
    else:
        num_trades = num_wins = num_losses = win_ratio = max_trade_dd = 0

    # Sharpe Ratio (Assuming Daily risk-free rate of 0 for simplicity)
    daily_returns = combined_equity.pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

    # Percentile Days (Percentage of days where equity > previous day)
    percentile_days = (daily_returns > 0).mean() * 100

    # 5. Output Dashboard Data
    print("\n" + "="*40)
    print("WALK-FORWARD BACKTEST RESULT DASHBOARD")
    print("="*40)
    print(f"Percentile Days (Profitable): {percentile_days:.2f}%")
    print(f"Net Profit:                   ${net_profit:.2f}")
    print(f"Annual Return ($):            ${annual_return_val:.2f}")
    print(f"% Annual Return:              {annual_return_pct:.2f}%")
    print(f"Sharpe Ratio:                 {sharpe_ratio:.4f}")
    print(f"Number of Trade:              {num_trades}")
    print(f"Max System Drawdown:          {max_sys_dd:.2f}%")
    print(f"Max Trade Drawdown:           {max_trade_dd:.2f}%")
    print(f"Number of Wins:               {num_wins}")
    print(f"Number of Losses:             {num_losses}")
    print(f"Win Ratio:                    {win_ratio:.2f}%")
    print("="*40)

    # 6. Plotting the Chart
    plt.figure(figsize=(14, 7))
    plt.plot(combined_equity.index, combined_equity.values, label='OOS Equity Curve', color='blue')
    plt.title('Walk-Forward Out-Of-Sample Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.fill_between(drawdowns.index, drawdowns.values * initial_equity + initial_equity, initial_equity, color='red', alpha=0.1, label='Drawdown Area')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. EXECUTION EXAMPLE (To be run by user)
# ==========================================
if __name__ == "__main__":
    # Note: Replace 'your_data.csv' with your actual absolute file path
    # dummy_file_path = "C:\\path\\to\\your_data.csv"
    
    """
    Uncomment below to run. Ensure you have a valid CSV file.
    
    walkTestStrategy1(
        strategy1=MovingAverageCrossStrategy,
        inFileName="your_data.csv",
        periodUnit="M",
        inSampleWindowStartDate="2024/01/01",
        inSampleWindowEndDate="2024/06/01",
        inSampleEndDate="2024/11/01",
        outSampleStep=1,
        outSampleStepUnit="M",
        optimize_params={'n1': range(5, 15, 2), 'n2': range(20, 60, 5)}
    )
    """