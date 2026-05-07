import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from pyiqfeed import IQFeed
import datetime

# Fetch historical data using IQFeed
def fetch_iqfeed_data(symbol, start_date, end_date):
    iq = IQFeed()
    data = iq.get_historical_data(symbol, start_date, end_date, interval='d')
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Define the trading strategy
class PercentileStrategy(Strategy):
    percentile_days = 100  # Number of days for percentile calculation
    sl1 = 3        # Stop loss multiplier
    pf1 = 3        # Stop profit multiplier

    def init(self):
        # Calculate ATR15
        self.atr15 = self.I(self.calculate_atr, 15)
        # Get @YM#C returns from data
        self.ym_return = self.data['ym_return']
        # Calculate 90th percentile of @YM#C returns
        self.percentile = self.I(self.calculate_percentile, self.ym_return, self.percentile_days)

    def calculate_atr(self, n):
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()
        return atr

    def calculate_percentile(self, returns, days):
        return returns.rolling(days).apply(lambda x: np.percentile(x.dropna(), 90))

    def next(self):
        if self.position:
            # Check stop loss and stop profit
            atr_yesterday = self.atr15[-2]
            stop_loss = self.data['open'][-1] - self.sl1 * atr_yesterday
            stop_profit = self.data['open'][-1] + self.pf1 * atr_yesterday
            if self.data['low'][-1] <= stop_loss:
                self.position.close()
            elif self.data['high'][-1] >= stop_profit:
                self.position.close()
        else:
            # Buy condition: @YM#C return >= 90th percentile
            if self.ym_return[-1] >= self.percentile[-1]:
                self.buy(size=1, sl=self.data['open'][-1] - self.sl1 * self.atr15[-1], 
                         tp=self.data['open'][-1] + self.pf1 * self.atr15[-1])

# Main execution
if __name__ == "__main__":
    # Fetch data
    start_date = '2011-01-01'
    end_date = '2025-04-30'
    dia_data = fetch_iqfeed_data('DIA', start_date, end_date)
    ym_data = fetch_iqfeed_data('@YM#C', start_date, end_date)

    # Calculate @YM#C daily returns
    ym_data['return'] = ym_data['close'].pct_change()

    # Combine data for backtesting
    combined_data = dia_data.copy()
    combined_data['ym_return'] = ym_data['return']

    # Run backtest
    bt = Backtest(combined_data, PercentileStrategy, cash=10000, commission=.002)

    # Optimize parameters
    stats_optimized = bt.optimize(
        percentile_days=range(50, 150, 10),  # Optimize from 50 to 140, step 10
        sl1=range(1, 5, 1),                 # Optimize from 1 to 4, step 1
        pf1=range(1, 5, 1),                 # Optimize from 1 to 4, step 1
        maximize='Sharpe Ratio'
    )

    # Print statistics
    print("Optimized Backtest Statistics:")
    print(f"Net Profit: ${stats_optimized['Equity Final [$]'] - stats_optimized['Equity Peak [$]']:.2f}")
    print(f"Annual Return: {stats_optimized['Return (Ann.) [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats_optimized['Sharpe Ratio']:.2f}")
    print(f"Number of Trades: {stats_optimized['# Trades']}")
    print(f"Max System Drawdown: {stats_optimized['Max. Drawdown [%]']:.2f}%")
    print(f"Max Trade Drawdown: {stats_optimized['Max. Drawdown [%]']:.2f}%")
    print(f"Number of Wins: {stats_optimized['Win Rate [%]'] * stats_optimized['# Trades'] / 100:.0f}")
    print(f"Number of Losses: {(1 - stats_optimized['Win Rate [%]'] / 100) * stats_optimized['# Trades']:.0f}")
    print(f"Win Ratio: {stats_optimized['Win Rate [%]']:.2f}%")

    # Plot equity curve
    bt.plot()