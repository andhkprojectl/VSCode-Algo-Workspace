import backtrader as bt
import pandas as pd
import numpy as np
from pyiqfeed import IQFeedConn, HistDataConn
from pyiqfeed.listeners import VerboseIQFeedListener
import datetime


# Function to fetch daily data using pyiqfeed
def fetch_daily_data(symbol, start_date, end_date):
    # Connect to IQFeed admin port
    admin_conn = IQFeedConn(name="Admin")
    admin_listener = VerboseIQFeedListener("AdminListener")
    admin_conn.add_listener(admin_listener)
    admin_conn.connect("username", "password", "product_id")  # Replace with actual credentials

    # Connect to historical data port
    hist_conn = HistDataConn(name="HistoricalData")
    hist_listener = VerboseIQFeedListener("HistListener")
    hist_conn.add_listener(hist_listener)
    hist_conn.connect("username", "password", "product_id")  # Replace with actual credentials

    # Request daily data
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    data = hist_conn.request_historical_data(
        ticker=symbol,
        start_time=start,
        end_time=end,
        interval_len=86400,  # 1 day in seconds
        interval_type='s'
    )

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open_p', 'high_p', 'low_p', 'close_p', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={
        'open_p': 'open',
        'high_p': 'high',
        'low_p': 'low',
        'close_p': 'close'
    })[['open', 'high', 'low', 'close', 'volume']]

    # Disconnect
    hist_conn.disconnect()
    admin_conn.disconnect()

    return df


# Fetch data (use dummy data if IQFeed unavailable)
try:
    df_dia = fetch_daily_data("DIA", "2011-01-01", "2025-04-30")
    df_ym = fetch_daily_data("@YM#C", "2011-01-01", "2025-04-30")
except Exception as e:
    print(f"Error fetching data: {e}. Using dummy data instead.")
    dates = pd.date_range('2011-01-01', '2025-04-30')
    df_dia = pd.DataFrame({
        'open': np.random.rand(len(dates)) * 100,
        'high': np.random.rand(len(dates)) * 100 + 5,
        'low': np.random.rand(len(dates)) * 100 - 5,
        'close': np.random.rand(len(dates)) * 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    df_ym = pd.DataFrame({
        'open': np.random.rand(len(dates)) * 10000,
        'high': np.random.rand(len(dates)) * 10000 + 50,
        'low': np.random.rand(len(dates)) * 10000 - 50,
        'close': np.random.rand(len(dates)) * 10000,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)


# Define the trading strategy
class MyStrategy(bt.Strategy):
    params = dict(
        percentile_days=100,
        sl1=3,
        pf1=3,
    )

    def __init__(self):
        self.atr15 = bt.indicators.ATR(self.data0, period=15)
        self.trade_day = None

    def next(self):
        if len(self.data1) > self.p.percentile_days:
            returns = [self.data1.close[i] - self.data1.close[i - 1] for i in range(-self.p.percentile_days, 0)]
            percentile_90 = np.percentile(returns, 90)
            current_return = self.data1.close[0] - self.data1.close[-1]
            if current_return >= percentile_90:
                self.buy_order = self.buy(data=self.data0, size=1, exectype=bt.Order.Market)
                self.trade_day = len(self)
        elif self.position and len(self) == self.trade_day + 1:
            self.sell_order = self.sell(data=self.data0, size=self.position.size, exectype=bt.Order.Market)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                entry_price = order.executed.price
                sl_price = entry_price - self.p.sl1 * self.atr15[0]
                tp_price = entry_price + self.p.pf1 * self.atr15[0]
                self.sell_order_sl = self.sell(data=self.data0, size=order.executed.size, exectype=bt.Order.Stop,
                                               price=sl_price)
                self.sell_order_tp = self.sell(data=self.data0, size=order.executed.size, exectype=bt.Order.Limit,
                                               price=tp_price)


# Set up cerebro for backtesting
cerebro = bt.Cerebro()
cerebro.broker.set_cash(10000)  # Initial capital

# Add data feeds
data_dia = bt.feeds.PandasData(dataname=df_dia)
data_ym = bt.feeds.PandasData(dataname=df_ym)
cerebro.adddata(data_dia)  # data0
cerebro.adddata(data_ym)  # data1

# Add strategy with optimization
cerebro.optstrategy(
    MyStrategy,
    percentile_days=range(50, 151, 10),
    sl1=range(1, 6, 1),
    pf1=range(1, 6, 1)
)

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')

# Run optimization
results = cerebro.run(maxcpus=1)  # Use 1 CPU to avoid threading issues with IQFeed

# Find the best run based on Sharpe ratio
best_sharpe = -float('inf')
best_strat = None
for strat in results:
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', -float('inf'))
    if sharpe and sharpe > best_sharpe:
        best_sharpe = sharpe
        best_strat = strat

if best_strat:
    # Extract statistics
    initial_cash = 10000
    final_value = best_strat.broker.getvalue()
    net_profit = final_value - initial_cash
    sharpe_ratio = best_strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
    max_drawdown = best_strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analysis = best_strat.analyzers.trades.get_analysis()
    num_trades = trade_analysis['total']['total']
    wins = trade_analysis['won']['total']
    losses = trade_analysis['lost']['total']
    win_ratio = wins / num_trades if num_trades > 0 else 0
    max_trade_drawdown = max([trade.get('pnl', {}).get('max', 0) for trade in trade_analysis.get('trades', [])],
                             default=0)
    annual_returns = best_strat.analyzers.annual_return.get_analysis()
    avg_annual_return = np.mean(list(annual_returns.values())) if annual_returns else 0
    pct_annual_return = avg_annual_return * 100

    # Print statistics
    print(f"\nBest Strategy Parameters:")
    print(f"Percentile Days: {best_strat.params.percentile_days}")
    print(f"Stop Loss Multiplier (sl1): {best_strat.params.sl1}")
    print(f"Stop Profit Multiplier (pf1): {best_strat.params.pf1}")
    print(f"\nPerformance Statistics:")
    print(f"Net Profit: ${net_profit:.2f}")
    print(f"Annual Return: {avg_annual_return:.4f}")
    print(f"% Annual Return: {pct_annual_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio if sharpe_ratio != 'N/A' else 'N/A'}")
    print(f"Number of Trades: {num_trades}")
    print(f"Max System Drawdown: {max_drawdown:.2f}%")
    print(f"Max Trade Drawdown: {max_trade_drawdown:.2f}")
    print(f"Number of Wins: {wins}")
    print(f"Number of Losses: {losses}")
    print(f"Win Ratio: {win_ratio:.2f}")

    # Plot equity curve (note: plotting may not work in all environments without a display)
    cerebro.plot(volume=False)

else:
    print("No valid backtest results obtained.")