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
            returns = [self.data1.close[i] - self.data1.close[i-1] for i in range(-self.p.percentile_days, 0)]
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
                self.sell_order_sl = self.sell(data=self.data0, size=order.executed.size, exectype=bt.Order.Stop, price=sl_price)
                self.sell_order