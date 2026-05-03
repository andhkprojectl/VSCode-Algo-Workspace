import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from pyiqfeed import FeedService, BarConn
from localconfig import passwords
import socket
import datetime
import os
from backtesting.lib import compute_stats

# IQFeed credentials from localconfig/passwords.py
PRODUCT_ID = passwords.dtn_product_id
LOGIN = passwords.dtn_login
PASSWORD = passwords.dtn_password

# Define symbols and date range
SYMBOLS = ["DIA", "@YM#C"]
START_DATE = "20110101 000000"
END_DATE = "20250430 235959"
DATA_DIR = "data"
LOOKBACK_DAYS = 100
PERCENTILE = 90

def fetch_iqfeed_data(symbol, start_date, end_date):
    """Fetch daily bar data from IQFeed and save to CSV."""
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    csv_file = f"{DATA_DIR}/{symbol}_daily.csv"
    if os.path.exists(csv_file):
        print(f"Loading cached data for {symbol} from {csv_file}")
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df

    # Initialize IQFeed service
    feed = FeedService(product=PRODUCT_ID, version="1.0", login=LOGIN, password=PASSWORD)
    feed.start_service()

    # Connect to BarConn for daily bars
    bar_conn = BarConn(name="BarConn")
    feed.add_listener(bar_conn)
    bar_conn.connect()

    # Request historical daily bars
    message = f"HID,{symbol},1D,{start_date},{end_date},,,000000,235959,1\n"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9100))  # Historical data port
    sock.sendall(message.encode())
    
    # Buffer data
    data = []
    while True:
        chunk = sock.recv(4096).decode()
        if not chunk or "!ENDMSG!" in chunk:
            break
        data.append(chunk)
    
    sock.close()
    feed.remove_listener(bar_conn)
    bar_conn.disconnect()
    feed.stop_service()

    # Process data
    lines = "".join(data).split("\n")
    records = []
    for line in lines:
        if line and not line.startswith("!"):
            fields = line.split(",")
            if len(fields) >= 6:
                try:
                    dt = pd.to_datetime(fields[0], format="%Y-%m-%d %H:%M:%S")
                    open_price = float(fields[1])
                    high = float(fields[2])
                    low = float(fields[3])
                    close = float(fields[4])
                    volume = int(fields[5])
                    records.append([dt, open_price, high, low, close, volume])
                except ValueError:
                    continue

    # Create DataFrame
    df = pd.DataFrame(records, columns=["datetime", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Save to CSV
    df.to_csv(csv_file)
    print(f"Saved data for {symbol} to {csv_file}")
    return df

def prepare_data():
    """Fetch and align DIA and @YM#C data."""
    dia_data = fetch_iqfeed_data("DIA", START_DATE, END_DATE)
    ym_data = fetch_iqfeed_data("@YM#C", START_DATE, END_DATE)

    # Align dates
    common_dates = dia_data.index.intersection(ym_data.index)
    dia_data = dia_data.loc[common_dates]
    ym_data = ym_data.loc[common_dates]

    # Calculate daily returns for @YM#C
    ym_data['Daily_Return'] = (ym_data['Close'] - ym_data['Close'].shift(1)) / ym_data['Close'].shift(1)
    
    return dia_data, ym_data

class YMPercentileStrategy(Strategy):
    lookback_days = LOOKBACK_DAYS
    percentile = PERCENTILE

    def init(self):
        # Load @YM#C data
        self.ym_data = pd.read_csv(f"{DATA_DIR}/@YM#C_daily.csv", parse_dates=['datetime'], index_col='datetime')
        self.ym_data = self.ym_data.loc[self.data.index]
        self.ym_returns = self.ym_data['Daily_Return'].values

    def next(self):
        # Skip if not enough data or no position
        if len(self.data) < self.lookback_days + 1:
            return

        # Get today's index
        today_idx = len(self.data) - 1
        
        # Calculate 90th percentile of past 100 days' @YM#C returns
        lookback_returns = self.ym_returns[today_idx - self.lookback_days:today_idx]
        if len(lookback_returns) < self.lookback_days or np.isnan(lookback_returns).any():
            return
        percentile_value = np.percentile(lookback_returns, self.percentile)

        # Check today's @YM#C return
        today_ym_return = self.ym_returns[today_idx]
        if np.isnan(today_ym_return):
            return

        # Buy DIA next day at open if condition met
        if today_ym_return >= percentile_value and not self.position:
            self.buy()
        
        # Sell at next day's open if holding a position
        elif self.position:
            self.position.close()

def main():
    # Fetch and prepare data
    dia_data, ym_data = prepare_data()

    # Run backtest
    bt = Backtest(dia_data, YMPercentileStrategy, cash=10_000, commission=.002, exclusive_orders=True)
    stats = bt.run()

    # Extract and print statistics
    net_profit = stats['Equity Final [$]'] - stats['Equity Initial [$]']
    annual_return = stats['Return (Ann.) [%]']
    percent_annual_return = stats['Return (Ann.) [%]']
    sharpe_ratio = stats['Sharpe Ratio']
    num_trades = stats['# Trades']
    max_system_drawdown = stats['Max. Drawdown [%]']
    max_trade_drawdown = stats['Worst Trade [%]']
    win_rate = stats['Win Rate [%]']
    num_wins = int(num_trades * win_rate / 100)
    num_losses = num_trades - num_wins

    print(f"Backtest Statistics:")
    print(f"Net Profit: ${net_profit:.2f}")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"% Annual Return: {percent_annual_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Max System Drawdown: {max_system_drawdown:.2f}%")
    print(f"Max Trade Drawdown: {max_trade_drawdown:.2f}%")
    print(f"Number of Wins: {num_wins}")
    print(f"Number of Losses: {num_losses}")
    print(f"Win Ratio: {win_rate:.2f}%")

    # Plot equity curve
    bt.plot()

if __name__ == "__main__":
    main()