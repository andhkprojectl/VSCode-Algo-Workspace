import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from pyiqfeed import FeedService, BarConn
from localconfig import passwords
import socket
import datetime
import os
from backtesting.lib import compute_stats
from dateutil.relativedelta import relativedelta

# IQFeed credentials from localconfig/passwords.py
PRODUCT_ID = passwords.dtn_product_id
LOGIN = passwords.dtn_login
PASSWORD = passwords.dtn_password

# Define symbols and date range
SYMBOLS = ["DIA", "@YM#C"]
START_DATE = "20110101 000000"
END_DATE = "20250430 235959"
DATA_DIR = "data"

def fetch_iqfeed_data(symbol, start_date, end_date):
    """Fetch daily bar data from IQFeed and save to CSV."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    csv_file = f"{DATA_DIR}/{symbol}_daily.csv"
    if os.path.exists(csv_file):
        print(f"Loading cached data for {symbol} from {csv_file}")
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df

    feed = FeedService(product=PRODUCT_ID, version="1.0", login=LOGIN, password=PASSWORD)
    feed.start_service()
    bar_conn = BarConn(name="BarConn")
    feed.add_listener(bar_conn)
    bar_conn.connect()

    message = f"HID,{symbol},1D,{start_date},{end_date},,,000000,235959,1\n"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 9100))
    sock.sendall(message.encode())
    
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

    df = pd.DataFrame(records, columns=["datetime", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(csv_file)
    print(f"Saved data for {symbol} to {csv_file}")
    return df

def prepare_data():
    """Fetch and align DIA and @YM#C data."""
    dia_data = fetch_iqfeed_data("DIA", START_DATE, END_DATE)
    ym_data = fetch_iqfeed_data("@YM#C", START_DATE, END_DATE)

    common_dates = dia_data.index.intersection(ym_data.index)
    dia_data = dia_data.loc[common_dates]
    ym_data = ym_data.loc[common_dates]

    ym_data['Daily_Return'] = (ym_data['Close'] - ym_data['Close'].shift(1)) / ym_data['Close'].shift(1)
    
    return dia_data, ym_data

class YMPercentileStrategy(Strategy):
    lookback_days = 100
    percentile = 90

    def init(self):
        self.ym_data = pd.read_csv(f"{DATA_DIR}/@YM#C_daily.csv", parse_dates=['datetime'], index_col='datetime')
        self.ym_data = self.ym_data.loc[self.data.index]
        self.ym_returns = self.ym_data['Daily_Return'].values

    def next(self):
        if len(self.data) < self.lookback_days + 1:
            return

        today_idx = len(self.data) - 1
        lookback_returns = self.ym_returns[today_idx - self.lookback_days:today_idx]
        if len(lookback_returns) < self.lookback_days or np.isnan(lookback_returns).any():
            return
        percentile_value = np.percentile(lookback_returns, self.percentile)

        today_ym_return = self.ym_returns[today_idx]
        if np.isnan(today_ym_return):
            return

        if today_ym_return >= percentile_value and not self.position:
            self.buy()
        elif self.position:
            self.position.close()

def walk_forward_test(dia_data, ym_data):
    in_sample_start = pd.to_datetime("2011-01-01")
    in_sample_end = pd.to_datetime("2011-12-31")
    out_of_sample_months = 1
    end_date = pd.to_datetime("2025-03-31")

    results = []
    while in_sample_end < end_date:
        out_of_sample_start = in_sample_end + pd.DateOffset(days=1)
        out_of_sample_end = out_of_sample_start + relativedelta(months=out_of_sample_months) - pd.DateOffset(days=1)

        if out_of_sample_end > end_date:
            out_of_sample_end = end_date

        in_sample_data = dia_data.loc[in_sample_start:in_sample_end]
        out_of_sample_data = dia_data.loc[out_of_sample_start:out_of_sample_end]

        bt = Backtest(in_sample_data, YMPercentileStrategy, cash=10_000, commission=.002, exclusive_orders=True)
        stats = bt.optimize(
            lookback_days=range(80, 151, 5),
            percentile=range(85, 96, 1),
            maximize='Sharpe Ratio',
            constraint=lambda p: p.lookback_days > 0 and p.percentile > 0
        )

        optimized_lookback_days = stats._strategy.lookback_days
        optimized_percentile = stats._strategy.percentile

        bt_out = Backtest(out_of_sample_data, YMPercentileStrategy, cash=10_000, commission=.002, exclusive_orders=True)
        stats_out = bt_out.run(lookback_days=optimized_lookback_days, percentile=optimized_percentile)
        results.append(stats_out)

        in_sample_start = out_of_sample_start
        in_sample_end = out_of_sample_end

    combined_stats = compute_stats(results)
    print(combined_stats)
    bt_out.plot()

def main():
    dia_data, ym_data = prepare_data()
    walk_forward_test(dia_data, ym_data)

if __name__ == "__main__":
    main()