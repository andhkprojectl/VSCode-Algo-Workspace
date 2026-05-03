import pandas as pd
import os
import time
from datetime import datetime, timedelta
# Import the specific connection classes instead
from pyiqfeed import HistoryConn

# --- CONFIGURATION ---
SYMBOL = 'NVDA'
CSV_FILE = 'c:/tmp/nvda_iqfeed_1min_fm_iqfeed.csv'


def log_iqfeed_data():
    # Instead of the PyConnector, we initialize the History Connection directly
    # Ensure IQFeed (the software) is already running on your PC
    hist_conn = HistoryConn(name="NVDA_History_Logger")

    try:
        hist_conn.connect()
        print(f"Connected to IQFeed. Monitoring {SYMBOL}...")
    except Exception as e:
        print(f"Failed to connect to IQFeed: {e}")
        print("Make sure the IQFeed Connection Manager is running and you are logged in.")
        return

    try:
        while True:
            # Timing Logic
            now = datetime.now()
            next_run = (now + timedelta(minutes=1)).replace(second=1, microsecond=0)
            wait_seconds = (next_run - now).total_seconds()

            print(f"Waiting {wait_seconds:.2f}s...")
            time.sleep(wait_seconds)

            # Requesting 1-minute bars (interval_len=60, interval_type='s')
            data = hist_conn.request_bars(
                symbol="NVDA",
                interval_len=60,
                interval_type='s',
                datapoints=1
            )

            if data is not None and len(data) > 0:
                # IQFeed returns a numpy structured array
                last_bar = data[-1]

                # IQFeed date/time is usually in the 'datetime' field of the record
                bar_time = last_bar['datetime']

                # 5. Data Formatting
                data_row = {
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'current_time': datetime.now().strftime('%H:%M:%S'),
                    'last_bar_time': bar_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'O': float(last_bar['open']),
                    'H': float(last_bar['high']),
                    'L': float(last_bar['low']),
                    'C': float(last_bar['close']),
                    'Volume': int(last_bar['period_volume'])
                }

                df = pd.DataFrame([data_row])

                # 6. Append to CSV
                file_exists = os.path.isfile(CSV_FILE)
                df.to_csv(CSV_FILE, mode='a', index=False, header=not file_exists)

                print(f"IQFeed Logged: {data_row['last_bar_time']} | Close: {data_row['C']}")
            else:
                print("IQFeed returned no data. Check symbol or connection.")

    except KeyboardInterrupt:
        print("Stopping logger...")
    finally:
        hist_conn.disconnect()


if __name__ == "__main__":
    log_iqfeed_data()