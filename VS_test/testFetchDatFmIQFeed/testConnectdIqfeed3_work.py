# Get daily data in regular hour from DTN iqfeed
# 1. Get DIA 5min data from Nasdaq but only include 9:30 to 16:00
# 2. Remove pre-trading and post-trading period
# 3. agg to daily bar

import datetime
from operator import truediv

import pandas as pd
import datetime as dt
from datetime import datetime, date, timedelta
from pyiqfeed import HistoryConn

# try get daily from 30Min to get ride of Nasdaq pre-trading and post-trading hour
# only keep daily bar from regular trading hour 9:30 to 16:00.
def fetchIqFeedDailyFm30Min(conn, symbol1, start_dt, end_dt, write_output_2_file=True):
    # Create a History Connection
    all_data = []
    isConnCreateHere = False
    if (conn is None):
        conn = HistoryConn(name="history")
        conn.connect()
        isConnCreateHere = True
    try:

        # data = get_bar_data(symbol, interval_len, interval_type,"20250501 044500", "20250502 133000")  # May 1–2, 2025
        data = conn.request_bars_in_period(
            ticker=symbol1,
            interval_len=1800,  # 30 minutes (1800 seconds)
            interval_type="s",  # Seconds
            bgn_prd=start_dt,
            end_prd=end_dt,
            bgn_flt=datetime.strptime("09:30", "%H:%M").time(),
            end_flt=datetime.strptime("16:00", "%H:%M").time(),
            ascend=True,  # Oldest to latest
            max_bars=None,  # Fetch all available bars
            timeout=None
        )
        print ("data output")
        print(data)
        # Print header names
        if data is not None and len(data) > 0:
            header_names = data.dtype.names
            print("Data Header Names:", header_names)
        else:
            print("No data returned")

        # Convert to list of dictionaries
        records = [
            {
                # "datetime": bar["datetime"],  # Keep full timestamp
                "Date1": bar["date"],
                "Time1": bar["time"],
                "Open": bar["open_p"],
                "High": bar["high_p"],
                "Low": bar["low_p"],
                "Close": bar["close_p"],
                "AccVolume": bar["tot_vlm"],
                "TradeVolume": bar["prd_vlm"],
                "num_trade": bar["num_trds"]
            }
            for bar in data
        ]

        all_data.extend(records)
        df = pd.DataFrame(all_data)

        # df['datetime'] = pd.to_datetime(df['Date1'] + ' ' + df['Time1'])
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str))
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='%Y-%m-%d %H:%M:%S')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='mixed', errors='coerce')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='ISO8601')

        # Convert Time1 (nanoseconds) to time string
        df['TimeSeconds'] = df['Time1'] / 1_000_000_000  # Convert to seconds
        df['TimeStr'] = pd.to_timedelta(df['TimeSeconds'], unit='s').apply(
            lambda x: x.components.hours * 3600 + x.components.minutes * 60 + x.components.seconds).apply(
            lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}")

        # Combine Date1 and TimeStr into datetime
        df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['TimeStr'], format='%Y-%m-%d %H:%M:%S',
                                        errors='coerce')

        df.set_index('datetime', inplace=True)
        #
        print(df.columns)
        print(df.head())
        #

        daily = df.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'TradeVolume': 'sum',
            'num_trade' : 'sum'

        })


        # Drop NaNs (non-trading days)
        daily.dropna(inplace=True)

        # Show sample output
        print(daily.head())

        # Save to CSV (optional)
        if (write_output_2_file):
            daily.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")

        # conn.disconnect()
        return daily

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()




# Get IQFeed DTN
start_dt1 = datetime(2025, 4, 1)
end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
dailyData = fetchIqFeedDailyFm30Min(None, "DIA", start_dt1, end_dt1, True)
