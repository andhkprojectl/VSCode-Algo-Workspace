# Get daily data in using
# 1. Get DIA 5min data from Nasdaq but only include 9:30 to 16:00
# 2. Remove pre-trading and post-trading period
# 3. agg to daily bar

import datetime
from operator import truediv

import pandas as pd
import datetime as dt
from datetime import datetime, date, timedelta
from pyiqfeed import HistoryConn

# try get daily bar directly, regardless of time, i.e. get from 00:00 to 23:59
def fetchIqFeedDaily(conn, symbol1, start_dt, end_dt, write_output_2_file=True):
    # Create a History Connection
    all_data = []
    isConnCreateHere = False
    if (conn is None):
        conn = HistoryConn(name="history")
        conn.connect()
        isConnCreateHere = True
    try:
        # print(1)
        # Get daily bars
        # bars = conn.request_daily_data(symbol1, start_dt, end_dt)
        bars = conn.request_daily_data_for_dates(
            ticker=symbol1,
            bgn_dt=start_dt,
            end_dt=end_dt,
            ascend=True,
            timeout=None
        )
        # conn.request_daily_data_for_dates()

        # print(2)

        # Convert to DataFrame
        df = pd.DataFrame(bars)

        # print(3)

        # Drop NaNs (non-trading days)
        df.dropna(inplace=True)

        # print(4)

        # Optional: Convert 'date' column to datetime and sort
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.sort_values('datetime')

        # print(5)

        # Show results
        # print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())
        print (df.head)

        # print(6)

        # Save to CSV (optional)
        if (write_output_2_file):
            df.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")

        # conn.disconnect()
        return df

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()



# Get IQFeed DTN
start_dt1 = datetime(2011, 1, 1)
end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
#start_dt1 = "2011-01-01"
#end_dt1 = "2025-04-30"
dailyBar = fetchIqFeedDaily(None, "@YM#C", start_dt1, end_dt1, True)
