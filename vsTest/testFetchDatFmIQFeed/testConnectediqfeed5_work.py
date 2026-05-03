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
        data = conn.request_daily_data_for_dates(symbol1, start_dt, end_dt)
        # print (data)
        # print (list(data))

        records = [
            {
                # "datetime": bar["datetime"],  # Keep full timestamp
                "Date": bar["date"],
                "Open": bar["open_p"],
                "High": bar["high_p"],
                "Low": bar["low_p"],
                "Close": bar["close_p"],
                "TradeVolume": bar["prd_vlm"],
                "OpenContract": bar["open_int"]
            }
            for bar in data
        ]
        # df = pd.DataFrame(data)
        df = pd.DataFrame(records)


        print (2)

        # conn.request_daily_data_for_dates()

        # Drop NaNs (non-trading days)
        df.dropna(inplace=True)

        # print(4)

        # Optional: Convert 'date' column to datetime and sort
        # df['datetime'] = pd.to_datetime(df['date'])
        # df = df.sort_values('datetime')

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
            conn.disconnect()    # Create a History Connection



# Get IQFeed DTN
start_dt1 = datetime(2011, 1, 1)
# end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
end_dt1 = datetime(2025, 4, 30)
#start_dt1 = "2011-01-01"
#end_dt1 = "2025-04-30"
dailyBar = fetchIqFeedDaily(None, "@YM#C", start_dt1, end_dt1, True)
