import PyIQFeed as iq
import pandas as pd
from datetime import datetime, time

# Define trading hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 30)

def fetch_30min_bars(start_date, end_date):
    """
    Fetch 30-minute bar data for DIA from IQFeed within trading hours.
    """
    # Launch IQFeed service
    iqfeed = iq.FeedService()
    iqfeed.launch(headless=True)

    # Connect to admin port
    admin_conn = iq.AdminConn()
    admin_conn.connect()

    # Connect to history port
    hist_conn = iq.HistoryConn()
    hist_conn.connect()

    # Request historical bars
    symbol = 'DIA'
    interval_len = 1800  # 30 minutes in seconds
    bars = hist_conn.request_interval_bars(
        symbol=symbol,
        interval_len=interval_len,
        interval_type='s',  # seconds
        begin_date=start_date,
        end_date=end_date
    )

    # Disconnect from IQFeed
    hist_conn.disconnect()
    admin_conn.disconnect()

    # Convert to DataFrame
    df = pd.DataFrame(bars)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.index = df.index.tz_localize('US/Eastern')

    # Filter to trading hours (10:00 to 16:30 for 30-min bars ending at these times)
    df = df.between_time('10:00', '16:30', include_start=True, include_end=True)

    return df

def main():
    # Define date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Fetch data
    bars = fetch_30min_bars(start_date, end_date)

    # Print sample data
    print(bars.head())

    # Save to CSV
    bars.to_csv('dia_30min_bars.csv')

if __name__ == "__main__":
    main()