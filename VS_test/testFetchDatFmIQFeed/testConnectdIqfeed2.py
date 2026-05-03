# test request_daily_data function. Get daily data DIA from Nasdaq but only include 9:30 to 16:00
# remove pre-trading and post-trading period
import datetime
import pandas as pd
import datetime as dt
from datetime import datetime, date, timedelta
from pyiqfeed import HistoryConn

# Create a History Connection
conn = HistoryConn(name="history")
conn.connect()

# Define parameters
symbol = "@YM#C"
start = dt.datetime(2011, 1, 1)
end = dt.datetime(2025, 4, 30)
interval = 1  # 1-minute bars

# Request 1-minute bars
try:
    bars = conn.request_bars(
        symbol=symbol,
        interval=interval,
        begin=start,
        end=end
    )
except Exception as e:
    print(f"An error occurred: {e}")

# Convert to DataFrame
df = pd.DataFrame(bars)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df.index = df.index.tz_localize('US/Eastern')  # IQFeed is in Eastern

# Filter regular trading hours (9:30 to 16:00)
df = df.between_time("09:30", "16:00")

# Resample to daily OHLC
daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'total_volume': 'sum'
})

# Drop NaNs (non-trading days)
daily.dropna(inplace=True)

# Show sample output
print(daily.head())

# Save to CSV (optional)
daily.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")