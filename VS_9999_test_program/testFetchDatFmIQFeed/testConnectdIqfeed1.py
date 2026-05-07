# test request_daily_data function, num_days as parameter
import datetime
import pandas as pd
from pyiqfeed import HistoryConn, ConnConnector

# Create a History Connection
conn = HistoryConn(name="history")
with ConnConnector([conn]):
    data = conn.request_daily_data("AAPL", num_days=10)

# Convert to pandas DataFrame
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print(df.columns)
print(df.head())
# print(df)