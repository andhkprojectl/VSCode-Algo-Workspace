#
# Description: c:\Project\ProjectLife\Strategies\1
# Study correlation of features with DIA return
# Using 1) General correlation. 2) Logistic regression. 3) Random forest
# Output:
# 1) General correlation: table show number between -1 to 1, DIA return and other features
# 2) Logistic regression
# 3) Random forest
#

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
# import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from pyiqfeed import HistoryConn
from sklearn.preprocessing import StandardScaler

# import plotly.graph_objects as go
import exchange_calendars as xcals
# from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


# e.g. time1 = 16:00:00
def get_close_price_by_time(inData1, time1):
    # Filter only 16:00:00 rows
    inData1_at_time1 = inData1[inData1['Time1'] == time1].copy()

    # Sort by Date in case it's not
    inData1_at_time1 = inData1_at_time1.sort_values(by='Date1')

    # Compute difference from previous day's close
    inData1_at_time1['close_diff_time1'] = inData1_at_time1['Close'].diff()
    inData1_at_time1['close_at_time1'] = inData1_at_time1['Close']

    return inData1_at_time1


# fetch daily data from 9:30 to 16:00
# return pd.DataFrame datatype
# e.g. start_time = "09:30"
# e.g. end_time = "16:00"
# e.g. period1 = 1800, 30 minutes
def fetch_data_with_time(conn, symbol1, start_dt, start_time, end_dt, end_time, period1):
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
            # interval_len=1800,  # 30 minutes (1800 seconds)
            interval_len=period1,  # 30 minutes (1800 seconds)
            interval_type="s",  # Seconds
            bgn_prd=start_dt,
            end_prd=end_dt,
            # bgn_flt=datetime.strptime("09:30", "%H:%M").time(),
            # end_flt=datetime.strptime("16:00", "%H:%M").time(),
            bgn_flt=datetime.strptime(start_time, "%H:%M").time(),
            end_flt=datetime.strptime(end_time, "%H:%M").time(),
            ascend=True,  # Oldest to latest
            max_bars=None,  # Fetch all available bars
            timeout=None
        )
        # print ("data output")

        # print(f"symbol: {symbol1}", data.dtype.names)
        # print 1st 5 rows
        # print(data[:5])
        # print(data["time"][:5])

        # Print header names
        # if data is not None and len(data) > 0:
        #     header_names = data.dtype.names
        #     print("Data Header Names:", header_names)
        # else:
        #     print("No data returned")

        # Convert to list of dictionaries
        all_data = [
            {
                # "datetime": bar["datetime"],  # Keep full timestamp
                "Date1": bar["date"],
                "Time1": bar["time"],
                "Open": bar["open_p"],
                "High": bar["high_p"],
                "Low": bar["low_p"],
                "Close": bar["close_p"],
                "AccVolume": bar["tot_vlm"],
                "Volume": bar["prd_vlm"],
                "numTrade": bar["num_trds"]
            }
            for bar in data
        ]

        # all_data.extend(records)
        df = pd.DataFrame(all_data)
        # print (df.columns)
        # print (df.head())

        # df['datetime'] = pd.to_datetime(df['Date1'] + ' ' + df['Time1'])
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str))
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='%Y-%m-%d %H:%M:%S')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='mixed', errors='coerce')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='ISO8601')

        # Convert Time1 (nanoseconds) to time string
        # df['TimeSeconds'] = df['Time1'] / 1_000_000_000  # Convert to seconds
        # df['TimeStr'] = pd.to_timedelta(df['TimeSeconds'], unit='s').apply(
        #    lambda x: x.components.hours * 3600 + x.components.minutes * 60 + x.components.seconds).apply(
        #    lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}")

        # Add new column datatime
        # Convert nanoseconds to timedelta
        df['time_delta'] = pd.to_timedelta(df['Time1'], unit='ns')
        # Convert timedelta to string formatted as HH:MM:SS
        # df['TimeStr'] = df['time_delta'].astype('timedelta64[s]').astype(str)
        df['TimeStr'] = df['time_delta'].dt.components.apply(
            lambda row: f"{int(row.hours):02}:{int(row.minutes):02}:{int(row.seconds):02}", axis=1
        )
        df['DateStr'] = df['Date1'].dt.strftime('%Y-%m-%d')
        # Combine Date1 and TimeStr into datetime
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['TimeStr'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['datetime'] = pd.to_datetime(df['Date1'].dt.strftime('%Y-%m-%d') + ' ' + df['TimeStr'],
                                        format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # add symbol name
        df['symbol'] = symbol1

        # print (f"fetch_data_with_time")
        # print(df.columns)
        # print(df.head())
        # print(df['datetime'].head())

        # print(df[['DateStr', 'TimeStr']].head())  # 確認合併前的時間格式
        # print(df[['DateStr']].head())  # 確認合併前的時間格式
        # print(df[['TimeStr']].head())  # 確認合併前的時間格式
        # print(df['datetime'].head())  # 確認轉換是否成功
        # if cannot convert datetime and errors='coerce', datetime field show na print how many show is convert succsess
        # print(df['datetime'].isna().sum())

        # print("111")
        # print (df.columns)
        # sort
        df = df.sort_values(by='datetime', ascending=True)

        # don't run below otherwise datetime field disappear
        # sort data by datetime
        # df.set_index('datetime', inplace=True)
        # df.sort_index(inplace=True)
        #

        # daily = df1.resample('1D').agg({
        #     'Open': 'first',
        #     'High': 'max',
        #     'Low': 'min',
        #     'Close': 'last',
        #     'Volume': 'sum',
        #     'numTrade': 'sum'

        # Save to CSV (optional)
        # if (write_output_2_file):
        #     daily.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")

        # conn.disconnect()

        # Drop NaNs (non-trading days)
        # df.dropna(inplace=True)

        return df

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()


def merge_2_dataFrame(df1, df2):
    # Assume df1 and df2 with same field. Key field is datetime
    # Only include some fields
    # 1) union all value of datetime field, 2) left merge df1 and df2 with datetime field (some datetimee row with blank value)
    # df1.to_csv("c:/tmp/df1.csv")
    # df2.to_csv("c:/tmp/df2.csv")
    # print (df1.columns)
    # print (df1['datetime'])
    # print (df2['datetime'])

    # df3_0 = pd.concat([df1['datetime'], df2['datetime']])
    # df3_1 = df3_0.drop_duplicates()
    # df3 = pd.DataFrame({'datetime': df3_1})
    df3 = pd.DataFrame({'datetime': pd.concat([df1['datetime'], df2['datetime']]).drop_duplicates()})

    # merge 2 dataFrame
    df4 = df3.merge(
        df1[['datetime', 'Date1', 'Time1', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']],
        on=['datetime'],
        how='left'
    )
    df4 = df4.rename(
        columns={'Date1': 'Date1_1', 'Time1': 'Time1_1', 'Open': 'Open_1', 'High': 'High_1', 'Low': 'Low_1',
                 'Close': 'Close_1', 'Volume': 'Volume_1', 'symbol': 'symbol_1'})

    df5 = df4.merge(
        df2[['datetime', 'Date1', 'Time1', 'Open', 'High', 'Low', 'Close', 'Volume', 'symbol']],
        on=['datetime'],
        how='left'
    )
    df5 = df5.rename(
        columns={'Date1': 'Date1_2', 'Time1': 'Time1_2', 'Open': 'Open_2', 'High': 'High_2', 'Low': 'Low_2',
                 'Close': 'Close_2', 'Volume': 'Volume_2', 'symbol': 'symbol_2'})

    # if datetime blank, remove row
    # print ("113")
    # print (df5.shape)
    # df5 = df5.dropna(subset=['datetime'])
    # print(df5.shape)
    df5 = df5.sort_values(by='datetime', ascending=True)

    return df5


def is_NYSE_trading_day(date1):
    """Check if the given date is not a trading day (NYSE holiday or weekend)."""
    try:
        nyse1 = xcals.get_calendar('XNYS')
        return nyse1.is_session(date1)
    except Exception as e:
        # return False
        date_string = "2025-08-24"
        date_object = datetime.strptime(date_string, "%Y-%m-%d").date()
        return nyse1.is_session(date_object)


def rmHoliday(df1):
    # df2 = df1[~df1['datetime'].date().isin(B2['B'])]
    try:
        # df1.to_csv("c:/tmp/df1.csv")
        # print ("112")
        # print (df1['datetime'].dt.date)
        df2 = df1[is_NYSE_trading_day(df1['datetime'].dt.date)]
        return df2
    except Exception as e:
        return df1


def fillMissingData(df1):
    # df1.to_csv("c:/tmp/fillMissingData.csv")

    # fill blank value with previous value
    df1[['Date1_1', 'Time1_1', 'Open_1', 'High_1', 'Low_1', 'Close_1', 'Volume_1', 'symbol_1', 'Date1_2', 'Time1_2',
         'Open_2', 'High_2', 'Low_2', 'Close_2', 'Volume_2', 'symbol_2']] = df1[
        ['Date1_1', 'Time1_1', 'Open_1', 'High_1', 'Low_1', 'Close_1', 'Volume_1', 'symbol_1', 'Date1_2', 'Time1_2',
         'Open_2', 'High_2', 'Low_2', 'Close_2', 'Volume_2', 'symbol_2']].ffill()

    # fill 1st several blank value with 1st row with value
    # first_index = df1.index[df1['Date1_1'] != ''][0]
    # Slice the DataFrame from that index onward
    # df1 = df1.iloc[first_index:]

    # remove preceding blank rows
    first_non_blank_pos = df1['Date1_1'].notna().values.argmax()
    df1 = df1.iloc[first_non_blank_pos:]

    # df1.to_csv("c:/tmp/fillMissingData.csv")
    return df1


# run after build
def featureScaling(df1):
    # @YM#C close price *100 of DIA, e.g. C of DIA 430 and C of @YM#C is 43000, hence /100
    df2 = df1.copy()
    # set @YM#C/100
    df2['Open_2'] = df2['Open_2'] / 100
    df2['High_2'] = df2['High_2'] / 100
    df2['Low_2'] = df2['Low_2'] / 100
    df2['Close_2'] = df2['Close_2'] / 100

    # df1.to_csv("c:/tmp/featureScaling.csv")

    return df2


# run after buildFeaturePd
def featureScaling2(df1):
    columns_to_scale = [
        'Open_1', 'High_1', 'Low_1', 'Close_1', 'Volume_1', 'Open_2', 'High_2', 'Low_2', 'Close_2', 'Volume_2'
        , 'rt_1', 'rt_2', 'ptl_top_1', 'ptl_bottom_1', 'ptl_top_reach_1', 'ptl_bottom_reach_1', 'ptl_top_2'
        , 'ptl_bottom_2', 'ptl_top_reach_2', 'ptl_bottom_reach_2', 'rt_1_p', 'rt_2_p', 'ptl_top_1_p', 'ptl_bottom_1_p'
        , 'ptl_top_2_p', 'ptl_bottom_2_p', 'vol_1_diff', 'vol_2_diff', 'vol_1_diff_p', 'vol_2_diff_p'
        , 'vol_2_diff_top_p', 'vol_2_diff_bot_p'
        , 'rtPre_2', 'rtReg_2', 'rtPst_2', 'rtPrePst_2', 'combine_2_1_p', 'combine_2_2_p', 'combine_2_3_p',
        'combine_2_4_p'
        , 'rtPre_1', 'rtReg_1', 'rtPst_1', 'rtPrePst_1'
        , 'ptl_top_reach_2_p', 'ptl_bottom_reach_2_p', 'vol2_top_reach_2_p'
        , 'rtHl_2_p'
    ]

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit and transform the selected columns
    # scaled_values = scaler.fit_transform(df1[columns_to_scale])
    # Create a DataFrame with scaled values and original column names
    # df_scaled = pd.DataFrame(scaled_values, columns=[f'{col}_scaled' for col in columns_to_scale])

    # Fit and transform selected columns
    df1[columns_to_scale] = scaler.fit_transform(df1[columns_to_scale])

    # df1.to_csv("c:/tmp/featureScaling2.csv")

    return df1

    # Combine with original DataFrame (optional: replace original columns or add new ones)
    # df = pd.concat([df, df_scaled], axis=1)

    # Verify the scaling (mean should be ~0, std should be ~1)
    # print("DataFrame with scaled columns:")
    # print(df)
    # print("\nMeans of scaled columns:")
    # print(df[[f'{col}_scaled' for col in columns_to_scale]].mean())
    # print("\nStandard deviations of scaled columns:")
    # print(df[[f'{col}_scaled' for col in columns_to_scale]].std())


def buildFeaturePd(df1):
    # print("nq_data head")
    # print(df1.head())
    # print(df1.shape)

    # ym_nvda_data.to_csv("c:/tmp/ym_dia_test2.csv")

    # add features and target to ym_nvda_data
    # df1['rt_1'] = df1['Close_1'].shift(-1) - df1['Close_1']
    # Create new columns with initial NaN values
    # Create new columns with initial NaN values
    df1['rt_1'] = np.nan  # Current close - previous close
    # df1['rt_reg'] = np.nan  # 9:30 close - previous day's 16:00 close
    # df1['rt_pst'] = np.nan  # 9:30 close - previous day's 20:00 close
    df1['rt_2'] = np.nan  # Current close - previous close
    # df1['rt_2_pre'] = np.nan  # 9:30 close - 04:00 close (same day)

    # Extract date and time components
    df1['date'] = df1['datetime'].dt.date
    df1['time'] = df1['datetime'].dt.time

    # Calculate Q (simple return)
    df1['rt_1'] = df1['Close_1'].diff()
    df1['rt_2'] = df1['Close_2'].diff()

    df1['rtHl_2'] = df1['High_2'] - df1['Open_2']

    df1['rt_1_2Bar'] = df1['Close_1'].diff(periods=2)
    df1['rt_2_2Bar'] = df1['Close_2'].diff(periods=2)

    df1['rt_1_3Bar'] = df1['Close_1'].diff(periods=3)
    df1['rt_2_3Bar'] = df1['Close_2'].diff(periods=3)

    df1['rt_1_4Bar'] = df1['Close_1'].diff(periods=4)
    df1['rt_2_4Bar'] = df1['Close_2'].diff(periods=4)

    df1['vol_1_diff'] = df1['Volume_1'].diff()
    df1['vol_2_diff'] = df1['Volume_2'].diff()

    # Create dictionaries to store daily reference prices
    daily_0400 = {}
    daily_0930 = {}
    daily_1600 = {}
    daily_2000 = {}

    daily_1_1530_C = {}  # yesterday regular Close
    daily_1_0930_O = {}  # today regular Open
    daily_1_1930_C = {}  # yesterday post trading close
    daily_1_0400_O = {}  # today pre-trading Open

    daily_2_1530_C = {}  # yesterday regular Close
    daily_2_0930_O = {}  # today regular Open
    daily_2_1930_C = {}  # yesterday post trading close
    daily_2_0400_O = {}  # today pre-trading Open

    # First pass: Collect reference prices for each day
    for date, group in df1.groupby('Date1_2'):
        # group.to_csv("c:/tmp/aa88.csv")
        # Get 04:00 price (approximate with first available price before 04:30)
        # morning_data = group[group['datetime'].dt.hour < 5]
        # if not morning_data.empty:
        #    daily_0400[date] = morning_data.iloc[0]['Close_1']
        # daily_2_1530_C[date] = morning_data.iloc[0]['Close_2']

        # Get exact times
        # for time, close in zip(group['time'], group['Close_1']):
        #    if time.hour == 9 and time.minute == 30:
        #        daily_0930[date] = close
        #    elif time.hour == 16 and time.minute == 0:
        #        daily_1600[date] = close
        #    elif time.hour == 20 and time.minute == 0:
        #        daily_2000[date] = close

        # for time, close, open in zip(group['Time1_2'], group['Close_2'], group['Open_2']):
        for time, close, open in zip(group['time'], group['Close_1'], group['Open_1']):
            if time.hour == 4 and time.minute == 0:
                daily_1_0400_O[date.date()] = open
            if time.hour == 9 and time.minute == 30:
                daily_1_0930_O[date.date()] = open
            elif time.hour == 15 and time.minute == 30:
                daily_1_1530_C[date.date()] = close
            elif time.hour == 19 and time.minute == 30:
                daily_1_1930_C[date.date()] = close

        # for time, close, open in zip(group['Time1_2'], group['Close_2'], group['Open_2']):
        for time, close, open in zip(group['time'], group['Close_2'], group['Open_2']):
            if time.hour == 4 and time.minute == 0:
                daily_2_0400_O[date.date()] = open
            if time.hour == 9 and time.minute == 30:
                daily_2_0930_O[date.date()] = open
            elif time.hour == 15 and time.minute == 30:
                daily_2_1530_C[date.date()] = close
            elif time.hour == 19 and time.minute == 30:
                daily_2_1930_C[date.date()] = close

    # Second pass: Calculate special returns
    for i, row in df1.iterrows():
        # current_date = row['Date1_2']
        # current_time = row['Time1_2']

        current_date = row['datetime'].date()
        current_time = row['datetime'].time()

        # Find previous trading day
        prev_date = None
        # dates = sorted(df1['Date1_2'].unique())
        # dates = sorted(df1['datetime'].unique())
        dates = sorted(df1['datetime'].dt.date.unique())
        # print(0)
        current_idx = dates.index(current_date)
        if current_idx > 0:
            prev_date = dates[current_idx - 1]

        # if current_time.hour == 9 and prev_date != None:
        #    print(1)
        # Calculate R (special case for 9:30)
        if current_time.hour == 9 and current_time.minute == 30:
            # print(0)
            # if prev_date and prev_date in daily_2_1530_C:
            if prev_date and prev_date in daily_2_1530_C:
                df1.at[i, 'rtReg'] = row['Open_2'] - daily_2_1530_C[prev_date]
                df1.at[i, 'rtReg_tmp_1'] = row['Open_1'] - daily_1_1530_C[prev_date]
            if prev_date and prev_date in daily_2_1930_C:
                df1.at[i, 'rtPst'] = row['Open_2'] - daily_2_1930_C[prev_date]
                df1.at[i, 'rtPst_tmp_1'] = row['Open_1'] - daily_1_1930_C[prev_date]
            if current_date in daily_2_0400_O:
                df1.at[i, 'rtPre'] = row['Open_2'] - daily_2_0400_O[current_date]
                df1.at[i, 'rtPre_tmp_1'] = row['Open_1'] - daily_1_0400_O[current_date]
            # if prev_date and prev_date in daily_1600:
            #    df1.at[i, 'rt_reg'] = row['Close_1'] - daily_1600[prev_date]
            # if prev_date and prev_date in daily_2000:
            #     df1.at[i, 'rt_pst'] = row['Close_1'] - daily_2000[prev_date]
            # if current_date in daily_2_1530_C:
            #    df1.at[i, 'rt_2_pre'] = row['Close_2'] - daily_2_1530_C[current_date]
        # else:
        #    # For non-9:30 rows, R equals Q
        #    df1.at[i, 'rtPre_2'] = df1.at[i - 1, 'rtPre_2']
        #    df1.at[i, 'rtReg_2'] = df1.at[i - 1, 'rtReg_2']
        #    df1.at[i, 'rtPst_2'] = df1.at[i - 1, 'rtPst_2']

        # Calculate R (special case for 4:00)
        if current_time.hour == 4 and current_time.minute == 0:
            if prev_date and prev_date in daily_2_1530_C:
                # df1.at[i, 'rtPrePst'] = row['Open_2'] - daily_2_1530_C[prev_date]
                df1.at[i, 'rtPrePst'] = row['Open_2'] - daily_2_1930_C[prev_date]
                df1.at[i, 'rtPrePst_tmp_1'] = row['Open_1'] - daily_1_1930_C[prev_date]
        # else:
        # For non-9:30 rows, R equals Q
        # df1.at[i, 'rtPrePst_2'] = df1.at[i - 1, 'rtPrePst_2']

    # fill
    df1[['rtPre_2']] = df1[['rtPre']].ffill()
    df1[['rtReg_2']] = df1[['rtReg']].ffill()
    df1[['rtPst_2']] = df1[['rtPst']].ffill()
    df1[['rtPrePst_2']] = df1[['rtPrePst']].ffill()

    df1[['rtPre_1']] = df1[['rtPre_tmp_1']].ffill()
    df1[['rtReg_1']] = df1[['rtReg_tmp_1']].ffill()
    df1[['rtPst_1']] = df1[['rtPst_tmp_1']].ffill()
    df1[['rtPrePst_1']] = df1[['rtPrePst_tmp_1']].ffill()

    # Drop temporary columns
    df1 = df1.drop(['date', 'time', 'rtPre', 'rtReg', 'rtPst', 'rtPrePst'
                       , 'rtPre_tmp_1', 'rtReg_tmp_1', 'rtPst_tmp_1', 'rtPrePst_tmp_1'
                    ], axis=1)

    # below are features
    topPercentileValue = 0.9
    bottomPercentileValue = 0.1
    topPercentileValue2 = 0.75
    bottomPercentileValue2 = 0.2
    percentilePeriod = 80  # days

    # find return percentile 10% and 90%
    df1['ptl_top_1'] = df1['rt_1'].rolling(window=percentilePeriod).quantile(topPercentileValue)
    df1['ptl_bottom_1'] = df1['rt_1'].rolling(window=percentilePeriod).quantile(bottomPercentileValue)
    df1['ptl_top_reach_1'] = df1['rt_1'] >= df1['ptl_top_1']
    df1['ptl_bottom_reach_1'] = df1['rt_1'] <= df1['ptl_bottom_1']

    # find return percentile 10% and 90%
    df1['ptl_top_2'] = df1['rt_2'].rolling(window=percentilePeriod).quantile(topPercentileValue)
    df1['ptl_bottom_2'] = df1['rt_2'].rolling(window=percentilePeriod).quantile(bottomPercentileValue)
    df1['ptl_top_reach_2'] = df1['rt_2'] >= df1['ptl_top_2']
    df1['ptl_bottom_reach_2'] = df1['rt_2'] <= df1['ptl_bottom_2']

    df1['vol_2_diff_top'] = df1['vol_2_diff'].rolling(window=percentilePeriod).quantile(topPercentileValue2)
    df1['vol_2_diff_bot'] = df1['vol_2_diff'].rolling(window=percentilePeriod).quantile(bottomPercentileValue2)
    df1['vol2_top_reach_2'] = df1['vol_2_diff'] > df1['vol_2_diff_top']

    # add prev value
    df1['rt_1_p'] = df1['rt_1'].shift(1)

    # df1['rt_1_pre_p'] = df1['rt_1_pre'].shift(1)
    df1['rt_2_p'] = df1['rt_2'].shift(1)
    df1['rtHl_2_p'] = df1['rtHl_2'].shift(1)

    df1['rt_1_2Bar_p'] = df1['rt_1_2Bar'].shift(1)
    df1['rt_1_3Bar_p'] = df1['rt_1_3Bar'].shift(1)
    df1['rt_1_4Bar_p'] = df1['rt_1_4Bar'].shift(1)
    df1['rt_2_2Bar_p'] = df1['rt_2_2Bar'].shift(1)
    df1['rt_2_3Bar_p'] = df1['rt_2_3Bar'].shift(1)
    df1['rt_2_4Bar_p'] = df1['rt_2_4Bar'].shift(1)

    df1['rt_1_2Bar_f'] = df1['rt_1_2Bar'].shift(-1)
    df1['rt_2_2Bar_f'] = df1['rt_2_2Bar'].shift(-1)


    df1['ptl_top_1_p'] = df1['ptl_top_1'].shift(1)
    df1['ptl_bottom_1_p'] = df1['ptl_bottom_1'].shift(1)
    df1['ptl_top_2_p'] = df1['ptl_top_2'].shift(1)
    df1['ptl_bottom_2_p'] = df1['ptl_bottom_2'].shift(1)

    df1['ptl_top_reach_2_p'] = df1['ptl_top_reach_2'].shift(1)
    df1['ptl_bottom_reach_2_p'] = df1['ptl_bottom_reach_2'].shift(1)

    df1['vol_2_diff_p'] = df1['vol_2_diff'].shift(1)
    df1['vol_1_diff_p'] = df1['vol_1_diff'].shift(1)
    df1['vol_2_diff_top_p'] = df1['vol_2_diff_top'].shift(1)
    df1['vol_2_diff_bot_p'] = df1['vol_2_diff_bot'].shift(1)
    df1['vol2_top_reach_2_p'] = df1['vol2_top_reach_2'].shift(1)


    # find correlation between 30min return from 10:00 to 12:30 and
    # - (rt_reg) O at 9:30 today - c regular close yesterday (close at 15:30)
    # - (rt_pre) O at 9:30 today - o pre-trading today (O at 04:00)
    # - (rt_pst) O at 9:30 today - c post-trading close yesterday (close at 19:30)

    # combine of 2 features
    df1['combine_2_1_p'] = df1['ptl_top_2_p'] * df1['vol_2_diff_top_p']
    df1['combine_2_2_p'] = df1['ptl_top_2_p'] * df1['vol_2_diff_top_p'] * df1['rtPre_2']
    df1['combine_2_3_p'] = df1['ptl_top_2_p'] * df1['vol_2_diff_top_p'] * df1['rtPre_1']
    df1['combine_2_4_p'] = df1['ptl_top_2_p'] * df1['vol_2_diff_top_p'] * df1['rtPst_1']

    df1['datetime_index'] = df1['datetime']
    df1.set_index('datetime_index', inplace=True)
    df1.sort_index(inplace=True)

    # remove all NaN row
    df1.dropna(inplace=True)  # Last row will have NaN 'Target'

    # Save to new CSV file
    df1.to_csv('c:/tmp/data1_with_new_columns.csv', index=False)

    return df1

    # remove 1st and last value
    # remove last
    # nq_data = nq_data.iloc[:-1]

    # remove 1st value. remove diff
    # nq_data = nq_data.iloc[1:]

    # remove last
    # nvda_data = nvda_data.iloc[:-1]
    # remove 1st value. remove diff
    # nvda_data = nvda_data.iloc[1:]

    # Remove rows with NaNs due to rolling windows
    # feature1.dropna(inplace=True)  # Last row will have NaN 'Target'

    # Target: Next day return

    # feature1['Target'] = nvda_data['nextDay_return1']

    # print("feature:", feature1.shape)
    # print("nvda_data:", nvda_data.shape)
    # print("nq_data:", nq_data.shape)

    # Features and target arrays
    # features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return1', 'VolumeChange']
    # X = feature1
    # y = nvda_data['nextDay_return1']

    # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train random forest
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)

    # Predictions
    # y_pred = rf.predict(X_test)

    # Evaluate
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"Test RMSE: {rmse:.2f}")

    # Feature importance
    # importances = pd.Series(rf.feature_importances_, index=features)
    # print(importances.sort_values(ascending=False))


# Target DIA and YM close price
def performCorrelation(df1):
    df2 = df1.copy()
    # remove string column or not related column before run correlatino
    df2 = df2.drop(columns=['Date1_1', 'Time1_1', 'symbol_1', 'Date1_2', 'Time1_2', 'symbol_2'])

    # Save to new CSV file
    df2.to_csv('c:/tmp/performCorrelationNQ.csv', index=False)

    corr_matrix = df2.corr()
    # display
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)  # Prevent line wrapping
    pd.set_option('display.max_colwidth', None)  # Full column text
    # print ("rt_1")
    #print (corr_matrix["rt_1"].sort_values(ascending=False))
    # print("")
    df2_rt2 = corr_matrix["rt_2"].sort_values(ascending=False)
    # df2_rt2 = corr_matrix["ptl_top_2_p"].sort_values(ascending=False)
    # print("rt_2")
    print (df2_rt2)
    df2_rt2.to_frame(name="Correlation").to_csv("c:/tmp/performCorrelationNQ2.csv", index_label="Field")


def scatterMatrixChart(df1):
    # write to csv
    df1.to_csv("c:/tmp/scatterMatrixChart.csv")

    # target: rt_1, rt_2
    # include pre-trading and post-trading hour
    # rt_1_col_name = [
    # 'rt_1',
    # 'rt_2',
    # 'rt_1_p',    'rt_1_pre_p',    'rt_2_p',    'rt_1_2Bar_p',
    # 'rt_1_3Bar_p', 'rt_1_4Bar_p', 'rt_2_2Bar_p',    'rt_2_3Bar_p','rt_2_4Bar_p',  'ptl_top_1_p',    'ptl_bottom_1_p',    'ptl_top_2_p',    'ptl_bottom_2_p'
    # ]

    #
    # exclude  pre-trading and post-trading hour, i.e. 09:30 to 16:00

    rt_1_col_name = [
        'rt_1',
        'rt_2',
        'rt_1_p', 'rt_2_p', 'rt_1_2Bar_p'
        , 'rtHl_2_p'
        , 'rt_1_3Bar_p', 'rt_1_4Bar_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rt_2_4Bar_p'
        # ,  'ptl_top_1_p',    'ptl_bottom_1_p',    'ptl_top_2_p',    'ptl_bottom_2_p'
    ]

    scatter_matrix1 = scatter_matrix(df1[rt_1_col_name], figsize=(12, 12), diagonal='hist')  # or diagonal='kde'

    # Customize font size for all axes
    for ax in scatter_matrix1.ravel():
        # for ax in scatter_matrix1.flatten():
        ax.tick_params(axis='both', labelsize=6)  # Increase tick label size
        ax.set_xlabel(ax.get_xlabel(), fontsize=5)  # Increase x-axis label size
        ax.set_ylabel(ax.get_ylabel(), fontsize=5)  # Increase y-axis label size
        # Rotate x-axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        # Rotate y-axis tick labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va='center')
    #
    # # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


def scatterPlot(df1):
    # write to csv
    # df1.to_csv("c:/tmp/scatterMatrixChart.csv")

    # Select target column and other columns
    rt_1_col_name = [
        'rt_1',
        'rt_2',
        'rt_1_p', 'rt_2_p', 'rt_1_2Bar_p',
        'rtHl_2_p',
        'rt_1_3Bar_p', 'rt_1_4Bar_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rt_2_4Bar_p'
        # ,  'ptl_top_1_p',    'ptl_bottom_1_p',    'ptl_top_2_p',    'ptl_bottom_2_p'
    ]
    df2 = df1[rt_1_col_name]

    # Select target column and other columns
    target_col = 'rt_2'
    other_cols = [col for col in df2.select_dtypes(include=[np.number]).columns if col != target_col]

    # Create subplots in a single row
    n_cols = len(other_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5), squeeze=False)

    # Plot scatter for each column against Close_1
    for idx, col in enumerate(other_cols):
        axes[0, idx].scatter(df2[col], df2[target_col], alpha=0.5)
        axes[0, idx].set_xlabel(col, fontsize=5)
        axes[0, idx].set_ylabel(target_col, fontsize=5)
        axes[0, idx].tick_params(axis='both', labelsize=6)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()


def scatterPlot2(df1):
    # write to csv
    # df1.to_csv("c:/tmp/scatterMatrixChart.csv")

    # Select target column and other columns
    rt_1_col_name = [
        'datetime',
        'rt_1',
        'rt_1_2Bar_f',
        'rt_2',
        'rt_1_p', 'rt_2_p', 'rt_1_2Bar_p','rtHl_2_p',
        'rt_1_3Bar_p', 'rt_1_4Bar_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rt_2_4Bar_p'
        # , 'ptl_top_1_p'
        # ,  'ptl_top_1_p',    'ptl_bottom_1_p',    'ptl_top_2_p',    'ptl_bottom_2_p'
    ]
    df2 = df1[rt_1_col_name]

    # Define the intervals to plot
    intervals = [
        ('09:30:00', '10:00:00'),
        ('10:00:00', '10:30:00'),
        ('10:30:00', '11:00:00'),
        ('11:00:00', '11:30:00'),
        ('11:30:00', '12:00:00')
    ]

    # Columns to plot against 'rt_1', excluding datetime and 'rt_1' itself
    columns_to_plot = [col for col in df2.columns if col != 'datetime' and col != 'rt_1']

    for start_time, end_time in intervals:
        # Filter the rows according to time of day
        mask = (df2['datetime'].dt.time >= pd.to_datetime(start_time).time()) & \
               (df2['datetime'].dt.time < pd.to_datetime(end_time).time())
        subset = df2.loc[mask]

        for col in columns_to_plot:
            plt.figure(figsize=(6, 4))
            plt.scatter(subset['rt_1'], subset[col])
            plt.xlabel('rt_1')
            plt.ylabel(col)
            plt.title(f'Scatter Plot of rt_1 vs {col}\nTime: {start_time} to {end_time}')
            plt.grid(True)
            plt.show()


def filter1(df1):
    # start_time3 = pd.to_datetime('09:30:00').time()
    # end_time3 = pd.to_datetime('16:30:00').time()
    start_time3 = pd.to_datetime('10:00:00').time()
    end_time3 = pd.to_datetime('11:30:00').time()
    df2 = df1[(df1['datetime'].dt.time >= start_time3) & (df1['datetime'].dt.time <= end_time3)]
    return df2


def filter2(df1):
    # start_time3 = pd.to_datetime('09:30:00').time()
    # end_time3 = pd.to_datetime('16:30:00').time()
    start_time3 = pd.to_datetime('12:00:00').time()
    end_time3 = pd.to_datetime('12:30:00').time()
    df2 = df1[(df1['datetime'].dt.time >= start_time3) & (df1['datetime'].dt.time <= end_time3)]
    return df2


def scatterPlot3(df1):
    # write to csv
    # df1.to_csv("c:/tmp/scatterMatrixChart.csv")

    # target_column_= 'rt_1'
    # target_column_= 'rt_1_2Bar_f'

    # Select target column and other columns
    rt_1_col_name = [
        'datetime',
        'rt_1',
        # 'rt_1_2Bar_f',
        # 'rt_2',
        'rt_1_p'
        # 'rt_1_2Bar_p' # ***
        # 'rt_1_3Bar_p' # **
        # 'rt_1_4Bar_p'
        # 'rt_2_p'
        # 'rt_2_2Bar_p' # ***
        # 'rt_2_3Bar_p' # ****
        # 'rt_2_4Bar_p'  # no need 4 bar, similar to 3 bar p
        # 'rt_1_2Bar_p', 'rt_1_3Bar_p', 'rt_1_4Bar_p'
        # , 'rt_2_2Bar_p',    'rt_2_3Bar_p','rt_2_4Bar_p'
        # 'ptl_top_1_p'
        # ,    'ptl_bottom_1_p',    'ptl_top_2_p',    'ptl_bottom_2_p'
    ]
    df2 = df1[rt_1_col_name]

    # Ensure datetime is datetime format
    # df['datetime'] = pd.to_datetime(df['datetime'])
    df2['time'] = df2['datetime'].dt.time

    # Define 30-min time windows
    time_windows = [
        ("09:30:00", "10:00:00"),
        ("10:00:00", "10:30:00"),
        ("10:30:00", "11:00:00"),
        ("11:00:00", "11:30:00"),
        ("11:30:00", "12:00:00"),
        ("12:00:00", "12:30:00")
    ]

    # Columns to compare against rt_1 (exclude datetime/time/rt_1)
    other_cols = [c for c in df2.columns if c not in ['datetime', 'time', 'rt_1']]

    # Loop over each time window
    for start, end in time_windows:
        mask = (df2['time'] >= pd.to_datetime(start).time()) & (df2['time'] < pd.to_datetime(end).time())
        subset = df2.loc[mask]

        if not subset.empty:
            plt.figure(figsize=(8, 6))
            for col in other_cols:
                plt.scatter(subset['rt_1'], subset[col], alpha=0.6, label=col)
                # plt.scatter(subset['rt_1_2Bar_f'], subset[col], alpha=0.6, label=col)

            plt.title(f"Scatter plots: rt_1 vs others ({start} to {end})")
            plt.xlabel("rt_1")
            # plt.xlabel("rt_1_2Bar_f")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.show()


def filterPrePstTradingHour(df1):
    # from 4:00 - 16:00, remove 16:00 - 20:00
    start_time1 = pd.to_datetime('04:00:00').time()
    end_time1 = pd.to_datetime('16:00:00').time()
    df_pre = df1[(df1['datetime'].dt.time >= start_time1) & (df1['datetime'].dt.time <= end_time1)]
    # from 9:30 - 16:00, remove 16:00 - 20:00, 04:00 - 09:30
    start_time2 = pd.to_datetime('09:30:00').time()
    end_time2 = pd.to_datetime('16:00:00').time()
    df_reg = df1[(df1['datetime'].dt.time >= start_time2) & (df1['datetime'].dt.time <= end_time2)]
    # from 9:30 - 20:00, remove 04:00 - 09:30
    start_time3 = pd.to_datetime('09:30:00').time()
    end_time3 = pd.to_datetime('20:00:00').time()
    df_pst = df1[(df1['datetime'].dt.time >= start_time3) & (df1['datetime'].dt.time <= end_time3)]

    # df_pre.to_csv("c:/tmp/df_pre.csv")
    # df_reg.to_csv("c:/tmp/df_reg.csv")
    # df_pst.to_csv("c:/tmp/df_pst.csv")

    return df_pre, df_reg, df_pst


# compare logisticsRegsssion and randomForest
# program get from chatGpt
def compare2Model1(X1, y1):
    tscv = TimeSeriesSplit(n_splits=5)

    # LogisticRegression pipeline (prob output)
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    # RandomForestRegressor (directly outputs numeric values)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    # Cross-validation with neg_mean_squared_error
    scores_logit = cross_val_score(
        logit, X1, y1,
        cv=tscv,
        scoring="neg_mean_squared_error"
    )

    scores_rf = cross_val_score(
        rf, X1, y1,
        cv=tscv,
        scoring="neg_mean_squared_error"
    )

    print()
    print("compare2Model1 begin")
    print("LogisticRegression MSE:", -np.mean(scores_logit))
    print("RandomForestRegressor MSE:", -np.mean(scores_rf))
    print("compare2Model1 end")
    print()


# same as program compare2Model2,  get from grok4
def compare2Model2(X1, y1):
    # Initialize models
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    # lr_clf = make_pipeline(
    #    StandardScaler(),
    #    LogisticRegression(max_iter=1000)
    # )

    # Cross-validation with 5 folds, scoring='neg_mean_squared_error'
    rf_scores = cross_val_score(rf_clf, X1, y1, cv=5, scoring='neg_mean_squared_error')
    lr_scores = cross_val_score(lr_clf, X1, y1, cv=5, scoring='neg_mean_squared_error')

    # Print scores
    print()
    print("compare2Model2 begin")
    print("RandomForestClassifier CV Scores (neg MSE):", rf_scores)
    print("Mean neg MSE for RandomForestClassifier: ", np.mean(rf_scores))

    print("LogisticRegression CV Scores (neg MSE):", lr_scores)
    print("Mean neg MSE for LogisticRegression: ", np.mean(lr_scores))

    # Determine which is better (higher mean neg MSE, i.e., less negative, means lower error)
    if np.mean(rf_scores) > np.mean(lr_scores):
        print("RandomForestClassifier is better with mean neg MSE:", np.mean(rf_scores))
    else:
        print("LogisticRegression is better with mean neg MSE:", np.mean(lr_scores))
    print("compare2Model2 end")
    print()


def logisticRegressionPreFilter(df1):
    start_time3 = pd.to_datetime('10:00:00').time()
    end_time3 = pd.to_datetime('12:00:00').time()
    ## start_time3 = pd.to_datetime('10:30:00').time()
    ## end_time3 = pd.to_datetime('10:30:00').time()

    df2 = df1[(df1['datetime'].dt.time >= start_time3) & (df1['datetime'].dt.time <= end_time3)]
    return df2

    #
    # filter weekday
    # dayofweek = 0: mon, 1: Tue, 6: Sun
    #
    df3 = df2[df2["datetime"].dt.dayofweek.isin([4])]

    return df3


# Target DIA close price
def performLogisticRegressionDIA(df1):
    # Compute features (X)
    # 1) Last 30min return = rt_1

    # 2) 14 bar RSI
    delta = df1['rt_1']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_days = 14
    avg_gain = gain.rolling(window=rsi_days, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_days, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df1['rsi_14'] = 100 - (100 / (1 + rs))
    df1['rsi_14_p'] = df1['rsi_14'].shift(1)

    # 3) minute return percentile (90% and 10%)
    # df1['ptl_top_1_p'], df1['ptl_bottom_1_p']

    # 4) 20 days EMA (assuming 13 30min bars per day, 20 bars)
    emaNumBars1 = 20
    emaNumBars2 = 50
    emaNumBars3 = 100
    df1['ema20'] = df1['Close_1'].ewm(span=emaNumBars1, adjust=False).mean()
    df1['ema50'] = df1['Close_1'].ewm(span=emaNumBars2, adjust=False).mean()
    df1['ema100'] = df1['Close_1'].ewm(span=emaNumBars3, adjust=False).mean()
    df1['ema20_p'] = df1['ema20'].shift(1)
    df1['ema50_p'] = df1['ema50'].shift(1)
    df1['ema100_p'] = df1['ema100'].shift(1)

    # 5) Top and bottom Bollinger Bands (20 periods, 2 std dev)
    bbNumBars = 20
    rolling_mean = df1['Close_1'].rolling(window=bbNumBars).mean()
    rolling_std = df1['Close_1'].rolling(window=bbNumBars).std()
    df1['top_bb'] = rolling_mean + (rolling_std * 2)
    df1['bottom_bb'] = rolling_mean - (rolling_std * 2)
    df1['top_bb_p'] = df1['top_bb'].shift(1)
    df1['bottom_bb_p'] = df1['bottom_bb'].shift(1)

    # Drop NaN rows. suppose data manipulate process done
    # nvda_data = nvda_data.dropna()

    df2 = logisticRegressionPreFilter(df1)

    # Define y: 1 if next return > 50, else 0 (assuming 'return' means price change)
    # df2['y'] = (df2['rt_1'] > 0.05).astype(int)
    df2['y'] = (df2['rt_1'] > 0).astype(int)
    df2 = df2.dropna()  # Drop last row

    df2.to_csv("c:/tmp/logisticRegressionDIA.csv")

    # Features X
    # X = df2[['rt_1_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p', 'vol_1_diff_p']]
    # X = df2[['rt_1_p', 'rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p',
    #          'vol_2_diff_p']]
    X = df2[[#'rt_1_p', 'rt_2_p'
             #'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p',
             # 'vol_2_diff_p'
            # 'rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ema20_p', 'top_bb_p'
             # 'rt_1_p', 'rsi_14_p', 'ptl_top_1_p', 'ema20_p', 'top_bb_p', 'vol_2_diff_p'
            'rt_1_p',  'vol_2_diff_p'
             ]]
    y = df2['y']

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_date = '2025-08-01'
    X_train, X_test = X.loc[:split_date], X.loc[split_date:]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]

    # Tune LogisticRegression with GridSearchCV
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],  # Use 'l2' to avoid solver issues
        'solver': ['lbfgs']  # Solver that supports l2
    }
    logreg = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Output
    print("Best Parameters:", best_params)
    print("Test Accuracy:", accuracy)
    # print("Coefficients:", best_model.coef_)
    print("Best CV Score:", grid_search.best_score_)
    print("Test Accuracy2:", grid_search.score(X_test, y_test))

    # -----------------------------
    # Feature importance interpretation
    # -----------------------------
    features = X.columns
    coefs = best_model.coef_[0]  # coefficients for each feature

    importance = pd.DataFrame({
        "feature": features,
        "coef": coefs
        # "abs_coef": abs(coefs)
    }).sort_values("feature", ascending=True)
    # }).sort_values("abs_coef", ascending=False)

    print("\nFeature importance:")
    # print(importance)
    # print(importance.to_string(index=False, justify="left"))
    # print(importance.astype(str).to_string(index=False, justify="left"))
    for _, row in importance.iterrows():
        print(f"{row['feature']} {row['coef']}")

    # Get coefficients for feature importance
    # features = ['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']
    # coef = model.coef_[0]
    # feature_importance = pd.DataFrame({
    #     'Feature': features,
    #     'Coefficient': coef,
    #     'Abs Coefficient': np.abs(coef),
    #     'Odds Ratio': np.exp(coef)
    # }).sort_values(by='Abs Coefficient', ascending=False)
    #
    # # Evaluate model
    # y_pred = model.predict(X_test_scaled)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # # Output
    # print("Model Accuracy:", accuracy)
    # print("\nFeature Importance:")
    # print(feature_importance)    # features = ['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']
    # coef = model.coef_[0]
    # feature_importance = pd.DataFrame({
    #     'Feature': features,
    #     'Coefficient': coef,
    #     'Abs Coefficient': np.abs(coef),
    #     'Odds Ratio': np.exp(coef)
    # }).sort_values(by='Abs Coefficient', ascending=False)
    #
    # # Evaluate model
    # y_pred = model.predict(X_test_scaled)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # # Output
    # print("Model Accuracy:", accuracy)
    # print("\nFeature Importance:")
    # print(feature_importance)

    # logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # logit.fit(X_train, y_train)
    # print("Logistic Regression Accuracy:", logit.score(X_test, y_test))

    # compare2Model1(X, y)
    # compare2Model2(X, y)


# Target YM close price
def performLogisticRegressionNQ(df1):
    # Compute features (X)
    # 1) Last 30min return = rt_1

    # 2) 14 bar RSI
    delta = df1['rt_2']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_days = 14
    avg_gain = gain.rolling(window=rsi_days, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_days, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df1['rsi_14'] = 100 - (100 / (1 + rs))
    df1['rsi_14_p'] = df1['rsi_14'].shift(1)

    # 3) minute return percentile (90% and 10%)
    # df1['ptl_top_1_p'], df1['ptl_bottom_1_p']

    # 4) 20 days EMA (assuming 13 30min bars per day, 20 bars)
    emaNumBars1 = 20
    emaNumBars2 = 50
    emaNumBars3 = 100
    df1['ema20'] = df1['Close_2'].ewm(span=emaNumBars1, adjust=False).mean()
    df1['ema50'] = df1['Close_2'].ewm(span=emaNumBars2, adjust=False).mean()
    df1['ema100'] = df1['Close_2'].ewm(span=emaNumBars3, adjust=False).mean()
    df1['ema20_p'] = df1['ema20'].shift(1)
    df1['ema50_p'] = df1['ema50'].shift(1)
    df1['ema100_p'] = df1['ema100'].shift(1)

    # 5) Top and bottom Bollinger Bands (20 periods, 2 std dev)
    bbNumBars = 20
    rolling_mean = df1['Close_2'].rolling(window=bbNumBars).mean()
    rolling_std = df1['Close_2'].rolling(window=bbNumBars).std()
    df1['top_bb'] = rolling_mean + (rolling_std * 2)
    df1['bottom_bb'] = rolling_mean - (rolling_std * 2)
    df1['top_bb_p'] = df1['top_bb'].shift(1)
    df1['bottom_bb_p'] = df1['bottom_bb'].shift(1)

    # Drop NaN rows. suppose data manipulate process done
    # nvda_data = nvda_data.dropna()

    df2 = logisticRegressionPreFilter(df1)

    # Define y: 1 if next return > 50, else 0 (assuming 'return' means price change)
    # df2['y'] = (df2['rt_1'] > 0.05).astype(int)
    df2['y'] = (df2['rt_2'] > 0).astype(int)
    df2 = df2.dropna()  # Drop last row

    # Features X
    X = df2[['rt_1_p', 'rt_2_p', 'rsi_14_p', 'ptl_top_2_p', 'ptl_bottom_2_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p', 'vol_2_diff_p'
            , 'ptl_top_reach_2_p', 'ptl_bottom_reach_2_p', 'vol2_top_reach_2_p', 'rtHl_2_p'
            ]]
    # X = df2[['rt_1_p', 'rt_2_p', 'rsi_14_p', 'ptl_top_2_p', 'ptl_bottom_2_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']]
    # X = df2[['rt_2_p', 'rsi_14_p', 'ptl_top_2_p', 'ptl_bottom_2_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']]
    y = df2['y']

    df2.to_csv("c:/tmp/logisticRegressionYM.csv")

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_date = '2025-08-01'
    X_train, X_test = X.loc[:split_date], X.loc[split_date:]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]

    # Tune LogisticRegression with GridSearchCV
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],  # Use 'l2' to avoid solver issues
        'solver': ['lbfgs']  # Solver that supports l2
    }
    logreg = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Output
    print("Best Parameters:", best_params)
    print("Test Accuracy:", accuracy)
    # print("Coefficients:", best_model.coef_)
    print("Best CV Score:", grid_search.best_score_)
    print("Test Accuracy2:", grid_search.score(X_test, y_test))

    # -----------------------------
    # Feature importance interpretation
    # -----------------------------
    features = X.columns
    coefs = best_model.coef_[0]  # coefficients for each feature

    importance = pd.DataFrame({
        "feature": features,
        "coef": coefs
        # "abs_coef": abs(coefs)
    }).sort_values("feature", ascending=True)
    # }).sort_values("abs_coef", ascending=False)

    print("\nFeature importance:")
    # print(importance)
    # print(importance.to_string(index=False, justify="left"))
    # print(importance.astype(str).to_string(index=False, justify="left"))
    for _, row in importance.iterrows():
        print(f"{row['feature']} {row['coef']}")

    # Get coefficients for feature importance
    # features = ['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']
    # coef = model.coef_[0]
    # feature_importance = pd.DataFrame({
    #     'Feature': features,
    #     'Coefficient': coef,
    #     'Abs Coefficient': np.abs(coef),
    #     'Odds Ratio': np.exp(coef)
    # }).sort_values(by='Abs Coefficient', ascending=False)
    #
    # # Evaluate model
    # y_pred = model.predict(X_test_scaled)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # # Output
    # print("Model Accuracy:", accuracy)
    # print("\nFeature Importance:")
    # print(feature_importance)    # features = ['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']
    # coef = model.coef_[0]
    # feature_importance = pd.DataFrame({
    #     'Feature': features,
    #     'Coefficient': coef,
    #     'Abs Coefficient': np.abs(coef),
    #     'Odds Ratio': np.exp(coef)
    # }).sort_values(by='Abs Coefficient', ascending=False)
    #
    # # Evaluate model
    # y_pred = model.predict(X_test_scaled)
    # accuracy = accuracy_score(y_test, y_pred)
    #
    # # Output
    # print("Model Accuracy:", accuracy)
    # print("\nFeature Importance:")
    # print(feature_importance)

    # logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # logit.fit(X_train, y_train)
    # print("Logistic Regression Accuracy:", logit.score(X_test, y_test))

    # compare2Model1(X, y)
    # compare2Model2(X, y)


# Target YM close price
def performLogisticRegressionNQ2(df0):
    df1 = df0
    df1 = logisticRegressionPreFilter(df0)
    # Step 1: parameter
    periods = 100

    # 4) 20 days EMA (assuming 13 30min bars per day, 20 bars)
    emaNumBars1 = 20
    emaNumBars2 = 50
    emaNumBars3 = 100
    df1['ema20'] = df1['Close_2'].ewm(span=emaNumBars1, adjust=False).mean()
    df1['ema50'] = df1['Close_2'].ewm(span=emaNumBars2, adjust=False).mean()
    df1['ema100'] = df1['Close_2'].ewm(span=emaNumBars3, adjust=False).mean()
    df1['ema20_p'] = df1['ema20'].shift(1)
    df1['ema50_p'] = df1['ema50'].shift(1)
    df1['ema100_p'] = df1['ema100'].shift(1)

    # 5) Top and bottom Bollinger Bands (20 periods, 2 std dev)
    bbNumBars = 30
    rolling_mean = df1['Close_2'].rolling(window=bbNumBars).mean()
    rolling_std = df1['Close_2'].rolling(window=bbNumBars).std()
    df1['top_bb'] = rolling_mean + (rolling_std * 2)
    df1['bottom_bb'] = rolling_mean - (rolling_std * 2)
    df1['top_bb_reach_2'] = df1['Close_2'] >= df1['top_bb']
    df1['top_bb_p'] = df1['top_bb'].shift(1)
    df1['bottom_bb_p'] = df1['bottom_bb'].shift(1)
    df1['top_bb_reach_2_p'] = df1['top_bb_reach_2'].shift(1)


    # data = pd.DataFrame({'close': closes, 'volume': volumes, 'return': returns, 'vol_diff': vol_diff})

    # Compute rolling 90th percentile (for original strategy)
    def rolling_percentile(series, window, percentile):
        return series.rolling(window).apply(lambda x: np.percentile(x[:-1], percentile) if len(x) >= window else np.nan)

    # data['ret_p90'] = rolling_percentile(data['return'], periods + 1, 90)
    # data['vol_diff_p90'] = rolling_percentile(data['vol_diff'], periods + 1, 90)
    # df1['ptl_top_2'] = df1['Close_2'].rolling(window=percentilePeriod).quantile(

    # Original strategy signals
    df1['signal'] = ((df1['rt_2_p'] > df1['ptl_top_2_p']) &
                     (df1['vol_2_diff_p'] > df1['vol_2_diff_top_p'])).astype(int)

    # Drop invalid rows
    df1_2 = df1.dropna(subset=['rt_2', 'vol_2_diff_p', 'rt_2_p', 'ptl_top_2_p', 'vol_2_diff_top_p'
                            , 'ptl_top_reach_2_p', 'top_bb_reach_2_p', 'rtHl_2_p'
                                ])

    # Step 2: Features - use percentile ranks for normalization
    def rolling_rank(series, window):
        return series.rolling(window + 1).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) >= window + 1 else np.nan)
        # return series.rolling(window + 1).apply(lambda x: (x[-1] > x[:-1]).mean() * 100 if len(x) >= window + 1 else np.nan)

    df1_2['rt_2_p_rank'] = rolling_rank(df1_2['rt_2_p'], periods)
    df1_2['rtHl_2_p_rank'] = rolling_rank(df1_2['rtHl_2_p'], periods)
    df1_2['vol_2_diff_p_rank'] = rolling_rank(df1_2['vol_2_diff_p'], periods)

    df2 = df1_2.dropna(subset=['rt_2_p_rank', 'vol_2_diff_p_rank', 'rtHl_2_p_rank'])

    # df2['combined_rank'] = (df2['rt_2_p_rank'] * df2['vol_2_diff_p_rank']) / 10000
    df2['combined_rank1'] = (df2['rt_2_p_rank'] * df2['vol_2_diff_p_rank']) / df2['rt_2_p_rank'].max() * df2[
        'vol_2_diff_p_rank'].max()
    df2['combined_rank2'] = (df2['rt_2_p_rank'] + df2['vol_2_diff_p_rank']) / (
                df2['rt_2_p_rank'].max() + df2['vol_2_diff_p_rank'].max())
    df2['combined_rank3'] = (df2['rt_2_p_rank'] * df2['vol_2_diff_p_rank']) / 10000
    df2['combined_rank4'] = (df2['rt_2_p_rank'] * df2['vol_2_diff_p_rank'])

    # Convert booleans to ints
    for col in ["ptl_top_reach_2_p", "top_bb_reach_2_p"]:
        df2[col] = df2[col].astype(float)

    df2.to_csv("c:/tmp/performLogisticRegressionNQ2.csv")

    # features = ['combined_rank4']
    # features = ['vol_2_diff_p_rank']
    # features = ['rt_2_p_rank', 'vol_2_diff_p_rank', 'combined_rank1', 'combined_rank2', 'combined_rank3', 'combined_rank4']
    features = [ # 'ptl_top_2_p', 'vol_2_diff_top_p', 'rtPre_2'
                # 'ptl_bottom_2_p', 'vol_2_diff_top_p'
                # 'ptl_top_2_p', 'vol_2_diff_top_p'
                'ptl_top_reach_2_p', 'top_bb_reach_2_p', 'ptl_top_2_p', 'top_bb_p', 'vol_2_diff_top_p', 'vol2_top_reach_2_p'
                # 'ptl_top_2_p'
                # 'vol_2_diff_top_p'
                # 'ptl_top_reach_2_p', 'vol_2_diff_top_p'
                # 'ptl_top_reach_2_p', 'ptl_bottom_reach_2_p', 'vol2_top_reach_2_p'
                # 'ptl_top_reach_2_p', 'vol2_top_reach_2_p'
                #, 'vol_2_diff_top_p'
                # 'ptl_top_2_p'
                # 'ptl_top_2_p', 'vol_2_diff_top_p', 'rtPre_2'
                # 'ptl_top_2_p', 'vol_2_diff_top_p', 'rtPst_2'
                # 'ptl_bottom_2_p', 'bottom_bb_p'
                # 'ptl_bottom_2_p', 'bottom_bb_p'
                # 'vol_2_diff_p', 'bottom_bb_p'
                # 'vol_2_diff_p', 'bottom_bb_p', 'ptl_bottom_2_p'

                # , 'rtPre_2', 'rtReg_2', 'rtPst_2', 'rtPrePst_2'
                # , 'rtPre_1', 'rtReg_1', 'rtPst_1', 'rtPrePst_1'
                # , 'combine_2_1_p'
                # 'combine_2_2_p'
                # 'combine_2_4_p'
                #, 'combine_2_3_p', 'combine_2_4_p'
                # , 'ptl_top_2_p', 'vol_2_diff_top_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ptl_bottom_2_p'
                # , 'rt_1_p'
                ]



    # 'Open_1', 'High_1', 'Low_1', 'Close_1', 'Volume_1', 'Open_2', 'High_2', 'Low_2', 'Close_2', 'Volume_2'
    # , 'rt_1', 'rt_2', 'ptl_top_1', 'ptl_bottom_1', 'ptl_top_reach_1', 'ptl_bottom_reach_1', 'ptl_top_2'
    # , 'ptl_bottom_2', 'ptl_top_reach_2', 'ptl_bottom_reach_2', 'rt_1_p', 'rt_2_p', 'ptl_top_1_p', 'ptl_bottom_1_p'
    # , 'ptl_top_2_p', 'ptl_bottom_2_p', 'vol_1_diff', 'vol_2_diff', 'vol_1_diff_p', 'vol_2_diff_p'
    # , 'vol_2_diff_top_p', 'vol_2_diff_bot_p'
    # , 'rtPre_2', 'rtReg_2', 'rtPst_2', 'rtPrePst_2', 'combine_2_1_p'
    # ,'rtPre_1', 'rtReg_1', 'rtPst_1', 'rtPrePst_1'

    # Reset index to make it consecutive
    # df2 = df2.reset_index(drop=True)
    # Prepare data safely
    df2 = df2.dropna().reset_index(drop=True)

    # print ("1")
    # print(df2.index[:10])  # Should print RangeIndex(start=0, stop=10, step=1)
    # print("2")
    # print(df2.head())  # Check structure
    # print("3")
    # print(type(df2.index))
    # print("4")

    X = df2[features]
    # y = (df2['rt_2'] > 0).astype(int)  # Target: 1 if next return positive
    # y = (df2['rt_2'] > 0.8).astype(int)  # Target: 1 if next return positive
    y = (df2['rtHl_2'] > 0.8).astype(int)  # Target: 1 if next return positive

    features2 = features + ['datetime']
    xFile = df2[features2]
    xFile.to_csv("c:/tmp/X1.csv")

    print ("100")
    print(df2["rtHl_2"].describe())
    print("101")
    print((df2["rtHl_2"] > 0).value_counts())
    print("102")
    print((df2["rtHl_2"] > 0.8).value_counts())
    print("200")

    # Step 3: Train/test split and fit model
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # split_date = '2025-08-01'
    # X_train, X_test = X.loc[:split_date], X.loc[split_date:]
    # y_train, y_test = y.loc[:split_date], y.loc[split_date:]
    # 5 fold
    # tscv = TimeSeriesSplit(n_splits=50, test_size = 3, max_train_size=6)
    tscv = TimeSeriesSplit(n_splits=5)
    best_auc = -np.inf
    best_C = None
    best_model = None

    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        y_train_fold = y.iloc[train_idx]
        print(f"Fold {i} classes:", np.unique(y_train_fold))

    for C in [0.01, 0.1, 1.0, 3.0, 10.0]:
        aucs = []
        # for train_idx, test_idx in tscv.split(X):
        #     X_tr, X_te = X[train_idx], X[test_idx]
        #     y_tr, y_te = y[train_idx], y[test_idx]
        for train_idx, test_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            y_te = y.iloc[test_idx]
            print("Unique classes in y_train:", np.unique(y_tr))

            model = LogisticRegression(
                penalty="l2",
                C=C,
                solver="lbfgs",
                max_iter=1000
            )
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, p))
        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_C = C

    # Fit final model on full dataset
    best_model = LogisticRegression(
        penalty="l2", C=best_C, solver="lbfgs", max_iter=1000
    )
    best_model.fit(X, y)

    print(f"Chosen C={best_C}, CV AUC={best_auc:.3f}")
    print("Betas (coef):")
    for name, coef in zip(features, best_model.coef_[0]):
        print(f"  {name}: {coef:.6f}")
    print(f"Intercept: {best_model.intercept_[0]:.6f}")

    # -----------------------------
    # Feature importance interpretation
    # -----------------------------
    features = X.columns
    coefs = best_model.coef_[0]  # coefficients for each feature

    importance = pd.DataFrame({
        "feature": features,
        "coef": coefs
        # "abs_coef": abs(coefs)
    }).sort_values("feature", ascending=True)
    # }).sort_values("abs_coef", ascending=False)

    print("\nFeature importance:")
    # print(importance)
    # print(importance.to_string(index=False, justify="left"))
    # print(importance.astype(str).to_string(index=False, justify="left"))
    for _, row in importance.iterrows():
        print(f"{row['feature']} {row['coef']}")

    # Compare to original strategy on test set
    # test_indices = X_test.index
    # original_positive_test = df2.loc[test_indices, 'rt_2'][
    # original_positive_test = df2.loc[test_indices, 'rtHl_2'][
    #                             df2.loc[test_indices, 'signal'] == 1] > 0
    # original_success_test = original_positive_test.mean() if not original_positive_test.empty else 0
    # print(f"Original Success Rate on Test: {original_success_test:.4f}")

    # -----------------------------
    # find AUC among each features
    # -----------------------------
    # 'ptl_top_2_p', 'vol_2_diff_top_p', 'rtPre_2'
    # logit_all = LogisticRegression().fit(X_train, y_train)
    # auc_all = roc_auc_score(y_test, logit_all.predict_proba(X_test)[:, 1])

    # logit_ptl_top_2_p = LogisticRegression().fit(X_train[['ptl_top_2_p']], y_train)
    # auc_ptl_top_2_p = roc_auc_score(y_test, logit_ptl_top_2_p.predict_proba(X_test[['ptl_top_2_p']])[:, 1])

    # logit_vol_2_diff_top_p = LogisticRegression().fit(X_train[['vol_2_diff_top_p']], y_train)
    # auc_vol_2_diff_top_p = roc_auc_score(y_test, logit_vol_2_diff_top_p.predict_proba(X_test[['vol_2_diff_top_p']])[:, 1])

    # logit_rtPre_2 = LogisticRegression().fit(X_train[['rtPre_2']], y_train)
    # auc_rtPre_2 = roc_auc_score(y_test, logit_rtPre_2.predict_proba(X_test[['rtPre_2']])[:, 1])

    #logit_4 = LogisticRegression().fit(X_train[['ptl_top_reach_2_p']], y_train)
    #auc_4 = roc_auc_score(y_test, logit_4.predict_proba(X_test[['ptl_top_reach_2_p']])[:, 1])

    #logit_5 = LogisticRegression().fit(X_train[['vol2_top_reach_2_p']], y_train)
    #auc_5 = roc_auc_score(y_test, logit_5.predict_proba(X_test[['vol2_top_reach_2_p']])[:, 1])


    # print()
    # print("AUC with all features:", auc_all)
    # print("AUC with auc_ptl_top_2_p only:", auc_ptl_top_2_p)
    # print("AUC with auc_vol_2_diff_top_p only:", auc_vol_2_diff_top_p)
    # AUC value interpretation:
    # AUC	    Meaning	Performance
    # 1.0	    Perfect classifier  Excellent
    # 0.9–1.0	Great separation	Strong
    # 0.7–0.9	Good separation	    Acceptable
    # 0.5–0.7	Weak separation	    Poor
    # 0.5	    Random guessing	    No predictive power
    # < 0.5	    Worse than random   Predicts oppositely




def performRandomForestDIA(df1):
    # Compute features (X)
    # 1) Last 30min return = rt_1

    # 2) 14 bar RSI
    delta = df1['rt_1']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_days = 14
    avg_gain = gain.rolling(window=rsi_days, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_days, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df1['rsi_14'] = 100 - (100 / (1 + rs))
    df1['rsi_14_p'] = df1['rsi_14'].shift(1)

    # 3) minute return percentile (90% and 10%)
    # df1['ptl_top_1_p'], df1['ptl_bottom_1_p']

    # 4) 20 days EMA (assuming 13 30min bars per day, 20 bars)
    emaNumBars1 = 20
    emaNumBars2 = 50
    emaNumBars3 = 100
    df1['ema20'] = df1['Close_1'].ewm(span=emaNumBars1, adjust=False).mean()
    df1['ema50'] = df1['Close_1'].ewm(span=emaNumBars2, adjust=False).mean()
    df1['ema100'] = df1['Close_1'].ewm(span=emaNumBars3, adjust=False).mean()
    df1['ema20_p'] = df1['ema20'].shift(1)
    df1['ema50_p'] = df1['ema50'].shift(1)
    df1['ema100_p'] = df1['ema100'].shift(1)

    # 5) Top and bottom Bollinger Bands (20 periods, 2 std dev)
    bbNumBars = 20
    rolling_mean = df1['Close_1'].rolling(window=bbNumBars).mean()
    rolling_std = df1['Close_1'].rolling(window=bbNumBars).std()
    df1['top_bb'] = rolling_mean + (rolling_std * 2)
    df1['bottom_bb'] = rolling_mean - (rolling_std * 2)
    df1['top_bb_p'] = df1['top_bb'].shift(1)
    df1['bottom_bb_p'] = df1['bottom_bb'].shift(1)

    # Drop NaN rows. suppose data manipulate process done
    # nvda_data = nvda_data.dropna()

    df2 = logisticRegressionPreFilter(df1)

    # Define y: 1 if next return > 50, else 0 (assuming 'return' means price change)
    df2['y'] = (df2['rt_1'] > 0).astype(int)
    df2 = df2.dropna()  # Drop last row



    # Features X
    # X = df2[['rt_1_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p', 'vol_1_diff_p']]
    # X = df2[['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']]
    X = df2[['rt_1_p', 'rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p',
             'vol_1_diff_p', 'rtHl_2_p']]
    y = df2['y']

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    split_date = '2025-08-01'
    X_train, X_test = X.loc[:split_date], X.loc[split_date:]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]

    # Tune RandomForestRegressor with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    #
    # below performRandomForest not accuracy because classification approach do not use
    #
    # libary RandomForestRegressor, instead, use RandomForestClassifier
    # below scoring use MSE is incorrect because output only 1/0, incorrect to find diff between y and predict y

    # rf = RandomForestRegressor(random_state=42)
    # grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ## grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    ## Best parameters and model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    ## Evaluate on test set
    # y_pred = best_model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)

    ## Output
    # print("Best Parameters:", best_params)
    # print("Test MSE:", mse)
    # print("Feature Importances:", best_model.feature_importances_)
    # print()

    #
    # below performRandomForest not accuracy because classification approach do not use
    #

    # Tune RandomForestClassifier with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf_clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Output
    print("Best Parameters:", best_params)
    print("Test Accuracy:", accuracy)

    # Create table for Feature Importances
    feature_names = X.columns
    # coefs = best_model.coef_[0]  # coefficients for each feature
    feature_importance = pd.DataFrame({
        'FeatureName': feature_names,
        'FeatureImportance': best_model.feature_importances_
    }).sort_values(by='FeatureName', ascending=True)

    print("\nFeature Importances (Table Format):")
    # print(feature_importance.to_string(index=False))
    for _, row in feature_importance.iterrows():
        print(f"{row['FeatureName']} {row['FeatureImportance']}")

    # Use cross_val_score with 5-fold CV and neg_mean_squared_error scoring
    # scores = cross_val_score(rf, X, y, cv=10, scoring='neg_mean_squared_error')
    # # Print results
    # print("Cross-Validation for random forest", scores)
    # print("Cross-Validation Scores (neg MSE):", scores)
    # print("Mean neg MSE:", np.mean(scores))
    # print("Mean RMSE:", np.sqrt(-np.mean(scores)))
    # compare2Model1(X, y)
    # compare2Model2(X, y)

    # -----------------------------
    # Logistic Regression (with scaling)
    # -----------------------------
    # logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # logit.fit(X_train, y_train)
    # print("Logistic Regression Accuracy:", logit.score(X_test, y_test))
    #
    # # -----------------------------
    # # Random Forest Classifier
    # # -----------------------------
    # rf = RandomForestClassifier(n_estimators=200, random_state=42)
    # rf.fit(X_train, y_train)
    # print("Random Forest Accuracy:", rf.score(X_test, y_test))
    #
    # # -----------------------------
    # # Cross-validation comparison
    # # -----------------------------
    # logit_cv = cross_val_score(logit, X, y, cv=5, scoring="accuracy").mean()
    # rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()
    #
    # print("CV LogisticRegression Accuracy:", logit_cv)
    # print("CV RandomForestClassifier Accuracy:", rf_cv)


def performRandomForestNQ(df1):
    # Compute features (X)
    # 1) Last 30min return = rt_1

    # 2) 14 bar RSI
    delta = df1['rt_2']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    rsi_days = 14
    avg_gain = gain.rolling(window=rsi_days, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_days, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df1['rsi_14'] = 100 - (100 / (1 + rs))
    df1['rsi_14_p'] = df1['rsi_14'].shift(1)

    # 3) minute return percentile (90% and 10%)
    # df1['ptl_top_1_p'], df1['ptl_bottom_1_p']

    # 4) 20 days EMA (assuming 13 30min bars per day, 20 bars)
    emaNumBars1 = 20
    emaNumBars2 = 50
    emaNumBars3 = 100
    df1['ema20'] = df1['Close_2'].ewm(span=emaNumBars1, adjust=False).mean()
    df1['ema50'] = df1['Close_2'].ewm(span=emaNumBars2, adjust=False).mean()
    df1['ema100'] = df1['Close_2'].ewm(span=emaNumBars3, adjust=False).mean()
    df1['ema20_p'] = df1['ema20'].shift(1)
    df1['ema50_p'] = df1['ema50'].shift(1)
    df1['ema100_p'] = df1['ema100'].shift(1)

    # 5) Top and bottom Bollinger Bands (20 periods, 2 std dev)
    bbNumBars = 20
    rolling_mean = df1['Close_2'].rolling(window=bbNumBars).mean()
    rolling_std = df1['Close_2'].rolling(window=bbNumBars).std()
    df1['top_bb'] = rolling_mean + (rolling_std * 2)
    df1['bottom_bb'] = rolling_mean - (rolling_std * 2)
    df1['top_bb_p'] = df1['top_bb'].shift(1)
    df1['bottom_bb_p'] = df1['bottom_bb'].shift(1)

    # Drop NaN rows. suppose data manipulate process done
    # nvda_data = nvda_data.dropna()

    df2 = logisticRegressionPreFilter(df1)

    # Define y: 1 if next return > 50, else 0 (assuming 'return' means price change)
    df2['y'] = (df2['rt_2'] > 0).astype(int)
    df2 = df2.dropna()  # Drop last row

    # Features X
    # X = df2[['rt_1_p', 'rt_2_2Bar_p', 'rt_2_3Bar_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p', 'vol_1_diff_p']]
    # X = df2[['rt_2_p', 'rsi_14_p', 'ptl_top_1_p', 'ptl_bottom_1_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p']]
    # X = df2[['rt_1_p', 'rt_2_p', 'rsi_14_p', 'ptl_top_2_p', 'ptl_bottom_2_p', 'ema20_p', 'top_bb_p', 'bottom_bb_p',
    #          'vol_2_diff_p'
    #          , 'rtPre_2', 'rtReg_2', 'rtPst_2', 'rtPrePst_2'
    #          , 'ptl_top_1_p'
    #          # , 'rtPre_1', 'rtReg_1', 'rtPst_1', 'rtPrePst_1'
    #          ]]
    # X = df2[['ptl_top_2_p', 'vol_2_diff_p' , 'ptl_top_reach_2_p', 'ptl_bottom_reach_2_p', 'vol2_top_reach_2_p']]
    X = df2[[#'ptl_top_2_p', 'vol_2_diff_p', 'rtPre_2'
             # 'ptl_top_2_p', 'top_bb_p'
             # 'ptl_top_2_p', 'top_bb_p', 'rtPre_2'
             # 'ptl_top_2_p', 'top_bb_p', 'rtPrePst_2'
             # 'top_bb_p', 'vol_2_diff_p'
             # 'ptl_bottom_2_p', 'bottom_bb_p'
              # 'bottom_bb_p', 'vol_2_diff_p'
            # 'ptl_bottom_2_p', 'vol_2_diff_p'
            'ptl_bottom_2_p', 'vol_2_diff_p', 'bottom_bb_p'
            # 'ptl_bottom_2_p', 'bottom_bb_p', 'rtPre_2'
             #, 'rtPre_2', 'rtReg_2', 'rtPst_2', 'rtPrePst_2'
             #, 'ptl_top_1_p'
             # , 'rtPre_1', 'rtReg_1', 'rtPst_1', 'rtPrePst_1'
             ]]
    y = df2['y']

    # Split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    split_date = '2025-08-01'
    X_train, X_test = X.loc[:split_date], X.loc[split_date:]
    y_train, y_test = y.loc[:split_date], y.loc[split_date:]

    # Tune RandomForestRegressor with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    #
    # below performRandomForest not accuracy because classification approach do not use
    #
    # libary RandomForestRegressor, instead, use RandomForestClassifier
    # below scoring use MSE is incorrect because output only 1/0, incorrect to find diff between y and predict y

    # rf = RandomForestRegressor(random_state=42)
    # grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    ## grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    # grid_search.fit(X_train, y_train)

    ## Best parameters and model
    # best_params = grid_search.best_params_
    # best_model = grid_search.best_estimator_

    ## Evaluate on test set
    # y_pred = best_model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)

    ## Output
    # print("Best Parameters:", best_params)
    # print("Test MSE:", mse)
    # print("Feature Importances:", best_model.feature_importances_)
    # print()

    #
    # below performRandomForest not accuracy because classification approach do not use
    #

    # Tune RandomForestClassifier with GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    rf_clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Output
    print("Best Parameters:", best_params)
    print("Test Accuracy:", accuracy)

    # Create table for Feature Importances
    feature_names = X.columns
    # coefs = best_model.coef_[0]  # coefficients for each feature
    feature_importance = pd.DataFrame({
        'FeatureName': feature_names,
        'FeatureImportance': best_model.feature_importances_
    }).sort_values(by='FeatureName', ascending=True)

    print("\nFeature Importances (Table Format):")
    # print(feature_importance.to_string(index=False))
    for _, row in feature_importance.iterrows():
        print(f"{row['FeatureName']} {row['FeatureImportance']}")

    # Use cross_val_score with 5-fold CV and neg_mean_squared_error scoring
    # scores = cross_val_score(rf, X, y, cv=10, scoring='neg_mean_squared_error')
    # # Print results
    # print("Cross-Validation for random forest", scores)
    # print("Cross-Validation Scores (neg MSE):", scores)
    # print("Mean neg MSE:", np.mean(scores))
    # print("Mean RMSE:", np.sqrt(-np.mean(scores)))
    # compare2Model1(X, y)
    # compare2Model2(X, y)

    # -----------------------------
    # Logistic Regression (with scaling)
    # -----------------------------
    # logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # logit.fit(X_train, y_train)
    # print("Logistic Regression Accuracy:", logit.score(X_test, y_test))
    #
    # # -----------------------------
    # # Random Forest Classifier
    # # -----------------------------
    # rf = RandomForestClassifier(n_estimators=200, random_state=42)
    # rf.fit(X_train, y_train)
    # print("Random Forest Accuracy:", rf.score(X_test, y_test))
    #
    # # -----------------------------
    # # Cross-validation comparison
    # # -----------------------------
    # logit_cv = cross_val_score(logit, X, y, cv=5, scoring="accuracy").mean()
    # rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()
    #
    # print("CV LogisticRegression Accuracy:", logit_cv)
    # print("CV RandomForestClassifier Accuracy:", rf_cv)


def main():
    # 3 months
    # start_dt1 = datetime(2025, 6, 1)
    #
    # 6 months
    start_dt1 = datetime(2025, 1, 1)
    #
    # 1 year
    # start_dt1 = datetime(2024, 9, 1)
    #
    # 2 year
    # start_dt1 = datetime(2023, 9, 1)
    #
    # end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
    # end_dt1 = datetime(2025, 10, 31)
    end_dt1 = datetime(2025, 11, 1)

    # symbol01= 'NVDA'
    # QQQ corr result better than NVDA
    symbol01 = 'QQQ'  # **
    # symbol01= 'SQQQ' #***
    # TQQQ corr result better than QQQ (vol_1_diff_p)
    # symbol01= 'TQQQ' #*
    # AAPL corr result good for (ptl_top_1_p, -ve relation)
    # symbol01= 'AAPL' #*
    # symbol01= 'META'
    # AMZN corr result good for (ptl_top_1_p, -ve relation)
    # symbol01= 'AMZN' #*
    # symbol01= 'MSFT'
    symbol02 = '@MNQ#C'

    # nvda_data = pd.DataFrame()
    try:
        #
        # 1. get data
        #
        #
        # Connect data source
        #
        conn = HistoryConn(name="history")
        conn.connect()

        #
        # Get data
        #
        # 300 sec = 5min
        # 7200 sec = 2 hr
        # 3600 sec = 1 hr
        # 1800 sec = 30min
        # period1 = 300
        period1 = 1800
        # period1 = 3600
        # period1 = 7200
        # period1 = 10800
        # period1 = 43200  # daily
        # Nasdaq stock, include pre-trading hour (04:00 - 09:30), regular hour (09:30 - 16:00), post-trading hour (16:00 -
        # get DIA all time data, then cut pre-trading, regular or post-trading hour
        # nvda_data = fetch_data_with_time(conn, symbol01, start_dt1, "03:00", end_dt1, "20:30", period1)
        nvda_data = fetch_data_with_time(conn, symbol01, start_dt1, "00:00", end_dt1, "23:30", period1)
        # ym is CME future from 00:00:00 to 23:59:59
        nq_data = fetch_data_with_time(conn, symbol02, start_dt1, "00:00", end_dt1, "23:30", period1)

        #
        # 1. end get data
        #

        #
        # *** up to here save data to dataFrame from external data source
        #

        #
        # 2. manipulate data
        #

        # post add data add symbol name
        # nvda_data['symbol'] = symbol01
        # nq_data['symbol'] = symbol02

        nvda_nq_combine_data1 = merge_2_dataFrame(nvda_data, nq_data)

        #
        # Data cleaning
        #
        # remove holiday
        nvda_nq_combine_data2 = rmHoliday(nvda_nq_combine_data1)

        # nvda_nq_combine_data1.to_csv("c:/tmp/nvda_nq_combine_data1.csv")
        # nvda_nq_combine_data2.to_csv("c:/tmp/nvda_nq_combine_data2.csv")

        # Fill blank (NaN) values in these columns with the previous row's values (forward fill)
        nvda_nq_combine_data3 = fillMissingData(nvda_nq_combine_data2)

        # nvda_nq_combine_data3 include pre-trading, regular and post-trading hour
        # (optional) below filter nvda_nq_combine_data3, nvda_nq_combine_data3 include all trading hour data
        # nvda_nq_combine_data3_pre, nvda_nq_combine_data3_reg, nvda_nq_combine_data3_pst = filterPrePstTradingHour(nvda_nq_combine_data3)
        # nvda_nq_combine_data3 = nvda_nq_combine_data3_reg
        # nvda_nq_combine_data3 = nvda_nq_combine_data3_pst

        #
        # Feature scaling
        #
        # only set @YM#C value /100, TBD
        # nvda_nq_combine_data4 = nvda_nq_combine_data3
        nvda_nq_combine_data4 = featureScaling(nvda_nq_combine_data3)

        #
        # (optional) cut pre-trading and post-trading period
        #
        # only keep from 9:30 to 16:00
        #
        # start_time3 = pd.to_datetime('09:30:00').time()
        # end_time3 = pd.to_datetime('16:30:00').time()
        # nvda_nq_combine_data4 = nvda_nq_combine_data4[(nvda_nq_combine_data4['datetime'].dt.time >= start_time3) & (nvda_nq_combine_data4['datetime'].dt.time <= end_time3)]

        #
        # build target features dataframe
        #
        nvda_nq_combine_data5 = buildFeaturePd(nvda_nq_combine_data4)

        nvda_nq_combine_data5_scaled = featureScaling2(nvda_nq_combine_data5)

        # filter1, try diff filter
        nvda_nq_combine_data6 = nvda_nq_combine_data5
        # nvda_nq_combine_data6 = filter1(nvda_nq_combine_data5)
        # nvda_nq_combine_data6 = filter1(nvda_nq_combine_data5_scaled)

        #
        # 2. end manipulate data
        #

        #
        # *** up to here done 1) data cleaning (e.e. dropna). 2) feature scaling. 3) build  features, e.g. rt_1 return between 2 bars. 4) combine dataFrame
        #

        #
        # 3. plot scatter chart (see relation)
        #

        # scatter matrix plot relation among all features
        # 1st time plot chart use this one to get a brief look
        # scatterMatrixChart(nvda_nq_combine_data6)

        # scatter chart among 1 target column and other columns
        # more detail plot between 1 feature with other after view above scattermatrix
        # scatterPlot(nvda_nq_combine_data6)

        # scatter chart among 1 target column and other columns, show multiple time frame
        # intraday Nasdaq/Dow use this one, limit time frame
        # scatterPlot3(nvda_nq_combine_data6)

        #
        # **** TBD. scatterPlot3 (may be new function scatterPlot4)
        #
        # 1) add filter weekday , e.g. Monday
        # 2) add filter date, e.g. 1st of each month
        #

        #
        # 3. end plot scatter chart (see relation)
        #

        #
        # *** up to here know which 2 features has relation
        #
        # --> e.g. rt_1 vs rt_2_3Bar_p
        #

        #
        # run ML model
        #
        # 4. correlation
        # check which item has digitally high correlation with return between 2 bars
        # e.g. abs(corr) >= 0.08
        #
        # filter2. intraday
        # test_corr1_df = filter2(nvda_nq_combine_data5)
        # performCorrelation(test_corr1_df)
        # performCorrelation(nvda_nq_combine_data6)
        #
        #
        # 4. end correlation
        #

        #
        # *** up to here know which 2 or 3 features has relation (view scatter chart and corr), in which time period
        #
        # e.g. abs(corr) >= 0.08
        # rt_1. from 11:00 to 11:30. -ve relation
        # rt_1_3Bar_p -0.127638
        # rt_2_3Bar_p -0.128724
        # rt_1. from 9:30 to 10:00
        # rt_2_p -0.091694
        # rt_1_p -0.095273
        # rt_2. from 9:30 to 10:00
        # rt_1_c_p 0.089293
        # rt_2_c_p 0.085159
        # rt_1_p -0.101435
        # rt_2_p -0.107444
        # rt_1. from 11:30 to 12:00
        # rt_1_pre_p            0.134943
        # ptl_bottom_2_p        0.108473
        # rt_2_c_p -0.239111
        # rt_2_p -0.239111
        # rt_1_c_p -0.242134
        # rt_1_p -0.242134
        # rt_2. from 11:30 to 12:00
        # rt_2_p -0.238809
        # rt_2_c_p -0.238809
        # rt_1_p -0.239885
        # rt_1_c_p -0.239885

        #
        # logistic regression
        #
        # use not scaled
        # performLogisticRegressionDIA (nvda_nq_combine_data5)
        # performLogisticRegressionNQ (nvda_nq_combine_data5)
        # performLogisticRegressionNQ2 (nvda_nq_combine_data6)
        # performLogisticRegressionNQ2 (dia_ym_combine_data5)
        performLogisticRegressionNQ2 (nvda_nq_combine_data5_scaled)


        # (not using this)
        # (not using this) performLogisticRegressionDIA(nvda_nq_combine_data5_scaled)

        #
        # random forest
        #
        # below use RandomForestClassifier
        # performRandomForestDIA(nvda_nq_combine_data5)
        # performRandomForestNQ(nvda_nq_combine_data5)
        # performRandomForestNQ(nvda_nq_combine_data5_scaled)







    except Exception as e:
        # logger.error(f"Failed to fetch data: {e}")
        print(f"Failed to fetch data: {e}")
    finally:
        # if (conn and conn.connected()):
        conn.disconnect()


# run main
main()