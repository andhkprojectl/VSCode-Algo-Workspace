import pandas as pd
from datetime import datetime
from ib_insync import IB, Stock, util

def getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1):
    """
    Fetches historical intraday market data from Interactive Brokers.
    """

    # Increase the global request timeout to 60 seconds (default is usually 10-20s)
    # IB.setConnectTimeout(120) # 120 s timeout
    
    # 1. Handle Connection
    # If conn1 is None or not connected, establish connection to TWS/Gateway
    if conn1 is None or not conn1.isConnected():
        conn1 = IB()
        # Connect to localhost on port 4002 (Paper trading / Gateway default)
        # Increase timeout to 5 minutes (or 0 for infinite)
        # conn1.RequestTimeout = 300        
        # clientId=1 is used to uniquely identify this connection
        conn1.connect('127.0.0.1', 4002, clientId=1)
    # set timeout
    # conn1.setTimeout(300) 


    # 2. Parse Dates and Times
    # Strip any colons from the time inputs to safely handle both '22:30' and '2230'
    start_time_clean = startTime.replace(':', '')
    end_time_clean = endTime.replace(':', '')
    
    # Combine into strings
    start_dt_str = f"{startDate}{start_time_clean}"
    end_dt_str = f"{endDate}{end_time_clean}"
    
    # Convert to Python datetime objects
    start_dt = datetime.strptime(start_dt_str, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end_dt_str, "%Y%m%d%H%M")
    
    # IB expects the endDateTime parameter formatted as 'YYYYMMDD HH:MM:SS'
    ib_end_dt = end_dt.strftime("%Y%m%d %H:%M:%S")
    
    # 3. Calculate Duration 
    # IB API fetches backwards from the endDateTime based on a duration string.
    duration_delta = end_dt - start_dt
    days = duration_delta.days
    
    if days < 1:
        durationStr = "1 D"
    elif days < 30:
        durationStr = f"{days + 1} D" # Add 1 day buffer to ensure coverage
    elif days < 365:
        durationStr = f"{duration_delta.days // 30 + 1} M"
    else:
        durationStr = f"{duration_delta.days // 365 + 1} Y"
        
    # 4. Format Bar Size (period1)
    if period1 == 1:
        barSizeSetting = "1 min"
    else:
        barSizeSetting = f"{period1} mins"
        
    # 5. Define the Contract
    # Assuming standard US Equities (SMART routing, USD)
    contract = Stock(symbolName, 'SMART', 'USD')

    



    
    print("before get data:", datetime.now())
    try:
        # 6. Fetch Data from IB
        # bars = conn1.reqHistoricalData(
        #     contract,
        #     endDateTime=ib_end_dt,
        #     durationStr=durationStr,
        #     barSizeSetting=barSizeSetting,
        #     whatToShow='TRADES',
        #     useRTH=False, # Set to False to include pre/post market data
        #     formatDate=1
        # )
        bars = conn1.run(conn1.reqHistoricalDataAsync(
            contract,
            endDateTime=ib_end_dt,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow='TRADES',
            useRTH=False, # Set to False to include pre/post market data
            formatDate=1,
            timeout=300))  

    except Exception as e:
        # Code that runs if an error happens (the "catch" part)
        print(f"An error occurred: {e}")
        bars = []  # Set bars to empty list on error to allow function to continue and return empty DataFrame
    finally:
        # Code that runs NO MATTER WHAT (success or failure)
        print("after get data:", datetime.now())  
    
    
    # Return empty DataFrame if no data is found
    if not bars:
        return pd.DataFrame()
        
    # 7. Data Processing into DataFrame
    # ib_insync provides a fast utility to convert bars to pandas
    df = util.df(bars)
    
    # Create the strict datetime column (yyyy-mm-dd hh24:mi:ss) for indexing
    df['index_datetime'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Set the new datetime column as the index
    df.set_index('index_datetime', inplace=True)
    
    # Ensure index is sorted in ascending order
    df.sort_index(ascending=True, inplace=True)
    
    # Filter the dataframe strictly by the requested start and end times 
    start_filter = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_filter = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.loc[start_filter:end_filter]
    
    # Prepare individual columns for final output
    df_date = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df_time = pd.to_datetime(df.index).strftime('%H:%M:%S')
    
    # Create the requested custom formatted datetime column: mm/dd/yyyy hh24:mi
    df_custom_datetime = pd.to_datetime(df.index).strftime('%m/%d/%Y %H:%M')
    
    

    # 8. Construct Final Output mapping to requested columns
    result_df = pd.DataFrame({
        'datetime': df_custom_datetime,  # <--- New column added here
        'date': df_date,
        'time': df_time,
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'open': df['open'],
        'volume': df['volume'],
        'symbolName': symbolName 
    }, index=df.index) 

    
    
    return result_df