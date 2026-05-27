import pandas as pd
from datetime import datetime, timedelta
from ib_insync import IB, Stock, util

def getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1):
    """
    Fetches historical intraday market data from Interactive Brokers, 
    chunking requests into maximum 16-day blocks to handle IB API limits.
    """
    
    # 1. Handle Connection
    if conn1 is None or not conn1.isConnected():
        conn1 = IB()
        conn1.connect('127.0.0.1', 4002, clientId=1)

    # 2. Parse Dates and Times
    start_time_clean = startTime.replace(':', '')
    end_time_clean = endTime.replace(':', '')
    
    start_dt_str = f"{startDate}{start_time_clean}"
    end_dt_str = f"{endDate}{end_time_clean}"
    
    start_dt = datetime.strptime(start_dt_str, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end_dt_str, "%Y%m%d%H%M")
    
    # 3. Format Bar Size (period1)
    if period1 == 1:
        barSizeSetting = "1 min"
    else:
        barSizeSetting = f"{period1} mins"
        
    # 4. Define the Contract
    contract = Stock(symbolName, 'SMART', 'USD')

    # 5. Loop and Chunk Data Fetching (Max 16 Days per request)
    all_dfs = []
    current_start = start_dt
    
    print(f"Starting data fetch for {symbolName} from {start_dt} to {end_dt}")
    
    while current_start < end_dt:
        # Determine the end date for this specific chunk (max 16 days forward)
        current_end = min(current_start + timedelta(days=16), end_dt)
        
        # IB expects endDateTime for this chunk
        ib_end_dt = current_end.strftime("%Y%m%d %H:%M:%S")
        
        # Calculate duration string for this specific chunk
        duration_delta = current_end - current_start
        days = duration_delta.days
        
        if days < 1:
            durationStr = "1 D"
        else:
            durationStr = f"{days + 1} D" # Add 1 day buffer to ensure edge coverage
            
        print(f"  -> Fetching chunk: {current_start} to {current_end} (Duration: {durationStr}) at {datetime.now().time().replace(microsecond=0)}")
        
        try:
            # Fetch Data from IB for this chunk
            bars = conn1.run(conn1.reqHistoricalDataAsync(
                contract,
                endDateTime=ib_end_dt,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow='TRADES',
                useRTH=False, 
                formatDate=1,
                timeout=300
            ))  
            
            if bars:
                chunk_df = util.df(bars)
                all_dfs.append(chunk_df)
                
        except Exception as e:
            print(f"  -> An error occurred during chunk {current_start} to {current_end}: {e}")
            # Continue to the next chunk even if one fails, or you could return/break here depending on your strictness
            
        finally:
            # Advance the start pointer for the next loop iteration
            current_start = current_end

    print("Data fetch completed:", datetime.now().time().replace(microsecond=0))  
    
    # 6. Check if any data was retrieved across all chunks
    if not all_dfs:
        print("Warning: No data found for the requested period.")
        return pd.DataFrame()
        
    # 7. Data Processing into a single DataFrame
    # Concatenate all chunks together
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates in case chunk boundaries overlapped slightly due to the 1 D buffer
    df.drop_duplicates(subset=['date'], inplace=True)
    
    # Create the strict datetime column (yyyy-mm-dd hh24:mi:ss) for indexing
    df['index_datetime'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df.set_index('index_datetime', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # Filter the aggregated dataframe strictly by the overall requested start and end times 
    start_filter = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_filter = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.loc[start_filter:end_filter]
    
    # Prepare individual columns for final output
    df_date = pd.to_datetime(df.index).strftime('%Y-%m-%d')
    df_time = pd.to_datetime(df.index).strftime('%H:%M:%S')
    df_custom_datetime = pd.to_datetime(df.index).strftime('%m/%d/%Y %H:%M')

    # 8. Construct Final Output mapping to requested columns
    `result_df` = pd.DataFrame({
        'datetime': df_custom_datetime,
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