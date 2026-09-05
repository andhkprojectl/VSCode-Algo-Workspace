import pandas as pd
from datetime import datetime, timedelta
from ib_insync import IB, ContFuture, Stock, util


# tickerType
# ST: stock
# FU: future
# other not support
#
# if future, provide futureExchange, e.g. "CME" for NQU6-CME-FUT
#
# if not continous future, provide futureExpireDate, e.g. "202609" for NQU6-CME-FUT
#
# include not only stock but also future
def getAllTypesTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1, tickerType, isConFuture= True, futureExpireDate=None, futureExchange=None):
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
    if (tickerType == "ST"):
        contract = Stock(symbolName, 'SMART', 'USD')
    elif (tickerType == "FU"):
        if (isConFuture == True):
            if futureExchange is None:
                print(f"ERROR: For continous futures, futureExchange must be provided.")
                return pd.DataFrame()
            contract = ContFuture(symbolName, futureExchange)      
        else:
            if futureExpireDate is None or futureExchange is None:
                print(f"ERROR: For non-continous futures, both futureExpireDate and futureExchange must be provided.")
                return pd.DataFrame()
            contract = ContFuture(symbolName, futureExpireDate, futureExchange)                  
            # contract = Future('NQ', '202609', 'CME')
        contract = ContFuture(symbolName, futureExchange)
        # This fetches the necessary details from IB to confirm the contract exists
        conn1.qualifyContracts(contract)
    else:
        print(f"ERROR: Unsupported tickerType '{tickerType}' for symbol '{symbolName}'.")
        return pd.DataFrame()

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
            if (tickerType == "ST"):
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
            elif (tickerType == "FU"):                
                bars = conn1.reqHistoricalData(
                        contract,
                        endDateTime=ib_end_dt,
                        durationStr=durationStr,
                        barSizeSetting='5 mins',
                        whatToShow='TRADES',
                        useRTH=True
                    )   
            else:
                print(f"ERROR: Unsupported tickerType '{tickerType}' for symbol '{symbolName}'.")
                return pd.DataFrame()                   
                          
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
    result_df = pd.DataFrame({
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


def saveDfToTicker5Min(df, tableName="ticker5Min", isOverride=True):
    """
    Save the market-data DataFrame returned by getTicketDataWithTimeFromIB
    into MariaDB table `ticker5Min` (database IBTradingDb) via mysql-connector-python.

    df columns expected: symbolName, date, time, open, high, low, close, volume
        - date : 'YYYY-MM-DD'
        - time : 'HH:MM:SS'
    The separate `date` and `time` values are combined into a single
    `datetime1` column ('YYYY-MM-DD HH:MM:SS') before being written.

    Behavior:
        - If a (ticker, datetime1) record does NOT exist in the table, it is
          inserted regardless of `isOverride`.
        - If a (ticker, datetime1) record already exists:
            * isOverride=True  -> update the existing record (upsert).
            * isOverride=False -> skip it (no action).

    The unique key uk_ticker_datetime1 (ticker, datetime1) prevents duplicates.
    Returns the number of rows processed.
    """
    import os
    import mysql.connector

    if df is None or df.empty:
        print("WARNING: DataFrame is empty - nothing to save to ticker5Min.")
        return 0

    df_db = df[['symbolName', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_db.columns = ['ticker', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']

    # Combine date + time into a single datetime value ('YYYY-MM-DD HH:MM:SS');
    # MySQL implicitly casts this string to DATETIME on insert.
    df_db['datetime1'] = df_db['date'].astype(str) + ' ' + df_db['time'].astype(str)

    for col in ['open', 'high', 'low', 'close']:
        df_db[col] = df_db[col].astype(float)
    df_db['volume'] = pd.to_numeric(df_db['volume'], errors='coerce').fillna(0).astype('int64')
    df_db = df_db.dropna(subset=['open', 'high', 'low', 'close'])

    records = df_db[['ticker', 'datetime1', 'open', 'high', 'low', 'close', 'volume']].values.tolist()

    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )

    if isOverride:
        # Upsert: insert new rows and update existing (ticker, datetime1) rows.
        action_sql = f"""
            INSERT INTO {tableName} (ticker, datetime1, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open   = VALUES(open),
                high   = VALUES(high),
                low    = VALUES(low),
                close  = VALUES(close),
                volume = VALUES(volume)
        """
        action_desc = "upserted on duplicate key (isOverride=True)"
    else:
        # Insert-or-skip: insert new rows, ignore existing (ticker, datetime1) rows.
        action_sql = f"""
            INSERT IGNORE INTO {tableName} (ticker, datetime1, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        action_desc = "inserted; existing rows skipped (isOverride=False)"

    cursor = conn.cursor()
    try:
        cursor.executemany(action_sql, records)
        conn.commit()
        print(f"Saved {len(records)} rows into {tableName} ({action_desc}).")
        return len(records)
    except Exception as e:
        conn.rollback()
        print(f"ERROR saving to {tableName}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1):
    return (getAllTypesTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1, "ST"))        