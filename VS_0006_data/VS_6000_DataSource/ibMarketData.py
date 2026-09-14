import time

import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from ib_insync import IB, Future, Stock, util

# All bar timestamps are normalized to naive US/Eastern before returning/saving,
# matching the existing ticker1Min data (CME Globex halt = 17:00-17:59 ET, no bars).
DB_TZ = ZoneInfo("America/New_York")


def _thirdFriday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1)
    first_friday = 1 + (4 - d.weekday()) % 7
    return datetime(year, month, first_friday + 14)


def frontMonthExpireCode(dt: datetime) -> str:
    """
    Front-month 'YYYYMM' for CME quarterly futures (Mar/Jun/Sep/Dec) as of date dt.
    CME rolls ~8 days before the 3rd-Friday expiry, so pick the nearest expiry >= dt + 8 days.
    e.g. 2026-05-01 -> '202606', 2026-06-15 -> '202609'.
    """
    dt = dt.replace(tzinfo=None)  # roll calendar only depends on the calendar date
    y, m = dt.year, ((dt.month - 1) // 3) * 3 + 3
    target = dt + timedelta(days=8)
    while True:
        if _thirdFriday(y, m) >= target:
            return f"{y}{m:02d}"
        m += 3
        if m > 12:
            m, y = 3, y + 1


# Columns of the final result_df, so empty results stay shape-compatible
_EMPTY_RESULT_COLS = ['datetime', 'date', 'time', 'high', 'low', 'close',
                      'open', 'volume', 'symbolName']


def _emptyResult() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_RESULT_COLS)


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
        conn1.connect('127.0.0.1', 4002, clientId=1) # IB gateway (paper)
        # conn1.connect('127.0.0.1', 7497, clientId=1) # IB TWS (paper)

    # 2. Parse Dates and Times (interpreted as US/Eastern, matching DB timestamps)
    start_time_clean = startTime.replace(':', '')
    end_time_clean = endTime.replace(':', '')

    start_dt_str = f"{startDate}{start_time_clean}"
    end_dt_str = f"{endDate}{end_time_clean}"

    start_dt = datetime.strptime(start_dt_str, "%Y%m%d%H%M").replace(tzinfo=DB_TZ)
    end_dt = datetime.strptime(end_dt_str, "%Y%m%d%H%M").replace(tzinfo=DB_TZ)
    
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
                return _emptyResult()
            # ContFuture always resolves to the front month of TODAY, so historical
            # requests would hit a back-month contract with near-zero volume.
            # Instead, resolve the front-month Future per chunk in the fetch loop below.
            contract = None
        else:
            if futureExpireDate is None or futureExchange is None:
                print(f"ERROR: For non-continous futures, both futureExpireDate and futureExchange must be provided.")
                return _emptyResult()
            # includeExpired=True is required for past/expired contracts (IB error 200 otherwise)
            contract = Future(symbolName, futureExpireDate, futureExchange, includeExpired=True)
            conn1.qualifyContracts(contract)
            if not contract.conId:
                print(f"ERROR: Could not qualify {symbolName} {futureExpireDate} on {futureExchange}.")
                return _emptyResult()
    else:
        print(f"ERROR: Unsupported tickerType '{tickerType}' for symbol '{symbolName}'.")
        return _emptyResult()

    # 5. Fetch Data
    all_dfs = []

    print(f"Starting data fetch for {symbolName} from {start_dt} to {end_dt}")

    current_start = start_dt
    cur_expiry = None
    while current_start < end_dt:
        # Determine the end date for this specific chunk (max 5 days for 1-min, 16 days otherwise)
        chunk_days = 5 if period1 == 1 else 16
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)

        # Continuous future: use the contract that was front month as of this chunk's date
        if (tickerType == "FU") and (isConFuture == True):
            expiry = frontMonthExpireCode(current_start)
            if expiry != cur_expiry:
                cur_expiry = expiry
                # includeExpired=True is required for past/expired contracts (IB error 200 otherwise)
                contract = Future(symbolName, expiry, futureExchange, includeExpired=True)
                conn1.qualifyContracts(contract)
                if not contract.conId:
                    print(f"ERROR: Could not qualify {symbolName} {expiry} on {futureExchange}.")
                    return _emptyResult()
                print(f"  -> Front month as of {current_start.date()}: {contract.localSymbol}")
        
        # IB expects endDateTime for this chunk (aware UTC so TWS interprets it
        # deterministically regardless of gateway timezone settings)
        ib_end_dt = current_end.astimezone(timezone.utc)

        # Calculate duration string for this specific chunk
        duration_delta = current_end - current_start
        days = duration_delta.days

        if days < 1:
            # Sub-day stub: use hours ("1 D" can truncate to the last session segment)
            hours = int(duration_delta.total_seconds() // 3600) + 1
            durationStr = f"{hours} H"
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
                    formatDate=2,
                    timeout=300
                ))
            elif (tickerType == "FU"):
                bars = conn1.reqHistoricalData(
                        contract,
                        endDateTime=ib_end_dt,
                        durationStr=durationStr,
                        barSizeSetting=barSizeSetting,
                        whatToShow='TRADES',
                        useRTH=False,
                        formatDate=2
                    )
                print(f"FU11")
            else:
                print(f"ERROR: Unsupported tickerType '{tickerType}' for symbol '{symbolName}'.")
                return _emptyResult()                   
                          
            if bars:
                chunk_df = util.df(bars)
                all_dfs.append(chunk_df)
                
        except Exception as e:
            print(f"  -> An error occurred during chunk {current_start} to {current_end}: {e}")
            # IB pacing: back off before repeating the same contract/barSize request
            time.sleep(15)
            # Continue to the next chunk even if one fails

        finally:
            # Advance the start pointer for the next loop iteration
            current_start = current_end
            # IB pacing: avoid 6+ requests for the same contract within 2 seconds
            time.sleep(2)

    print("Data fetch completed:", datetime.now().time().replace(microsecond=0))  
    
    # 6. Check if any data was retrieved across all chunks
    if not all_dfs:
        print("Warning: No data found for the requested period.")
        return _emptyResult()
        
    # 7. Data Processing into a single DataFrame
    # Concatenate all chunks together
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Drop duplicates in case chunk boundaries overlapped slightly due to the 1 D buffer
    df.drop_duplicates(subset=['date'], inplace=True)

    # formatDate=2 returns UTC; normalize to naive US/Eastern to match existing DB data
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(DB_TZ).dt.tz_localize(None)
    
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


def getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1):
    return (getAllTypesTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1, "ST"))        