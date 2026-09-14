import os
import pandas as pd
import mysql.connector


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
