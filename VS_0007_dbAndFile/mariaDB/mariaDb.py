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


def _connectDb():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )


def _validateTableName(tableName):
    if not isinstance(tableName, str) or not tableName.isidentifier():
        raise ValueError(f"Invalid MariaDB table name: {tableName!r}")


def ensureTradeExecutionTable(tableName="tradeExecution"):
    """Create the live-trade execution table when it does not already exist."""
    _validateTableName(tableName)
    conn = _connectDb()
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {tableName} (
                id                BIGINT       NOT NULL AUTO_INCREMENT,
                strategy          VARCHAR(20)  NOT NULL,
                ticker            VARCHAR(20)  NOT NULL,
                orderId           BIGINT       NOT NULL,
                execId            VARCHAR(128) NOT NULL,
                executionTimeUtc  DATETIME     NOT NULL,
                side              VARCHAR(10)  NOT NULL,
                quantity          DECIMAL(18,4) NOT NULL,
                price             DECIMAL(18,4) NOT NULL,
                exchangeName      VARCHAR(64)  NOT NULL DEFAULT '',
                account           VARCHAR(64)  NOT NULL DEFAULT '',
                created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                UNIQUE KEY uk_tradeExecution_execId (execId),
                KEY idx_tradeExecution_strategy_ticker (strategy, ticker),
                KEY idx_tradeExecution_orderId (orderId),
                KEY idx_tradeExecution_time (executionTimeUtc)
            ) ENGINE=InnoDB
              DEFAULT CHARSET=utf8mb4
              COLLATE=utf8mb4_unicode_ci
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def saveTradeExecution(execution, tableName="tradeExecution"):
    """Insert one IB execution, ignoring an already-seen IB execution ID."""
    _validateTableName(tableName)
    required = ("strategy", "ticker", "orderId", "execId", "executionTimeUtc",
                "side", "quantity", "price")
    missing = [key for key in required if execution.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing execution fields: {missing}")
    conn = _connectDb()
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            INSERT IGNORE INTO {tableName}
                (strategy, ticker, orderId, execId, executionTimeUtc, side,
                 quantity, price, exchangeName, account)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            str(execution["strategy"]), str(execution["ticker"]),
            int(execution["orderId"]), str(execution["execId"]),
            execution["executionTimeUtc"], str(execution["side"]),
            float(execution["quantity"]), float(execution["price"]),
            str(execution.get("exchangeName", "")),
            str(execution.get("account", "")),
        ))
        conn.commit()
        return cursor.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
