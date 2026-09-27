# MariaDB Integration — Requirements

Consolidated from `requirement1.txt` and `requirement2.txt`.

## 1. Aim

- AmiBroker reads ticker information (OHLCV) from a local MariaDB database using ODBC.
- Create table(s) in MariaDB database `IBTradingDb` to store ticker information.
- Ticker information is fetched from Interactive Brokers (IB).

## 2. Background

### Software

- Python 3.8+
- `mysql-connector-python`
- `pandas`
- `numpy`
- `ibapi` (for IB Gateway/TWS integration)
- `pyodbc` (if using ODBC connection)

### Database

- MariaDB 10.5+
- IB Trading Database (`IBTradingDb`) with user `ibUser1` (see `sql/createIBDbUser.sql`)
- ODBC Driver for MySQL/MariaDB (MySQL Connector/ODBC)

### AmiBroker

- AmiBroker 6.30.5
- ODBC driver configuration for `IBTradingDb`
- Ticker data import/export capabilities

## 3. Requirement 1 — Database Structure and Data Ingestion

### 3.1 Develop database structure

- Design and create a ticker table that stores ticker information and is readable from AmiBroker via ODBC.
- The table stores: date, time, ticker code, OHLCV.
- The table lives in the IB Trading Database (`IBTradingDb`) with user `ibUser1` (see `sql/createIBDbUser.sql`).
- Date and time are stored in **one single column**, called `datetime1`.

As built, this requirement is implemented per timeframe:

| Table | DDL script |
|-------|-----------|
| `ticker5Min` | `sql/createTicker5Min.sql` |
| `ticker1Min` | `sql/createTicker1Min.sql` |

`ticker5Min` schema (`sql/createTicker5Min.sql`):

| Column | Type | Notes |
|--------|------|-------|
| `id` | BIGINT AUTO_INCREMENT | Surrogate primary key (also needed for updatable ODBC result sets) |
| `ticker` | VARCHAR(20) | Ticker / symbol code, e.g. NVDA |
| `datetime1` | DATETIME | Combined trading date and time (`YYYY-MM-DD HH:MM:SS`) |
| `open` | DECIMAL(12,4) | Open price |
| `high` | DECIMAL(12,4) | High price |
| `low` | DECIMAL(12,4) | Low price |
| `close` | DECIMAL(12,4) | Close price |
| `volume` | BIGINT | Bar volume, default 0 |
| `created_at` | TIMESTAMP | Row insert timestamp |

- Unique key `uk_ticker_datetime1 (ticker, datetime1)` prevents duplicates.
- Indexes: `idx_ticker (ticker)`, `idx_datetime1 (datetime1)`.
- `sql/migrateTicker5Min_datetime1.sql` migrates an older version of the table (separate `date` + `time` columns) to the single `datetime1` column; it is guarded and safe to re-run.

### 3.2 Update existing Python program

- Program: `VS_0006_dataFunc/VS_6003_GetMarketDataToDb/5min/DB_NVDA_20250101_20260430_5Min.py`
- The existing program already reads NVDA data from IB by calling `ibMarketData.getTicketDataWithTimeFromIB(...)` and stores the result in variable `df`.
- Modify the program to save the data from `df` into the MariaDB ticker table (`ticker5Min`) in database `IBTradingDb`.

## 4. Requirement 2 — `saveDfToTicker5Min` Function

### 4.1 Move the function to a shared module

- Move function `saveDfToTicker5Min` out of `DB_NVDA_20250101_20260430_5Min.py` into a shared module; `DB_NVDA_20250101_20260430_5Min.py` then calls the function from that module.
- The requirement originally named `ibMarketData.py` as the destination; as built, the function lives in `VS_0007_dbAndFile/mariaDB/mariaDb.py` and is imported as `mariaDb.saveDfToTicker5Min(df, tableName="ticker5Min", isOverride=True)`.

### 4.2 Updated logic

Add a parameter `isOverride` (boolean, default `True`) to `saveDfToTicker5Min`:

| Condition | `isOverride` | Action |
|-----------|--------------|--------|
| `(datetime1, ticker)` record already exists in the table | `True` | Update the existing record |
| `(datetime1, ticker)` record already exists in the table | `False` | No action (skip) |
| `(datetime1, ticker)` record does not exist in the table | `True` or `False` | Insert the record |

As built in `mariaDb.py`, this is implemented with:

- `isOverride=True` → `INSERT ... ON DUPLICATE KEY UPDATE` (upsert, relies on unique key `uk_ticker_datetime1`)
- `isOverride=False` → `INSERT IGNORE` (existing rows skipped)

Input `df` columns expected: `symbolName, date, time, open, high, low, close, volume` (`date` = `YYYY-MM-DD`, `time` = `HH:MM:SS`); `date` + `time` are combined into `datetime1` before writing. DB connection settings come from environment variables `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` (defaults: `localhost`, `3306`, `ibUser1`, empty password, `IBTradingDb`).

## 5. File References

| File | Role |
|------|------|
| `sql/createIBDbUser.sql` | Create database `IBTradingDb` and user `ibUser1` |
| `sql/createTicker5Min.sql` | Create table `ticker5Min` |
| `sql/createTicker1Min.sql` | Create table `ticker1Min` |
| `sql/migrateTicker5Min_datetime1.sql` | Migrate `date` + `time` columns into single `datetime1` |
| `mariaDb.py` | `saveDfToTicker5Min(df, tableName, isOverride)` implementation |
| `VS_0006_dataFunc/VS_6003_GetMarketDataToDb/5min/DB_NVDA_20250101_20260430_5Min.py` | Fetches NVDA data from IB and saves it to `ticker5Min` |
