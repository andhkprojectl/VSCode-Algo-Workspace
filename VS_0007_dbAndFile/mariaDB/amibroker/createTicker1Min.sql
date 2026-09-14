-- ============================================================
-- File    : createTicker1Min.sql
-- Database: IBTradingDb  (user: ibUser1)
-- Purpose : Create table `ticker1Min` to store ticker OHLCV data
--           fetched from IB. The table is read by AmiBroker via
--           ODBC (MySQL/MariaDB Connector/ODBC DSN).
--
-- Columns : identical to `ticker5Min` (datetime1 + ticker + OHLCV)
-- ============================================================

USE IBTradingDb;

-- Remove the existing table if it needs to be recreated
-- DROP TABLE IF EXISTS ticker1Min;

CREATE TABLE IF NOT EXISTS ticker1Min (
    id          BIGINT        NOT NULL AUTO_INCREMENT COMMENT 'Surrogate primary key (also needed for updatable ODBC result sets)',
    ticker      VARCHAR(20)   NOT NULL                COMMENT 'Ticker / symbol code, e.g. NVDA',
    datetime1   DATETIME      NOT NULL                COMMENT 'Combined trading date and time (YYYY-MM-DD HH:MM:SS)',
    open        DECIMAL(12,4) NOT NULL                COMMENT 'Open price',
    high        DECIMAL(12,4) NOT NULL                COMMENT 'High price',
    low         DECIMAL(12,4) NOT NULL                COMMENT 'Low price',
    close       DECIMAL(12,4) NOT NULL                COMMENT 'Close price',
    volume      BIGINT        NOT NULL DEFAULT 0      COMMENT 'Bar volume',
    created_at  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Row insert timestamp',
    PRIMARY KEY (id),
    UNIQUE KEY uk_ticker_datetime1 (ticker, datetime1),
    KEY idx_ticker (ticker),
    KEY idx_datetime1 (datetime1)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Ticker OHLCV data from IB, read by AmiBroker via ODBC';

-- ============================================================
-- Verification
-- ============================================================
-- SHOW CREATE TABLE ticker1Min;
-- DESCRIBE ticker1Min;
