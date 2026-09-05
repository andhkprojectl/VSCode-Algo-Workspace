-- ============================================================
-- File    : migrateTicker5Min_datetime1.sql
-- Database: IBTradingDb  (user: ibUser1)
-- Purpose : Migrate table `ticker5Min` from separate `date` + `time`
--           columns into a single combined `datetime1` column.
--
-- Steps   : 1) Add `datetime1 DATETIME` column
--           2) Backfill it from existing date + time rows
--           3) Drop old unique key uk_ticker_date_time + idx_date
--           4) Drop `date` and `time` columns
--           5) Add new unique key uk_ticker_datetime1 + idx_datetime1
--
-- Safe to re-run: each step is guarded so it only runs if the
-- object still exists / does not yet exist.
-- ============================================================

USE IBTradingDb;

-- ------------------------------------------------------------
-- 1. Add the new combined column (skip if it already exists)
-- ------------------------------------------------------------
SET @col_exists := (
    SELECT COUNT(*) FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = DATABASE()
      AND TABLE_NAME   = 'ticker5Min'
      AND COLUMN_NAME  = 'datetime1'
);
SET @sql := IF(@col_exists = 0,
    'ALTER TABLE ticker5Min ADD COLUMN datetime1 DATETIME NOT NULL COMMENT ''Combined trading date and time (YYYY-MM-DD HH:MM:SS)'' AFTER ticker',
    'SELECT ''datetime1 column already exists'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ------------------------------------------------------------
-- 2. Backfill datetime1 from the existing date + time values
--    (only rows that are not yet populated)
-- ------------------------------------------------------------
UPDATE ticker5Min
   SET datetime1 = CAST(CONCAT(date, ' ', time) AS DATETIME)
 WHERE date IS NOT NULL
   AND time IS NOT NULL;

-- ------------------------------------------------------------
-- 3. Drop indexes that depend on the old date/time columns
--    (guarded so re-run does not error)
-- ------------------------------------------------------------
SET @sql := IF(EXISTS(SELECT 1 FROM information_schema.STATISTICS
                        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ticker5Min'
                          AND INDEX_NAME = 'uk_ticker_date_time'),
    'ALTER TABLE ticker5Min DROP INDEX uk_ticker_date_time',
    'SELECT ''uk_ticker_date_time already gone'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql := IF(EXISTS(SELECT 1 FROM information_schema.STATISTICS
                        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ticker5Min'
                          AND INDEX_NAME = 'idx_date'),
    'ALTER TABLE ticker5Min DROP INDEX idx_date',
    'SELECT ''idx_date already gone'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ------------------------------------------------------------
-- 4. Drop the now-obsolete date and time columns
-- ------------------------------------------------------------
SET @sql := IF(@col_exists = 0,
    'ALTER TABLE ticker5Min DROP COLUMN date, DROP COLUMN time',
    'SELECT ''date/time already dropped'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ------------------------------------------------------------
-- 5. Add the replacement unique key + lookup index on datetime1
-- ------------------------------------------------------------
SET @sql := IF(NOT EXISTS(SELECT 1 FROM information_schema.STATISTICS
                            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ticker5Min'
                              AND INDEX_NAME = 'uk_ticker_datetime1'),
    'ALTER TABLE ticker5Min ADD UNIQUE KEY uk_ticker_datetime1 (ticker, datetime1)',
    'SELECT ''uk_ticker_datetime1 already exists'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

SET @sql := IF(NOT EXISTS(SELECT 1 FROM information_schema.STATISTICS
                            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'ticker5Min'
                              AND INDEX_NAME = 'idx_datetime1'),
    'ALTER TABLE ticker5Min ADD KEY idx_datetime1 (datetime1)',
    'SELECT ''idx_datetime1 already exists'' AS msg');
PREPARE stmt FROM @sql; EXECUTE stmt; DEALLOCATE PREPARE stmt;

-- ============================================================
-- Verification
-- ============================================================
-- SHOW CREATE TABLE ticker5Min;
-- DESCRIBE ticker5Min;
