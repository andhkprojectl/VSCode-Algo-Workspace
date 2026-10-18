-- ============================================================
-- File    : createTradeExecution.sql
-- Database: IBTradingDb  (user: ibUser1)
-- Purpose : Store actual Interactive Brokers execution/fill details
--           reported while a live strategy is running.
-- ============================================================

USE IBTradingDb;

CREATE TABLE IF NOT EXISTS tradeExecution (
    id                BIGINT        NOT NULL AUTO_INCREMENT,
    strategy          VARCHAR(20)   NOT NULL COMMENT 'Strategy id, e.g. S4003',
    ticker            VARCHAR(20)   NOT NULL COMMENT 'IB contract symbol',
    orderId           BIGINT        NOT NULL COMMENT 'IB order id',
    execId            VARCHAR(128)  NOT NULL COMMENT 'Globally unique IB execution id',
    executionTimeUtc  DATETIME      NOT NULL COMMENT 'IB execution time normalized to naive UTC',
    side              VARCHAR(10)   NOT NULL COMMENT 'IB execution side, e.g. BOT or SLD',
    quantity          DECIMAL(18,4) NOT NULL COMMENT 'Executed quantity',
    price             DECIMAL(18,4) NOT NULL COMMENT 'Executed price',
    exchangeName      VARCHAR(64)   NOT NULL DEFAULT '' COMMENT 'Execution exchange',
    account           VARCHAR(64)   NOT NULL DEFAULT '' COMMENT 'IB account',
    created_at        TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uk_tradeExecution_execId (execId),
    KEY idx_tradeExecution_strategy_ticker (strategy, ticker),
    KEY idx_tradeExecution_orderId (orderId),
    KEY idx_tradeExecution_time (executionTimeUtc)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Interactive Brokers live-trade execution details';

-- Verification:
-- SHOW CREATE TABLE tradeExecution;
-- DESCRIBE tradeExecution;
