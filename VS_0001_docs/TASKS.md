# TASKS — For whole workspace

## Phase 0 — Foundation 

- [x] T0.1 Create `config.yaml` (workspace root): global strategy configuration with section `Ingest` (T6.1). Parameter list per section follows PRD.md T6.1. Never store credentials in config.yaml; credentials stay in `VS_0002_config/.env`.

## Phase 1 — Ingest
- [x] T1.1 implement `VS_6003_GetMarketDataToDb/1min/ingest.py`: logic and function same as DB_NQ_20250101_20260702_1Min.py. When call function ibMarketData.getAllTypesTicketDataWithTimeFromIB, instead of hard code, move parameters symbolName, startDate, startTime, endDate, endTime, period1, tickerType, isConFuture, futureExpireDate, futureExchange to global configuration file `config.yaml` (workspace root), under section Ingest

## Phase 7 — CLI command py file

- [x] T7.1 Implement `ingest.py`: wrapper for ingest mode. When run this program, actually call `VS_6003_GetMarketDataToDb/1min/ingest.py`, see T6.1

- [x] T7.2 Implement `backtest.py`: Wrapper for backtesting mode. CLI arguments: `--strategy <id>`, actually call backtest file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `backtest.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/3 BackTest/S4003_backtest.py`

- [x] T7.3 Implement `walkforward.py`: Wrapper for walkforward mode. CLI arguments: `--strategy <id>`, actually call walkforward test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `walkforward.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py`

- [x] T7.4 Implement `lookAhead.py`: Wrapper for lookAhead mode. CLI arguments: `--strategy <id>`, actually call lookAhead test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `lookAhead.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/5 OtherTest/S4003_lookAhead.py`

- [x] T7.5 Implement `monteCarlo.py`: Wrapper for monteCarlo mode. CLI arguments: `--strategy <id>`, actually call monteCarlo test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `monteCarlo.py` with parameter --strategy S4003, actually run file `VS_4003_20260827_ES/5 OtherTest/S4003_monteCarlo.py`

- [x] T7.6 Implement `liveTrade.py`: Wrapper for liveTrade mode. CLI arguments: `--strategy <id>`, actually call liveTrade test file with folder prefix <id> under under folder VS_4000_strategy. E.g. When run `liveTrade.py` with parameter --strategy S4003, actually run file `VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py`


## STatus Overview
| Subsystem  | File | Progress |
| --- | --- | --- |
| VS_4003 | [TASKS/VS_4003.md](TASKS/VS_4003.md) | 6/10 |