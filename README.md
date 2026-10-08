# VSCode Algo Workspace

Automated trading system: multiple strategies, each trading its own symbols,
timeframe and broker. Market data comes from Interactive Brokers (IB) via
TWS/Gateway; data is stored in MariaDB.

## Setup

- Python 3.13+, virtual env in `.venv` (`pip install -r` the deps used by the
  programs: pandas, numpy, pyyaml, ib_insync, mysql-connector-python,
  scikit-learn, joblib, python-dotenv, pytest).
- Credentials live in `VS_0002_config/.env` (`DB_HOST`, `DB_PORT`, `DB_USER`,
  `DB_PASSWORD`, `DB_NAME` — see `VS_0002_config/.env.example`). Never commit it.
- MariaDB `IBTradingDb` must exist (schema/user: `VS_0007_dbAndFile/mariaDB/sql`).
- For IB access, TWS or IB Gateway must be running and logged in:
  paper ports 7497 (TWS) / 4002 (Gateway); live ports 7496 / 4001.

All commands below run from the workspace root, e.g.
`.venv/Scripts/python.exe <command>` (Windows) or `python <command>`.

## The five modes

### 1. Data ingestion

Fetches 1-min bars from IB and saves them to MariaDB. Parameters from the
`Ingest` section of the root `config.yaml` (symbol, date/time range, bar size,
contract type, output table).

```
python ingest.py
```

### 2. Backtesting

```
python backtest.py --strategy S4003
```

Runs `VS_4000_strategy/VS_4003_20260827_ES/3 BackTest/S4003_backtest.py`.
Parameters from the strategy's `config.yaml` section `Backtest Test`; data from
MariaDB `ticker1Min`. Extra args are forwarded (e.g. `--config <path>`).
Output: the strategy's `backTestResult` folder.

### 3. Walk-forward test

```
python walkforward.py --strategy S4003
```

Runs `VS_4000_strategy/VS_4003_20260827_ES/4 Walk Forward Test/S4003_walkforward.py`.
Parameters from section `WalkForward Test`; data from MariaDB `ticker1Min`.
Extra args forwarded (e.g. `--ticker ES`). Output: the strategy's
`walkForwardTestResult` folder.

### 4. Other tests

Look-ahead bias test:

```
python lookAhead.py --strategy S4003
```

Monte Carlo simulation:

```
python monteCarlo.py --strategy S4003
```

Both run the programs in `5 OtherTest/`, read parameters from sections
`LookAhead Test` / `MonteCarlo Test`, use MariaDB `ticker1Min`, and write html
reports to the strategy's `lookAheadTestResult` / `monteCarloResult` folders.

### 5. Live trading

```
python "VS_4000_strategy/VS_4003_20260827_ES/6 liveTrade/S4003_live.py"
```

(A root `liveTrade.py --strategy <id>` wrapper is planned as TASKS.md T7.6.)

Streams 1-min bars from IB (`reqHistoricalData` with `keepUpToDate=True`),
evaluates the strategy class on each new bar and places orders via `doTrade0`
in `VS_0008_liveTrade/ibAutotrade.py`. Parameters from section `Live Trade`
(incl. `tradePort`). Logs to the strategy's `liveTrade` folder
(`S4003_live_signals.log`, `S4003_doTrade0_debug.log`).

SAFETY: `Live Trade.liveTrade: false` = OBSERVE mode — signals are computed and
logged but NO orders are placed. Flip to `true` only after backtest,
walk-forward and other tests pass, and use paper ports before live ones.

## Strategy id convention

`--strategy <id>` (e.g. `S4003`, case-insensitive) selects the strategy folder
`VS_4000_strategy/VS_<num>_*/` and the program `<id>_<mode>.py` inside the
mode's subfolder (`3 BackTest`, `4 Walk Forward Test`, `5 OtherTest`).
Currently available: `S4003` (ES 1-min intraday).

## Tests

```
python -m pytest
```

See `AGENTS.md` for the full project structure and `VS_0001_docs/TASKS.md`
for the build order.
