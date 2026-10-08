import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add VS_6000_DataSource dir to sys.path so ibMarketData can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "VS_6000_DataSource"))

# Add VS_0007_dbAndFile/mariaDB dir to sys.path so mariaDb can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "VS_0007_dbAndFile" / "mariaDB"))

import ibMarketData
import mariaDb

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"


def loadIngestConfig(configPath=CONFIG_PATH):
    """Read the `Ingest` section from global strategy config config.yaml (workspace root)."""
    with open(configPath, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ingest = cfg.get("Ingest")
    if not ingest:
        raise KeyError(f"Section 'Ingest' not found in {configPath}")
    return ingest


# ==========================================
# Ingest: fetch market data from IB and save to MariaDB.
# Logic and function same as DB_NQ_20250101_20260702_1Min.py, but all
# parameters come from config.yaml (workspace root) section `Ingest` (T6.1).
# ==========================================
if __name__ == "__main__":
    # Load credentials (DB_HOST, DB_USER, DB_PASSWORD, ...) from VS_0002_config/.env
    dotenv_path = WORKSPACE_ROOT / "VS_0002_config" / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"ERROR: .env file not found at {dotenv_path}")

    cfg = loadIngestConfig()
    print(f"Ingest parameters from {CONFIG_PATH}:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Fetch data from IB; conn1=None -> the function creates the connection
    df = ibMarketData.getAllTypesTicketDataWithTimeFromIB(
        conn1=None,
        symbolName=cfg["symbolName"],
        startDate=str(cfg["startDate"]),
        startTime=str(cfg["startTime"]),
        endDate=str(cfg["endDate"]),
        endTime=str(cfg["endTime"]),
        period1=cfg["period1"],
        tickerType=cfg["tickerType"],       # "ST" for stock, "FU" for future
        isConFuture=cfg["isConFuture"],     # True for continuous future
        futureExpireDate=cfg["futureExpireDate"],  # e.g. "202609" for specific contract
        futureExchange=cfg["futureExchange"],      # NQ/ES/RTY = CME; YM = CBOT
    )

    print(df.head())
    if df.empty:
        print("ERROR: No data fetched from IB - nothing saved to DB.")
        sys.exit(1)

    # --------------------------------------------------------------------------------------------
    # Save to MariaDB (IBTradingDb)
    # isOverride=True  -> update existing (ticker, datetime1) rows; insert new rows
    # isOverride=False -> skip existing (ticker, datetime1) rows; insert new rows
    # --------------------------------------------------------------------------------------------
    mariaDb.saveDfToTicker5Min(df, tableName=cfg["tableName"], isOverride=cfg["isOverride"])
