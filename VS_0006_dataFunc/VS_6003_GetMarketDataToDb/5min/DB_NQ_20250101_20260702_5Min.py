import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add VS_6000_DataSource dir to sys.path so ibMarketData can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "VS_6000_DataSource"))

# Add VS_0007_dbAndFile/mariaDB dir to sys.path so IBDb can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "VS_0007_dbAndFile" / "mariaDB"))

import ibMarketData
import mariaDb

# --------------------------------------------
# Load environment variables from VS_0002_config/.env
# --------------------------------------------
current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parents[2] / "VS_0002_config" / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Loaded .env from: {dotenv_path}")
else:
    print(f"ERROR: .env file not found at {dotenv_path}")

csv_excel_path = os.getenv("csvExcelPath")
if csv_excel_path:
    print(f"Successfully retrieved csvExcelPath: {csv_excel_path}")
else:
    print("ERROR: csvExcelPath not found in the .env file.")

# outcsvFileName = Path(csv_excel_path) / "NVDA_20250101_20260430_5Min.csv"
# print(f"outcsvFileName: {outcsvFileName}")


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # You can pass None to let the function create the connection
    my_ib_connection = None


    # Fetch 5-minute NVDA data
    df = ibMarketData.getAllTypesTicketDataWithTimeFromIB(
        conn1=my_ib_connection,
        # symbolName="NQ",  # nasdaq e-mini future continuous contract
        symbolName="ES",  # S&P e-mini future continuous contract
        # symbolName="RTY", # Russell e-mini future continuous contract
        # symbolName="YM",    # DOW e-mini future continuous contract        
        # startDate="20251001",
        startDate="20260701",
        startTime="0000",
        endDate="20260719",
        endTime="2355",
        period1=5,
        tickerType = "FU",  # tickerType: "ST" for stock, "FU" for future
        isConFuture=True,  # True for continuous future, False for specific future contract
        futureExpireDate=None,  # e.g., "202609" for specific future contract
        futureExchange="CME"  # e.g., NQ, ES, RTY come from CME; YM comes from CBOT
        # futureExchange="CBOT"  # e.g., NQ, ES, RTY come from CME; YM comes from CBOT
    )


    print(df.head())
    # df.to_csv(outcsvFileName, index=False)

    # --------------------------------------------------------------------------------------------
    # format import amibroker
    # --------------------------------------------------------------------------------------------
    # Select the columns in the exact requested order
    df2_amibroker = df[['symbolName', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Rename columns to match the target standard
    df2_amibroker.columns = ['Ticker', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    # --------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Save to MariaDB table ticker5Min (IBTradingDb) for AmiBroker ODBC read
    # isOverride=True  -> update existing (ticker, datetime1) rows; insert new rows
    # isOverride=False -> skip existing (ticker, datetime1) rows; insert new rows
    # --------------------------------------------------------------------------------------------
    mariaDb.saveDfToTicker5Min(df, tableName="ticker5Min", isOverride=True)
