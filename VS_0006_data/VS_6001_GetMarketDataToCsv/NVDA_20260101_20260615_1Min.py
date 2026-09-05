import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add VS_6000_DataSource (VS_0006_data) to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VS_6000_DataSource'))

import ibMarketData
import pandas as pd

# --------------------------------------------
current_dir = Path(__file__).resolve().parent
dotenv_path = current_dir.parents[1] / "VS_0002_config" / ".env"
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

outcsvFileName = Path(csv_excel_path) / "NVDA_20260101_20260615_1Min.csv"
print(f"outcsvFileName: {outcsvFileName}")
# --------------------------------------------

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # You can pass None to let the function create the connection
    my_ib_connection = None 
    
    # Fetch 1-minute NVDA data 
    df = ibMarketData.getTicketDataWithTimeFromIB(
        conn1=my_ib_connection,
        symbolName="NVDA",
        startDate="20260101",
        startTime="0355",
        endDate="20260615",
        endTime="2005",
        period1=1
    )
    
    print(df.head())
    df.to_csv(outcsvFileName, index=False)

    # --------------------------------------------------------------------------------------------
    # format import amibroker
    # --------------------------------------------------------------------------------------------
    # Select the columns in the exact requested order
    df2_amibroker = df[['symbolName', 'date', 'time', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Rename columns to match the target standard
    df2_amibroker.columns = ['Ticker', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    # --------------------------------------------------------------------------------------------
