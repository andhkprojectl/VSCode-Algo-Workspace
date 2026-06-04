from dotenv import load_dotenv
import os
import sys

# Add VS_0002_data to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../VS_0002_data'))

import ibMarketData
# from folder2.b import some_function
import pandas as pd


# conn1 = None
# symbolName = "NVDA"
# startDate = "20260101"
# startTime = "0930"
# endDate = "20260430"
# endTime = "1600"
# period1 = 1
# # a1 = ibMarketData.getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1)
# a1 = ibMarketData.getTicketDataWithTimeFromIB(conn1, symbolName, startDate, startTime, endDate, endTime, period1)
# a1.to_csv("c:/tmp/NVDA_1min_data01.csv", index=False)

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
        startDate="20240501",
        startTime="0930",
        endDate="20241231",
        endTime="1600",
        period1=5
    )
    
    print(df.head())

    df.to_csv("c:/tmp/NVDA_1min_data01.csv", index=False)