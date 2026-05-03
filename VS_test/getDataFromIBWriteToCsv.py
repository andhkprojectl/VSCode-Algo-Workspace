import pandas as pd
import asyncio
from ib_insync import *
from datetime import datetime, timedelta
import os

# --- CONFIGURATION ---
SYMBOL = 'NVDA'
CSV_FILE = 'c:/tmp/nvda_1min_data_fm_ib.csv'
HOST = '127.0.0.1'
PORT = 4002  # Use 7496 for live
# PORT = 7497  # Use 7496 for live
CLIENT_ID = 1


async def log_nvda_data():
    ib = IB()
    try:
        # Connect to IBKR
        await ib.connectAsync(HOST, PORT, clientId=CLIENT_ID)
        print(f"Connected to IBKR. Logging {SYMBOL} every minute...")

        ib.reqMarketDataType(1)
        print("Market Data Type set to: LIVE")

        # Define the Contract
        contract = Stock(SYMBOL, 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)

        while True:
            # 1. Timing Logic: Wait for the 1st second of the next minute
            now = datetime.now()
            next_run = (now + timedelta(minutes=1)).replace(second=1, microsecond=0)
            wait_seconds = (next_run - now).total_seconds()

            print(f"Sleeping {wait_seconds:.2f}s until {next_run.strftime('%H:%M:%S')}...")
            await asyncio.sleep(wait_seconds)

            # 2. Get Historical Data (Requesting the last 2 bars to ensure the most recent is closed)
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='120 S',  # Request 120 seconds to get the last finished 1min bar
                barSizeSetting='1 min',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if bars:
                last_bar = bars[-1]  # The most recent completed bar

                # 3. Data Formatting
                data_row = {
                    'current_date': datetime.now().strftime('%Y-%m-%d'),
                    'current_time': datetime.now().strftime('%H:%M:%S'),
                    'last_bar_time': last_bar.date.strftime('%Y-%m-%d %H:%M:%S'),
                    'O': last_bar.open,
                    'H': last_bar.high,
                    'L': last_bar.low,
                    'C': last_bar.close,
                    'Volume': last_bar.volume
                }

                df = pd.DataFrame([data_row])

                # 4. Write to CSV (Append mode)
                file_exists = os.path.isfile(CSV_FILE)
                df.to_csv(CSV_FILE, mode='a', index=False, header=not file_exists)

                print(f"Logged Bar: {data_row['last_bar_time']} | Close: {data_row['C']}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        ib.disconnect()


# Run the async loop
if __name__ == "__main__":
    asyncio.run(log_nvda_data())