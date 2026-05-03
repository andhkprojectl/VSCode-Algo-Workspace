import socket
import pandas as pd
from datetime import datetime
from ib_insync import *

# Function to connect to IQFeed and get historical data
def fetch_iqfeed_data(symbol, start_date, end_date, interval_seconds):
    host = '127.0.0.1'  # Default for IQFeed
    port = 9100         # Default IQFeed historical data port
    request_id = "HIST" # Identifier for the request

    # Construct request string for historical data
    request = f"HIT,{symbol},{interval_seconds},{start_date},{end_date},,,,1\n"

    # Connect to IQFeed
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(request.encode('utf-8'))

        # Receive data
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk

    # Decode and parse the data
    lines = data.decode('utf-8').splitlines()
    parsed_data = []
    for line in lines:
        if line.startswith(request_id) or line == "":  # Skip header or empty lines
            continue
        fields = line.split(',')
        if len(fields) >= 8:
            parsed_data.append({
                'timestamp': fields[0],
                'open': float(fields[1]),
                'high': float(fields[2]),
                'low': float(fields[3]),
                'close': float(fields[4]),
                'volume': int(fields[5]),
                'tick_count': int(fields[6]),
                'bid_volume': int(fields[7]) if fields[7] else 0,
                'ask_volume': int(fields[8]) if fields[8] else 0,
            })

    # Convert to Pandas DataFrame
    df = pd.DataFrame(parsed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Function to connect to IB TWS and place a buy order
def place_ib_order(symbol, close_price):
    # Connect to TWS
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Default TWS paper trading port

    # Define the NVDA stock contract
    contract = Stock(symbol, 'SMART', 'USD')

    # Market order for 100 shares
    order = MarketOrder('BUY', 100)

    # Place the order
    trade = ib.placeOrder(contract, order)
    print(f"Order placed: {trade}")

    # Disconnect from IB
    ib.disconnect()

# Main function
def main():
    # Parameters
    symbol = "NVDA"
    start_date = "20110101"  # Format: YYYYMMDD
    end_date = "20241231"
    interval_seconds = 900  # 15 minutes = 900 seconds

    # Fetch data from IQFeed
    print("Fetching historical data from IQFeed...")
    df = fetch_iqfeed_data(symbol, start_date, end_date, interval_seconds)
    print("Data fetched successfully.")

    # Display the last few rows of data
    print(df.tail())

    # Get the most recent close price
    last_close_price = df.iloc[-1]['close']
    print(f"Most recent close price for {symbol}: {last_close_price}")

    # Place a buy order at the close price using IB TWS
    print("Placing buy order at Interactive Brokers TWS...")
    place_ib_order(symbol, last_close_price)
    print("Order placed successfully.")

if __name__ == "__main__":
    main()