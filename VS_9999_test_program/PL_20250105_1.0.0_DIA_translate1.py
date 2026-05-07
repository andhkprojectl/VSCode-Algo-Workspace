import yfinance as yf
import pandas as pd
import numpy as np
import talib
from ib_insync import *
import logging
from datetime import datetime, timedelta
import time

# Set up logging
logging.basicConfig(
    filename='TWSTrade111.log',
    level=logging.INFO,
    format='%(asctime)s;%(message)s'
)
trade_logger = logging.getLogger('TradeLogger')
trade_handler = logging.FileHandler('TWSTradeTrade.log')
trade_handler.setFormatter(logging.Formatter('%(asctime)s;inCubation_DIA.py;DIA;%(message)s'))
trade_logger.addHandler(trade_handler)
trade_logger.setLevel(logging.INFO)

# Parameters
YM_RETURN_PERIOD1 = 160
YM_RETURN_TH2 = 140  # Optimized value from AFL
ATR_TH = 3.0         # Stop loss multiplier
PROFIT_ATR_TH = 3.0  # Profit take multiplier
STOP_PERIOD1 = 4     # N-bar exit in days
NUM_CONTRACTS = 80   # Number of DIA shares

# Initialize Interactive Brokers connection
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust port/clientId as needed

def fetch_data():
    """Fetch daily data for YM=F (Dow futures) and DIA."""
    ym = yf.download('YM=F', period='2y', interval='1d')
    dia = yf.download('DIA', period='2y', interval='1d')
    return ym, dia

def calculate_indicators(ym, dia):
    """Calculate returns, percentiles, and ATR."""
    # YM returns
    ym['return'] = ym['Close'].diff()
    
    # Rolling percentiles
    ym['ymReturn_91'] = ym['return'].rolling(window=YM_RETURN_PERIOD1).apply(
        lambda x: np.percentile(x.dropna(), 91)
    )
    
    # ATR for DIA
    dia['ATR'] = talib.ATR(dia['High'], dia['Low'], dia['Close'], timeperiod=15)
    
    return ym, dia

def generate_signals(ym, dia):
    """Generate buy and sell signals based on AFL logic."""
    # Buy signal: ymReturn_91 >= ymReturnTh2 on previous day
    ym['buy00'] = ym['return'] >= ym['ymReturn_91']
    ym['Buy'] = ym['buy00'].shift(1).fillna(False)
    
    # Short signal: Disabled as per AFL (short00 = False)
    ym['Short'] = False
    
    # N-bar exit
    dia['Sell'] = False
    dia['Cover'] = False
    for i in range(STOP_PERIOD1, len(dia)):
        if ym['Buy'].iloc[i - STOP_PERIOD1]:
            dia.iloc[i, dia.columns.get_loc('Sell')] = True
        # No short logic as Short is False
    
    return ym, dia

def place_bracket_order(symbol, action, quantity, entry_price, stop_loss_price, profit_take_price):
    """Place a bracket order with IB."""
    contract = Stock(symbol, 'SMART', 'USD')
    
    # Parent order
    parent = Order()
    parent.action = action
    parent.orderType = 'LMT'
    parent.lmtPrice = entry_price
    parent.totalQuantity = quantity
    parent.transmit = False
    
    # Stop loss
    stop_loss = Order()
    stop_loss.action = 'SELL' if action == 'BUY' else 'BUY'
    stop_loss.orderType = 'STP'
    stop_loss.auxPrice = stop_loss_price
    stop_loss.totalQuantity = quantity
    stop_loss.parentId = parent.orderId
    stop_loss.transmit = False
    
    # Profit take
    profit_take = Order()
    profit_take.action = 'SELL' if action == 'BUY' else 'BUY'
    profit_take.orderType = 'LMT'
    profit_take.lmtPrice = profit_take_price
    profit_take.totalQuantity = quantity
    profit_take.parentId = parent.orderId
    profit_take.transmit = True
    
    bracket = [parent, stop_loss, profit_take]
    for order in bracket:
        ib.placeOrder(contract, order)
    
    return bracket

def monitor_nbar_exit(positions, stop_period):
    """Monitor positions and close after stop_period days."""
    current_date = datetime.now().date()
    for pos in positions:
        entry_date = pos['entry_date']
        if (current_date - entry_date).days >= stop_period:
            contract = Stock(pos['symbol'], 'SMART', 'USD')
            order = MarketOrder('SELL' if pos['action'] == 'BUY' else 'BUY', pos['quantity'])
            ib.placeOrder(contract, order)
            trade_logger.info(f"Closed position for {pos['symbol']} after {stop_period} days")

def main():
    """Main trading logic."""
    # Fetch and process data
    ym, dia = fetch_data()
    ym, dia = calculate_indicators(ym, dia)
    ym, dia = generate_signals(ym, dia)
    
    # Get latest signals
    latest_buy = ym['Buy'].iloc[-1]
    latest_sell = dia['Sell'].iloc[-1]
    latest_open = dia['Open'].iloc[-1]
    latest_atr = dia['ATR'].iloc[-2]  # Previous day's ATR
    
    # Trading logic
    positions = []  # Track open positions
    if latest_buy:
        entry_price = latest_open
        stop_loss_price = entry_price - (ATR_TH * latest_atr)
        profit_take_price = entry_price + (PROFIT_ATR_TH * latest_atr)
        
        bracket = place_bracket_order(
            'DIA', 'BUY', NUM_CONTRACTS, entry_price, stop_loss_price, profit_take_price
        )
        positions.append({
            'symbol': 'DIA',
            'action': 'BUY',
            'quantity': NUM_CONTRACTS,
            'entry_date': datetime.now().date()
        })
        
        log_msg = (f"Buy;DIA;entry_price={entry_price:.2f};stop_loss={stop_loss_price:.2f};"
                  f"profit_take={profit_take_price:.2f};quantity={NUM_CONTRACTS}")
        trade_logger.info(log_msg)
        logging.info(f"Placed buy order: {log_msg}")
    
    # Monitor n-bar exit (simplified for daily run)
    if positions:
        monitor_nbar_exit(positions, STOP_PERIOD1)
    
    ib.disconnect()

if __name__ == "__main__":
    main()