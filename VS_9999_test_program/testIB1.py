from ib_insync import *
import math


def connect_ib():
    """
    Connect to Interactive Brokers TWS or IB Gateway.
    """
    ib = IB()
    ib.connect('127.0.0.1', 4002, clientId=1)  # Ensure TWS or IB Gateway is running
    return ib


def create_bracket_order(symbol, buy_price, quantity, tag, profit_percent=2, loss_percent=2):
    """
    Create a bracket order:
    - Buy order at `buy_price`
    - Take-profit order at `buy_price + profit_percent`
    - Stop-loss order at `buy_price - loss_percent`

    Parameters:
        symbol (str): Stock symbol, e.g., 'NVDA'
        buy_price (float): Price at which to place the buy order
        quantity (int): Number of shares to buy
        tag (str): Text to associate with the order
        profit_percent (float): Percent above buy price for take-profit
        loss_percent (float): Percent below buy price for stop-loss
    Returns:
        list: List of orders (parent + child orders)
    """
    # Calculate profit and loss levels
    take_profit_price = round(buy_price * (1 + profit_percent / 100), 2)
    stop_loss_price = round(buy_price * (1 - loss_percent / 100), 2)

    # Define NVDA stock contract
    contract = Stock(symbol=symbol, exchange='SMART', currency='USD')

    # Parent buy order
    parent_order = LimitOrder(
        action='BUY',
        totalQuantity=quantity,
        lmtPrice=buy_price,
        orderId=None,  # Order ID will be assigned by IB
        tif='GTC'  # Good till canceled
    )
    parent_order.transmit = False  # This prevents transmitting until child orders are added
    parent_order.orderRef = tag  # Add custom tag for identification

    # Take-profit order
    take_profit_order = LimitOrder(
        action='SELL',
        totalQuantity=quantity,
        lmtPrice=take_profit_price,
        orderId=None,
        tif='GTC'
    )
    take_profit_order.parentId = parent_order.orderId  # Link to parent order
    take_profit_order.transmit = False  # Transmit with stop-loss

    # Stop-loss order
    stop_loss_order = StopOrder(
        action='SELL',
        totalQuantity=quantity,
        stopPrice=stop_loss_price,
        orderId=None,
        tif='GTC'
    )
    stop_loss_order.parentId = parent_order.orderId  # Link to parent order
    stop_loss_order.transmit = True  # Stop-loss transmits the entire bracket

    return contract, [parent_order, take_profit_order, stop_loss_order]


def main():
    # Parameters
    symbol = "NVDA"  # Stock symbol
    buy_price = 138.85  # Buy price
    quantity = 100  # Number of shares to buy
    tag = "BUY1"  # Text tag for the order
    profit_percent = 2  # Profit target (percent above buy price)
    loss_percent = 2  # Stop-loss (percent below buy price)

    # Connect to IB
    ib = connect_ib()
    print("Connected to IB TWS.")

    # Create bracket order
    contract, orders = create_bracket_order(
        symbol=symbol,
        buy_price=buy_price,
        quantity=quantity,
        tag=tag,
        profit_percent=profit_percent,
        loss_percent=loss_percent
    )

    # Place order
    print(f"Placing bracket order for {symbol}...")
    ib.qualifyContracts(contract)  # Ensure the contract is valid
    trades = ib.placeOrder(contract, orders[0])  # Place the parent order

    # Place child orders once parent order ID is assigned
    orders[1].parentId = trades.order.orderId  # Assign parent ID to take-profit order
    orders[2].parentId = trades.order.orderId  # Assign parent ID to stop-loss order
    ib.placeOrder(contract, orders[1])  # Place take-profit order
    ib.placeOrder(contract, orders[2])  # Place stop-loss order

    print(f"Bracket order placed. Parent Order ID: {trades.order.orderId}")

    # Disconnect from IB after placing the order
    ib.disconnect()
    print("Disconnected from IB TWS.")


if __name__ == "__main__":
    main()