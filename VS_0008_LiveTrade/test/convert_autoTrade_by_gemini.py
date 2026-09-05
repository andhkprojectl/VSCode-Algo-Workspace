import datetime
import math
from ib_insync import *

class AutoTradeIB:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initializes connection to Interactive Brokers TWS or IB Gateway.
        Port 7497 is default for paper trading, 7496 for live trading.
        """
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
        self.log_file = "TWSTrade111.log"
        self.max_open_positions = 15 # Replicates StaticVarGet("MAX_OPEN_POSITION")
        self.quota = 20

    def write_line(self, message):
        """
        Equivalent to writeline() in AFL. Logs to a permanent logfile.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp};autoTrade.py;{message}\n"
        
        with open(self.log_file, "a") as fh:
            fh.write(log_entry)
        print(log_entry.strip())

    def dump_all_open_positions(self):
        """
        Equivalent to dumAllOpenPositions(). Fetches and logs open positions from IB.
        """
        positions = self.ib.positions()
        is_open_position_found = len(positions) > 0
        
        for pos in positions:
            self.write_line(f"(testing env only) Dumped existing open position: Symbol[{pos.contract.symbol}]")
            
        if not is_open_position_found:
            self.write_line("(testing env only) Dumped all existing filled open positions. No open position found.")
            
        return positions

    def get_nearest_round_to_price(self, price, tick_size):
        """
        Helper equivalent to getNearestRoundToPrice in AFL. 
        Rounds a given price dynamically based on the instrument's tick size to prevent IB API rejection.
        """
        if tick_size == 0 or price == 0:
            return price
        return round(round(price / tick_size) * tick_size, 4)

    def do_trade_00(self, contract, buy_signal, sell_signal, short_signal, cover_signal, 
                    limit_price, stop_loss_price_0, profit_take_price_0, num_contracts, 
                    ib_tick_size, is_intra_day, is_mkt_order, 
                    max_num_tick_size_slippage, symbol_type):
        """
        Core logic engine: Equivalent to doTrade00(). 
        Evaluates signals, normalizes prices, validates stop losses, and executes trades via bracket orders.
        """
        self.write_line(f"begin2[{contract.symbol}]")

        stop_loss_price = stop_loss_price_0
        profit_take_price = profit_take_price_0
        is_perform_stp_loss = True
        
        # 1. Audit & Validate Stop Loss Price Logic
        if not is_mkt_order:
            if (buy_signal and stop_loss_price >= limit_price) or \
               (short_signal and stop_loss_price <= limit_price):
                is_perform_stp_loss = False

        is_perform_buy = True
        is_perform_sell = True
        is_perform_short = True
        is_perform_cover = True
        
        # 2. Safety filter: Do not open position if stop loss is invalid
        if not is_perform_stp_loss:
            is_perform_buy = False
            is_perform_short = False

        # 3. Conflicting signals filter
        if buy_signal and sell_signal:
            is_perform_buy = False
            is_perform_sell = False
            
        if short_signal and cover_signal:
            is_perform_short = False
            is_perform_cover = False

        # 4. Process Tick Size and Slippage Parameters
        # AFL converts tickSize internally. E.g., ibControllerTickSize/10000 
        actual_tick_size = ib_tick_size / 10000.0 if symbol_type == 1 else ib_tick_size
        max_allow_slippage = max_num_tick_size_slippage * actual_tick_size

        limit_price_buy = limit_price + max_allow_slippage
        limit_price_sell = limit_price - max_allow_slippage

        # Tick-adjusting stop loss and take profit
        stop_loss_price = self.get_nearest_round_to_price(stop_loss_price_0, actual_tick_size)
        profit_take_price = self.get_nearest_round_to_price(profit_take_price_0, actual_tick_size)
        limit_price_buy = self.get_nearest_round_to_price(limit_price_buy, actual_tick_size)
        limit_price_sell = self.get_nearest_round_to_price(limit_price_sell, actual_tick_size)

        # 5. Order Execution
        order_action = ""
        trade_limit_price = 0.0

        if buy_signal and is_perform_buy:
            order_action = "BUY"
            trade_limit_price = limit_price_buy
        elif short_signal and is_perform_short:
            order_action = "SELL"
            trade_limit_price = limit_price_sell
        elif sell_signal and is_perform_sell:
            order_action = "SELL" # Exiting Long
            trade_limit_price = limit_price_sell
        elif cover_signal and is_perform_cover:
            order_action = "BUY" # Exiting Short
            trade_limit_price = limit_price_buy

        if order_action:
            self.write_line(f"Executing {order_action} on {contract.symbol} | MKT: {is_mkt_order} | LMT: {trade_limit_price}")
            
            # Form the Parent Order
            if is_mkt_order:
                parent = MarketOrder(order_action, num_contracts)
            else:
                parent = LimitOrder(order_action, num_contracts, trade_limit_price)
            
            parent.orderId = self.ib.client.getReqId()
            parent.transmit = True

            # Standard Bracket logic if it is an Entry Order (Buy / Short)
            bracket_orders = [parent]
            if (buy_signal or short_signal) and is_perform_stp_loss:
                parent.transmit = False  # Hold transmission until children are attached

                # Take Profit (Child)
                if profit_take_price > 0:
                    take_profit = LimitOrder(
                        "SELL" if order_action == "BUY" else "BUY",
                        num_contracts,
                        profit_take_price
                    )
                    take_profit.orderId = self.ib.client.getReqId()
                    take_profit.parentId = parent.orderId
                    take_profit.transmit = False
                    bracket_orders.append(take_profit)

                # Stop Loss (Child)
                if stop_loss_price > 0:
                    stop_loss = StopOrder(
                        "SELL" if order_action == "BUY" else "BUY",
                        num_contracts,
                        stop_loss_price
                    )
                    stop_loss.orderId = self.ib.client.getReqId()
                    stop_loss.parentId = parent.orderId
                    stop_loss.transmit = True # Final child transmits the batch
                    bracket_orders.append(stop_loss)

            # Place Orders in IB
            for order in bracket_orders:
                self.ib.placeOrder(contract, order)

            self.write_line(f"Successfully placed bracket orders for {contract.symbol}")
            return True

        return False

    def do_trade_stock(self, symbol, buy, sell, short, cover, limit_price, stop_loss_0, profit_take_0, 
                       num_contracts, tick_size, is_intra_day, is_mkt_order, max_slippage):
        """ Equivalent to doTradeStock0() -> symbol_type=2 (Stock) """
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return self.do_trade_00(contract, buy, sell, short, cover, limit_price, stop_loss_0, profit_take_0, 
                                num_contracts, tick_size, is_intra_day, is_mkt_order, max_slippage, symbol_type=2)

    def do_trade_future(self, symbol, exchange, buy, sell, short, cover, limit_price, stop_loss_0, profit_take_0, 
                        num_contracts, tick_size, is_intra_day, is_mkt_order, max_slippage):
        """ Equivalent to doTradeFuture0() -> symbol_type=1 (Future) """
        # Note: Futures require specifying the expiry and exchange.
        contract = Future(symbol, exchange=exchange)
        self.ib.qualifyContracts(contract)
        return self.do_trade_00(contract, buy, sell, short, cover, limit_price, stop_loss_0, profit_take_0, 
                                num_contracts, tick_size, is_intra_day, is_mkt_order, max_slippage, symbol_type=1)

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Connect to IB TWS/Gateway
    algo = AutoTradeIB(host='127.0.0.1', port=7497) # Use 7497 for paper trading

    # 2. Dump all open positions just like in the AFL validation phase
    algo.dump_all_open_positions()

    # 3. Simulate a Stock Trade (Buy Signal generated on AAPL)
    # Params: symbol, buy, sell, short, cover, limitPrice, stpLoss, profit, numContracts, tickSize, isIntraday, isMkt, maxSlippage
    algo.do_trade_stock(
        symbol="AAPL",
        buy=True,
        sell=False,
        short=False,
        cover=False,
        limit_price=180.50,
        stop_loss_0=178.00,
        profit_take_0=185.00,
        num_contracts=10,
        tick_size=0.01,
        is_intra_day=True,
        is_mkt_order=False,
        max_slippage=3
    )

    # Allow IB network traffic to process before script terminates
    algo.ib.sleep(2)