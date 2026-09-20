"""
VS_8101_autotrade.py
===================
Python port of the AmiBroker AFL trade-execution function `doTrade00`
(autoTrade.afl lines 1480-2176) as function `doTrade0`, per requirement
autotradeURV1.txt / L8001_autotrade_UR_V1.txt (see VS_8101_autotradeV1.md).

Safety: test against paper TWS/Gateway only (port 7497 / 4002); never
transmit real orders from tests. autoTrade.afl remains the single source
of truth for the trading logic.
"""

import datetime
import time

from ib_insync import IB, LimitOrder, MarketOrder, StopLimitOrder, StopOrder

LOG_FILE = "TWSTrade111.log"
PROGRAM_NAME = "VS_8101_autotrade.py"


def _ib_is_connected(ib):
    # ib_insync >= 1.0 renamed isConnected() -> is_connected(); support both
    if hasattr(ib, "is_connected"):
        return bool(ib.is_connected())
    return bool(ib.isConnected())


class AutoTrade:
    """Holds the connected ib_insync.IB plus the AFL helper functions."""

    def __init__(self, host="127.0.0.1", port=4002, client_id=1, auto_connect=True,
                 max_open_position=15):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.max_open_position = max_open_position
        self.ibvar_qta = 20
        self.semaphore1 = 0
        self.ib = IB()
        if auto_connect:
            self.ib.connect(host, port, clientId=client_id)

    def ensure_connected(self, host=None, port=None, client_id=None,
                         retries=3, sleep_sec=2):
        """Connection gate with retry logic (AFL: IBcStatus check).
        Connects when not connected and returns True once connected;
        returns False after `retries` failed attempts."""
        host = self.host if host is None else host
        port = self.port if port is None else port
        client_id = self.client_id if client_id is None else client_id
        for attempt in range(1, retries + 1):
            try:
                if _ib_is_connected(self.ib):
                    self._set_delayed_market_data()
                    return True
                self.write_line(f"connect attempt {attempt}/{retries} to "
                                f"{host}:{port} clientId={client_id}")
                self.ib.connect(host, port, clientId=client_id)
                if _ib_is_connected(self.ib):
                    self.write_line(f"connect OK to {host}:{port} "
                                    f"clientId={client_id}")
                    self._set_delayed_market_data()
                    return True
                self.write_line(f"connect attempt {attempt}/{retries}: not connected")
            except Exception as e:
                self.write_line(f"connect error attempt {attempt}/{retries}: [{e}]")
            time.sleep(sleep_sec)
        return bool(_ib_is_connected(self.ib))

    def _set_delayed_market_data(self):
        """Accounts without a live subscription: request delayed market data
        so log-only quote snapshots do not raise errors 354/10168."""
        req = getattr(self.ib, "reqMarketDataType", None)
        if req is not None:
            try:
                req(3)  # 3 = delayed data when live is not subscribed
            except Exception as e:
                self.write_line(f"reqMarketDataType(3) error. [{e}]")

    # ---------------------------------------------------------------
    # AFL writeline(s1): append <Now>;autoTrade.afl;<msg> to TWSTrade111.log
    # ---------------------------------------------------------------
    def write_line(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp};{PROGRAM_NAME};{message}\n"
        with open(LOG_FILE, "a") as fh:
            fh.write(line)
        print(line.strip())

    def reconnect(self):
        try:
            self.ib.disconnect()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception as e:
            self.write_line(f"reconnect error. [{e}]")

    # ---------------------------------------------------------------
    # AFL getNearestRoundToPrice(price1, roundTo, type1), line 1441
    # type 1 = nearest, 2 = round up, 3 = round down
    # ---------------------------------------------------------------
    @staticmethod
    def get_nearest_round_to_price(price, tick, round_type):
        round_to_price = price
        frp = price % tick
        if round_type == 1:
            rounded_no = price - frp
            if abs(rounded_no - price) > abs(rounded_no + tick - price):
                round_to_price = rounded_no + tick
        elif round_type == 2:
            round_to_price = price - frp + tick
        elif round_type == 3:
            round_to_price = price - frp
        return round_to_price

    # ---------------------------------------------------------------
    # AFL cancelPendingOrder(ibc1, inSymbol1), line 125
    # match exact symbol or common prefix >= 2 chars; cancel only
    # PreSubmitted/Submitted; returns True if anything was cancelled
    # ---------------------------------------------------------------
    def cancel_pending_order(self, symbol):
        self.write_line(f"cancelPendingOrder begin. [{symbol}]")
        rs = False
        for trade in self.ib.openTrades():
            order_symbol = trade.contract.symbol
            status = trade.orderStatus.status
            self.write_line(f"cancelPendingOrder. compare symbol. input symbol=[{symbol}]. "
                            f"pending symbol=[{order_symbol}][{status}]")
            min_len = min(len(symbol), len(order_symbol))
            symbol_match = (order_symbol == symbol) or (
                symbol[:min_len] == order_symbol[:min_len] and min_len >= 2)
            if symbol_match and status in ("PreSubmitted", "Submitted"):
                self.ib.cancelOrder(trade.order)
                rs = True
                self.write_line(f"cancelPendingOrder. order cancelled. symbol=[{symbol}]. "
                                f"order id=[{trade.order.orderId}]")
        return rs

    def position_size(self, symbol):
        return float(sum(p.position for p in self.ib.positions()
                         if p.contract.symbol == symbol))

    # ---------------------------------------------------------------
    # AFL GetPositionSize block, lines 1718-1754: retry when sell/cover
    # is requested but the position reads 0, then reconnect and retry again
    # ---------------------------------------------------------------
    def get_position_size_with_retry(self, symbol, sell, cover):
        size = self.position_size(symbol)
        if (sell or cover) and size == 0:
            for _ in range(2):
                self.ib.sleep(1)
                size = self.position_size(symbol)
                self.write_line("try call GetPositionSize again. sell/cover but "
                                "GetPositionSize return 0 position")
            if size == 0:
                self.write_line("try 3 times. sell or cover but currPositionSize is "
                                "still 0, why???. reconnect and then try another 3 times")
                self.reconnect()
                for _ in range(2):
                    self.ib.sleep(1)
                    size = self.position_size(symbol)
                    self.write_line("try call GetPositionSize again 2. sell/cover but "
                                    "GetPositionSize return 0 position")
            if size == 0:
                self.write_line("try 3 times. sell or cover but currPositionSize is "
                                "still 0, why???. Dummed all positions")
                for p in self.ib.positions():
                    self.write_line(f"Symbol name[{p.contract.symbol}]")
        return size

    # ---------------------------------------------------------------
    # AFL execution/pending list dumps, lines 1770-1793
    # ---------------------------------------------------------------
    @staticmethod
    def _fill_quantity(fill):
        # ib_insync 1.0: fill.quantity; 0.9.x: fill.execution.shares
        q = getattr(fill, "quantity", None)
        if q is None:
            q = getattr(getattr(fill, "execution", None), "shares", 0)
        return float(q or 0)

    def log_order_status(self):
        for trade in self.ib.trades():
            if trade.fills:
                filled = sum(self._fill_quantity(f) for f in trade.fills)
                avg = getattr(trade.orderStatus, "avgFillPrice",
                              getattr(trade, "avgFillPrice", 0.0))
                self.write_line(f"Execution List. Order Id: {trade.order.orderId}. "
                                f"Symbol: {trade.contract.symbol}. Filled: {filled}. "
                                f"Avg. price: {avg}. "
                                f"Order status: [{trade.orderStatus.status}]")
        for trade in self.ib.openTrades():
            self.write_line(f"Pending List. Order Id: {trade.order.orderId}. "
                            f"Symbol: [{trade.contract.symbol}]. "
                            f"Order status: [{trade.orderStatus.status}]")

    # ---------------------------------------------------------------
    # AFL GetRTData("Last"/"Ask"/"Bid"): fetched for logging only, the
    # bid/ask-based limit logic is commented out in the AFL
    # ---------------------------------------------------------------
    def get_rt_data(self, contract, wait_sec=8.0):
        """Last/Ask/Bid snapshot (delayed data when live is not subscribed).
        Polls up to `wait_sec` seconds for the first valid delayed ticks."""
        import math

        quotes = {"last": float("nan"), "ask": float("nan"), "bid": float("nan")}

        def _valid(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v)) and v > 0

        req = getattr(self.ib, "reqMktData", None)
        cancel = getattr(self.ib, "cancelMktData", None)
        if req is None:
            return quotes
        try:
            ticker = req(contract, "", False, False)
            for _ in range(int(wait_sec / 0.5)):
                self.ib.sleep(0.5)
                quotes = {"last": ticker.last, "ask": ticker.ask, "bid": ticker.bid}
                if _valid(ticker.last) and _valid(ticker.bid) and _valid(ticker.ask):
                    break
            if cancel is not None:
                cancel(contract)
        except Exception as e:
            self.write_line(f"get_rt_data error. [{e}]")
        return quotes


def _round_child_prices(price, tick):
    """Equivalent of the tick-size normalization IB Controller performs when
    ibControllerTickSize is passed to ibc.PlaceOrder."""
    return AutoTrade.get_nearest_round_to_price(price, tick, 1)


def _place_bracket(auto_trade, contract, parent, stop_child, profit_child):
    """Parent transmit=False; children chained via parentId (AFL bracket
    semantics); the LAST child placed is transmitted (AFL: when profit-take
    is invalid, the stop child itself carries transmit=True)."""
    ib = auto_trade.ib
    parent.transmit = False
    parent_trade = ib.placeOrder(contract, parent)
    parent_id = parent_trade.order.orderId
    stop_child.transmit = profit_child is None
    stop_child.parentId = parent_id
    ib.placeOrder(contract, stop_child)
    if profit_child is not None:
        profit_child.transmit = True
        profit_child.parentId = parent_id
        ib.placeOrder(contract, profit_child)
    return parent_trade


def _stop_child(action, qty, stop_price, stp_lmt_order_price, tif):
    """AFL: plain buy/short entries use stpLmtOrderPrice (0 => pure STP)."""
    if stp_lmt_order_price:
        order = StopLimitOrder(action, qty, stp_lmt_order_price, stop_price, tif=tif)
    else:
        order = StopOrder(action, qty, stop_price, tif=tif)
    return order


def doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price,
             stop_loss_price, profit_take_price, num_contracts,
             ib_controller_tick_size, is_intra_day, is_mkt_order,
             max_num_tick_size4_buy_short_slippage,
             max_num_tick_size4_stp_lmt_order_slippage, symbol_type,
             *, port=4002, host="127.0.0.1", client_id=1,
             connect_retries=3, connect_sleep_sec=2):
    """Faithful port of AFL doTrade00 (autoTrade.afl lines 1480-2176).

    `auto_trade` may be None (an `AutoTrade` is then created and connected
    to the IB Gateway, default port 4002, with retry logic), a connected
    `AutoTrade` instance, or a raw ib_insync `IB` instance (wrapped).
    Connection errors are retried `connect_retries` times with
    `connect_sleep_sec` seconds between attempts.

    Returns 1 if any order action was placed, else 0.
    """
    if auto_trade is None:
        auto_trade = AutoTrade(host, port, client_id, auto_connect=False)
    elif isinstance(auto_trade, IB):
        holder = AutoTrade(host, port, client_id, auto_connect=False)
        holder.ib = auto_trade
        auto_trade = holder
    if not auto_trade.ensure_connected(host, port, client_id,
                                       connect_retries, connect_sleep_sec):
        auto_trade.write_line(f"Cannot connect IB after {connect_retries} attempts. "
                              f"IBcStatus = [0]")
        return 0
    ib = auto_trade.ib
    if not getattr(contract, "conId", 0):
        ib.qualifyContracts(contract)
    symbol1 = contract.symbol
    rs = 0
    quota = 20

    auto_trade.write_line(f"begin2[{symbol1}]")

    is_perform_stp_loss = True
    # audit. validate stopLossPrice
    if (not is_mkt_order
            and ((buy and stop_loss_price >= limit_price)
                 or (short and stop_loss_price <= limit_price))):
        is_perform_stp_loss = False

    IS_PERFORM_BUY = True
    IS_PERFORM_SELL = True
    IS_PERFORM_SHORT = True
    IS_PERFORM_COVER = True

    # not open new position if stop loss is invalid, i.e. every new position
    # must follow with stp loss order
    if not is_perform_stp_loss:
        IS_PERFORM_BUY = False
        IS_PERFORM_SHORT = False

    # if both buy and sell == true disable buy
    if buy and sell:
        IS_PERFORM_BUY = False
        IS_PERFORM_SELL = False
    # if both short and cover == true, disable short
    if short and cover:
        IS_PERFORM_SHORT = False
        IS_PERFORM_COVER = False

    rt = auto_trade.get_rt_data(contract)

    tick = ib_controller_tick_size / 10000.0
    max_allow_slippage = max_num_tick_size4_buy_short_slippage * tick

    LIMIT_PRICE_BUY = limit_price + max_allow_slippage
    LIMIT_PRICE_SELL = limit_price - max_allow_slippage

    stp_lmt_order_price = 0.0
    stop_loss_price_raw = stop_loss_price  # AFL uses raw stopLossPrice0 for stpLmtOrderPrice
    if buy:
        stop_loss_price = auto_trade.get_nearest_round_to_price(stop_loss_price, tick, 3)
        if profit_take_price > 0:
            profit_take_price = auto_trade.get_nearest_round_to_price(profit_take_price, tick, 2)
        if max_num_tick_size4_stp_lmt_order_slippage >= 1:
            stp_lmt_order_price = auto_trade.get_nearest_round_to_price(
                stop_loss_price_raw - tick * max_num_tick_size4_stp_lmt_order_slippage, tick, 3)
    elif short:
        stop_loss_price = auto_trade.get_nearest_round_to_price(stop_loss_price, tick, 2)
        if profit_take_price > 0:
            profit_take_price = auto_trade.get_nearest_round_to_price(profit_take_price, tick, 3)
        if max_num_tick_size4_stp_lmt_order_slippage >= 1:
            stp_lmt_order_price = auto_trade.get_nearest_round_to_price(
                stop_loss_price_raw + tick * max_num_tick_size4_stp_lmt_order_slippage, tick, 2)

    STOP_PRICE_BUY = stop_loss_price
    STOP_PRICE_SELL = stop_loss_price
    PROFIT_TAKING_PRICE_BUY = profit_take_price
    PROFIT_TAKING_PRICE_SHORT = profit_take_price
    is_perform_profit_taking = profit_take_price > 0

    if not _ib_is_connected(ib):
        auto_trade.write_line("Cannot connect IB. IBcStatus = [0]")
        return rs

    already_reset_semaphore = False
    curr_position_size = auto_trade.get_position_size_with_retry(symbol1, sell, cover)

    auto_trade.write_line(
        f"currPositionSize. [{symbol1}]. currPositionSize=[{curr_position_size}]. "
        f"LIMIT_PRICE_SELL=[{LIMIT_PRICE_SELL:.4f}]. LIMIT_PRICE_BUY=[{LIMIT_PRICE_BUY:.4f}]. "
        f"lastTradePrice=[{rt['last']:.4f}]. lastAskPrice=[{rt['ask']:.4f}]. "
        f"lastBidPrice=[{rt['bid']:.4f}]. buy1=[{buy}]. sell1=[{sell}]. short1=[{short}]. "
        f"cover1=[{cover}]. IS_PERFORM_BUY=[{IS_PERFORM_BUY}]. IS_PERFORM_SELL=[{IS_PERFORM_SELL}]. "
        f"isPerformProfitTaking=[{is_perform_profit_taking}]. IS_PERFORM_SHORT=[{IS_PERFORM_SHORT}]. "
        f"IS_PERFORM_COVER=[{IS_PERFORM_COVER}]. maxNumTickSize4BuyShortSlippage="
        f"[{max_num_tick_size4_buy_short_slippage}]. ibControllerTickSize="
        f"[{ib_controller_tick_size}]. actualFutureTickSize=[{tick}]. "
        f"maxAllowSlippage=[{max_allow_slippage:.4f}]. limitPrice=[{limit_price:.4f}]. "
        f"numContracts=[{num_contracts}]. stpLmtOrderPrice=[{stp_lmt_order_price:.4f}]")

    auto_trade.log_order_status()

    order_id = -2
    tif = "Day" if is_intra_day else "GTC"

    def _writeback_quota():
        auto_trade.ibvar_qta = max(quota, 0)
        auto_trade.semaphore1 = 0

    # 1. place sell order. sell and no short
    if sell and not short and IS_PERFORM_SELL and curr_position_size > 0:
        quota += 1
        _writeback_quota()
        already_reset_semaphore = True
        auto_trade.write_line(f"sell. [{symbol1}][{curr_position_size}][{LIMIT_PRICE_SELL:.4f}]")
        if auto_trade.cancel_pending_order(symbol1):
            ib.sleep(0.5)
        qty = int(abs(curr_position_size))
        if is_mkt_order:
            sell_trade = ib.placeOrder(contract, MarketOrder("SELL", qty, tif=tif))
        else:
            sell_trade = ib.placeOrder(
                contract, LimitOrder("SELL", qty, _round_child_prices(LIMIT_PRICE_SELL, tick), tif=tif))
        order_id = sell_trade.order.orderId
        auto_trade.write_line(f"After sell. return rs = 1. Sell order id = [{order_id}]")
        rs = 1

    # 2. place cover order. cover and no buy
    if cover and not buy and IS_PERFORM_COVER and curr_position_size < 0:
        quota += 1
        _writeback_quota()
        already_reset_semaphore = True
        auto_trade.write_line(f"cover. [{symbol1}][{curr_position_size}][{LIMIT_PRICE_BUY:.4f}]")
        if auto_trade.cancel_pending_order(symbol1):
            ib.sleep(0.5)
        qty = int(abs(curr_position_size))
        if is_mkt_order:
            cover_trade = ib.placeOrder(contract, MarketOrder("BUY", qty, tif=tif))
        else:
            cover_trade = ib.placeOrder(
                contract, LimitOrder("BUY", qty, _round_child_prices(LIMIT_PRICE_BUY, tick), tif=tif))
        order_id = cover_trade.order.orderId
        auto_trade.write_line(f"After cover. return rs = 1. Cover order id = [{order_id}]")
        rs = 1

    # 3. place buy order. Together with stop loss order
    qty1 = num_contracts
    if buy and IS_PERFORM_BUY and curr_position_size == 0:
        quota -= 1
        _writeback_quota()
        already_reset_semaphore = True
        auto_trade.write_line(f"buy. [{symbol1}][{qty1}][{LIMIT_PRICE_BUY:.4f}]"
                              f"[{STOP_PRICE_BUY:.4f}][{PROFIT_TAKING_PRICE_BUY:.4f}]"
                              f"[{stop_loss_price:.4f}][{profit_take_price:.4f}]"
                              f"[{stp_lmt_order_price:.4f}]")
        if is_mkt_order:
            parent = MarketOrder("BUY", qty1, tif=tif)
        else:
            parent = LimitOrder("BUY", qty1,
                                _round_child_prices(LIMIT_PRICE_BUY, tick), tif=tif)
        stop_child = _stop_child("SELL", qty1, STOP_PRICE_BUY, stp_lmt_order_price, tif)
        profit_child = (LimitOrder("SELL", qty1, PROFIT_TAKING_PRICE_BUY, tif=tif)
                        if is_perform_profit_taking else None)
        parent_trade = _place_bracket(auto_trade, contract, parent, stop_child, profit_child)
        order_id = parent_trade.order.orderId
        auto_trade.write_line(f"After buy. return rs = 1. Buy order id = [{order_id}]")
        rs = 1

    # 4. place short order. Together with stop loss order
    qty1 = num_contracts
    if short and IS_PERFORM_SHORT and curr_position_size == 0:
        quota -= 1
        _writeback_quota()
        already_reset_semaphore = True
        auto_trade.write_line(f"short. [{symbol1}][{qty1}][{LIMIT_PRICE_SELL:.4f}]"
                              f"[{STOP_PRICE_SELL:.4f}][{PROFIT_TAKING_PRICE_SHORT:.4f}]"
                              f"[{stop_loss_price:.4f}][{profit_take_price:.4f}]"
                              f"[{stp_lmt_order_price:.4f}]")
        if is_mkt_order:
            parent = MarketOrder("SELL", qty1, tif=tif)
        else:
            parent = LimitOrder("SELL", qty1,
                                _round_child_prices(LIMIT_PRICE_SELL, tick), tif=tif)
        stop_child = _stop_child("BUY", qty1, STOP_PRICE_SELL, stp_lmt_order_price, tif)
        profit_child = (LimitOrder("BUY", qty1, PROFIT_TAKING_PRICE_SHORT, tif=tif)
                        if is_perform_profit_taking else None)
        parent_trade = _place_bracket(auto_trade, contract, parent, stop_child, profit_child)
        order_id = parent_trade.order.orderId
        auto_trade.write_line(f"After short. return rs = 1. Short order id = [{order_id}]")
        rs = 1

    # 5. place buy order and place cover at the same time
    qty1 = num_contracts
    if (buy and cover and IS_PERFORM_BUY and IS_PERFORM_COVER
            and curr_position_size < 0):
        auto_trade.write_line("buy and cover at the same time begin")
        _writeback_quota()
        already_reset_semaphore = True
        if auto_trade.cancel_pending_order(symbol1):
            ib.sleep(0.5)
        auto_trade.write_line(f"buy and cover. [{symbol1}][{qty1}][{LIMIT_PRICE_BUY:.4f}]"
                              f"[{STOP_PRICE_BUY:.4f}][{PROFIT_TAKING_PRICE_BUY:.4f}]"
                              f"[{stop_loss_price:.4f}][{profit_take_price:.4f}]"
                              f"[{stp_lmt_order_price:.4f}]")
        qty2 = 2 * qty1
        if is_mkt_order:
            parent = MarketOrder("BUY", qty2, tif=tif)
        else:
            parent = LimitOrder("BUY", qty2,
                                _round_child_prices(LIMIT_PRICE_BUY, tick), tif=tif)
        # reversal children use a plain stop (AFL places STP with limit 0)
        stop_child = StopOrder("SELL", qty1, STOP_PRICE_BUY, tif=tif)
        profit_child = (LimitOrder("SELL", qty1, PROFIT_TAKING_PRICE_BUY, tif=tif)
                        if is_perform_profit_taking else None)
        parent_trade = _place_bracket(auto_trade, contract, parent, stop_child, profit_child)
        order_id = parent_trade.order.orderId
        auto_trade.write_line(f"After buy and cover. return rs = 1. Buy and cover order id = [{order_id}]")
        rs = 1

    # 6. place short and sell order at the same time
    qty1 = num_contracts
    if (short and sell and IS_PERFORM_SHORT and IS_PERFORM_SELL
            and curr_position_size > 0):
        _writeback_quota()
        already_reset_semaphore = True
        auto_trade.write_line(f"short and sell. [{symbol1}][{curr_position_size}]"
                              f"[{LIMIT_PRICE_SELL:.4f}]")
        if auto_trade.cancel_pending_order(symbol1):
            ib.sleep(0.5)
        auto_trade.write_line(f"short and sell. [{symbol1}][{2 * qty1}]"
                              f"[{LIMIT_PRICE_SELL:.4f}][{STOP_PRICE_SELL:.4f}]"
                              f"[{PROFIT_TAKING_PRICE_SHORT:.4f}][{stop_loss_price:.4f}]"
                              f"[{profit_take_price:.4f}]")
        qty2 = 2 * qty1
        if is_mkt_order:
            parent = MarketOrder("SELL", qty2, tif=tif)
        else:
            parent = LimitOrder("SELL", qty2,
                                _round_child_prices(LIMIT_PRICE_SELL, tick), tif=tif)
        # reversal children use a plain stop (AFL places STP with limit 0)
        stop_child = StopOrder("BUY", qty1, STOP_PRICE_SELL, tif=tif)
        profit_child = (LimitOrder("BUY", qty1, PROFIT_TAKING_PRICE_SHORT, tif=tif)
                        if is_perform_profit_taking else None)
        parent_trade = _place_bracket(auto_trade, contract, parent, stop_child, profit_child)
        order_id = parent_trade.order.orderId
        auto_trade.write_line(f"After short and sell. return rs = 1. Short and sell order id = [{order_id}]")
        rs = 1

    # final check make sure release semaphore
    if not already_reset_semaphore:
        auto_trade.semaphore1 = 0

    return rs


def do_trade_stock0(auto_trade, contract, buy, sell, short, cover, limit_price,
                    stop_loss_price, profit_take_price, num_contracts,
                    ib_controller_tick_size, is_intra_day, is_mkt_order,
                    max_num_tick_size4_buy_short_slippage,
                    max_num_tick_size4_stp_lmt_order_slippage):
    """AFL doTradeStock0 wrapper: symbol_type = 2 (stock)."""
    return doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price,
                    stop_loss_price, profit_take_price, num_contracts,
                    ib_controller_tick_size, is_intra_day, is_mkt_order,
                    max_num_tick_size4_buy_short_slippage,
                    max_num_tick_size4_stp_lmt_order_slippage, 2)


def do_trade_future0(auto_trade, contract, buy, sell, short, cover, limit_price,
                     stop_loss_price, profit_take_price, num_contracts,
                     ib_controller_tick_size, is_intra_day, is_mkt_order,
                     max_num_tick_size4_buy_short_slippage,
                     max_num_tick_size4_stp_lmt_order_slippage):
    """AFL doTradeFuture0 wrapper: symbol_type = 1 (future)."""
    return doTrade0(auto_trade, contract, buy, sell, short, cover, limit_price,
                    stop_loss_price, profit_take_price, num_contracts,
                    ib_controller_tick_size, is_intra_day, is_mkt_order,
                    max_num_tick_size4_buy_short_slippage,
                    max_num_tick_size4_stp_lmt_order_slippage, 1)
