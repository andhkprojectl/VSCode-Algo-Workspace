#
#  1.0.2
#
# Trade Dia according to #YM#C
# - Similar to 1.0.1
# - Modify code, simulate same result as inCubation_DIA_30min.afl
#
# - Note:
#   - optReturn.stra.broker not found because
#     - need to set bt.cerebro(optreturn=True)
#     - use addStrategy (simple backtest) instead of optStrategy
#
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import datetime as dt

# from pkg_resources import non_empty_lines
from pyiqfeed import HistoryConn
import logging
import backtrader as bt
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout
from bokeh.layouts import column

import exchange_calendars as xcals

from pathlib import Path
import csv


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

defaultParams = {
    'percentile_threshold': 0.3,
    'atr_period': 15,
    'n1': 2.0,
    'n2': 1.0,
    'initial_capital': 40000,
    'in_sample_years': 1,
    'out_sample_months': 1,
    'LOOKBACK_DAYS' : 100,
    'PERCENTILE' : 90,
    'SELL_NUM_DAYS': 4,
    'SL1' : 3,
    'PF1' : 3
}


# Function to check if a date is a non-trading day (NYSE holiday or weekend)
def is_NYSE_trading_day(date1):
    """Check if the given date is not a trading day (NYSE holiday or weekend)."""
    nyse1 = xcals.get_calendar('XNYS')
    return nyse1.is_session(date1)

def convert2Daily(df1):
    daily = df1.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
        'numTrade': 'sum'

    })

    # Drop NaNs (non-trading days)
    daily.dropna(inplace=True)

    # Show sample output
    # print(daily.head())
    return daily

# fetch daily data using function request_daily_data_for_dates
def fetch_daily_data(conn, symbol1, start_dt, end_dt):
    # Create a History Connection
    all_data = []
    isConnCreateHere = False
    if (conn is None):
        conn = HistoryConn(name="history")
        conn.connect()
        isConnCreateHere = True
    try:
        data = conn.request_daily_data_for_dates(symbol1, start_dt, end_dt)
        # print (data)
        # print (list(data))

        records = [
            {
                # "datetime": bar["datetime"],  # Keep full timestamp
                "datetime": bar["date"],
                "Open": bar["open_p"],
                "High": bar["high_p"],
                "Low": bar["low_p"],
                "Close": bar["close_p"],
                "Volume": bar["prd_vlm"],
                "OpenContract": bar["open_int"]
            }
            for bar in data
        ]
        # df = pd.DataFrame(data)
        df = pd.DataFrame(records)


        print (2)

        # conn.request_daily_data_for_dates()

        # Drop NaNs (non-trading days)
        df.dropna(inplace=True)

        # print(4)

        # Optional: Convert 'date' column to datetime and sort
        # df['datetime'] = pd.to_datetime(df['date'])
        # df = df.sort_values('datetime')

        # print(5)

        # Show results
        # print(df[['datetime', 'open', 'high', 'low', 'close', 'volume']].head())
        print (df.head)

        # print(6)

        # Save to CSV (optional)
        #if (write_output_2_file):
        #    df.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        # conn.disconnect()
        return df

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()    # Create a History Connection

# fetch daily data from 9:30 to 16:00
# return pd.DataFrame datatype
# e.g. start_time = "09:30"
# e.g. end_time = "16:00"
# e.g. period1 = 1800, 30 minutes
def fetch_data_with_time(conn, symbol1, start_dt, start_time, end_dt, end_time, period1):
    # Create a History Connection
    all_data = []
    isConnCreateHere = False
    if (conn is None):
        conn = HistoryConn(name="history")
        conn.connect()
        isConnCreateHere = True
    try:

        # data = get_bar_data(symbol, interval_len, interval_type,"20250501 044500", "20250502 133000")  # May 1–2, 2025
        data = conn.request_bars_in_period(
            ticker=symbol1,
            # interval_len=1800,  # 30 minutes (1800 seconds)
            interval_len=period1,  # 30 minutes (1800 seconds)
            interval_type="s",  # Seconds
            bgn_prd=start_dt,
            end_prd=end_dt,
            # bgn_flt=datetime.strptime("09:30", "%H:%M").time(),
            # end_flt=datetime.strptime("16:00", "%H:%M").time(),
            bgn_flt=datetime.strptime(start_time, "%H:%M").time(),
            end_flt=datetime.strptime(end_time, "%H:%M").time(),
            ascend=True,  # Oldest to latest
            max_bars=None,  # Fetch all available bars
            timeout=None
        )
        # print ("data output")

        # print(f"symbol: {symbol1}", data.dtype.names)
        # print 1st 5 rows
        # print(data[:5])
        # print(data["time"][:5])

        # Print header names
        # if data is not None and len(data) > 0:
        #     header_names = data.dtype.names
        #     print("Data Header Names:", header_names)
        # else:
        #     print("No data returned")

        # Convert to list of dictionaries
        all_data = [
            {
                # "datetime": bar["datetime"],  # Keep full timestamp
                "Date1": bar["date"],
                "Time1": bar["time"],
                "Open": bar["open_p"],
                "High": bar["high_p"],
                "Low": bar["low_p"],
                "Close": bar["close_p"],
                "AccVolume": bar["tot_vlm"],
                "Volume": bar["prd_vlm"],
                "numTrade": bar["num_trds"]
            }
            for bar in data
        ]

        # all_data.extend(records)
        df = pd.DataFrame(all_data)
        # print (df.columns)
        # print (df.head())

        # df['datetime'] = pd.to_datetime(df['Date1'] + ' ' + df['Time1'])
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str))
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='%Y-%m-%d %H:%M:%S')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='mixed', errors='coerce')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='ISO8601')

        # Convert Time1 (nanoseconds) to time string
        # df['TimeSeconds'] = df['Time1'] / 1_000_000_000  # Convert to seconds
        # df['TimeStr'] = pd.to_timedelta(df['TimeSeconds'], unit='s').apply(
        #    lambda x: x.components.hours * 3600 + x.components.minutes * 60 + x.components.seconds).apply(
        #    lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}")

        # Add new column datatime
        # Convert nanoseconds to timedelta
        df['time_delta'] = pd.to_timedelta(df['Time1'], unit='ns')
        # Convert timedelta to string formatted as HH:MM:SS
        # df['TimeStr'] = df['time_delta'].astype('timedelta64[s]').astype(str)
        df['TimeStr'] = df['time_delta'].dt.components.apply(
            lambda row: f"{int(row.hours):02}:{int(row.minutes):02}:{int(row.seconds):02}", axis=1
        )
        df['DateStr'] = df['Date1'].dt.strftime('%Y-%m-%d')
        # Combine Date1 and TimeStr into datetime
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['TimeStr'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df['datetime'] = pd.to_datetime(df['Date1'].dt.strftime('%Y-%m-%d') + ' ' + df['TimeStr'], format='%Y-%m-%d %H:%M:%S', errors='coerce')



        print (f"fetch_data_with_time")
        print(df.columns)
        print(df.head())
        print(df['datetime'].head())

        # print(df[['DateStr', 'TimeStr']].head())  # 確認合併前的時間格式
        # print(df[['DateStr']].head())  # 確認合併前的時間格式
        # print(df[['TimeStr']].head())  # 確認合併前的時間格式
        # print(df['datetime'].head())  # 確認轉換是否成功
        # if cannot convert datetime and errors='coerce', datetime field show na print how many show is convert succsess
        # print(df['datetime'].isna().sum())


        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        #

        #

        # daily = df1.resample('1D').agg({
        #     'Open': 'first',
        #     'High': 'max',
        #     'Low': 'min',
        #     'Close': 'last',
        #     'Volume': 'sum',
        #     'numTrade': 'sum'


        # Save to CSV (optional)
        # if (write_output_2_file):
        #     daily.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")

        # conn.disconnect()

        return df

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()

def fetch_daily_data_00930_to_1600_30min(conn, symbol1, start_dt, start_time, end_dt, end_time):
    df1 = fetch_data_with_time(conn, symbol1, start_dt, "09:30", end_dt, "16:00", 1800)
    daily_data = convert2Daily(df1)
    return daily_data

def prepare_data(start_dt1, end_dt1):
    ym_data = pd.DataFrame()
    dia_data = pd.DataFrame()
    combine_data = pd.DataFrame()
    try:
        conn = HistoryConn(name="history")
        conn.connect()
        ym_data = fetch_daily_data(conn, "@YM#C", start_dt1, end_dt1)
        # dia_data = fetch_daily_data_00930_to_1600_30min(conn, "DIA", start_dt1, end_dt1)
        # ym_data1 = fetch_data_with_time(conn, "@YM#C", start_dt1, "00:00", end_dt1, "23:59", 1800)
        dia_data1 = fetch_data_with_time(conn, "DIA", start_dt1, "04:00", end_dt1, "20:30", 1800)

        # convert daily
        # ym_data = convert2Daily(ym_data1)
        dia_data = convert2Daily(dia_data1)

        # fetch_daily_data_with_time(conn, symbol1, start_dt, "09:30", end_dt, "16:00", 1800)
        # ym_data = pd.DataFrame()
        # dia_data = pd.DataFrame()
        # Align dates
        common_dates = dia_data.index.intersection(ym_data.index)
        dia_data = dia_data.loc[common_dates]
        ym_data = ym_data.loc[common_dates]

        # Calculate daily returns for @YM#C
        ym_data['Daily_Return'] = (ym_data['Close'] - ym_data['Close'].shift(1)) / ym_data['Close'].shift(1)

        # Combine dia_data, ym_data together
        dia_data_prefix = dia_data.add_prefix('D_')
        ym_data_prefix = ym_data.add_prefix('Y_')
        combine_data = pd.concat([dia_data_prefix, ym_data_prefix], axis=1)

    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
    finally:
        #if (conn and conn.connected()):
        conn.disconnect()
    return dia_data, ym_data, combine_data


# Define the trading strategy
class ymDiaStrategy1(bt.Strategy):
    params = dict(
        percentile_days=160,
        percentile_th = 140,
        start_trade_dt = datetime(2021,1,1)
        # sl1=3,
        # pf1=3,
    )
    #self.data0: dia_data
    #self.data1: ym_data

    def __init__(self):
        self.atr15 = bt.indicators.ATR(self.data0, period=15)
        # self.atr =   bt.ind.ATR(self.data, period=self.p.atr_period)
        # self.trade_day = None
        # self.buy_order = None
        # self.sell_next_open = False
        self.trade_count = 0
        self.portfolio_values = []
        self.traded = False
        self.order_history = []  # Store order details
        self.buy_price = None  # Track buy price for stop/take profit
        # self.buy_bar = None  # Track bar index of buy
        self.sell_price = None
        self.trading_day_count = 0
        self.buy_trading_day = None

        self.stopLossPrice = None
        self.stopLossLimitPrice = None
        self.stopProfitPrice = None
        self.stopProfitLimitPrice = None

        self.active_orders = []  # Track main, stop, take profit, normal sell orders
        self.normal_sell_order = None  # Track normal sell limit order
        self.order_ids = {}  # Store order IDs: main, stop, take_profit, normal_sell
        self.order_type1 = None

        self.current_trade = {}
        self.completed_trades = []

    def getclosereason(self, order):
        # # Example logic — extend as needed
        # if order.exectype == bt.Order.Stop:
        #     return "Stop Loss Hit"
        # elif order.exectype == bt.Order.Limit:
        #     return "Take Profit Hit"
        # elif order.exectype == bt.Order.Market:
        #     return "Timed Exit / Manual Close"
        # else:
        #     return "Unknown"
        return "TBD"

    def next(self):
        current_date = self.data0.datetime.date(0)
        print(f"Date: {current_date}, Bar: {len(self)}, DIA Close: {self.data0.close[0]:.2f}, "
              f"Cash: {self.broker.get_cash():.2f}, Position: {self.position.size}, "
              f"Params: days={self.params.percentile_days}")

        # Check if current date is a holiday or weekend
        is_trading_day = is_NYSE_trading_day(current_date)

        # Increment trading day counter only on trading days
        if is_trading_day:
            self.trading_day_count += 1

        if not is_trading_day:
            print(f"Skipping trading on holiday/weekend: {current_date}")
            return

        # currDate1 =

        # Check for trigger conditions for active orders
        for order in self.active_orders:
            if order.alive():
                order_type = (
                    "Buy" if order.isbuy() else
                    "Normal Sell" if order is self.normal_sell_order else
                    "Stop Loss" if order.exectype == bt.Order.Stop else
                    "Take Profit" if order.exectype == bt.Order.Limit else "Unknown"
                )
                if order_type == "Stop Loss" and self.data0.low[0] <= order.price:
                    print(f"*** Stop Loss Triggered: Date={current_date}, OrderID={order.ref}, Price=${order.price:.2f}, Size={order.size} ***")
                elif order_type in ["Take Profit", "Normal Sell"] and self.data0.high[0] >= order.price:
                    print(f"*** {order_type} Triggered: Date={current_date}, OrderID={order.ref}, Price=${order.price:.2f}, Size={order.size} ***")

        # self.buy_order = self.buy(data=self.data0, size=1, exectype=bt.Order.Market)
        # self.data0
        # self.data1
        # tradePrice1 = self.data0.open[0]
        self.buy_price = self.data0.open[0]
        # for backtest, stop limit price = trade price
        stopLimitPrice = self.buy_price
        positionSize1 = 82 # 35000/425 (USD amount/DIA stock price)

        # assume
        # 1) stop loss and stop profit are limit price
        # 2) for testing, stop loss and stop profit same as limit price
        self.stopLossPrice = self.buy_price - defaultParams['SL1'] * self.atr15[-1]
        self.stopLossLimitPrice = self.stopLossPrice  # same for testing only
        self.stopProfitPrice = self.buy_price + defaultParams['PF1'] * self.atr15[-1]
        self.stopProfitLimitPrice = self.stopProfitPrice
        # isSell1 = self.position and len(self) == defaultParams['SELL_NUM_DAYS']
        # isBuy1 =

        """
        dt = self.datas[0].datetime.datetime(0)
        if dt >= datetime(2025, 5, 12):
            print(0)
            a = self.position
            b = len(self)
            d = defaultParams['SELL_NUM_DAYS']
        """

        if (len(self.data1) > self.p.percentile_days
            and self.data0.datetime.date(0) >= self.params.start_trade_dt.date()
                ):

#            if (self.data0.datetime.date(0)  >= datetime(2021, 2, 25).date()):
#                print(111)

            # find return of yesterday
            returns = [self.data1.close[i] - self.data1.close[i - 1] for i in range(-self.p.percentile_days -1, -1)]
            percentileValue = 91
            percentile_90 = np.percentile(returns, percentileValue)
            yesterday_return = self.data1.close[-1] - self.data1.close[-2]
            if (self.position):
                print ("self.position not None")
            else:
                print("self.position is None")

            if (yesterday_return >= percentile_90
                and yesterday_return >= self.p.percentile_th
                # and not self.position
                and self.position.size == 0
                # and self.buy_bar is None
                and self.buy_trading_day is None

            ):
                print ("Order Bought 1")
                a1 = self.data0.close[1] # next bar
                a2 = self.data0.close[0] # current bar
                a3 = self.data0.close[-1] # previous bar
                """
                main_order = self.buy(
                    data=self.data0,
                    size=positionSize1,
                    price=self.buy_price,
                    plimit=stopLimitPrice,
                    exectype=bt.Order.Limit,
                    transmit=False
                )
                stop_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=self.stopLossPrice,
                    plimit=self.stopLossLimitPrice,
                    exectype=bt.Order.Limit,
                    parent=main_order,
                    transmit=False
                )
                take_profit_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=self.stopProfitPrice,
                    plimit=self.stopProfitLimitPrice,
                    exectype=bt.Order.Limit,
                    parent=main_order,
                    transmit=True
                )
                """
                # testing change to mkt order. above place order on 1-Feb-2021 but filled order on 4-Mar-2021 , why??
                main_order = self.buy(
                    data=self.data0,
                    size=positionSize1,
                    price=self.buy_price,
                    exectype=bt.Order.Market,
                    transmit=False
                )
                stop_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=self.stopLossPrice,
                    plimit=self.stopLossLimitPrice,
                    exectype=bt.Order.Stop,
                    parent=main_order,
                    transmit=False
                )
                take_profit_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=self.stopProfitPrice,
                    exectype=bt.Order.Limit,
                    # exectype=bt.Order.Market,
                    parent=main_order,
                    transmit=True
                )

                self.active_orders = [main_order, stop_order, take_profit_order]
                self.order_ids = {
                    'main': main_order.ref,
                    'stop': stop_order.ref,
                    'take_profit': take_profit_order.ref
                }

                # self.buyer_order = main_order
                # self.buy_order = self.buy()
                # self.buyer =  self.buy()
                # self.buy_order = self.buy(data=self.data0, size=positionSize1, price=tradePrice1, plimit=stopLimitPrice, exectype=bt.Order.Limit)
                # self.sell_next_open = True  # 在下一根K線開盤賣出
                #
                # self.trade_day = len(self)

                self.trade_count += 1
                self.buy_trading_day = self.trading_day_count  # Record trading day of buy
                # self.buy_bar = len(self)  # Record buy bar index
                # self.buy_price = self.data0.close[0]  # Record buy price
                print(f"Submitted Buy DIA: Size=100, Price={self.data0.close[0]:.2f}")

                # a1 = self.position
                # a2 = len (self)
                # a3 = self.buy_bar


        # if len(self.data1) > self.p.percentile_days:
        # elif self.position and len(self) == self.trade_day + params1.SELL_NUM_DAYS:
        # if self.position and len(self) == self.trade_day + params1.SELL_NUM_DAYS:
        # if self.position and len(self) >= 1:
        # if self.position and len(self) == defaultParams['SELL_NUM_DAYS']:
        # if self.position and len(self) > defaultParams['SELL_NUM_DAYS']:
        # if self.buy_bar is not None and len(self) == self.buy_bar + defaultParams['SELL_NUM_DAYS'] and self.position:
        # normal sell order
        if (self.buy_trading_day is not None
                # and self.trading_day_count >= self.buy_trading_day + defaultParams['SELL_NUM_DAYS']
                and self.trading_day_count >= (self.buy_trading_day + 4)
                and self.position.size != 0
        ):
            #len(self) >= self.buy_bar + 4:
            self.sell_price = self.data0.open[0]
            self.normal_sell_order = self.sell(
                data=self.data0,
                size=positionSize1,
                price=self.sell_price,
                plimit=stopLimitPrice,
                exectype=bt.Order.Limit,
                transmit=True
            )
            # self.sell(exectype=bt.Order.Market)
            self.trade_count += 1
            self.buy_trading_day = None  # Reset buy trading day
            # Schedule next buy 1 bar after sell
            # self.next_buy_bar = len(self) + 1
            # self.sell_next_open = False
            # self.buy_bar = None  # Reset buy bar
            self.order_ids['normal_sell'] = self.normal_sell_order.ref
            self.active_orders.append(self.normal_sell_order)

            print(f"Submitted Normal Sell (Limit): OrderID={self.normal_sell_order.ref}, Size={self.position.size}, "
                  f"Price={self.sell_price:.2f}, Trading Day={self.trading_day_count}")
            # print("Submitted Normal Sell order")

            print("2. Normal sell. Canceling remaining open orders (stop loss and take profit order).")
            for o in self.active_orders:
                if o.alive() and o.ref != order.ref:
                    self.cancel(o)

        self.portfolio_values.append(self.broker.getvalue())

    def log(self, txt):
        """Log messages with timestamp"""
        dt = self.datas[0].datetime.datetime(0)
        print(f'[{dt}]: {txt}')

    def notify_order(self, order):
        # a22 = order.status
        if order.status in [order.Completed]:
            print("Order Completed1")
            try:
                executed_dt = bt.num2date(order.executed.dt)
            except Exception as e:
                print(f"Error converting datetime: {e}, order.executed.dt={order.executed.dt}")

            executed_dt = self.data0.datetime.datetime()
            executed_dt2 = bt.num2date(order.executed.dt)
            executed_price = order.executed.price
            order_type = "Buy" if order.isbuy() else (
                "Normal Sell" if self.normal_sell_order is not None else
                "Stop Loss" if order.exectype == bt.Order.Stop else
                #"Take Profit" if order.exectype == bt.Order.Limit else "Unknown Sell"
                # "Take Profit" if bt.Order.Market and order is not self.normal_sell_order else "Unknown Sell"
                "Take Profit" if order.exectype == bt.Order.Limit and self.normal_sell_order is None else "Unknown Sell"

            )
            self.order_history.append({
                'type': order_type,
                'datetime': executed_dt,
                'price': executed_price,
                'size': order.size
            })
            if order_type == "Take Profit":
                print(f"*** Take Profit Executed: Datetime={executed_dt}, Price=${executed_price:.2f}, Size={order.size}, Value={order.executed.value:.2f}, atr={self.atr15[0]}, self.stopLossPrice={self.stopLossPrice}, self.stopProfitPrice={self.stopProfitPrice}. position size={self.position.size} ***")
            elif order_type == "Stop Loss":
                print(f"*** Stop Loss Executed: Datetime={executed_dt}, Price=${executed_price:.2f}, Size={order.size}, Value={order.executed.value:.2f}, atr={self.atr15[0]}, self.stopLossPrice={self.stopLossPrice}, self.stopProfitPrice={self.stopProfitPrice}. position size={self.position.size} ***")
            elif order_type == "Normal Sell":
                print(f"*** Normal Sell Executed: Datetime={executed_dt}, Price=${executed_price:.2f}, Size={order.size}, Value={order.executed.value:.2f}, atr={self.atr15[0]}, self.stopLossPrice={self.stopLossPrice}, self.stopProfitPrice={self.stopProfitPrice}. position size={self.position.size} ***")
            else:
                print(f"Order Completed2: {order_type}, Datetime={executed_dt}, Price=${executed_price:.2f}, Value={order.executed.value:.2f}, atr={self.atr15[0]}, self.stopLossPrice={self.stopLossPrice}, self.stopProfitPrice={self.stopProfitPrice}. position size={self.position.size}")


            currDate2 = self.data0.datetime.date(0)
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')
                self.position_size = order.executed.size
                order_type = "Buy"
                self.current_trade = {
                    'buy_dt': executed_dt,
                    'buy_price': executed_price,
                    'size': order.executed.size,
                    'order_type': 'Buy'
                }
            elif order.issell():
                # self.buy_bar = None  # Reset so 4-day logic doesn't re-trigger
                self.buy_trading_day = None
                print("set self.buy_trading_day to None")

                # cancel bracket order stop order and take profit order, to prevent sell and these order triggered at the same bar
                #if self.position.size == 0:
                # if self.normal_sell_order is not None:
                print("1. Position closed. Canceling remaining open orders.")
                for o in self.active_orders:
                    if o.alive() and o.ref != order.ref:
                        self.cancel(o)
                print(f"Cancelled sibling orders after sell execution: OrderID={order.ref}")
                self.active_orders = []
                self.normal_sell_order = None
                self.order_ids = {}


                # reason = "Take-Profit" if order.exectype == bt.Order.Limit else "Trailing Stop"
                # self.log(
                #     f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Reason: {reason}')
                # # self.position_size = 0
                # # Determine sell type
                # if order.exectype == bt.Order.Market:
                #     order_type = "Normal Sell"
                # elif order.exectype == bt.Order.Stop:
                #     order_type = "Stop Loss"
                # elif order.exectype == bt.Order.Limit:
                #     order_type = "Take Profit"
                # else:
                #     order_type = "Unknown Sell"
                if order.ref == self.order_ids.get('stop'):
                    self.order_type1 = 'Stop Loss'
                elif order.ref == self.order_ids.get('take_profit'):
                    self.order_type1 = 'Take Profit'
                elif order.ref == self.order_ids.get('normal_sell'):
                    self.order_type1 = 'Normal Sell'
                else:
                    self.order_type1 = 'Unknown'

                sell_price = executed_price
                sell_dt = executed_dt2
                buy_price = self.current_trade['buy_price']
                buy_dt = self.current_trade['buy_dt']
                size = self.current_trade['size']
                profit = (sell_price - buy_price) * size
                period = (sell_dt - buy_dt).days

                self.completed_trades.append({
                    'buy_dt': buy_dt,
                    'buy_price': buy_price,
                    'sell_dt': sell_dt,
                    'sell_price': sell_price,
                    'order_type': order_type,
                    'period': period,
                    'profit': profit,
                    'size': size
                })

                self.current_trade = {}



            # elif order.status in [order.Completed, order.Canceled, order.Rejected]:
            #     self.buy_order = None  # 無論成功或失敗都清除掛單
            #     self.order = None
            # self.order = None
            # Capture order details
            # executed_dt = self.datas[0].datetime.num2date(order.executed.dt)
            executed_dt = bt.num2date(order.executed.dt)
            executed_price = order.executed.price

            self.order_history.append({
                'type': order_type,
                'datetime': executed_dt,
                'price': executed_price,
                'size': order.size
            })

            # above normal sell perform below, below is for bracket order
            # if not order.isbuy() and self.position.size == 0:
            #     print("Position closed. Canceling remaining open orders.")
            #     for o in self.active_orders:
            #         if o.alive() and o != order:
            #             self.cancel(o)
            #     self.active_orders = []
            #     self.normal_sell_order = None
            #     self.order_ids = {}


            print(f"Order Completed3: {order_type}, Datetime: {executed_dt}, "
                  f"Price: ${executed_price:.2f}, Value: {order.executed.value}")


        elif order.status in [order.Close]:
            print("Order Close")
            currDate3 = self.data0.datetime.date(0)
            self.close_reason = self.getclosereason(order)
            # print(f"Order Close Reason: {self.close_reason}")
            print(f"Order Closed: Reason={self.close_reason}, Type={order.ordtypename()}, OrderID={order.ref}, Size={order.size}")
            # TBD. not self.position
            # if not order.isbuy() and not self.position:
            #     self.active_orders = []
            #     self.normal_sell_order = None
            #     self.order_ids = {}

            # if not order.isbuy() and not self.position:
                # self.next_buy_bar = len(self) + 1
                # self.buy_bar = None
                # do not set to None as bracket order order.status = order.close
                # self.buy_trading_day = None


    def notify_trade(self, trade):
        print("Notify Trade")
        if trade.isclosed:
            self.log(f'TRADE CLOSED, Gross P/L: {trade.pnl:.2f}, Net P/L: {trade.pnlcomm:.2f}')
            print(f"Now position: {self.position.size}")  # will be 0 here

    def stop(self):
        print(f"Strategy completed. Total trades: {self.trade_count}, Params: days={self.params.percentile_days}")
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            print(f"Portfolio returns std: {np.std(returns):.6f}, mean: {np.mean(returns):.6f}")
            if np.std(returns) > 0:
                manual_sharpe = np.mean(returns) / np.std(returns)
                print(f"Manual Sharpe (daily): {manual_sharpe:.2f}")

        if self.completed_trades:
            output_file = Path("C:/tmp/backtrader_backtest_trade_log.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists
            with open(output_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'buy_dt', 'buy_price', 'sell_dt', 'sell_price',
                    'order_type', 'period', 'profit', 'size'
                ])
                writer.writeheader()
                for trade in self.completed_trades:
                    writer.writerow({
                        'buy_dt': trade['buy_dt'].strftime('%Y-%m-%d'),
                        'buy_price': f"{trade['buy_price']:.2f}",
                        'sell_dt': trade['sell_dt'].strftime('%Y-%m-%d'),
                        'sell_price': f"{trade['sell_price']:.2f}",
                        'order_type': trade['order_type'],
                        'period': trade['period'],
                        'profit': f"{trade['profit']:.2f}",
                        'size': trade['size']
                    })
            print(f"Trade log written to {output_file}")
        else:
            print("No trades to log.")


def performBacktestOptimization(inDataList, optTarget, isPlot):
    if (len(inDataList) != 2):
        print(f"performBacktestOptimization. Invalid data length: {len(inDataList):2f}")
        return None
    # Default paramenter
    # params1 = DEFAULT_PARAMS.copy()
    # Run backtest
    # Set up cerebro for backtesting
    ymDiaCerebro = bt.Cerebro(optreturn = True)
    # ymDiaCerebro = bt.Cerebro(optreturn = False)

    #ymDiaCerebro.broker.set_cash(defaultParams['initial_capital'])  # Initial capital
    ymDiaCerebro.broker.set_cash(4000000)  # Initial capital
    # ymDiaCerebro.broker.setcommission(commission=0.0)

    # Add data feeds
    # data_dia = bt.feeds.PandasData(dataname=df_dia)
    # data_ym = bt.feeds.PandasData(dataname=df_ym)

    # a1 = len(inDataList[0])
    # a2 = len(inDataList[1])


    ymDiaCerebro.adddata(inDataList[0])  #data_dia
    ymDiaCerebro.adddata(inDataList[1])  # data_ym

    # bt.Cerebro(optreturn=False)

    # Add strategy with optimization
    ymDiaCerebro.optstrategy(
        ymDiaStrategy1,
        percentile_days=range(50, 151, 10)
        # percentile_th = 140
        # percentile_th = range(80, 140, 5)
        # sl1=range(1, 6, 1),
        # pf1=range(1, 6, 1)
    )

    # Add analyzers
    # ymDiaCerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
    ymDiaCerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years
                , riskfreerate=0.03  # Example: 3% annual risk-free rate
    )
    # ymDiaCerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=False, riskfreerate=0.0)
    ymDiaCerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    ymDiaCerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    ymDiaCerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    ymDiaCerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # ymDiaCerebro.addanalyzer(PortfolioValue, _name='portfolio')

    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=False, riskfreerate=0.0)
    # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # cerebro.addanalyzer(PortfolioValue, _name='portfolio')

    # Run optimization
    results = ymDiaCerebro.run(maxcpus=1)  # Use 1 CPU to avoid threading issues with IQFeed
    # best_strat = results[0]
    # for strat_list in results:
    #     strat = strat_list[0]  # first (and only) strategy in the list
    #     print (1223)
    #     final_value = strat.broker.getvalue()

    # optimize target is Sharpe Ratio, may be MDD, CAR/MDD, etc
    if (optTarget=='sharpe_ratio'):
        # Find the best run based on Sharpe ratio
        best_sharpe = -float('inf')
        best_strat = None
        # Flatten results
        flattened_results = [strat for sublist in results for strat in sublist]
        for optReturn in flattened_results:
            # strategy = getattr(optReturn, 'strat', None)
            # if strategy is None:
            #     # Alternative: check if optreturn itself is the strategy
            #     if hasattr(strat, 'broker'):
            #         strategy = strat
            #     else:
            #         print(f"Warning: No strategy instance found for params: ")
            #         continue
            # a1 = strat.analyzers.sharpe.get_analysis()
            sharpe_dict = optReturn.analyzers.sharpe.get_analysis()
            # sharpe = optReturn.analyzers.sharpe.get_analysis().get('sharperatio', -float('inf'))

            if sharpe_dict is None:
                # print(f"Warning: Sharpe analysis returned None for strategy with params: "
                #       f"percentile_days={strat.params.percentile_days}, sl1={strat.params.sl1}, pf1={strat.params.pf1}")
                sharpe = -float('inf')
            else:
                sharpe = sharpe_dict.get('sharperatio', -float('inf'))
                # print(f"Sharpe Ratio: {sharpe:.2f} for params: "
                #      f"percentile_days={strat.params.percentile_days}, sl1={strat.params.sl1}, pf1={strat.params.pf1}")

            print(f"Sharpe: {sharpe}")
            print(optReturn.analyzers.drawdown.get_analysis()['max']['drawdown'])
            print(optReturn.analyzers.annual_return.get_analysis())
            if sharpe and sharpe > best_sharpe:
                # print (1)
                best_sharpe = sharpe
                best_optReturn = optReturn

    #if (isPlot):
    #    ymDiaCerebro.plot(volume=False)

    # strategy = best_strat
    # print (1)
    # final_value = strategy.broker.getvalue()  # Correct access

    # final_value11 = best_strat.broker.getvalue()
    showOptBackTestResult(best_optReturn)

    # plot curve

    return results, best_optReturn


def performSimpleBacktest(start_trade_dt1, inDataList, optTarget, isPlot):
    if (len(inDataList) != 2):
        print(f"performSimpleBacktest. Invalid data length: {len(inDataList):2f}")
        return None

    # Default paramenter
    # params1 = DEFAULT_PARAMS.copy()
    # Run backtest
    # Set up cerebro for backtesting
    ymDiaCerebro = bt.Cerebro(optreturn=True)
    # ymDiaCerebro = bt.Cerebro(optreturn = False)

    # ymDiaCerebro.broker.set_cash(defaultParams['initial_capital'])  # Initial capital
    ymDiaCerebro.broker.set_cash(4000000)  # Initial capital
    # ymDiaCerebro.broker.setcommission(commission=0.0)

    # Add data feeds
    # data_dia = bt.feeds.PandasData(dataname=df_dia)
    # data_ym = bt.feeds.PandasData(dataname=df_ym)

    # a1 = len(inDataList[0])
    # a2 = len(inDataList[1])

    ymDiaCerebro.adddata(inDataList[0])  # data_dia
    ymDiaCerebro.adddata(inDataList[1])  # data_ym

    # bt.Cerebro(optreturn=False)

    # Add strategy with optimization
    ymDiaCerebro.addstrategy(
        ymDiaStrategy1,
        percentile_days=140,
        start_trade_dt = start_trade_dt1
        # percentile_th = 140
        # percentile_th = range(80, 140, 5)
        # sl1=range(1, 6, 1),
        # pf1=range(1, 6, 1)
    )

    # Add analyzers
    ymDiaCerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Years)
    # ymDiaCerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=False, riskfreerate=0.0)
    ymDiaCerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    ymDiaCerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    ymDiaCerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    ymDiaCerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # ymDiaCerebro.addanalyzer(PortfolioValue, _name='portfolio')

    # cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=False, riskfreerate=0.0)
    # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    # cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    # cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    # cerebro.addanalyzer(PortfolioValue, _name='portfolio')

    # Run optimization
    # results = ymDiaCerebro.run(maxcpus=1)  # Use 1 CPU to avoid threading issues with IQFeed
    results = ymDiaCerebro.run()
    backTestReturn = results[0]  # Single strategy instance
    showSimpleBackTestResult(backTestReturn)

    # Plot equity curve
    # ymDiaCerebro.plot(volume=False, style='candlestick', iplot=False)
    ymDiaCerebro.plot(volume=True, style='candlestick', iplot=False, numfigs = 1)
    plt.show()


    # cerebro.plot(
    #     style='candlestick',
    #     numfigs=1,
    #     width=0.6,
    #     height=8,
    #     dpi=120,
    #     volume=True,
    #     barup='blue',
    #     bardown='orange',
    #     subplots=True
    # )

    return results, backTestReturn

def showOptBackTestResult(best_optReturn):
    if best_optReturn:
        # Extract statistics
        initial_cash = 10000
        # strategy = best_optReturn.strat


        # final_value1 = best_optReturn.strat.broker.getvalue()
        portfolio_dict = getattr(best_optReturn.analyzers, 'portfolio', None)
        # final_value = best_optReturn.broker.getvalue()
        if portfolio_dict:
            portfolio_dict = portfolio_dict.get_analysis()
            net_profit = portfolio_dict.get('net_profit', 'None')
            final_value = portfolio_dict.get('final_value', 'None')
        else:
            returns_dict = best_optReturn.analyzers.returns.get_analysis()
            final_value = initial_cash * (1 + returns_dict.get('rnorm100', 0) / 100) if returns_dict else 'None'
            net_profit = final_value - initial_cash if final_value != 'None' else 'None'

        # final_value = strategy.broker.getvalue()  # Correct access
        # final_value = best_optReturn.getvalue()
        # net_profit = final_value - initial_cash
        sharpe_ratio = best_optReturn.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')
        max_drawdown = best_optReturn.analyzers.drawdown.get_analysis()['max']['drawdown']
        trade_analysis = best_optReturn.analyzers.trades.get_analysis()
        num_trades = trade_analysis['total']['total']
        # wins = trade_analysis['won']['total']
        wins = trade_analysis['won']['total'] if trade_analysis and 'won' in trade_analysis else 0
        # losses = trade_analysis['lost']['total']
        # if trade_analysis and 'lost' in trade_analysis:
        losses = trade_analysis['lost']['total'] if trade_analysis and 'lost' in trade_analysis else 0
        win_ratio = wins / num_trades if num_trades > 0 else 0

        max_trade_drawdown = max([trade.get('pnl', {}).get('max', 0) for trade in trade_analysis.get('trades', [])],
                                 default=0)
        annual_returns = best_optReturn.analyzers.annual_return.get_analysis()
        avg_annual_return = np.mean(list(annual_returns.values())) if annual_returns else 0
        pct_annual_return = avg_annual_return * 100

        # Print statistics
        print(f"\nBest Strategy Parameters:")
        print(f"Percentile Days: {best_optReturn.params.percentile_days}")
        # print(f"Stop Loss Multiplier (sl1): {best_optReturn.params.sl1}")
        # print(f"Stop Profit Multiplier (pf1): {best_optReturn.params.pf1}")
        #
        print(f"\nPerformance Statistics:")
        print(f"Net Profit: ${net_profit:.2f}")
        print(f"Annual Return: {avg_annual_return:.4f}")
        print(f"% Annual Return: {pct_annual_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio if sharpe_ratio != 'N/A' else 'N/A'}")
        print(f"Number of Trades: {num_trades}")
        print(f"Max System Drawdown: {max_drawdown:.2f}%")
        print(f"Max Trade Drawdown: {max_trade_drawdown:.2f}")
        print(f"Number of Wins: {wins}")
        print(f"Number of Losses: {losses}")
        print(f"Win Ratio: {win_ratio:.2f}")


        # # Print statistics
        # print(f"\nBest Strategy Parameters:")
        # print(f"Percentile Days: {best_strat.params.percentile_days}")
        # print(f"Stop Loss Multiplier (sl1): {best_strat.params.sl1}")
        # print(f"Stop Profit Multiplier (pf1): {best_strat.params.pf1}")
        # print(f"\nPerformance Statistics:")
        # print(f"Net Profit: ${net_profit:.2f}")
        # print(f"Annual Return: {avg_annual_return:.4f}")
        # print(f"% Annual Return: {pct_annual_return:.2f}%")
        # print(f"Sharpe Ratio: {sharpe_ratio if sharpe_ratio != 'N/A' else 'N/A'}")
        # print(f"Number of Trades: {num_trades}")
        # print(f"Max System Drawdown: {max_drawdown:.2f}%")
        # print(f"Max Trade Drawdown: {max_trade_drawdown:.2f}")
        # print(f"Number of Wins: {wins}")
        # print(f"Number of Losses: {losses}")
        # print(f"Win Ratio: {win_ratio:.2f}")
    else:
        print("No valid backtest results obtained.")


def showSimpleBackTestResult(strategy):
    # Debug analyzer names
    analyzer_names = [name for name in dir(strategy.analyzers) if not name.startswith('_')]
    print(f"Analyzer names: {analyzer_names}")

    # Parameters
    params = strategy.params._getkwargs()

    # Analyzer outputs
    sharpe_dict = strategy.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_dict.get('sharperatio', 'None') if sharpe_dict else 'None'
    annual_return_dict = strategy.analyzers.annual_return.get_analysis()
    annual_return = np.mean(list(annual_return_dict.values())) if annual_return_dict else 'None'
    drawdown_dict = strategy.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown_dict['max']['drawdown'] if drawdown_dict and 'max' in drawdown_dict else 'None'
    returns_dict = strategy.analyzers.returns.get_analysis()
    trade_analysis = strategy.analyzers.trades.get_analysis()

    # Portfolio value
    initial_cash = 1000000
    # portfolio_dict = getattr(strategy.analyzers, 'portfolio', None)
    # if portfolio_dict:
    #     portfolio_dict = portfolio_dict.get_analysis()
    #     final_value = portfolio_dict.get('final_value', 'None')
    #     net_profit = portfolio_dict.get('net_profit', 'None')
    # else:
    #     print("Portfolio analyzer missing, using Returns analyzer for net_profit")
    #     final_value = initial_cash * (1 + returns_dict.get('rnorm100', 0) / 100) if returns_dict else 'None'
    #     net_profit = final_value - initial_cash if final_value != 'None' else 'None'

    start_cash = strategy.broker.startingcash
    end_value = strategy.broker.getvalue()
    net_profit = end_value - start_cash
    final_value = end_value

    # Trade stats
    num_trades = trade_analysis['total']['total'] if trade_analysis and 'total' in trade_analysis else 0
    wins = trade_analysis['won']['total'] if trade_analysis and 'won' in trade_analysis else 0
    losses = trade_analysis['lost']['total'] if trade_analysis and 'lost' in trade_analysis else 0
    win_ratio = (wins / num_trades * 100) if num_trades > 0 else 0

    # Print order details
    print("\nOrder Details:")
    if strategy.order_history:
        for order in strategy.order_history:
            print(f"Order Type: {order['type']}, Datetime: {order['datetime'].strftime('%Y-%m-%d')}, "
                  f"Price: ${order['price']:.2f}, Size: {order['size']}")
    else:
        print("No orders executed.")

    # Print results
    print(f"\nBacktest Results: days={params['percentile_days']}")
    print(f"Net Profit: ${net_profit:.2f}" if net_profit != 'None' else "Net Profit: None")
    print(f"Final Value: ${final_value:.2f}" if final_value != 'None' else "Final Value: None")
    print(f"Annual Return: {annual_return}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Max Drawdown: {max_drawdown}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Winning Trades: {wins}")
    print(f"Losing Trades: {losses}")
    print(f"Win Ratio: {win_ratio:.2f}%")
    print(f"Returns: {returns_dict}")


def main():
    # Fetch data
    # start_date = '2011-01-01'
    # end_date = '2025-04-30'

    start_dt1 = datetime(2020, 1, 1)
    # end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
    end_dt1 = datetime(2025, 5, 31)

    start_trade_dt1 = datetime(2021, 2, 1)

    # Fetch and prepare data
    dia_data, ym_data, combine_data = prepare_data(start_dt1, end_dt1)

    #
    inDataList = [bt.feeds.PandasData(dataname=dia_data), bt.feeds.PandasData(dataname=ym_data)]



    # backTest type
    # 1: optimization
    # 2: Simple backtest
    # 3: walkforward test (moving window)
    backTestType = 2

    if backTestType==1:
        results, best_optReturn = performBacktestOptimization (inDataList, 'sharpe_ratio', True)
        # showOptBackTestResult(best_optReturn)
    elif backTestType==2:
        results, best_optReturn = performSimpleBacktest(start_trade_dt1, inDataList, 'sharpe_ratio', True)
        # showSimpleBackTestResult(best_optReturn)

    # # Create Bokeh figure
    # p = figure(
    #     x_axis_type="datetime",
    #     title="YM vs DIA",
    #     height=400,
    #     width=800,
    #     x_axis_label='Date',
    #     y_axis_label='Price (USD)'
    # )
    #
    # # Plot close and SMA
    # # p.line(dia_data['datetime'], dia_data['close'], color='blue', legend_label='DIAClose', line_width=2)
    # # p.line(ym_data['datetime'], ym_data['close'], color='orange', legend_label='YMclose', line_width=2)
    # # p.line('date', 'value1', source=source1, legend_label="Value 1", line_width=2, color="blue")
    # # p.line('date', 'value2', source=source2, legend_label="Value 2", line_width=2, color="red")
    #
    #
    # print (11)
    # dia_data['datetime'] = dia_data.index
    # ym_data['datetime'] = ym_data.index
    #
    # print (dia_data.head)
    # print(ym_data.head)
    #
    #
    # # Plot close
    # p.line(dia_data['datetime'], dia_data['Close'], color='blue', legend_label='DIA Close', line_width=2)
    # p.line(ym_data['datetime'], ym_data['Close'], color='orange', legend_label='YM close', line_width=2)
    #
    # # Customize
    # p.legend.location = "top_left"
    # p.legend.click_policy = "hide"
    # p.grid.grid_line_alpha = 0.3
    #
    #
    # show (p)


if __name__ == "__main__":
    main()