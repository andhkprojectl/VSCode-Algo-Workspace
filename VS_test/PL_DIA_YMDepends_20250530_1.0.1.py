#
#  1.0.1
#
# Trade Dia according to #YM#C
# - Similar to 1.0.0
# - use bt.Cerebro() (return optReturn Object (no broker, no full strategy instances), assume optreturn = true
# - use optStragey
# - instead of bt.Cerebro(optreturn = False)
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
from pyiqfeed import HistoryConn
import logging
import backtrader as bt
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout
from bokeh.layouts import column


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
def fetch_daily_data_00930_to_1600(conn, symbol1, start_dt, end_dt):
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
            interval_len=1800,  # 30 minutes (1800 seconds)
            interval_type="s",  # Seconds
            bgn_prd=start_dt,
            end_prd=end_dt,
            bgn_flt=datetime.strptime("09:30", "%H:%M").time(),
            end_flt=datetime.strptime("16:00", "%H:%M").time(),
            ascend=True,  # Oldest to latest
            max_bars=None,  # Fetch all available bars
            timeout=None
        )
        print ("data output")
        print(data)
        # Print header names
        if data is not None and len(data) > 0:
            header_names = data.dtype.names
            print("Data Header Names:", header_names)
        else:
            print("No data returned")

        # Convert to list of dictionaries
        records = [
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

        all_data.extend(records)
        df = pd.DataFrame(all_data)

        # df['datetime'] = pd.to_datetime(df['Date1'] + ' ' + df['Time1'])
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str))
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='%Y-%m-%d %H:%M:%S')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='mixed', errors='coerce')
        # df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['Time1'].astype(str), format='ISO8601')

        # Convert Time1 (nanoseconds) to time string
        df['TimeSeconds'] = df['Time1'] / 1_000_000_000  # Convert to seconds
        df['TimeStr'] = pd.to_timedelta(df['TimeSeconds'], unit='s').apply(
            lambda x: x.components.hours * 3600 + x.components.minutes * 60 + x.components.seconds).apply(
            lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}")

        # Combine Date1 and TimeStr into datetime
        df['datetime'] = pd.to_datetime(df['Date1'].astype(str) + ' ' + df['TimeStr'], format='%Y-%m-%d %H:%M:%S',
                                        errors='coerce')

        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        #
        print(df.columns)
        print(df.head())
        #

        daily = df.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'numTrade' : 'sum'

        })


        # Drop NaNs (non-trading days)
        daily.dropna(inplace=True)

        # Show sample output
        print(daily.head())

        # Save to CSV (optional)
        # if (write_output_2_file):
        #     daily.to_csv("c:/tmp/DIA_RTH_Daily_OHLC.csv")

        # conn.disconnect()

        return daily

    except Exception as e:
        # print(f"Error fetching chunk {current_start} to {chunk_end}: {e}")
        print("Exception:")
        print(e)
    finally:
        if (isConnCreateHere):
            conn.disconnect()


def prepare_data(start_dt1, end_dt1):
    ym_data = pd.DataFrame()
    dia_data = pd.DataFrame()
    combine_data = pd.DataFrame()
    try:
        conn = HistoryConn(name="history")
        conn.connect()
        ym_data = fetch_daily_data(conn, "@YM#C", start_dt1, end_dt1)
        dia_data = fetch_daily_data_00930_to_1600(conn, "DIA", start_dt1, end_dt1)
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
        # if (conn.connected()):
        conn.disconnect()
    return dia_data, ym_data, combine_data


# Define the trading strategy
class ymDiaStrategy1(bt.Strategy):
    params = dict(
        percentile_days=160,
        percentile_th = 140
        # sl1=3,
        # pf1=3,
    )
    #self.data0: dia_data
    #self.data1: ym_data

    def __init__(self):
        self.atr15 = bt.indicators.ATR(self.data0, period=15)
        self.trade_day = None
        self.buy_order = None
        self.sell_next_open = False

    def next(self):
        # self.buy_order = self.buy(data=self.data0, size=1, exectype=bt.Order.Market)
        tradePrice1 = self.data0.open[0]
        # for backtest, stop limit price = trade price
        stopLimitPrice = tradePrice1
        positionSize1 = 10
        # assume
        # 1) stop loss and stop profit are limit price
        # 2) for testing, stop loss and stop profit same as limit price
        stopLossPrice = tradePrice1 - defaultParams['SL1'] * self.atr15[0]
        stopLossLimitPrice = stopLossPrice  # same for testing only
        stopProfitPrice = tradePrice1 + defaultParams['PF1'] * self.atr15[0]
        stopProfitLimitPrice = stopProfitPrice

        if len(self.data1) > self.p.percentile_days:

            # find return of yesterday
            returns = [self.data1.close[i] - self.data1.close[i - 1] for i in range(-self.p.percentile_days -1, -1)]
            percentile_90 = np.percentile(returns, 90)
            yesterday_return = self.data1.close[-1] - self.data1.close[-2]
            if (yesterday_return >= percentile_90
                and yesterday_return >= self.p.percentile_th
            ):
                print ("Order Bought 1")
                main_order = self.buy(
                    data=self.data0,
                    size=positionSize1,
                    price=tradePrice1,
                    plimit=stopLimitPrice,
                    exectype=bt.Order.Limit,
                    transmit=False
                )
                stop_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=stopLossPrice,
                    plimit=stopLossLimitPrice,
                    parent=main_order,
                    transmit=False
                )
                take_profit_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=stopProfitPrice,
                    plimit=stopProfitLimitPrice,
                    exectype=bt.Order.Limit,
                    parent=main_order,
                    transmit=True
                )
                self.buyer_order = main_order
                self.buy_order = self.buy()
                self.buyer =  self.buy()
                # self.buy_order = self.buy(data=self.data0, size=positionSize1, price=tradePrice1, plimit=stopLimitPrice, exectype=bt.Order.Limit)
                self.buy_order = self.buy()
                self.sell_next_open = True  # 在下一根K線開盤賣出
                self.trade_day = len(self)
                a1 = self.position
                a2 = len (self)
                a3 = self.trade_day


            # elif self.position and len(self) == self.trade_day + params1.SELL_NUM_DAYS:
            # if self.position and len(self) == self.trade_day + params1.SELL_NUM_DAYS:
            # if self.position and len(self) >= 1:
            elif self.position and len(self) == 1:
                print("Order sold 1")
                self.sell_order = self.sell(
                    data=self.data0,
                    size=positionSize1,
                    price=tradePrice1,
                    plimit=stopLimitPrice,
                    exectype=bt.Order.Limit,
                    transmit=True
                )
                self.sell(exectype=bt.Order.Market)
                self.sell_next_open = False
                print("Order sold")

    def log(self, txt):
        """Log messages with timestamp"""
        dt = self.datas[0].datetime.datetime(0)
        print(f'[{dt}]: {txt}')

    def notify_order(self, order):
        a22 = order.status
        if order.status in [order.Completed]:
            print("Order Completed")
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')
                self.position_size = order.executed.size
            elif order.issell():
                reason = "Take-Profit" if order.exectype == bt.Order.Limit else "Trailing Stop"
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Reason: {reason}')
                self.position_size = 0
            elif order.status in [order.Completed, order.Canceled, order.Rejected]:
                self.buy_order = None  # 無論成功或失敗都清除掛單
                self.order = None
            # self.order = None
        elif order.status in [order.Close]:
            print("Order Close")


    def notify_trade(self, trade):
        print("Order notified")
        if trade.isclosed:
            self.log(f'TRADE CLOSED, Gross P/L: {trade.pnl:.2f}, Net P/L: {trade.pnlcomm:.2f}')




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


def performSimpleBacktest(inDataList, optTarget, isPlot):
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
        percentile_days=50
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

    start_dt1 = datetime(2023, 1, 1)
    # end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
    end_dt1 = datetime(2025, 4, 30)
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
        results, best_optReturn = performSimpleBacktest(inDataList, 'sharpe_ratio', True)
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