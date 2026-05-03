import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import datetime as dt
from pyiqfeed import HistoryConn
import logging
from backtesting import Backtest, Strategy


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_PARAMS = {
    'percentile_period': 120,
    'percentile_threshold': 0.3,
    'atr_period': 15,
    'n1': 2.0,
    'n2': 1.0,
    'initial_capital': 100000,
    'in_sample_years': 1,
    'out_sample_months': 1,
    'LOOKBACK_DAYS' : 100,
    'PERCENTILE' : 90
}
# LOOKBACK_DAYS = 100
# PERCENTILE = 90



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
                "Date": bar["date"],
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
        df.set_index('Date', inplace=True)
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


class YMPercentileStrategy(Strategy):
    # lookback_days = LOOKBACK_DAYS
    # percentile = PERCENTILE
    params = DEFAULT_PARAMS.copy()
    lookback_days = params.get('LOOKBACK_DAYS')
    percentile = params.get('PERCENTILE')

    def init(self):
        # Load @YM#C data
        # self.ym_data = pd.read_csv(f"{DATA_DIR}/@YM#C_daily.csv", parse_dates=['datetime'], index_col='datetime')
        # self.ym_data = ym_data
        # self.ym_data = self.ym_data.loc[self.data.index]
        # self.ym_returns = self.ym_data['Daily_Return'].values
        print(1)
        # Seperate 2 DataFrame
        # Identify columns for A and B based on prefixes
        dia_colummns = [col for col in data.columns if col.startswith('D_')]
        ym_columns = [col for col in data.columns if col.startswith('Y_')]

        # Reconstruct DataFrame A by selecting A_ columns and removing prefix
        dia_data = combine_data[dia_colummns].copy()
        dia_data.columns = [col.replace('D_', '') for col in dia_data.columns]

        # Reconstruct DataFrame B by selecting B_ columns and removing prefix
        ym_data = combine_data[ym_columns].copy()
        ym_data.columns = [col.replace('Y_', '') for col in ym_data.columns]

    def next(self):
        # Skip if not enough data or no position
        if len(self.data) < self.lookback_days + 1:
            return

        # Get today's index
        today_idx = len(self.data) - 1

        # Calculate 90th percentile of past 100 days' @YM#C returns
        lookback_returns = self.ym_returns[today_idx - self.lookback_days:today_idx]
        if len(lookback_returns) < self.lookback_days or np.isnan(lookback_returns).any():
            return
        percentile_value = np.percentile(lookback_returns, self.percentile)

        # Check today's @YM#C return
        today_ym_return = self.ym_returns[today_idx]
        if np.isnan(today_ym_return):
            return

        # Buy DIA next day at open if condition met
        if today_ym_return >= percentile_value and not self.position:
            self.buy()

        # Sell at next day's open if holding a position
        elif self.position:
            self.position.close()


def calculate_atr(data, period=15):
    """
    Calculate Average True Range (ATR) for the given period.
    """
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

#To be develop
def walk_forward_test(ym_data, dia_data, params):
    """
    Perform walk-forward test with given parameters.
    """
    start_date = pd.to_datetime('2011-01-01')
    end_date = pd.to_datetime('2025-04-30')
    current_date = start_date

    portfolio = pd.DataFrame(index=dia_data.index)
    portfolio['total'] = params['initial_capital']
    trades = []
    signal = 'hold'  # Initialize signal as 'hold'

    while current_date + timedelta(days=365 * params['in_sample_years']) <= pd.to_datetime('2025-03-31'):
        # Define in-sample and out-of-sample periods
        in_sample_end = current_date + timedelta(days=365 * params['in_sample_years'])
        out_sample_end = in_sample_end + pd.offsets.MonthEnd(params['out_sample_months'])

        if out_sample_end > end_date:
            out_sample_end = end_date

        # In-sample data
        in_sample_ym = ym_data[(ym_data.index >= current_date) & (ym_data.index < in_sample_end)]
        in_sample_dia = dia_data[(dia_data.index >= current_date) & (dia_data.index < in_sample_end)]

        # Out-of-sample data
        out_sample_ym = ym_data[(ym_data.index >= in_sample_end) & (ym_data.index < out_sample_end)]
        out_sample_dia = dia_data[(dia_data.index >= in_sample_end) & (dia_data.index < out_sample_end)]

        if len(out_sample_ym) < params['percentile_period'] or len(out_sample_dia) < params['atr_period']:
            logger.warning(f"Insufficient data for period starting {current_date}. Skipping.")
            current_date = out_sample_end
            continue

        # Calculate returns and percentile for @YM#C
        out_sample_ym['return'] = out_sample_ym['close'].pct_change()
        out_sample_ym['percentile'] = out_sample_ym['return'].rolling(params['percentile_period']).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )

        # Calculate ATR15 for DIA
        out_sample_dia['atr15'] = calculate_atr(out_sample_dia, params['atr_period'])
        out_sample_dia['next_open'] = out_sample_dia['open'].shift(-1)

        # Trading logic
        position = 0
        entry_price = 0
        atr_at_entry = 0
        entry_date = None

        for i in range(1, len(out_sample_dia)):
            date = out_sample_dia.index[i]
            if date not in out_sample_ym.index:
                # If no data for YM on this day, carry forward the previous signal
                portfolio.loc[date, 'total'] = portfolio.loc[out_sample_dia.index[i - 1], 'total']
                continue

            ym_idx = out_sample_ym.index.get_loc(date)
            if pd.isna(out_sample_ym['percentile'].iloc[ym_idx]):
                # If percentile is NaN (insufficient data), carry forward previous signal
                portfolio.loc[date, 'total'] = portfolio.loc[out_sample_dia.index[i - 1], 'total']
                continue

            # Generate signal based on percentile
            current_percentile = out_sample_ym['percentile'].iloc[ym_idx]
            if current_percentile < params['percentile_threshold']:
                signal = 'buy'
            else:
                signal = signal  # Carry forward previous signal if no new condition met

            if signal == 'buy' and position == 0:
                entry_price = out_sample_dia['next_open'].iloc[i - 1]
                atr_at_entry = out_sample_dia['atr15'].iloc[i - 1]
                position = 1
                entry_date = date
                trade = {'entry_date': entry_date, 'entry_price': entry_price}
            elif position == 1:
                current_price = out_sample_dia['close'].iloc[i]
                if current_price > entry_price + params['n1'] * atr_at_entry or current_price < entry_price - params[
                    'n2'] * atr_at_entry:
                    position = 0
                    trade['exit_date'] = date
                    trade['exit_price'] = current_price
                    trade['profit'] = current_price - entry_price
                    trades.append(trade)
                    portfolio.loc[date, 'total'] += trade['profit']
                else:
                    portfolio.loc[date, 'total'] = portfolio.loc[out_sample_dia.index[i - 1], 'total']
            else:
                portfolio.loc[date, 'total'] = portfolio.loc[out_sample_dia.index[i - 1], 'total']

        # Handle unclosed trades
        if position == 1:
            last_price = out_sample_dia['close'].iloc[-1]
            trade['exit_date'] = out_sample_dia.index[-1]
            trade['exit_price'] = last_price
            trade['profit'] = last_price - entry_price
            trades.append(trade)
            portfolio.loc[out_sample_dia.index[-1], 'total'] += trade['profit']

        current_date = out_sample_end

    return portfolio, trades


def calculate_metrics(portfolio, trades, params):
    """
    Calculate performance metrics.
    """
    initial_capital = params['initial_capital']
    # portfolio['total'] = portfolio['total'].fillna(method='ffill')
    portfolio['total'] = portfolio['total'].ffill()

    net_profit = portfolio['total'].iloc[-1] - initial_capital
    days = (portfolio.index[-1] - portfolio.index[0]).days
    years = days / 365.25
    annual_return = (portfolio['total'].iloc[-1] / initial_capital) ** (1 / years) - 1
    percent_annual_return = annual_return * 100

    returns = portfolio['total'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0

    rolling_max = portfolio['total'].cummax()
    drawdowns = (portfolio['total'] - rolling_max) / rolling_max
    max_system_drawdown = drawdowns.min() * initial_capital
    percent_max_system_drawdown = drawdowns.min() * 100

    trade_profits = [trade['profit'] for trade in trades]
    num_trades = len(trades)
    win_rate = len([p for p in trade_profits if p > 0]) / num_trades if num_trades > 0 else 0
    average_trade_duration = np.mean(
        [(trade['exit_date'] - trade['entry_date']).days for trade in trades]) if num_trades > 0 else 0
    profit_factor = sum([p for p in trade_profits if p > 0]) / abs(sum([p for p in trade_profits if p < 0])) if sum(
        [p for p in trade_profits if p < 0]) != 0 else np.inf

    max_trade_drawdown = min(trade_profits) if trade_profits else 0
    percent_max_trade_drawdown = (max_trade_drawdown / trades[trade_profits.index(min(trade_profits))][
        'entry_price']) * 100 if trade_profits else 0

    return {
        'Net Profit': net_profit,
        'Annual Return': annual_return,
        '% Annual Return': percent_annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max System Drawdown': max_system_drawdown,
        '% Max System Drawdown': percent_max_system_drawdown,
        'Max Trade Drawdown': max_trade_drawdown,
        '% Max Trade Drawdown': percent_max_trade_drawdown,
        'Number of Trades': num_trades,
        'Win Rate': win_rate,
        'Average Trade Duration (days)': average_trade_duration,
        'Profit Factor': profit_factor
    }


def plot_equity_curve(portfolio):
    """
    Plot equity curve with drawdowns.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio.index, portfolio['total'], label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    plt.close()


def main():
    # Fetch data
    # start_date = '2011-01-01'
    # end_date = '2025-04-30'

    start_dt1 = datetime(2025, 4, 1)
    # end_dt1 = datetime(2025, 4, 30, 23, 59, 59)
    end_dt1 = datetime(2025, 4, 30)
    # Fetch and prepare data
    dia_data, ym_data, combine_data = prepare_data(start_dt1, end_dt1)


    # Run backtest
    bt = Backtest(combine_data, YMPercentileStrategy, cash=10_000, commission=.002, exclusive_orders=True)
    stats = bt.run()

    # Extract and print statistics
    net_profit = stats['Equity Final [$]'] - stats['Equity Initial [$]']
    annual_return = stats['Return (Ann.) [%]']
    percent_annual_return = stats['Return (Ann.) [%]']
    sharpe_ratio = stats['Sharpe Ratio']
    num_trades = stats['# Trades']
    max_system_drawdown = stats['Max. Drawdown [%]']
    max_trade_drawdown = stats['Worst Trade [%]']
    win_rate = stats['Win Rate [%]']
    num_wins = int(num_trades * win_rate / 100)
    num_losses = num_trades - num_wins

    print(f"Backtest Statistics:")
    print(f"Net Profit: ${net_profit:.2f}")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"% Annual Return: {percent_annual_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Max System Drawdown: {max_system_drawdown:.2f}%")
    print(f"Max Trade Drawdown: {max_trade_drawdown:.2f}%")
    print(f"Number of Wins: {num_wins}")
    print(f"Number of Losses: {num_losses}")
    print(f"Win Ratio: {win_rate:.2f}%")

    # Plot equity curve
    bt.plot()




    # Run walk-forward test
    # params = DEFAULT_PARAMS.copy()
    # portfolio, trades = walk_forward_test(ym_data, dia_data, params)
    #
    # # Calculate metrics
    # metrics = calculate_metrics(portfolio, trades, params)
    #
    # # Display results
    # for key, value in metrics.items():
    #     print(f"{key}: {value:.2f}")
    #
    # # Plot equity curve
    # plot_equity_curve(portfolio)


if __name__ == "__main__":
    main()