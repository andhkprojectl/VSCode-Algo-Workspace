import pyiqfeed as iq
from pyiqfeed import HistoryConn, ConnConnector
import numpy as np
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt

# Step 1: Set up IQFeed connection
# from localconfig.passwords import dtn_product_id, dtn_login, dtn_password

dtn_product_id="1"
dtn_login="2"
dtn_password="3"

hist_conn = HistoryConn(name="pyiqfeed-Backtest")

with ConnConnector([hist_conn], product=dtn_product_id, login=dtn_login, password=dtn_password) as connector:
    bgn_dt = date(2015, 1, 1)
    end_dt = date(2025, 3, 31)

    # Fetch DIA data
    try:
        dia_data = hist_conn.request_daily_data_for_dates("DIA", bgn_dt, end_dt, ascend=True)
        dia_df = pd.DataFrame(dia_data)
        dia_df['date'] = dia_df['date'].astype(str).str.decode('utf-8')
        dia_df['date'] = pd.to_datetime(dia_df['date'], format='%Y-%m-%d')
        dia_df.set_index('date', inplace=True)
        dia_df.sort_index(inplace=True)
    except (iq.NoDataError, iq.UnauthorizedError) as err:
        print(f"Error fetching DIA data: {err}")
        exit()

    # Fetch @YM#C data
    try:
        ym_data = hist_conn.request_daily_data_for_dates("@YM#C", bgn_dt, end_dt, ascend=True)
        ym_df = pd.DataFrame(ym_data)
        ym_df['date'] = ym_df['date'].astype(str).str.decode('utf-8')
        ym_df['date'] = pd.to_datetime(ym_df['date'], format='%Y-%m-%d')
        ym_df.set_index('date', inplace=True)
        ym_df.sort_index(inplace=True)
    except (iq.NoDataError, iq.UnauthorizedError) as err:
        print(f"Error fetching @YM#C data: {err}")
        exit()

    # Step 2: Align @YM#C data with DIA trading days
    trading_days = dia_df.index
    try:
        ym_close_trading = ym_df.loc[trading_days]['close']
    except KeyError as err:
        print(f"Error aligning data: {err}")
        exit()

    # Step 3: Calculate daily price differences for @YM#C
    delta_ym = ym_close_trading.diff()

    # Step 4: Initialize backtest variables
    initial_cash = 100000
    cash = initial_cash
    equity_list = []
    date_list = []
    daily_pnl = []

    for i in range(len(dia_df)):
        t = dia_df.index[i]
        if i < 100:
            pnl_today = 0
        else:
            window = delta_ym.iloc[i-100:i]
            if len(window.dropna()) == 100:
                P90 = window.quantile(0.9)
                R_t1 = delta_ym.iloc[i-1]
                if pd.notna(R_t1) and R_t1 > P90:
                    open_price = dia_df['open'].iloc[i]
                    close_price = dia_df['close'].iloc[i]
                    shares = int(cash // open_price)
                    if shares > 0:
                        cost = shares * open_price
                        revenue = shares * close_price
                        pnl_today = revenue - cost
                        cash += pnl_today
                    else:
                        pnl_today = 0
                else:
                    pnl_today = 0
            else:
                pnl_today = 0
        equity_list.append(cash)
        date_list.append(t)
        daily_pnl.append(pnl_today)

    # Step 5: Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(date_list, equity_list, label='Equity')
    plt.title('Equity Curve (2015-01-01 to 2025-03-31)')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig('equity_curve.png')
    plt.close()

    # Step 6: Calculate statistics
    final_equity = equity_list[-1]
    net_return = (final_equity / initial_cash) - 1
    years = (end_dt - bgn_dt).days / 365.25
    annual_return = (final_equity / initial_cash) ** (1 / years) - 1

    daily_returns = [(equity_list[j] / equity_list[j-1]) - 1 for j in range(1, len(equity_list))]
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns) if std_daily_return != 0 else 1e-10
    sharpe_ratio = mean_daily_return / std_daily_return * np.sqrt(252)

    print(f"Net Return: {net_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")