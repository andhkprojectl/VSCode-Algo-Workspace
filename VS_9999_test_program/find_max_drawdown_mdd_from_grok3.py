import pandas as pd
import numpy as np

# Load data
# df = pd.read_csv('C:/Project/ProjectLife/Feasible Study/calcuate_statistics/aa2.csv')
df = pd.read_csv('C:/Project/ProjectLife/Feasible Study/calcuate_statistics/aa2_sort_sell_datetime.csv')

# Parse dates using pandas
df['Buy Date Time'] = pd.to_datetime(df['Buy Date Time'], format='%d/%m/%Y %H:%M')
df['Sell Date Time'] = pd.to_datetime(df['Sell Date Time'], format='%d/%m/%Y %H:%M')

# Sort by Sell Date Time to ensure correct trade sequence
df = df.sort_values(by='Sell Date Time')

# Calculate equity curve
initial_capital = 40000
df['Equity'] = initial_capital + df['Profit'].cumsum()
df['Peak'] = df['Equity'].cummax()
df['Drawdown'] = df['Peak'] - df['Equity']

# Max Drawdown and Max Drawdown %
max_drawdown = df['Drawdown'].max()
max_drawdown_idx = df['Drawdown'].idxmax()
peak_equity = df['Peak'].iloc[max_drawdown_idx]
trough_equity = df['Equity'].iloc[max_drawdown_idx]
max_drawdown_percent = (max_drawdown / peak_equity) * 100

# Annual Return
total_profit = df['Profit'].sum()
period_days = (df['Sell Date Time'].max() - df['Sell Date Time'].min()).days
years = period_days / 365.25
annual_return = total_profit / years

# Annual Return %
final_equity = initial_capital + total_profit
annual_return_percent = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

# Exposure %
trade_days = sum((df['Sell Date Time'] - df['Buy Date Time']).dt.total_seconds() / (24 * 3600))
exposure_percent = (trade_days / period_days) * 100

# Output
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Final Equity: ${final_equity:,.2f}")
print(f"Max Drawdown: ${max_drawdown:,.2f}")
print(f"Max Drawdown %: {max_drawdown_percent:.2f}%")
print(f"Peak Equity: ${peak_equity:,.2f} at {df['Sell Date Time'].iloc[max_drawdown_idx]}")
print(f"Trough Equity: ${trough_equity:,.2f} at {df['Sell Date Time'].iloc[max_drawdown_idx]}")
print(f"Annual Return: ${annual_return:,.2f}")
print(f"Annual Return %: {annual_return_percent:.2f}%")
print(f"Exposure %: {exposure_percent:.2f}%")