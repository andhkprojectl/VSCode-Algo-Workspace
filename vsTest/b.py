import pandas as pd
import numpy as np
from datetime import datetime
import math

# Parse the provided data into a DataFrame
# Read data from aa2.xlsx
df = pd.read_excel('C:/Project/ProjectLife/Feasible Study/calcuate_statistics/aa2.xlsx', sheet_name='Sheet2')


# Convert date columns to datetime
df['Buy Date Time'] = pd.to_datetime(df['Buy Date Time'])
df['Sell Date Time'] = pd.to_datetime(df['Sell Date Time'])

# Sort by buy date (oldest first)
df = df.sort_values('Buy Date Time').reset_index(drop=True)

# Initial capital
initial_capital = 40000

# 1. Calculate basic metrics
num_trades = len(df)
winning_trades = len(df[df['Profit'] >= 0])
losing_trades = len(df[df['Profit'] < 0])
win_ratio = winning_trades / num_trades
net_profit = df['Profit'].sum()
ending_capital = initial_capital + net_profit
net_profit_pct = (net_profit / initial_capital) * 100

# 2. Calculate trade-based drawdown metrics
trade_costs = df['Buy price'] * df['Shares']
trade_returns = df['Profit'] / trade_costs

# Max trade drawdown (dollar)
max_trade_drawdown = abs(df['Profit'].min()) if losing_trades > 0 else 0

# Max trade % drawdown
if losing_trades > 0:
    losing_mask = df['Profit'] < 0
    max_trade_pct_drawdown = (abs(df.loc[losing_mask, 'Profit']) / trade_costs[losing_mask]).max() * 100
else:
    max_trade_pct_drawdown = 0

# 3. Calculate equity curve and system drawdown
equity = [initial_capital]
dates = [df['Buy Date Time'].min()]  # Start with first buy date

for i, row in df.iterrows():
    equity.append(equity[-1] + row['Profit'])
    dates.append(row['Sell Date Time'])

# Calculate drawdowns
peak = equity[0]
max_system_drawdown = 0
max_system_pct_drawdown = 0

for value in equity:
    if value > peak:
        peak = value

    drawdown = peak - value
    drawdown_pct = (drawdown / peak) * 100

    if drawdown > max_system_drawdown:
        max_system_drawdown = drawdown

    if drawdown_pct > max_system_pct_drawdown:
        max_system_pct_drawdown = drawdown_pct

# 4. Calculate annual return
start_date = dates[0]
end_date = dates[-1]
total_days = (end_date - start_date).days
total_years = total_days / 365.25
cagr = ((ending_capital / initial_capital) ** (1 / total_years) - 1) * 100

# 5. Calculate Sharpe ratio (risk-free rate = 3%)
risk_free_rate = 0.03
excess_returns = []

for i, row in df.iterrows():
    holding_days = (row['Sell Date Time'] - row['Buy Date Time']).days
    trade_return = row['Profit'] / (row['Buy price'] * row['Shares'])
    # rf_return = risk_free_rate * (holding_days / 365.25)
    rf_return = risk_free_rate * (holding_days / 252.00)
    excess_returns.append(trade_return - rf_return)

if len(excess_returns) > 1:
    sharpe_ratio = (np.mean(excess_returns) / np.std(excess_returns)) * math.sqrt(252)
else:
    sharpe_ratio = 0

# 6. Calculate other metrics
profit_factor = abs(df[df['Profit'] >= 0]['Profit'].sum() / df[df['Profit'] < 0]['Profit'].sum())
standard_error = np.std(trade_returns) / math.sqrt(num_trades)

# 7. Calculate CAR/MaxDD and RAR/MaxDD
car_maxdd = (cagr / 100) / (max_system_pct_drawdown / 100) if max_system_pct_drawdown > 0 else 0
rar_maxdd = car_maxdd  # Using same calculation as CAR/MaxDD

# Compile results
results = {
    "Initial capital": f"${initial_capital:,.2f}",
    "Ending capital": f"${ending_capital:,.2f}",
    "Net Profit": f"${net_profit:,.2f}",
    "Net Profit %": f"{net_profit_pct:.2f}%",
    "Annual Return": f"{cagr:.2f}%",
    "Sharpe Ratio": f"{sharpe_ratio:.4f}",
    "Number of Trades": num_trades,
    "Winning Trades": winning_trades,
    "Losing Trades": losing_trades,
    "Win Ratio": f"{win_ratio:.2%}",
    "Max trade drawdown": f"${max_trade_drawdown:,.2f}",
    "Max trade % drawdown": f"{max_trade_pct_drawdown:.2f}%",
    "Max system drawdown": f"${max_system_drawdown:,.2f}",
    "Max system % drawdown": f"{max_system_pct_drawdown:.2f}%",
    "CAR/MaxDD": f"{car_maxdd:.4f}",
    "RAR/MaxDD": f"{rar_maxdd:.4f}",
    "Profit Factor": f"{profit_factor:.4f}",
    "Standard Error": f"{standard_error:.6f}",
    "Sharpe Ratio (annualized)": f"{sharpe_ratio:.4f}"
}

# Display results
for metric, value in results.items():
    print(f"{metric}: {value}")