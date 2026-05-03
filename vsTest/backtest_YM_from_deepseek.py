import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyiqfeed import PassiveIQFeed
from datetime import datetime, timedelta

# ======================
# IQFeed Configuration
# ======================
feed = PassiveIQFeed()
feed.connect()


# ======================
# Data Retrieval
# ======================
def get_iqfeed_data(symbol, start, end, interval='1d', session='24x5'):
    """Retrieve historical data from IQFeed"""
    data = feed.request_historical_data(
        symbol=symbol,
        begin_date=start.strftime('%Y%m%d'),
        end_date=end.strftime('%Y%m%d'),
        interval=interval,
        interval_type='d' if interval == '1d' else 's',
        max_bars=10000,
        session=session
    )
    return pd.DataFrame(data)


# Get data
start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 3, 31)

# DIA (NYSE hours)
dia = get_iqfeed_data('DIA', start_date, end_date, session='NYSE')
dia = dia[['Timestamp', 'Open', 'High', 'Low', 'Close']].set_index('Timestamp')

# YM Continuous Contract (24-hour session)
ym = get_iqfeed_data('@YM#C', start_date, end_date, session='24x5')
ym = ym[['Timestamp', 'Close']].rename(columns={'Close': 'YM_Close'}).set_index('Timestamp')

# ======================
# Data Processing
# ======================
# Merge datasets
merged = pd.merge(dia, ym, how='inner', left_index=True, right_index=True)

# Calculate YM returns
merged['YM_Return'] = merged['YM_Close'].pct_change()

# Calculate rolling 90th percentile
merged['YM_90th'] = merged['YM_Return'].rolling(100).quantile(0.9)

# Generate signals
merged['Signal'] = np.where(merged['YM_Return'] > merged['YM_90th'], 1, 0)
merged['Signal'] = merged['Signal'].shift()  # Trade next day

# ======================
# Backtest Engine
# ======================
equity = 100000  # Starting capital
position_size = 10000  # Per trade size
commission = 1.0  # Per trade commission
equity_curve = []
trade_log = []

for i in range(1, len(merged)):
    current_date = merged.index[i]

    # Get prices
    dia_open = merged.iloc[i]['Open']
    dia_close = merged.iloc[i]['Close']

    # Check signal
    if merged.iloc[i - 1]['Signal'] == 1:
        # Buy at open
        shares = position_size / dia_open
        equity -= position_size
        equity -= commission

        # Sell at close
        equity += shares * dia_close
        equity -= commission

        trade_log.append({
            'Date': current_date,
            'Action': 'BUY',
            'Price': dia_open,
            'Shares': shares,
            'Equity': equity
        })

    equity_curve.append(equity)

# Create equity dataframe
results = pd.DataFrame({
    'Date': merged.index[1:],
    'Equity': equity_curve
}).set_index('Date')


# ======================
# Performance Statistics
# ======================
def calculate_stats(equity_series):
    returns = equity_series.pct_change().dropna()

    stats = {
        'Net Return': equity_series.iloc[-1] / equity_series.iloc[0] - 1,
        'Annual Return %': (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(equity_series)) - 1,
        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252),
        'Max Drawdown': (equity_series / equity_series.cummax() - 1).min(),
        'Number of Trades': len(trade_log)
    }
    return stats


stats = calculate_stats(results['Equity'])

# ======================
# Output Results
# ======================
print(f"\nBacktest Results ({len(results)} trading days)")
print("=====================================")
for k, v in stats.items():
    print(f"{k:>20}: {v:>10.2f}")

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(results['Equity'])
plt.title('Strategy Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity ($)')
plt.grid(True)
plt.show()

feed.disconnect()