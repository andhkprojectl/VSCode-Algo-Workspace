import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyiqfeed as iq
from pyiqfeed import HistoryConn, ConnConnector

# ─── CONFIG ────────────────────────────────────────────────────────────────
START_DATE = datetime.datetime(2015, 1, 1)
END_DATE   = datetime.datetime(2025, 3, 31)
# Fetch enough days to cover the range
NUM_DAYS = (END_DATE - START_DATE).days + 50

# ─── 1) FETCH DAILY @YM#C ───────────────────────────────────────────────────
hist_ym = HistoryConn(name="hist-YM")
with ConnConnector([hist_ym]):
    # request_daily_data(symbol, num_days) :contentReference[oaicite:0]{index=0}
    ym_array = hist_ym.request_daily_data("@YM#C", NUM_DAYS)

# Convert to DataFrame
ym = pd.DataFrame(ym_array)
# Example dtype.names might be ('date','open_p','high_p','low_p','close_p','prd_vlm',…)
# Adapt field names if different:
ym['date'] = pd.to_datetime(ym['date'])
ym.set_index('date', inplace=True)
ym = ym.sort_index()

# Clip to our date range
ym = ym.loc[START_DATE:END_DATE]

# ─── 2) FETCH DAILY DIA ─────────────────────────────────────────────────────
hist_dia = HistoryConn(name="hist-DIA")
with ConnConnector([hist_dia]):
    dia_array = hist_dia.request_daily_data("DIA", NUM_DAYS)

dia = pd.DataFrame(dia_array)
dia['date'] = pd.to_datetime(dia['date'])
dia.set_index('date', inplace=True)
dia = dia.sort_index()
dia = dia.loc[START_DATE:END_DATE]

# ─── 3) SIGNAL GENERATION ──────────────────────────────────────────────────
# YM return = today's close – yesterday's close
ym['Ret'] = ym['close_p'].diff()

# 90th percentile of last 100 returns
ym['Pct90'] = ym['Ret'].rolling(100).quantile(0.9)

# Signal: 1 when Ret > Pct90, else 0
ym['Signal'] = (ym['Ret'] > ym['Pct90']).astype(int)

# ─── 4) BACKTEST ON DIA ─────────────────────────────────────────────────────
# Align dates
common = ym.index.intersection(dia.index)
ym   = ym.reindex(common)
dia  = dia.reindex(common)

# Shift signal to next day (we buy DIA at next open)
signal = ym['Signal'].shift(1).fillna(0)

# Strategy daily return: when signal=1, buy at Open, sell at Close
strat_ret = signal * (dia['close_p'] / dia['open_p'] - 1)

# Equity curve
equity = (1 + strat_ret).cumprod()

# ─── 5) PERFORMANCE METRICS ─────────────────────────────────────────────────
net_return   = equity.iloc[-1] - 1
total_days   = (equity.index[-1] - equity.index[0]).days
years        = total_days / 365.25
ann_return   = (1 + net_return) ** (1/years) - 1
sharpe_ratio = strat_ret.mean() / strat_ret.std() * np.sqrt(252)

print(f"Net Return       : {net_return:.2%}")
print(f"Annualized Return: {ann_return:.2%}")
print(f"Sharpe Ratio     : {sharpe_ratio:.2f}")

# ─── 6) PLOT EQUITY CURVE ───────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(equity.index, equity, label="Strategy Equity", linewidth=1.5)
plt.title("Equity Curve: DIA Strategy (2015–2025)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.show()
