import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf

# Load NVDA 30-minute data
df = yf.download("NVDA", interval="30m", start="2023-01-01", end="2025-01-01")
df = df.dropna()

# Compute indicators
df['ema25'] = df['Close'].ewm(span=25).mean()
df['ema50'] = df['Close'].ewm(span=50).mean()
df['ema_diff'] = df['ema25'] - df['ema50']
df['ema_cross'] = (df['ema25'] > df['ema50']).astype(int)

# Momentum and volatility features
df['returns'] = df['Close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()
df['rsi'] = df['returns'].rolling(14).apply(lambda x: 100 - 100/(1 + (x[x>0].mean() / (-x[x<0].mean() if -x[x<0].mean()!=0 else 1))), raw=False)
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# Define target variable: next bar up/down
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.dropna()

# Choose features
features = ['ema25','ema50','ema_diff','ema_cross','rsi','volatility','hour','dayofweek']
X = df[features]
y = df['target']

# Split into train/test sets (time-based)
split_date = '2024-07-01'
X_train, X_test = X.loc[:split_date], X.loc[split_date:]
y_train, y_test = y.loc[:split_date], y.loc[split_date:]

# Train Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

#  Predict probability of next-bar going up
df.loc[X_test.index, 'prob_up'] = rf.predict_proba(X_test)[:,1]

# Evaluate model
auc = roc_auc_score(y_test, df.loc[X_test.index, 'prob_up'])
acc = accuracy_score(y_test, (df.loc[X_test.index, 'prob_up'] > 0.5).astype(int))
print(f"AUC: {auc:.3f}, Accuracy: {acc:.3f}")

# Use model to filter EMA crossover trades
df['signal'] = 0
df.loc[(df['ema_cross'] == 1) & (df['prob_up'] > 0.6), 'signal'] = 1     # buy
df.loc[(df['ema_cross'] == 0) & (df['prob_up'] < 0.4), 'signal'] = -1    # sell

# Simulate simple backtest (no transaction cost)
df['strategy_ret'] = df['signal'].shift(1) * df['returns']
cumulative_ret = (1 + df[['returns','strategy_ret']]).cumprod()

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot(cumulative_ret['returns'], label='Buy & Hold')
plt.plot(cumulative_ret['strategy_ret'], label='EMA+RF Strategy')
plt.title('NVDA 30-Minute EMA+RandomForest Strategy')
plt.legend()
plt.show()

# View feature importance
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(fi)
