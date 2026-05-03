import pandas as pd
import numpy as np
# import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import warnings
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# 1) download NVDA data
# df = yf.download("NVDA", start="2016-01-01", progress=False)
# Load data
# df = pd.read_csv('c:/tmp/aaa11_negative.csv')
df = pd.read_csv('c:/tmp/aaa.csv')
# df = df.rename(columns=str.title)  # Close
df = df.dropna()

# 2) 建立技術特徵（均線、斜率、偏離度等）
def SMA(s, n): return s.rolling(n).mean()
def EMA(s, n): return s.ewm(span=n, adjust=False).mean()

close = df["Close"]
vol = df["Volume"]

df["SMA5"]  = SMA(close, 5)
df["SMA10"] = SMA(close, 10)
df["SMA20"] = SMA(close, 20)
df["SMA50"] = SMA(close, 50)

df["EMA12"] = EMA(close, 12)
df["EMA26"] = EMA(close, 26)
df["MACD"]  = df["EMA12"] - df["EMA26"]
df["MACDsig"] = EMA(df["MACD"], 9)
df["MACDhist"] = df["MACD"] - df["MACDsig"]

for n in [5, 10, 20, 50]:
    df[f"slope_SMA{n}"] = df[f"SMA{n}"].diff()
    df[f"dev_SMA{n}"]   = (close - df[f"SMA{n}"]) / df[f"SMA{n}"]

# 均線間關係
df["spread_5_20"] = df["SMA5"] - df["SMA20"]
df["ratio_5_20"]  = df["SMA5"] / df["SMA20"] - 1

# 簡易波動（20日）
df["ret"] = close.pct_change()
df["rv_20"] = df["ret"].rolling(20).std()

# 3) 目標：下一根報酬
# df["y_next"] = df["ret"].shift(-1)
df["y_next"] = df["maxRtPst"]

df.to_csv('c:/tmp/aaa_tmp.csv', index=False)

# 4) 特徵集合與清洗
feat_cols = [
    "SMA5","SMA10","SMA20","SMA50",
    "EMA12","EMA26","MACD","MACDsig","MACDhist",
    "slope_SMA5","slope_SMA10","slope_SMA20","slope_SMA50",
    "dev_SMA5","dev_SMA10","dev_SMA20","dev_SMA50",
    "spread_5_20","ratio_5_20","rv_20","Volume"
]
df = df.dropna(subset=feat_cols + ["y_next"]).copy()

X = df[feat_cols].values
y = df["y_next"].values



# 5) 時序交叉驗證 + ElasticNet
tscv = TimeSeriesSplit(n_splits=5)
maes = []
coefs = []

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(l1_ratio=[0.2,0.5,0.8], alphas=None, cv=3, max_iter=5000))
])

for fold, (tr, va) in enumerate(tscv.split(X)):
    Xtr, Xva = X[tr], X[va]
    ytr, yva = y[tr], y[va]
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xva)
    mae = mean_absolute_error(yva, pred)
    maes.append(mae)
    coefs.append(pipe.named_steps["model"].coef_)
    print(f"Fold {fold+1} MAE: {mae:.6f}")

print("Avg MAE:", np.mean(maes))
coef_mean = np.mean(np.vstack(coefs), axis=0)
imp = pd.Series(coef_mean, index=feat_cols).sort_values(key=np.abs, ascending=False)
print("\nElasticNet 平均重要度(按絕對係數):")
print(imp.head(20))


