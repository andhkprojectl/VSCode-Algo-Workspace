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
df = pd.read_csv('c:/tmp/aaa2.csv')

# Drop non-numeric column 'Date1'
df = df.drop(columns=['Date1'])
# df = df.drop(columns=['Time1'])
df = df.drop(columns=['Time1'], errors='ignore')


# df = df.rename(columns=str.title)  # Close
df = df.dropna()


# 3) 目標：下一根報酬
# df["y_next"] = df["ret"].shift(-1)
df["y_next"] = df["maxRtPst"]

df.to_csv('c:/tmp/aaa_tmp.csv', index=False)

# 4) 特徵集合與清洗
feat_cols = df.columns.tolist()

# feat_cols = [
#    "SMA5","SMA10","SMA20","SMA50",
#    "EMA12","EMA26","MACD","MACDsig","MACDhist",
#    "slope_SMA5","slope_SMA10","slope_SMA20","slope_SMA50",
#    "dev_SMA5","dev_SMA10","dev_SMA20","dev_SMA50",
#    "spread_5_20","ratio_5_20","rv_20","Volume"
# ]
df = df.dropna(subset=feat_cols + ["y_next"]).copy()

# df2 = df.copy()
# df2 = df2.drop(columns=['y_next'], errors='ignore')
# # df = df.drop(columns=['Time1'])
# df2 = df2.drop(columns=['maxRtPst'], errors='ignore')

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


