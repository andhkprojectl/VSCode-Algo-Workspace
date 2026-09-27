# S0001_GeneralFeature_2_glm — Specification

## Background

`S0001_GeneralFeature_1_kimi.py` is a feature-engineering program for 1-minute ES futures. It builds a feature DataFrame from OHLCV data (class `ESFeatureEngineer`) covering: price action & momentum, volatility, volume, order flow/microstructure, time & seasonality, statistical/distribution features, and market structure.

This specification describes how to create and extend **`S0001_GeneralFeature_2_glm.py`** based on it.

## Phase 1 — Create `S0001_GeneralFeature_2_glm.py`

Source: modify a copy of the existing `S0001_GeneralFeature_1_kimi.py`.
**Do not change the existing file `S0001_GeneralFeature_1_kimi.py`.** Output goes to `S0001_GeneralFeature_2_glm.py`.

### 1. Fix look-ahead bias
- Solve any feature that has a look-ahead issue (e.g. swing high/low detection uses future bars via negative `shift`).

### 2. Normalize features
- General features: use a **standard scaler**.
- Date and time features: use **sine and cosine transformations** (cyclical encoding).

### 3. Data source
- Get data from MariaDB. Refer to `VS_0007_dbAndFile\createIBDbUser.sql`.
  - Database: `IBDb`
  - Table: `ticker1min`
  - Filter: `ticker` column = `'ES'`

### 4. New features
- **Linear regression slope** from the last 15 bars (include the current bar) for:
  - close
  - EMA 6 period
  - EMA 9 period
  - EMA 20 period
  - EMA 50 period
- **RSI divergence with close price**: RSI 6 period, RSI 9 period
- **RSI bullish divergence with high price**: RSI 6 period, RSI 9 period
- **RSI bearish divergence with low price**: RSI 6 period, RSI 9 period

## Phase 2 — Extend `S0001_GeneralFeature_2_glm.py`

Run after Phase 1. Just update the existing `S0001_GeneralFeature_2_glm.py`; no need to create a new output file.

### New features
- **Market regime**
- **Rolling VWAP of close price**: 3, 6, 9, 20 period
- **Banded rolling VWAP of close price** (3, 6, 9, 20 period):
  - vwap + 1 × vwap std
  - vwap − 1 × vwap std
  - vwap + 2 × vwap std
  - vwap − 2 × vwap std
