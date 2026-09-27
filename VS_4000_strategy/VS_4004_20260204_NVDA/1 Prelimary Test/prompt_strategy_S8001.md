# strategyIRB1000_V1 — NVDA 5-min Algo Trade Strategy

Consolidated requirement — `prompt_strategy_S8001_V1.md` + `V2` + `V3`
updates merged into one document. Target program:
`VS_4004_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py`
(class `strategyIRB1000_V1`).

## 1. Program

- Filename: `VS_4004_20260204_NVDA/1 Prelimary Test And Strategy/strategyIRB1000_V1.py`
- Class `strategyIRB1000_V1` with the strategy logic inside the class
- Can be passed as a parameter to the test programs
  `VS_0003_test`: `backtest.py`, `otherTest.py`, `monteCarloSimulation.py`
  and `walkforwardTest.py`
- Can also be passed as a parameter of the 5-minute live trade program
  `VS_9000_LiveTrade/5min/inCubation_intraday1.py` (TBD)

## 2. Background

- Time frame: 5-minute
- Period: 09:30 – 16:00, regular NASDAQ trading hours, US NY time zone
- Symbol: NVDA

## 3. Variables and indicators

- `H` high price, `L` low price, `O` open price, `C` close price, `V` volume
- Range-reversal bars:

```
iRbhLRange = H - L
iRbUpperTh = H - iRbhLRange * 0.45
iRbLowerTh = L + iRbhLRange * 0.45
iRbBullish = IIf(C > L AND O > L AND C < iRbLowerTh AND O < iRbLowerTh, 1, 0)
iRbBearish = IIf(C < H AND O < H AND C > iRbUpperTh AND O > iRbUpperTh, 1, 0)
```

- `ema10` = EMA of C, 10 bars; `ema20` = EMA of C, 20 bars
- `regSlopema10` = linear regression slope of `ema10`, 3 bars
- Difference between close price and `ema20`, absolute, ×100, rounded to 0
  decimals:

```
ema20DiffCAbsRound0 = round((abs(ema20 - C)) * 100, 0)
```

- Divide `ema20DiffCAbsRound0` into 5 ranks (value 1–5), saved in
  `ema20DiffCAbsRank`:

| Rank | ema20DiffCAbsRound0 range |
|------|---------------------------|
| 1 | 0 <= x < 10 |
| 2 | 10 <= x < 21 |
| 3 | 21 <= x < 35 |
| 4 | 35 <= x < 66 |
| 5 | 66 <= x |

- `sumEma20DiffCAbsRank` = number of bars of each rank within the last 100
  bars of `ema20DiffCAbsRank`
  (e.g. if the last 100 bars contain 30 rank-1 bars, `sumEma20DiffCAbsRank`
  for rank 1 is 30)
- `rt10` = difference of the close price 5 bars ahead vs the current bar
  close price
- `sumEma20DiffCAbsUpRank` = number of bars of each rank that are up
  (rt10 > 0) within the last 100 bars
  (e.g. if the last 100 bars contain 30 rank-1 bars and 20 of them have
  rt10 > 0, `sumEma20DiffCAbsUpRank` for rank 1 is 20)
- Percentage per rank:

```
ema20DiffCAbsUpRankper = sumEma20DiffCAbsUpRank * 100 / sumEma20DiffCAbsRank
```

## 4. Trading signals

### Buy signal

```
buy00 = iRbBullish
    AND regSlopema10 > Ref(regSlopema10, -1)
    AND pRtEma20Rankper > 50
    AND ema20DiffCAbsRank >= 3
    AND pRtEma20Rankper >= Ref(pRtEma20Rankper, -1)
    AND pRtEma20Rankper >= Ref(pRtEma20Rankper, -2)
```

- Buy signal is triggered when the previous bar's `buy00` is true
- Live trade: buy price is the current close price, slippage = 0.03
- Backtest: buy price is the current open price

### Sell signal

- 5 bars after the buy bar
- Live trade: buy price is the current close price, slippage = 0.03
- Backtest: buy price is the current open price

### Short signal

```
short00 = iRbBearish
    AND regSlopema10 < Ref(regSlopema10, -1)
    AND pRtEma20Rankper < 50
    AND ema20DiffCAbsRank >= 3
    AND pRtEma20Rankper <= Ref(pRtEma20Rankper, -1)
    AND pRtEma20Rankper <= Ref(pRtEma20Rankper, -2)
```

- Short signal is triggered when the previous bar's `short00` is true
- Live trade: short price is the current close price, slippage = 0.03
- Backtest: short price is the current open price

### Cover signal

- 5 bars after the short bar
- Live trade: buy price is the current close price, slippage = 0.03
- Backtest: buy price is the current open price

## 5. Stop loss

- Buy stop loss — stop limit order:
  - Live trade: stop price = current close price − previous bar
    `ATR(7) * 2.5`; stop limit price = stop price − 0.06
  - Backtest: stop price = current close price − previous bar
    `ATR(7) * 2.5`; stop limit price = stop price − 0.06
- Short stop loss — stop limit order:
  - Live trade: stop price = current close price + previous bar
    `ATR(7) * 2.5`; stop limit price = stop price + 0.06
  - Backtest: stop price = current close price + previous bar
    `ATR(7) * 2.5`; stop limit price = stop price + 0.06

## 6. Position size

- 110 NVDA stocks per buy/short trade; sell/cover all stocks for each trade

## 7. V2 update — trading window and forced flat

Added to `VS_4004_20260204_NVDA/2 Strategy/S8001_4_GenerateFromPromptQwen37Max.py`:

1. One more condition for buy and short signals: only allow buy or short
   between 09:30:00 and 14:55:00
2. One more condition for sell and cover signals: if there is an open NVDA
   position at 15:55:00 — if the position is long, force sell all; if the
   position is short, force cover all

## 8. V3 update — rank statistics

Update function `_calculate_rank_statistics`:

1. `sum_rank` becomes an array of series. There are 5 ranks of
   `ema20DiffCAbsRank` (calculated by `_calculate_rank`); for each rank,
   calculate `sum_rank` — i.e. 5 series of `sum_rank`, one per rank
2. Similar to `sum_rank`, `sum_up_rank` becomes 5 series, one per rank
3. The returned percentage depends on the current bar's
   `ema20DiffCAbsRank` value. For example, if the current bar's
   `ema20DiffCAbsRank = 1`, return rank 1 of
   `sum_up_rank * 100 / sum_rank`; if it is 2, return rank 2; the same for
   ranks 3 to 5

## Version history

- **V1** — original strategy requirement (sections 1–6)
- **V2** — trading window (09:30–14:55) and forced-flat at 15:55
- **V3** — rank statistics: per-rank series for `sum_rank` / `sum_up_rank`
  and rank-dependent percentage selection
