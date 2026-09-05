# Plan: NVDA 1-min Alpha Finder (S6002_1)

## TL;DR
Build `S6002_1_GenStatisticsRelationCsvFile.py` to read NVDA 1-min CSV, compute ~114 statistics columns (ATR/RSI/BB/EMA/close-diff/volume/IRB + rolling 90/10 percentiles) and 4 forward revenue targets (rt1/3/5/8), then find single & 2-feature combinations with strong relation to revenue via correlation (|r|>=0.5) and R-squared (>=0.25). Output enriched CSV + relation summary CSV + plotly scatter HTML. pytest suite alongside.

## Confirmed decisions (from user)
1. **ATR formula** = True Range (H/L/C): TR=max(H-L, |H-prevC|, |L-prevC|); atr=TR.rolling(n).mean(). Matches existing `talib.ATR` / manual pattern in `S8001_4_GenerateFromPromptQwen37Max.py`.
2. **BB cross value** = save the **close price** at the cross bar (NaN otherwise). Diff1/3/5 = **cross_close - close.shift(N)** (backward-looking: cross bar close minus close N bars before the cross). *(Updated 2026-06-20: changed from forward-looking to backward-looking per user request.)*
3. **EMA** = cDiffEma5/10/20 = current bar close price - previous N bars EMA close price (i.e., `close - EMA(close, span=N)`). Percentile vars named `cDiffEma20_90` / `cDiffEma20_10`.
4. **Combination scope** = singles + all 2-feature pairs. Use closed-form R^2 from correlation matrix (no 25k sklearn fits).
5. **Test framework** = pytest.
6. **Rows** = read ALL rows in input CSV (no trimming).
7. **IRB** = inside-range-bar (bullish OR bearish per `S8001_2_ConvertFromGemini.py` threshold 0.45 logic). Verified correct 2026-06-20.
8. **Output extra columns** = add `bbTop` (Bollinger Band top) and `bbBottom` (Bollinger Band bottom) to the statistics output CSV. *(Added 2026-06-20.)*

## Steps

### Phase A — Scaffolding & I/O
1. Create `VS_0006_data\VS_6002_findAlpha\2 Strategy\S6002_1_GenStatisticsRelationCsvFile.py`.
2. Constants at top: `INPUT_CSV`, `OUTPUT_DIR` (explicit absolute paths from requirement, NOT .env `csvExcelPath` since VS_6002 subfolder differs). `onlyRegularPeriod="Y"` parameter with default Y.
3. `load_data()`: `pd.read_csv(INPUT_CSV)`; rename cols A-I → Datetime/Date/Time/High/Low/Close/Open/Volume/Symbol (reuse pattern from `lookAheadTest_S8004_v1.py` L120-136); `pd.to_datetime(Datetime, format='%m/%d/%Y %H:%M')`; set index; sort.
4. `filter_regular(df)`: if onlyRegularPeriod=Y, keep 09:30 <= time < 16:00 using index time. **Decision: treat filtered series as continuous — rolling windows span overnight gap (no per-day reset).**

### Phase B — Statistics (4c) — *parallelizable groups*
5. **ATR group**: compute atr3/5/10/15 via TR rolling mean. Then atr{N}Diff1 = atr.diff(1); atr{N}Diff3 = atr.diff(3). Then percentile helper `rolling_pct(s,100,q)` → atr{N}_90/_10 and atr{N}Diff{D}_90/_10.
6. **Close-diff group**: cDiff1/2/3/5 = close.diff(N). Percentiles cDiff{N}_90/_10.
7. **EMA group**: ema5/10/20 = close.ewm(span=N, adjust=False).mean(); cDiffEma{N} = close - ema{N} (current bar close price - previous N bars EMA close price). Percentiles cDiffEma{N}_90/_10 (N=5,10,20).
8. **BB group**: BB period=20, std=2 (reuse `test_dia_ym_correlation1.py` pattern). top=mean+2std, bot=mean-2std. Store `bbTop` and `bbBottom` as output columns. Cross-up: prev close<=top & cur close>top → cCrossUpBBTop=close. Cross-down: prev close>=bot & cur close<bot → cCrossDownBBBottom=close. Diff1/3/5 = **cross_close - close.shift(N)** (backward-looking: cross bar close minus close N bars before the cross). Percentiles for all 8 BB cols.
9. **RSI group**: rsi2/6/14 via manual delta/gain/loss rolling mean (reuse `NVDA_irb_20260209_V1.py` `rsi()`). rsi{N}Diff1=rsi.diff(1); rsi{N}Diff5=rsi.diff(5). Percentiles for rsi + diffs.
10. **Volume group**: vDiff1 = (vol - vol.shift(1))/vol.shift(1) (percent different). Percentiles vDiff1_90/_10.
11. **IRB**: irb1 = 1 if inside-range-bar (bullish OR bearish per `S8001_2_ConvertFromGemini.py` threshold 0.45 logic) else 0.
12. `rolling_pct(s, window=100, q)`: `s.rolling(window, min_periods=1).apply(lambda x: np.nanpercentile(x, q), raw=True)` — handles sparse NaN (BB cross cols). _90 = q=90, _10 = q=10.

### Phase C — Revenue (4d) — *depends on B*
13. rt1 = open.shift(-2) - open.shift(-1); rt3 = open.shift(-4) - open.shift(-1); rt5 = open.shift(-6) - open.shift(-1); rt8 = open.shift(-9) - open.shift(-1). (future open minus next-bar open).

### Phase D — Relation analysis (4f) — *depends on B,C*
14. Build feature list (all 114 stat cols). Build clean df = drop rows with any NaN in features or rt targets.
15. **Singles**: corr matrix `clean[features+rt].corr()`; for each (feature, rt): r=corr[feature][rt]; r2=r*r. Strong if |r|>=0.5 (implies r2>=0.25).
16. **Pairs**: closed-form 2-feature R^2 from correlations: `R2 = (r_y1^2 + r_y2^2 - 2*r_y1*r_y2*r_12)/(1-r_12^2)`. Iterate all C(114,2)=6441 pairs × 4 rt. Strong if R2>=0.25. (Validate closed-form vs sklearn LinearRegression on 5 random pairs in tests.)
17. Collect strong relations into summary records: {type: single/pair, features, rt, corr (single only), r2, passes}.

### Phase E — Output (5) — *depends on D*
18. `S6002_1_statistics_revenue.csv`: input cols + all stats (including bbTop, bbBottom) + rt cols (per-row).
19. `S6002_1_relation_summary.csv`: strong relations table.
20. `S6002_1_scatter_plots.html`: plotly scatter for each strong single feature-vs-rt (reuse `fig.to_html` pattern from `lookAheadTest_S8004_v1.py`); for strong pairs add 3D scatter (2 features + rt color). Combine into one HTML.
21. All outputs to `C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_6002_findAlpha\csvExcel`.
22. `main()` orchestrates A→E with progress prints.

### Phase F — Tests (4g) — *parallel with E once functions exist*
23. Create `VS_0006_data\VS_6002_findAlpha\2 Strategy\test_S6002_1_GenStatisticsRelationCsvFile.py` (pytest).
24. Tests (synthetic small dataframes with known values):
    - test_atr_true_range: known H/L/C → expected atr.
    - test_rsi: monotonic up series → rsi~100.
    - test_rolling_pct: known series → 90/10 percentile.
    - test_bb_cross_up/down: constructed cross → cCrossUpBBTop = close, Diff1/3/5 = cross_close - close.shift(N) (backward-looking).
    - test_revenue_rt: known future opens → rt1/3/5/8.
    - test_filter_regular: rows outside 9:30-16:00 dropped.
    - test_pair_r2_closed_form: compare closed-form R^2 vs sklearn LinearRegression on random 2-feature data (assert abs diff < 1e-9).
    - test_irb_flag: constructed inside-range bar → irb1=1.
    - test_bbTop_bbBottom_columns: verify bbTop and bbBottom columns present in output.
25. Run `pytest` and ensure all pass.

## Relevant files
- `VS_0006_data\VS_6002_findAlpha\2 Strategy\S6002_1_GenStatisticsRelationCsvFile.py` — NEW main program.
- `VS_0006_data\VS_6002_findAlpha\2 Strategy\test_S6002_1_GenStatisticsRelationCsvFile.py` — NEW pytest suite.
- `VS_8000_Strategy\VS_8001_20260204_NVDA\2 Strategy\S8001_2_ConvertFromGemini.py` — IRB threshold pattern (iRbBullish/Bearish, 0.45).
- `VS_8000_Strategy\VS_8001_20260204_NVDA\2 Strategy\S8001_4_GenerateFromPromptQwen37Max.py` — manual ATR (TR) pattern.
- `VS_9999_test_program\MachineLearning\NVDA_irb_20260209_V1.py` — manual RSI + BB (20,2std) pattern.
- `VS_9999_test_program\MachineLearning\test_dia_ym_correlation1.py` — correlation matrix + rolling quantile pattern; `performCorrelation()` L663.
- `VS_9999_test_program\percentile_strategy_backtest.py` — `rolling(days).apply(lambda x: np.percentile(x.dropna(),90))` pattern.
- `VS_8000_Strategy\VS_8001_20260204_NVDA\5 Limited Test\lookAheadTest_S8004_v1.py` L120-136 — CSV col rename pattern; L452-508 — plotly `fig.to_html` pattern.
- `VS_0006_data\VS_6001_GetMarketDataToCsv\NVDA_20260101_20260615_1Min.py` — source of input CSV (confirms col layout).

## Verification
1. `python S6002_1_GenStatisticsRelationCsvFile.py` runs end-to-end, prints row counts after filter and after dropna.
2. `pytest VS_0006_data\VS_6002_findAlpha\2 Strategy\test_S6002_1_GenStatisticsRelationCsvFile.py -v` — all tests pass.
3. Confirm 3 output files exist in `...\VS_6002_findAlpha\csvExcel`: `S6002_1_statistics_revenue.csv` (cols = 9 input + 114 stats + 4 rt = 127), `S6002_1_relation_summary.csv`, `S6002_1_scatter_plots.html`.
4. Open `S6002_1_statistics_revenue.csv` — first ~100 rows have NaN in percentile cols (warmup), later rows populated; last 9 rows have NaN in rt8. Verify `bbTop` and `bbBottom` columns present and populated.
5. Open `S6002_1_scatter_plots.html` in browser — scatter plots render for strong relations; if none pass threshold, HTML states "no strong relation found" (possible on noisy 1-min data).
6. Spot-check: `atr3` at row 105 ≈ mean of last 3 TR values; `rsi14` row 20 in 0-100 range; `cDiffEma20_90` finite after row 100; `bbTop` > `bbBottom` at all rows after warmup.

## Decisions
- ATR = True Range (H/L/C), not close-only.
- BB cross stores close price at cross bar; Diff = cross_close - close.shift(N) (backward-looking: cross bar close minus close N bars before the cross).
- EMA set = {5,10,20}; cDiffEma{N} = current bar close price - previous N bars EMA close price.
- Combinations = singles + all pairs via closed-form R^2 (fast, no 25k sklearn fits).
- Tests = pytest, synthetic data, alongside main file.
- Read all CSV rows (no trim).
- Indicators computed on filtered continuous series (overnight gap not reset).
- Paths hardcoded per requirement (not .env) since VS_6002 subfolder differs from `csvExcelPath`.
- Output CSV names prefixed `S6002_1_` to avoid overwriting input `NVDA_..._1Min.csv` in same folder.
- Output CSV includes `bbTop` and `bbBottom` columns (Bollinger Band top/bottom values).

## Further Considerations
1. **Overnight gap in rolling windows**: with onlyRegularPeriod=Y, a 100-bar window at 09:35 includes prior-day bars. Acceptable? Recommend keep continuous (current plan). Alternative: reset windows at each day open (more complex, drops first 100 bars/day).
2. **No strong relations found**: 1-min data is noisy; thresholds now relaxed to |r|>=0.5 / R^2>=0.25 per user request, which should yield more candidate relations. If summary CSV is still empty, recommend further lowering in a follow-up.
3. **Pair scatter legibility**: ~6k pairs × 4 rt could yield many strong pairs. Cap scatter HTML to top 20 by R^2 to keep HTML manageable. Recommend cap=20.