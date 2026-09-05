# Plan: Replace BB Cross Diffs with BB Band Diffs in S6002_1

**Target**: `VS_0006_data/VS_6002_findAlpha/2 Strategy/S6002_1_GenStatisticsRelationCsvFile.py`
**Function**: `compute_statistics()`, lines 197-225 (Bollinger Band group section)

---

## Step 1 — Remove cross-diff columns (lines 208-218)

Delete these 6 column assignments:
- `cCrossUpBBTopDiff1`, `cCrossUpBBTopDiff3`, `cCrossUpBBTopDiff5`
- `cCrossDownBBBottomDiff1`, `cCrossDownBBBottomDiff3`, `cCrossDownBBBottomDiff5`

## Step 2 — Remove the `add_pct_cols` loop for BB cross columns (lines 220-225)

This removes all `_90`/`_10` percentile columns for:
- `cCrossUpBBTop`, `cCrossUpBBTopDiff1/3/5`
- `cCrossDownBBBottom`, `cCrossDownBBBottomDiff1/3/5`

Total removed: 22 columns (6 base diffs + 16 percentiles from the loop, but the loop also covers `cCrossUpBBTop` and `cCrossDownBBBottom` which each have `_90`/`_10` = 4 more, so 6 + 16 = 22).

## Step 3 — Add 6 new BB band diff columns

```python
out['bbTopDiff1'] = bb_top - bb_top.shift(1)
out['bbTopDiff3'] = bb_top - bb_top.shift(3)
out['bbTopDiff5'] = bb_top - bb_top.shift(5)
out['bbBottomDiff1'] = bb_bot - bb_bot.shift(1)
out['bbBottomDiff3'] = bb_bot - bb_bot.shift(3)
out['bbBottomDiff5'] = bb_bot - bb_bot.shift(5)
```

## Step 4 — Add `add_pct_cols` loop for the 6 new BB diff columns

Generates 12 percentile columns:
- `bbTopDiff1_90`, `bbTopDiff1_10`
- `bbTopDiff3_90`, `bbTopDiff3_10`
- `bbTopDiff5_90`, `bbTopDiff5_10`
- `bbBottomDiff1_90`, `bbBottomDiff1_10`
- `bbBottomDiff3_90`, `bbBottomDiff3_10`
- `bbBottomDiff5_90`, `bbBottomDiff5_10`

## Step 5 — Output files automatically pick up new columns

`get_feature_columns()` dynamically discovers all non-input, non-rt columns, so the output CSV and relation analysis automatically include the new columns.

---

## Exact replacement

Replace the Bollinger Band group block (lines 197-225):

```python
    # --- Bollinger Band group ---
    bb_top, bb_bot = calc_bollinger(out, BB_PERIOD, BB_STD)
    out['bbTop'] = bb_top
    out['bbBottom'] = bb_bot
    close = out['Close']
    prev_close = close.shift(1)

    # Cross up BB top: prev close <= top & cur close > top -> save close at cross
    cross_up = (prev_close <= bb_top) & (close > bb_top)
    out['cCrossUpBBTop'] = close.where(cross_up, np.nan)

    # Cross down BB bottom: prev close >= bot & cur close < bot -> save close at cross
    cross_down = (prev_close >= bb_bot) & (close < bb_bot)
    out['cCrossDownBBBottom'] = close.where(cross_down, np.nan)

    # BB band diffs (current - N bars ago)
    out['bbTopDiff1'] = bb_top - bb_top.shift(1)
    out['bbTopDiff3'] = bb_top - bb_top.shift(3)
    out['bbTopDiff5'] = bb_top - bb_top.shift(5)
    out['bbBottomDiff1'] = bb_bot - bb_bot.shift(1)
    out['bbBottomDiff3'] = bb_bot - bb_bot.shift(3)
    out['bbBottomDiff5'] = bb_bot - bb_bot.shift(5)

    # Percentile columns for BB band diffs
    for bb_diff_col in [
        'bbTopDiff1', 'bbTopDiff3', 'bbTopDiff5',
        'bbBottomDiff1', 'bbBottomDiff3', 'bbBottomDiff5',
    ]:
        add_pct_cols(out, bb_diff_col)
```

---

## Net column change

| Action | Count |
|--------|-------|
| Removed | 22 columns |
| Added | 18 columns (6 base + 12 percentiles) |
| **Net** | **-4 columns** |

---

## Verification checklist

- [ ] Run script, confirm no `KeyError` on removed column names
- [ ] Output CSV contains `bbTopDiff1`, `bbTopDiff1_90`, `bbTopDiff1_10`, ..., `bbBottomDiff5_90`, `bbBottomDiff5_10`
- [ ] `cCrossUpBBTop` and `cCrossDownBBBottom` still exist (base cross columns kept)
- [ ] None of the removed columns appear in the output
- [ ] Relation summary and scatter plots include new columns