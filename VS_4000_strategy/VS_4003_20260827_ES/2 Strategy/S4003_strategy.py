"""
S4003_strategy.py
=================
Strategy class strategyS4003 for VS_4003_20260827_ES (ES 1-minute price-action
strategy converted from AmiBroker AFL
`VS_4003_20260827_ES/1 Prelimary Test/amibroker/backtest_4003_1 trim_4_python.afl`).

Port of class strategyS4003V1 (from `1 Prelimary Test/S4003_1_glm.py`), renamed
strategyS4003, logic unchanged. TASKS.md T1.1. No imports from `1 Prelimary Test`.

Parameters come from `VS_4000_strategy/VS_4003_20260827_ES/config.yaml` section
`Strategy` (see loadStrategyConfig / strategyS4003.applyStrategyConfig); the
class-level attributes remain as defaults identical to strategyS4003V1 and are
used when the config file or a key is missing.

Class strategyS4003 implements (active AFL branches only):
  Buy  : buy00 = priceActionUp4Bar2, entered next bar at Open
  Short: short00 = priceActionDown3Bar1 OR short02, entered next bar at Open
  Exits: stop loss (1x max ATR14 of prev 3 bars), profit take (1.5x),
         15-bar stop (long side shifts on consecutive buy signals), and
         force exit at Open on opposite entry signal (same-bar cancel per AFL).
  Size : 1 ES contract, point value 50, commission 14.50 per round turn.

Interfaces:
  - Subclasses backtesting.Strategy for VS_0003_test/backtest.py and walkForwardTest.py
  - generate_signals(df) for VS_0003_test/otherTest.py (look-ahead bias test)
  - run_backtest(df, init_balance, position_size) for VS_0003_test/monteCarloSimulation.py
  - run_full_backtest(df) drives the custom AFL-faithful engine and the 3 output files

Import-safe: no DB/IB connections and no config access at import time.
"""

import os
import io
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from backtesting import Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    class Strategy:  # fallback base so this module imports without backtesting lib
        pass

OUTPUT_DIR = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES\backTestResult"

DT_FORMAT = '%d/%m/%Y %H:%M'


# ===========================================================================
# Indicator helpers (pure pandas, AmiBroker-faithful initialization)
# ===========================================================================
def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14, mode: str = 'wilder') -> pd.Series:
    """True Range ATR. mode 'wilder': classic Wilder recursion seeded with SMA of
    first `period` TRs (Null before). mode 'sma': rolling mean of TR."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1, skipna=True)
    tr = tr.to_numpy(dtype=float)
    n = len(tr)
    atr = np.full(n, np.nan)
    if n == 0:
        return pd.Series(atr, index=high.index)
    if mode == 'sma':
        atr = pd.Series(tr).rolling(period).mean().to_numpy()
    else:
        if n >= period:
            atr[period - 1] = np.nanmean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=high.index)


def _lin_reg_slope(s: pd.Series, period: int) -> pd.Series:
    """Rolling OLS slope of the last `period` values (AmiBroker LinRegSlope)."""
    t = np.arange(period, dtype=float)
    tm = t.mean()
    denom = float(((t - tm) ** 2).sum())

    def _slope(y):
        y = np.asarray(y, dtype=float)
        if np.isnan(y).any():
            return np.nan
        return float(((y - y.mean()) * (t - tm)).sum() / denom)

    return s.rolling(period).apply(_slope, raw=True)


def _bars_since_min(s: pd.Series, period: int) -> pd.Series:
    """AmiBroker LLVBars: bars passed since the lowest value in the window
    (partial windows allowed, oldest occurrence wins ties, 0 = current bar)."""
    def _f(w):
        return float(len(w) - 1 - np.argmin(w))
    return s.rolling(period, min_periods=1).apply(_f, raw=True)


def _cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    """AmiBroker Cross(a, b): a crosses above b."""
    res = (a > b) & (a.shift(1) <= b.shift(1))
    return res.fillna(False)


def _parse_dt_series(sr: pd.Series) -> pd.Series:
    for fmt in ('%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M:%S',
                '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return pd.to_datetime(sr, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(sr)


# ===========================================================================
# Config: VS_4000_strategy/VS_4003_20260827_ES/config.yaml, section Strategy
# ===========================================================================
def loadStrategyConfig(config_path=None) -> dict:
    """Return the `Strategy` section of the local strategy config as a dict.
    Missing file/section yields {} so callers fall back to class defaults."""
    if config_path is None:
        root = Path(__file__).resolve().parents[3]
        config_path = root / 'VS_4000_strategy' / 'VS_4003_20260827_ES' / 'config.yaml'
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('Strategy') or {}


# ===========================================================================
# Strategy class
# ===========================================================================
class strategyS4003(Strategy):
    """
    ES 1-min price action strategy V1 (converted from AFL backtest_4003_1).

    Entry signals evaluated on completed bar i, entry at bar i+1 Open:
      buy00   = priceActionUp4Bar2 (bull body + vwap50 cross-up 3-4 bars ago,
                bear body + vwap50 cross-down 1 bar ago, current bar lowest of 5)
      short00 = priceActionDown3Bar1 OR short02 (vwap50-slope reversal)

    Exits (AFL loop, isTesting=True path):
      stop loss / profit take anchored at entry-bar Open using
      maxATR14(prev 3 bars) at entry bar (1x stop, 1.5x profit),
      15-bar stop at Open (long side shifts while consecutive Buy signals),
      force exit at Open on opposite entry (same-bar entries cancelled).

    Entry fill (AmiBroker price-bound checking, validated trade-by-trade):
      long  = max(Open - ATR14[signal bar]/4, Low)
      short = min(Open + ATR14[signal bar]/4, High)
    """

    # --- data window (requirement #7 parity window) ---
    symbol = 'ES'
    start_date = '2026-07-13 00:00:00'   # trading start (entries gated to >= start)
    end_date = '2026-08-17 23:59:59'
    end_day_inclusive = True      # False -> data through 2026-08-16 23:59:59
    warmup_days = 120             # extra history loaded for indicator warm-up
                                   # (AmiBroker computes indicators on all loaded bars)

    # --- calibration levers (validated against AmiBroker report) ---
    entry_price_mode = 'limit'     # 'limit': O -/+ ATR14[i-1]/4 clamped to bar Low/High
                                   # (AmiBroker price-bound checking, parity-validated)
                                   # 'open': plain next-bar Open
    atr_mode = 'wilder'            # 'wilder' | 'sma'
    percentile_min_periods = 100   # AmiBroker Percentile full-window default

    # --- strategy parameters (from AFL) ---
    stop_period1 = 15             # N-bar stop
    profit_mult = 1.5             # profit1Pt = 1.5 * max3BarAtr14_1
    stop_mult = 1.0               # stopLossPt = 1.0 * max3BarAtr14_1
    entry_limit_atr_frac = 4.0    # BuyPrice offset = ATR14[i-1] / 4

    # --- account / costs ---
    num_contracts = 1
    point_value = 50.0            # ES point value
    commission_per_trade = 14.50  # round turn (report: 1073 / 74 trades)
    initial_capital = 80000.0
    risk_free_annual = 0.03       # Sharpe ratio risk-free rate
    ulcer_period = 14

    # fraction-based commission attr for backtesting.py harness (backtest.py reads it)
    commission = 0.0

    # config.yaml section `Strategy` key -> class attribute
    _CONFIG_ATTR_MAP = {
        'symbolName': 'symbol',
        'startDate': 'start_date',
        'endDate': 'end_date',
        'endDayInclusive': 'end_day_inclusive',
        'warmupDays': 'warmup_days',
        'entryPriceMode': 'entry_price_mode',
        'atrMode': 'atr_mode',
        'percentileMinPeriods': 'percentile_min_periods',
        'stopPeriod': 'stop_period1',
        'profitMult': 'profit_mult',
        'stopMult': 'stop_mult',
        'entryLimitAtrFrac': 'entry_limit_atr_frac',
        'positionSize': 'num_contracts',
        'pointValue': 'point_value',
        'commissionPerTrade': 'commission_per_trade',
        'initCapital': 'initial_capital',
        'riskFreeAnnual': 'risk_free_annual',
        'ulcerPeriod': 'ulcer_period',
    }

    def __init__(self, *args, config=None, **kwargs):
        type(self).applyStrategyConfig(config)
        super().__init__(*args, **kwargs)

    @classmethod
    def applyStrategyConfig(cls, config=None) -> dict:
        """Set class attributes from config.yaml section `Strategy` (camelCase
        keys). Keys absent from the section keep the class-level defaults.
        Returns the config dict used."""
        cfg = loadStrategyConfig() if config is None else (config or {})
        for key, attr in cls._CONFIG_ATTR_MAP.items():
            if key in cfg and cfg[key] is not None:
                setattr(cls, attr, cfg[key])
        return cfg

    # ------------------------------------------------------------------
    # DataFrame normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            for col in ('datetime', 'datetime1', 'DateTime', 'Date'):
                if col in out.columns:
                    out[col] = _parse_dt_series(out[col])
                    out = out.set_index(col)
                    break
            else:
                out.index = _parse_dt_series(pd.Series(out.index.astype(str)))
        rename = {}
        for c in out.columns:
            cl = str(c).lower()
            if cl == 'open':
                rename[c] = 'Open'
            elif cl == 'high':
                rename[c] = 'High'
            elif cl == 'low':
                rename[c] = 'Low'
            elif cl == 'close':
                rename[c] = 'Close'
            elif cl in ('volume', 'vol'):
                rename[c] = 'Volume'
        out = out.rename(columns=rename)
        if 'Volume' not in out.columns:
            out['Volume'] = 1.0
        out = out.sort_index()
        return out[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    # ------------------------------------------------------------------
    # Indicators (AFL section 1-6, active branches only)
    # ------------------------------------------------------------------
    @classmethod
    def _compute_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        O, H = out['Open'], out['High']
        L, C = out['Low'], out['Close']
        V = out['Volume']

        out['atr14'] = _atr_series(H, L, C, 14, cls.atr_mode)
        out['max3BarAtr14_1'] = out['atr14'].rolling(3, min_periods=1).max().shift(1)

        out['maxCO'] = pd.concat([C, O], axis=1).max(axis=1)
        out['minCO'] = pd.concat([C, O], axis=1).min(axis=1)
        out['isBullish'] = C > O
        out['isBearish'] = C < O

        body_size = (C - O).abs()
        rng = H - L
        body_size_per = body_size / rng.where(rng != 0, np.nan)
        body_size_per = body_size_per.fillna(0.0)
        rank_top = body_size.rolling(100, min_periods=cls.percentile_min_periods).quantile(0.70)
        is_large_body = (body_size >= rank_top) & (body_size_per >= 0.65)
        out['largeBullishBody'] = out['isBullish'] & is_large_body
        out['largeBearishBody'] = out['isBearish'] & is_large_body

        # vwap50 = Sum(C*V, 50) / Sum(V, 50), Nz -> 0 (partial sums from bar 0)
        pv_sum = (C * V).rolling(50, min_periods=1).sum()
        v_sum = V.rolling(50, min_periods=1).sum()
        out['vwap50'] = (pv_sum / v_sum.replace(0, np.nan)).fillna(0.0)

        out['regSlop3Vwap50Bar'] = _lin_reg_slope(out['vwap50'], 3)
        out['regSlop5vwap50Bar'] = _lin_reg_slope(out['vwap50'], 5)
        l0 = (out['regSlop5vwap50Bar'] < -0.01).fillna(False)
        out['regSlop5vwap50BarL0Sum15'] = l0.rolling(5, min_periods=1).sum()

        out['crossUpVWap50'] = (_cross_over(C, out['vwap50'])
                                | ((O < out['vwap50']) & (C > out['vwap50']))).fillna(False)
        out['crossDownVWap50'] = (_cross_over(out['vwap50'], C)
                                  | ((O > out['vwap50']) & (C < out['vwap50']))).fillna(False)

        out['llvBar5_1'] = _bars_since_min(L, 5)
        out['llvBar5_2'] = _bars_since_min(out['minCO'], 5)
        return out

    # ------------------------------------------------------------------
    # Entry signals (AFL buy00 / short00) and next-bar Buy/Short arrays
    # ------------------------------------------------------------------
    @classmethod
    def _compute_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        out = cls._compute_indicators(df)

        lb = out['largeBullishBody']
        lbb = out['largeBearishBody']
        cu = out['crossUpVWap50']
        cd = out['crossDownVWap50']
        is5_low = (out['llvBar5_1'] == 0) | (out['llvBar5_2'] == 0)

        # priceActionUp4Bar2 (AFL buy01_3_2 -> buy00)
        buy00 = ((lb.shift(3) | lb.shift(2))
                 & (cu.shift(3) | cu.shift(2))
                 & lbb.shift(1) & cd.shift(1)
                 & is5_low & out['isBullish']).fillna(False)

        # priceActionDown3Bar1
        pa_down = (out['isBullish'].shift(2) & out['isBearish'].shift(1)
                   & ((out['minCO'].shift(2) > out['minCO'].shift(1))
                      | (out['Low'].shift(2) > out['Low'].shift(1)))
                   & ((out['maxCO'].shift(2) < out['maxCO'].shift(1))
                      | (out['High'].shift(2) < out['High'].shift(1)))
                   & out['isBearish'] & cd).fillna(False)

        # short02
        cd_sum4 = cd.rolling(4, min_periods=1).sum()
        short02 = ((out['regSlop3Vwap50Bar'].shift(1) > 0)
                   & (out['regSlop3Vwap50Bar'] < 0)
                   & (cd_sum4 > 0)
                   & out['isBearish'] & out['isBearish'].shift(1)
                   & (out['regSlop5vwap50BarL0Sum15'] >= 4)).fillna(False)

        out['buy00'] = buy00
        out['short00'] = pa_down | short02

        # AFL: Buy = Ref(buy00, -1) AND timeFilterBuy(True) AND allowOpenNewPosition(True)
        out['buySig'] = out['buy00'].shift(1).fillna(False).astype(int)
        out['shortSig'] = out['short00'].shift(1).fillna(False).astype(int)
        return out

    @classmethod
    def _prepare(cls, df: pd.DataFrame) -> pd.DataFrame:
        return cls._compute_signals(cls._normalize_df(df))

    # ------------------------------------------------------------------
    # AFL-faithful trade simulation loop
    # ------------------------------------------------------------------
    @classmethod
    def _simulate(cls, sig: pd.DataFrame, contracts=None, trade_start_ts=None):
        """Returns (trades list of dict, equity ndarray, in_market ndarray).

        Mirrors the AFL for-loop exactly, including:
          - order per bar: buy entry, short entry, long exits, short exits
          - Buy/Short array mutation (Buy[i]=0 on stop/profit/force long exit)
          - same-bar entry stop/profit -> trade cancelled (AFL zeroes Buy[i])
          - long N-bar anchor shifts on consecutive buys; short side fixed 15
        trade_start_ts gates entries (AmiBroker From date) while indicators
        keep their full warm-up history.
        """
        ct = cls.num_contracts if contracts is None else contracts
        n = len(sig)
        O = sig['Open'].to_numpy(float)
        H = sig['High'].to_numpy(float)
        L = sig['Low'].to_numpy(float)
        C = sig['Close'].to_numpy(float)
        dts = sig.index
        buy = sig['buySig'].to_numpy(int).copy()
        shr = sig['shortSig'].to_numpy(int).copy()
        if trade_start_ts is not None:
            gate = dts < trade_start_ts
            buy[gate] = 0
            shr[gate] = 0
        atr_off = (sig['atr14'].shift(1) / cls.entry_limit_atr_frac).to_numpy(float)
        max3 = sig['max3BarAtr14_1'].to_numpy(float)

        pv = cls.point_value
        comm = cls.commission_per_trade
        limit_mode = (cls.entry_price_mode == 'limit')
        sp = cls.stop_period1

        trades = []
        realized = 0.0
        equity = np.full(n, cls.initial_capital)
        in_market = np.zeros(n, dtype=int)

        isInLong = isInShort = False
        long_fill = long_stop = long_profit = 0.0
        short_fill = short_stop = short_profit = 0.0
        sell_stop_n = sp
        buy_force_cover = short_force_sell = False
        cur = None      # open long trade dict
        cur_s = None    # open short trade dict

        def _record_long(i, price, reason):
            nonlocal realized, cur
            pnl = (price - long_fill) * pv * ct - comm
            realized += pnl
            trades.append({
                'type': 'Long', 'entry_dt': dts[cur['entry_bar']],
                'entry_price': long_fill, 'exit_dt': dts[i], 'exit_price': price,
                'entry_bar': cur['entry_bar'], 'exit_bar': i,
                'pnl': pnl, 'pnl_pct': pnl / (long_fill * pv * ct) * 100.0,
                'mae': cur['mae'], 'mfe': cur['mfe'], 'reason': reason,
            })

        def _record_short(i, price, reason):
            nonlocal realized, cur_s
            pnl = (short_fill - price) * pv * ct - comm
            realized += pnl
            trades.append({
                'type': 'Short', 'entry_dt': dts[cur_s['entry_bar']],
                'entry_price': short_fill, 'exit_dt': dts[i], 'exit_price': price,
                'entry_bar': cur_s['entry_bar'], 'exit_bar': i,
                'pnl': pnl, 'pnl_pct': pnl / (short_fill * pv * ct) * 100.0,
                'mae': cur_s['mae'], 'mfe': cur_s['mfe'], 'reason': reason,
            })

        for i in range(n):
            was_long, was_short = isInLong, isInShort

            # ---------- Buy entry ----------
            if buy[i] == 1 and not isInLong:
                stop_pt = max3[i]
                if limit_mode and not np.isnan(atr_off[i]):
                    long_fill = max(O[i] - atr_off[i], L[i])  # AB price-bound clamp to Low
                else:
                    long_fill = O[i]
                long_stop = O[i] - stop_pt
                long_profit = O[i] + cls.profit_mult * stop_pt
                isInLong = True
                sell_stop_n = sp
                cur = {'entry_bar': i, 'mae': 0.0, 'mfe': 0.0}
                if isInShort:
                    buy_force_cover = True

            # ---------- Short entry ----------
            if shr[i] == 1 and not isInShort:
                stop_pt = max3[i]
                if limit_mode and not np.isnan(atr_off[i]):
                    short_fill = min(O[i] + atr_off[i], H[i])  # AB price-bound clamp to High
                else:
                    short_fill = O[i]
                short_stop = O[i] + stop_pt
                short_profit = O[i] - cls.profit_mult * stop_pt
                isInShort = True
                cur_s = {'entry_bar': i, 'mae': 0.0, 'mfe': 0.0}
                if isInLong:
                    short_force_sell = True

            # ---------- Long exits ----------
            if isInLong and cur is not None:
                cur['mae'] = min(cur['mae'], L[i] - long_fill)
                cur['mfe'] = max(cur['mfe'], H[i] - long_fill)
                exit_price = None
                if (L[i] <= long_stop) or short_force_sell:
                    if short_force_sell:
                        exit_price, reason = O[i], 'force'
                        short_force_sell = False
                    else:
                        exit_price, reason = long_stop, 'stop'
                elif H[i] >= long_profit:
                    exit_price, reason = long_profit, 'profit'
                if exit_price is not None:
                    buy[i] = 0  # AFL cancels same-bar entries (Buy[i] = 0)
                    if cur['entry_bar'] != i:
                        _record_long(i, exit_price, reason)
                    isInLong = False
                    cur = None
                else:
                    j = i - sell_stop_n
                    if j >= 0 and buy[j] == 1 and buy[i] != 1:
                        # N-bar stop at Open
                        _record_long(i, O[i], 'nbar')
                        isInLong = False
                        cur = None
                    else:
                        if j >= 0 and buy[j] == 1 and buy[i] == 1:
                            sell_stop_n += 1

            # ---------- Short exits ----------
            if isInShort and cur_s is not None:
                cur_s['mae'] = min(cur_s['mae'], short_fill - H[i])
                cur_s['mfe'] = max(cur_s['mfe'], short_fill - L[i])
                exit_price = None
                if (H[i] >= short_stop) or buy_force_cover:
                    if buy_force_cover:
                        exit_price, reason = O[i], 'force'
                        buy_force_cover = False
                    else:
                        exit_price, reason = short_stop, 'stop'
                elif L[i] <= short_profit:
                    exit_price, reason = short_profit, 'profit'
                if exit_price is not None:
                    shr[i] = 0
                    if cur_s['entry_bar'] != i:
                        _record_short(i, exit_price, reason)
                    isInShort = False
                    cur_s = None
                else:
                    j = i - sp
                    if (j >= 0 and (shr[j] == 1 or buy_force_cover) and shr[i] != 1):
                        _record_short(i, O[i], 'nbar')
                        isInShort = False
                        cur_s = None

            # ---------- equity / exposure ----------
            unreal = 0.0
            if isInLong:
                unreal = (C[i] - long_fill) * pv * ct
            elif isInShort:
                unreal = (short_fill - C[i]) * pv * ct
            equity[i] = cls.initial_capital + realized + unreal
            in_market[i] = 1 if (was_long or was_short or isInLong or isInShort) else 0

        # close any open position at last bar close (AmiBroker end-of-data behavior)
        if n > 0:
            if isInLong and cur is not None:
                cur['mae'] = min(cur['mae'], L[n - 1] - long_fill)
                cur['mfe'] = max(cur['mfe'], H[n - 1] - long_fill)
                _record_long(n - 1, C[n - 1], 'eod')
            if isInShort and cur_s is not None:
                cur_s['mae'] = min(cur_s['mae'], short_fill - H[n - 1])
                cur_s['mfe'] = max(cur_s['mfe'], short_fill - L[n - 1])
                _record_short(n - 1, C[n - 1], 'eod')
            equity[n - 1] = cls.initial_capital + realized

        return trades, equity, in_market

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @classmethod
    def _compute_stats(cls, trades, equity, in_market, dt_index):
        n = len(equity)
        td = pd.DataFrame(trades)
        total = len(td)
        initial = cls.initial_capital

        stats = {}
        stats['Initial Capital'] = initial
        net = float(td['pnl'].sum()) if total else 0.0
        stats['End Capital'] = initial + net
        stats['Net Profit'] = net
        stats['Net Profit %'] = net / initial * 100.0 if initial else 0.0
        stats['Exposure %'] = float(in_market.sum()) / n * 100.0 if n else 0.0

        years = max((dt_index[-1] - dt_index[0]).total_seconds() / (365.25 * 86400.0), 1e-9) if n > 1 else 1e-9
        end_eq = equity[-1]
        stats['Annual Return %'] = ((end_eq / initial) ** (1.0 / years) - 1.0) * 100.0 if initial > 0 and end_eq > 0 else 0.0

        stats['Total Commission Cost'] = cls.commission_per_trade * total
        stats['Number of trades'] = total
        stats['Number of Long trades'] = int((td['type'] == 'Long').sum()) if total else 0
        stats['Number of Short trades'] = int((td['type'] == 'Short').sum()) if total else 0

        wins = td[td['pnl'] > 0] if total else td
        losses = td[td['pnl'] <= 0] if total else td
        stats['Number of wins'] = len(wins)
        stats['Number of wins %'] = len(wins) / total * 100.0 if total else 0.0
        stats['Total win amount'] = float(wins['pnl'].sum()) if total else 0.0
        stats['Number of loss'] = len(losses)
        stats['Number of loss %'] = len(losses) / total * 100.0 if total else 0.0
        stats['Total loss amount'] = float(losses['pnl'].sum()) if total else 0.0
        stats['Average Profit/loss'] = float(td['pnl'].mean()) if total else 0.0
        stats['Average Profit/loss %'] = float(td['pnl_pct'].mean()) if total else 0.0
        stats['Max trade drawdown'] = float(td['pnl_pct'].min()) if total else 0.0

        peak = np.maximum.accumulate(equity)
        mdd = float(((peak - equity) / peak).max() * 100.0) if n else 0.0
        stats['Maximum system drawdown'] = mdd
        stats['CAR/MaxDD'] = stats['Annual Return %'] / mdd if mdd > 0 else float('inf')

        gross_loss = stats['Total loss amount']
        stats['Profit Factor'] = (stats['Total win amount'] / abs(gross_loss)) if gross_loss < 0 else float('inf')

        rets = np.diff(equity) / equity[:-1] if n > 1 else np.array([])
        if len(rets) > 1 and np.std(rets, ddof=1) > 0:
            bpy = len(rets) / years
            ex = rets - cls.risk_free_annual / bpy
            stats['Sharpe Ratio'] = float(ex.mean() / ex.std(ddof=1) * np.sqrt(bpy))
        else:
            stats['Sharpe Ratio'] = 0.0

        eq_s = pd.Series(equity)
        roll_max = eq_s.rolling(cls.ulcer_period, min_periods=1).max()
        dd_pct = (eq_s / roll_max - 1.0) * 100.0
        stats['Ulcer Index'] = float(np.sqrt((dd_pct ** 2).mean()))

        if n > 2 and np.all(equity > 0):
            y = np.log(equity)
            t = np.arange(n, dtype=float)
            slope, intercept = np.polyfit(t, y, 1)
            sse = float(((y - (slope * t + intercept)) ** 2).sum())
            sxx = float(((t - t.mean()) ** 2).sum())
            se = np.sqrt(sse / (n - 2) / sxx) if sse > 0 and sxx > 0 else np.nan
            stats['K-Ratio'] = float(slope / se) if se and se > 0 else float('nan')
        else:
            stats['K-Ratio'] = float('nan')

        stats['Avg Bars Held'] = float((td['exit_bar'] - td['entry_bar']).mean()) if total else 0.0
        stats['Start Date'] = str(dt_index[0])
        stats['End Date'] = str(dt_index[-1])
        stats['Bars'] = n
        return stats

    # ------------------------------------------------------------------
    # Full backtest (custom engine) + output files
    # ------------------------------------------------------------------
    @classmethod
    def run_full_backtest(cls, df: pd.DataFrame) -> dict:
        sig = cls._prepare(df)
        trade_start = pd.Timestamp(cls.start_date)
        gate = trade_start if sig.index[0] < trade_start else None
        trades, equity, in_market = cls._simulate(sig, trade_start_ts=gate)
        # restrict stats/outputs to the trading window (AmiBroker From-To range)
        if gate is not None:
            keep = sig.index >= gate
            sig = sig[keep]
            equity = equity[keep]
            in_market = in_market[keep]
        stats = cls._compute_stats(trades, equity, in_market, sig.index)
        return {
            'signals': sig,
            'trades': pd.DataFrame(trades),
            'equity': pd.Series(equity, index=sig.index),
            'stats': stats,
        }

    @classmethod
    def write_output_files(cls, results: dict, out_dir: str = OUTPUT_DIR) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        sig = results['signals']
        td = results['trades'].copy()
        pv, ct = cls.point_value, cls.num_contracts

        # ---- File 1: statistics HTML (with embedded equity chart) ----
        html_path = os.path.join(out_dir, f'S4003_1_{ts}.html')
        chart_html = cls._equity_chart_html(results['equity'])
        rows = ''.join(
            f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v:,.2f}</td></tr>'
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
            else f'<tr><th style="text-align:left">{k}</th><td style="text-align:right">{v}</td></tr>'
            for k, v in results['stats'].items())
        html = f"""<html><head><title>S4003_1 Backtest Statistics</title>
<style>body{{font-family:Arial;}} table{{border-collapse:collapse;}}
th,td{{padding:4px 12px;border:1px solid #999;}} th{{background:#eee;}}</style></head>
<body><h2>S4003_1 Backtest Statistics</h2>
<p>Symbol: {cls.symbol} | Mode: {cls.entry_price_mode} | ATR: {cls.atr_mode} |
Period: {results['stats']['Start Date']} - {results['stats']['End Date']} |
Generated: {datetime.now().strftime(DT_FORMAT)}</p>
<table>{rows}</table>
<h3>Equity Chart</h3>{chart_html}
</body></html>"""
        with open(html_path, 'w') as f:
            f.write(html)

        # ---- File 2: trades CSV (descending by entry datetime) ----
        trades_path = os.path.join(out_dir, f'S4003_1_{ts}.csv')
        if len(td):
            td = td.sort_values('entry_dt', ascending=False)
            td['cum_pnl'] = td.sort_values('entry_dt')['pnl'].cumsum().reindex(td.index)
            if_dir = np.where(td['type'] == 'Long', 1, -1)
            pct_change = (td['exit_price'] - td['entry_price']) / td['entry_price'] * 100.0 * if_dir
            out_trades = pd.DataFrame({
                'Symbol': cls.symbol,
                'Trade Type': td['type'],
                'Entry DateTime': td['entry_dt'].dt.strftime(DT_FORMAT),
                'Entry Price': td['entry_price'].round(2),
                'Exit DateTime': td['exit_dt'].dt.strftime(DT_FORMAT),
                'Exit Price': td['exit_price'].round(2),
                '% Change': pct_change.round(4),
                'Profit': td['pnl'].round(2),
                '% Profit': td['pnl_pct'].round(4),
                'Shares': ct,
                'Position Value': (td['entry_price'] * pv * ct).round(2),
                'Cumulative Profit': td['cum_pnl'].round(2),
                'Bars Held': td['exit_bar'] - td['entry_bar'],
                'MAE': (td['mae'] * pv * ct).round(2),
                'MFE': (td['mfe'] * pv * ct).round(2),
                'Exit Reason': td['reason'],
            })
        else:
            out_trades = pd.DataFrame(columns=[
                'Symbol', 'Trade Type', 'Entry DateTime', 'Entry Price',
                'Exit DateTime', 'Exit Price', '% Change', 'Profit', '% Profit',
                'Shares', 'Position Value', 'Cumulative Profit', 'Bars Held',
                'MAE', 'MFE', 'Exit Reason'])
        out_trades.to_csv(trades_path, index=False)

        # ---- File 3: explore CSV (every 1-min bar, descending) ----
        explore_path = os.path.join(out_dir, f'S4003_1_explore_{ts}.csv')
        exp = pd.DataFrame({
            'Symbol': cls.symbol,
            'Trade': np.where(sig['buy00'], 'buyer00', np.where(sig['short00'], 'short00', '')),
            'DateTime': sig.index.strftime(DT_FORMAT),
            'Open': sig['Open'].round(2),
            'Close': sig['Close'].round(2),
            'High': sig['High'].round(2),
            'Low': sig['Low'].round(2),
            'cCrossUpBBTop': sig['buy00'].astype(int),
            'cCrossDownBBBottom': sig['short00'].astype(int),
        }).iloc[::-1]
        exp.to_csv(explore_path, index=False)

        return {'html': html_path, 'trades_csv': trades_path, 'explore_csv': explore_path}

    @staticmethod
    def _equity_chart_html(equity: pd.Series) -> str:
        """Return an HTML fragment embedding the equity chart.
        Uses matplotlib (PNG base64) when available; otherwise falls back to a
        dependency-free inline SVG chart so output writing never fails."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(equity.index, equity.to_numpy(), linewidth=0.9, color='#1f77b4')
            ax.set_title('S4003_1 Equity Curve')
            ax.set_ylabel('Equity ($)')
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{b64}" alt="equity chart"/>'
        except ImportError:
            return strategyS4003._equity_chart_svg(equity)

    @staticmethod
    def _equity_chart_svg(equity: pd.Series) -> str:
        """Dependency-free SVG line chart of the equity curve."""
        w, h = 960, 420
        pad_l, pad_r, pad_t, pad_b = 80, 25, 20, 45
        vals = equity.to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            return '<p>(no equity data)</p>'
        step = max(1, n // 2000)          # downsample for a compact SVG
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        iw, ih = w - pad_l - pad_r, h - pad_t - pad_b

        def X(i):
            return pad_l + iw * (i / max(n - 1, 1))

        def Y(v):
            return pad_t + ih * (1 - (v - vmin) / (vmax - vmin))

        pts = ' '.join(f'{X(i):.1f},{Y(vals[i]):.1f}' for i in idx)
        grid = []
        for k in range(5):
            v = vmin + (vmax - vmin) * k / 4
            gy = Y(v)
            grid.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" '
                        f'stroke="#dddddd" stroke-width="1"/>'
                        f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                        f'font-size="11" fill="#555555">{v:,.0f}</text>')
        t0 = equity.index[0].strftime('%d/%m/%Y %H:%M')
        t1 = equity.index[-1].strftime('%d/%m/%Y %H:%M')
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                f'xmlns="http://www.w3.org/2000/svg">'
                f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'
                f'<text x="{w / 2}" y="14" text-anchor="middle" font-size="13" '
                f'font-weight="bold" fill="#333333">S4003_1 Equity Curve</text>'
                + ''.join(grid) +
                f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1"/>'
                f'<text x="{pad_l}" y="{h - 8}" font-size="11" fill="#555555">{t0}</text>'
                f'<text x="{w - pad_r}" y="{h - 8}" text-anchor="end" font-size="11" '
                f'fill="#555555">{t1}</text>'
                f'<text x="{pad_l - 8}" y="{pad_t + 4}" text-anchor="end" font-size="11" '
                f'fill="#555555">Equity ($)</text>'
                f'</svg>')

    # ------------------------------------------------------------------
    # Interface for VS_0003_test/otherTest.py (look-ahead bias test)
    # ------------------------------------------------------------------
    @classmethod
    def generate_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with 'signal' column: 1 = buy entry bar, -1 = short entry
        bar, 0 = hold. Signal at bar T derives only from bar T-1 data."""
        out = cls._prepare(df)
        signal = np.where(out['buySig'] == 1, 1, np.where(out['shortSig'] == 1, -1, 0))
        out['signal'] = signal.astype(int)
        return out

    # ------------------------------------------------------------------
    # Interface for VS_0003_test/monteCarloSimulation.py
    # ------------------------------------------------------------------
    @classmethod
    def run_backtest(cls, df: pd.DataFrame, init_balance: float, position_size: int) -> list:
        """Manual bar-by-bar backtest; returns per-trade P&L in dollars
        (net of commission). init_balance kept for signature compatibility."""
        sig = cls._prepare(df)
        trades, _equity, _im = cls._simulate(sig, contracts=int(position_size))
        return [t['pnl'] for t in trades]

    # ------------------------------------------------------------------
    # backtesting.py Strategy interface (backtest.py / walkForwardTest.py)
    # Approximation: entries fill next bar at Open (matches AFL), sl/tp orders
    # replicate stop/profit fills, N-bar exit and force exit emit market orders.
    # ------------------------------------------------------------------
    def init(self):
        df = pd.DataFrame({
            'Open': self.data.Open, 'High': self.data.High,
            'Low': self.data.Low, 'Close': self.data.Close,
            'Volume': self.data.Volume if self.data.Volume is not None else 1.0,
        }, index=self.data.index)
        sig = type(self)._prepare(df)
        self._buy00 = sig['buy00'].to_numpy(bool)
        self._short00 = sig['short00'].to_numpy(bool)
        self._max3 = sig['max3BarAtr14_1'].to_numpy(float)
        self.vwap50 = self.I(lambda: sig['vwap50'].to_numpy(float), name='vwap50')
        self._anchored = False
        self._entry_bar = None

    def next(self):
        cls = type(self)
        i = len(self.data) - 1
        if i < 1:
            return
        buy_sig = self._buy00[i]
        short_sig = self._short00[i]

        # anchor sl/tp on the bar the trade filled (fill = this bar's open)
        if self.position and not self._anchored:
            stop_pt = self._max3[i] if not np.isnan(self._max3[i]) else 0.0
            for t in self.trades:
                if t.sl is None and t.tp is None:
                    if t.is_long:
                        t.sl = t.entry_price - cls.stop_mult * stop_pt
                        t.tp = t.entry_price + cls.profit_mult * stop_pt
                    else:
                        t.sl = t.entry_price + cls.stop_mult * stop_pt
                        t.tp = t.entry_price - cls.profit_mult * stop_pt
            self._anchored = True
            self._entry_bar = i

        if self.position:
            # force exit on opposite signal (fills next bar open, like AFL)
            if self.position.is_long and short_sig:
                self.position.close()
                self._anchored = False
                self.sell(size=cls.num_contracts)
                return
            if self.position.is_short and buy_sig:
                self.position.close()
                self._anchored = False
                self.buy(size=cls.num_contracts)
                return
            # N-bar stop: AFL exits at open of entry+15; close() placed at the
            # close of entry+14 fills there. Skip while entry signals keep firing.
            if self._entry_bar is not None:
                elapsed = i - self._entry_bar
                sig_now = buy_sig if self.position.is_long else short_sig
                if elapsed >= cls.stop_period1 - 1 and not sig_now:
                    self.position.close()
                    self._anchored = False
                    return
        else:
            if buy_sig:
                self.buy(size=cls.num_contracts)
                self._anchored = False
            elif short_sig:
                self.sell(size=cls.num_contracts)
                self._anchored = False
