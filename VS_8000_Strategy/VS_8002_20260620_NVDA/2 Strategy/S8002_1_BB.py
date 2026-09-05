"""
S8002_1_BB.py
=============
Bollinger Band Cross Strategy for NVDA 5-minute bars.

Class strategyS8002BBV1 implements:
  - BB(20, 2) cross-up -> buy; BB(20, 2) cross-down -> short
  - 5-bar exit (sell/cover)
  - ATR(7)*2.5 stop loss (previous bar ATR)
  - Force close at 15:55
  - Entry window 09:30 - 14:55
  - Position size = 110 shares

Dual-mode:
  - Subclasses backtesting.Strategy for backtest.py / walkForwardTest.py / monkeyTest
  - Exposes generate_signals(df) for lookAheadBiasTest
  - Exposes run_backtest(df, init_balance, position_size) for monteCarloSimulation

Plan: .github/prompts/plan-strategy_S8002Bb_V1.prompt.md
"""

import pandas as pd
import numpy as np
from backtesting import Strategy


# ===========================================================================
# Indicator helpers (pure pandas, no talib dependency)
# ===========================================================================
def _calc_bollinger(close: pd.Series, period: int = 20, std_mult: int = 2):
    """Return (top_band, bottom_band) Series."""
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    top = rolling_mean + (rolling_std * std_mult)
    bot = rolling_mean - (rolling_std * std_mult)
    return top, bot


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 7) -> pd.Series:
    """True Range ATR: TR=max(H-L, |H-prevC|, |L-prevC|); atr=TR.rolling(n).mean()."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


# ===========================================================================
# Strategy class
# ===========================================================================
class strategyS8002BBV1(Strategy):
    """
    Bollinger Band Cross Strategy V1 (NVDA 5-min).

    Buy  : prev bar cCrossUpBBTop true  & time in [09:30, 14:55]
    Short: prev bar cCrossDownBBBottom true & time in [09:30, 14:55]
    Sell : 5 bars after buy, OR ATR stop hit, OR force close at 15:55
    Cover: 5 bars after short, OR ATR stop hit, OR force close at 15:55
    Stop : long  stop = Close - prevATR(7)*2.5
           short stop = Close + prevATR(7)*2.5
    Size : 110 shares per trade
    """

    # --- Strategy Parameters (class attributes, optimizable) ---
    bb_period = 20
    bb_std = 2
    atr_period = 7
    atr_mult = 2.5
    hold_bars = 5
    position_size = 110

    # --- Costs / mode ---
    commission = 0.0
    slippage = 0.03       # live only; not applied in backtest
    is_live = False

    # --- Time filters ---
    entry_start = '09:30'
    entry_end = '14:55'
    force_close_time = '15:55'

    # ------------------------------------------------------------------
    # Phase 2 - init() for backtesting.py
    # ------------------------------------------------------------------
    def init(self):
        # Compute indicators and register with self.I() so they are
        # available as sliced arrays in next().
        self.bbTop = self.I(
            lambda: _calc_bollinger(pd.Series(self.data.Close), self.bb_period, self.bb_std)[0].values
        )
        self.bbBottom = self.I(
            lambda: _calc_bollinger(pd.Series(self.data.Close), self.bb_period, self.bb_std)[1].values
        )
        self.atr7 = self.I(
            lambda: _calc_atr(
                pd.Series(self.data.High),
                pd.Series(self.data.Low),
                pd.Series(self.data.Close),
                self.atr_period,
            ).values
        )

        # Track entry bar index for N-bar exit
        self._entry_bar_index = None

    # ------------------------------------------------------------------
    # Phase 3 - next() for backtesting.py
    # ------------------------------------------------------------------
    def next(self):
        price = self.data.Close[-1]
        current_time = self.data.index[-1].time()

        # Parse time boundaries once (cached on class for speed)
        entry_start_t = pd.Timestamp(self.entry_start).time()
        entry_end_t = pd.Timestamp(self.entry_end).time()
        force_close_t = pd.Timestamp(self.force_close_time).time()

        # --- Force close at 15:55 ---
        if current_time == force_close_t and self.position:
            self.position.close()
            self._entry_bar_index = None
            return

        # --- 5-bar exit ---
        if self.position and self._entry_bar_index is not None:
            bars_since_entry = len(self.data) - 1 - self._entry_bar_index
            if bars_since_entry >= self.hold_bars:
                self.position.close()
                self._entry_bar_index = None
                return

        # --- Entry signals (use previous bar's cross) ---
        # Need at least 2 bars of indicator history
        if len(self.data) < 2:
            return

        prev_cross_up = (self.data.Close[-2] <= self.bbTop[-2]) and (self.data.Close[-1] > self.bbTop[-1])
        prev_cross_down = (self.data.Close[-2] >= self.bbBottom[-2]) and (self.data.Close[-1] < self.bbBottom[-1])

        # Actually the spec says: signal triggered when PREVIOUS bar buy00 is true.
        # buy00 = cCrossUpBBTop at that bar. So we check the cross that occurred on
        # the previous completed bar (index -2 relative to current).
        # cCrossUpBBTop[i] = (Close[i-1] <= bbTop[i-1]) & (Close[i] > bbTop[i])
        # For "previous bar buy00 true" we look at bar [-2]:
        if len(self.data) >= 3:
            buy00_prev = (self.data.Close[-3] <= self.bbTop[-3]) and (self.data.Close[-2] > self.bbTop[-2])
            short00_prev = (self.data.Close[-3] >= self.bbBottom[-3]) and (self.data.Close[-2] < self.bbBottom[-2])
        else:
            buy00_prev = False
            short00_prev = False

        in_entry_window = entry_start_t <= current_time <= entry_end_t

        # --- Buy ---
        if buy00_prev and in_entry_window and not self.position:
            # ATR stop uses previous bar ATR (atr7[-2])
            prev_atr = self.atr7[-2] if not np.isnan(self.atr7[-2]) else 0.0
            stop_price = price - prev_atr * self.atr_mult
            self.buy(size=self.position_size, sl=stop_price)
            self._entry_bar_index = len(self.data) - 1
            return

        # --- Short ---
        if short00_prev and in_entry_window and not self.position:
            prev_atr = self.atr7[-2] if not np.isnan(self.atr7[-2]) else 0.0
            stop_price = price + prev_atr * self.atr_mult
            self.sell(size=self.position_size, sl=stop_price)
            self._entry_bar_index = len(self.data) - 1
            return

    # ------------------------------------------------------------------
    # Phase 4 - generate_signals(df) for lookAheadBiasTest
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Pure-pandas indicator computation. Returns df with new columns."""
        out = df.copy()
        bb_top, bb_bot = _calc_bollinger(out['Close'], 20, 2)
        out['bbTop'] = bb_top
        out['bbBottom'] = bb_bot
        out['atr7'] = _calc_atr(out['High'], out['Low'], out['Close'], 7)

        prev_close = out['Close'].shift(1)
        # cCrossUpBBTop[i] = (prev Close <= bbTop) & (cur Close > bbTop)
        out['cCrossUpBBTop'] = (prev_close <= out['bbTop']) & (out['Close'] > out['bbTop'])
        # cCrossDownBBBottom[i] = (prev Close >= bbBottom) & (cur Close < bbBottom)
        out['cCrossDownBBBottom'] = (prev_close >= out['bbBottom']) & (out['Close'] < out['bbBottom'])
        return out

    @classmethod
    def generate_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry signals for lookAheadBiasTest.
        Returns df with 'signal' column: 1=buy, -1=short, 0=hold.

        Signal at bar T is based on buy00/short00 at bar T-1 (shifted),
        so no look-ahead bias.
        """
        # Ensure datetime index
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if 'datetime' in out.columns:
                out['datetime'] = pd.to_datetime(out['datetime'], format='%m/%d/%Y %H:%M')
                out = out.set_index('datetime')
            else:
                out.index = pd.to_datetime(out.index)

        # Normalize column names to capitalized (backtesting.py convention)
        rename = {}
        for c in out.columns:
            cl = c.lower()
            if cl == 'open':
                rename[c] = 'Open'
            elif cl == 'high':
                rename[c] = 'High'
            elif cl == 'low':
                rename[c] = 'Low'
            elif cl == 'close':
                rename[c] = 'Close'
            elif cl == 'volume':
                rename[c] = 'Volume'
        out = out.rename(columns=rename)

        out = cls._compute_indicators(out)

        # buy00 / short00 at each bar
        out['buy00'] = out['cCrossUpBBTop']
        out['short00'] = out['cCrossDownBBBottom']

        # Signal triggers when PREVIOUS bar buy00/short00 is true
        out['signal'] = 0
        buy_signal = out['buy00'].shift(1).fillna(False).astype(bool)
        short_signal = out['short00'].shift(1).fillna(False).astype(bool)

        # Time filter: entry only between 09:30 and 14:55
        t = out.index.time
        entry_start_t = pd.Timestamp(cls.entry_start).time()
        entry_end_t = pd.Timestamp(cls.entry_end).time()
        in_window = (t >= entry_start_t) & (t <= entry_end_t)

        out.loc[buy_signal & in_window, 'signal'] = 1
        out.loc[short_signal & in_window, 'signal'] = -1

        return out

    # ------------------------------------------------------------------
    # Phase 5 - run_backtest(df, init_balance, position_size) for monteCarloSimulation
    # ------------------------------------------------------------------
    @classmethod
    def run_backtest(cls, df: pd.DataFrame, init_balance: float, position_size: int):
        """
        Manual backtest loop returning a list of per-trade P&L (monetary).

        Pattern follows S8001_2_ConvertFromGemini.py:
          - Entry when prev-bar signal triggers; entry_price = Open[current] (backtest)
          - Exits: 5-bar exit (at Open), ATR stop, 15:55 force close
          - P&L = (exit - entry) * position_size  (long)
                = (entry - exit) * position_size  (short)
        """
        # Prepare dataframe with indicators and signals
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if 'datetime' in out.columns:
                out['datetime'] = pd.to_datetime(out['datetime'], format='%m/%d/%Y %H:%M')
                out = out.set_index('datetime')
            else:
                out.index = pd.to_datetime(out.index)

        rename = {}
        for c in out.columns:
            cl = c.lower()
            if cl == 'open':
                rename[c] = 'Open'
            elif cl == 'high':
                rename[c] = 'High'
            elif cl == 'low':
                rename[c] = 'Low'
            elif cl == 'close':
                rename[c] = 'Close'
            elif cl == 'volume':
                rename[c] = 'Volume'
        out = out.rename(columns=rename)

        out = cls._compute_indicators(out)
        out['buy00'] = out['cCrossUpBBTop']
        out['short00'] = out['cCrossDownBBBottom']

        # Shift signals by 1 (signal triggers on prev bar)
        out['buy_signal'] = out['buy00'].shift(1).fillna(False).astype(bool)
        out['short_signal'] = out['short00'].shift(1).fillna(False).astype(bool)

        entry_start_t = pd.Timestamp(cls.entry_start).time()
        entry_end_t = pd.Timestamp(cls.entry_end).time()
        force_close_t = pd.Timestamp(cls.force_close_time).time()

        trade_pnl = []
        isInLong = False
        isInShort = False
        entry_bar_index = 0
        entry_price = 0.0
        stop_price = 0.0

        for i in range(len(out)):
            cur_time = out.index[i].time()
            in_window = entry_start_t <= cur_time <= entry_end_t
            is_force_close = (cur_time == force_close_t)

            # --- Exits (checked before entries on the same bar) ---
            if isInLong:
                exit_price = None
                # ATR stop: Low <= stop_price
                if out['Low'].iloc[i] <= stop_price:
                    exit_price = stop_price
                # 5-bar exit
                elif (i - entry_bar_index) >= cls.hold_bars:
                    exit_price = out['Open'].iloc[i]
                # Force close at 15:55
                elif is_force_close:
                    exit_price = out['Open'].iloc[i]

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * position_size
                    trade_pnl.append(pnl)
                    isInLong = False

            if isInShort:
                exit_price = None
                # ATR stop: High >= stop_price
                if out['High'].iloc[i] >= stop_price:
                    exit_price = stop_price
                # 5-bar exit
                elif (i - entry_bar_index) >= cls.hold_bars:
                    exit_price = out['Open'].iloc[i]
                # Force close at 15:55
                elif is_force_close:
                    exit_price = out['Open'].iloc[i]

                if exit_price is not None:
                    pnl = (entry_price - exit_price) * position_size
                    trade_pnl.append(pnl)
                    isInShort = False

            # --- Entries (only if flat) ---
            if not isInLong and not isInShort:
                if out['buy_signal'].iloc[i] and in_window:
                    entry_price = out['Open'].iloc[i]
                    # ATR stop uses previous bar ATR
                    prev_atr = out['atr7'].iloc[i - 1] if i > 0 and not np.isnan(out['atr7'].iloc[i - 1]) else 0.0
                    stop_price = entry_price - prev_atr * cls.atr_mult
                    isInLong = True
                    entry_bar_index = i
                elif out['short_signal'].iloc[i] and in_window:
                    entry_price = out['Open'].iloc[i]
                    prev_atr = out['atr7'].iloc[i - 1] if i > 0 and not np.isnan(out['atr7'].iloc[i - 1]) else 0.0
                    stop_price = entry_price + prev_atr * cls.atr_mult
                    isInShort = True
                    entry_bar_index = i

        return trade_pnl


# ===========================================================================
# Example / smoke test
# ===========================================================================
if __name__ == "__main__":
    import os
    import sys

    # Locate a sample NVDA 5-min CSV
    candidate = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'VS_0006_data', 'VS_6001_GetMarketDataToCsv', 'NVDA_20250101_20260430_5Min.csv'
    )
    # Fallback to data file directory
    if not os.path.exists(candidate):
        candidate = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_6002_findAlpha\csvExcel\NVDA_20250101_20260430_5Min_0930_1600.csv"

    if not os.path.exists(candidate):
        print("No sample CSV found; skipping smoke test.")
        sys.exit(0)

    print(f"Smoke test with: {candidate}")

    # Test generate_signals
    df = pd.read_csv(candidate)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
    df = df.set_index('datetime')
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
    })
    df = df.sort_index()

    sig_df = strategyS8002BBV1.generate_signals(df)
    n_buy = (sig_df['signal'] == 1).sum()
    n_short = (sig_df['signal'] == -1).sum()
    print(f"generate_signals: {n_buy} buy signals, {n_short} short signals")

    # Test run_backtest
    pnl = strategyS8002BBV1.run_backtest(df, init_balance=100000, position_size=110)
    print(f"run_backtest: {len(pnl)} trades, total P&L = ${sum(pnl):.2f}")
    if pnl:
        print(f"  avg P&L/trade = ${sum(pnl) / len(pnl):.2f}")
        print(f"  max win = ${max(pnl):.2f}, max loss = ${min(pnl):.2f}")
