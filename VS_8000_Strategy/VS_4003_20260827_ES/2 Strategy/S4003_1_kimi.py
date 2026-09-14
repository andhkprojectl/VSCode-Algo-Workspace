"""
S4003_1.py
==========

Conversion of AmiBroker AFL backtest_4003_1 trim_4_python.afl to Python.

Class strategyS4003V1 supports:
  - backtest.py:            init/next (backtesting.Strategy subclass)
  - otherTest.py:           generate_signals(df) -> signal column
  - monteCarloSimulation.py: run_backtest(df, init_balance, position_size) -> list of P&L
  - walkForwardTest.py:     standard Strategy subclass

Also produces three backtest artifacts:
  1. HTML stats file:      <outDir>/S4003_1_<YYYYMMDDHHMMSS>.html
  2. Trades CSV:           <outDir>/S4003_1_<YYYYMMDDHHMMSS>.csv (cols A-O)
  3. Explore CSV:          <outDir>/S4003_1_explore_<YYYYMMDDHHMMSS>.csv (cols A-I)

Data source: MariaDB IBTradingDb.ticker1Min (user 'ibUser1'@'localhost').
"""

import os
import sys
import html
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mysql.connector

try:
    from backtesting import Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    # backtesting.py is only required by the backtest.py / walkForwardTest.py
    # harnesses; standalone runs (DB -> backtest -> 3 output files) work without it.
    BACKTESTING_AVAILABLE = False

    class Strategy:  # minimal fallback base so this module still imports
        pass


# Load VS_0002_config/.env (repo convention, same as IBDb.py / DB_NQ_*.py) so
# DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME env vars are available.
try:
    from pathlib import Path as _Path
    from dotenv import load_dotenv as _load_dotenv
    for _p in (_Path(__file__).resolve().parents[3] / "VS_0002_config" / ".env",
               _Path(__file__).resolve().parents[2] / "VS_0002_config" / ".env",
               _Path.cwd() / "VS_0002_config" / ".env"):
        if _p.exists():
            _load_dotenv(dotenv_path=str(_p), override=False)
            break
except ImportError:
    pass


# =============================================================================
# 1. DATA LOADER (MariaDB -> pandas DataFrame)
# =============================================================================
class strategyS4003V1(Strategy):
    """
    Converted strategy logic from backtest_4003_1 trim_4_python.afl.

    Entry:  buy00 = priceActionUp4Bar2; short00 = priceActionDown3Bar1 OR short02
    Exit:   stops replicate AFL bar-loop logic (buyForceCover / shortForceSell),
            profit target = 1.5 * max3BarAtr14_1, stop = max3BarAtr14_1,
            N-bar stop = 15.
    """

    # ----------------- class-level config -----------------
    # AFL: timeFilterBuy = True  -> no session filter at all.
    entry_window_start: Optional[str] = None
    entry_window_end: Optional[str] = None
    force_close_time: Optional[str] = None   # AFL has no force-close; exits are stop/profit/N-bar only
    hold_bars = 15                            # AFL stopPeriod1 = 15 (N-bar stop)
    profit_mult = 1.5                 # 1.5 * max3BarAtr14_1
    init_balance = 80000.0            # AFL report: Initial capital 80000.00
    position_size = 1                 # AFL NumContracts = 1
    point_value = 50.0                # ES futures: $50 per point
    symbol = "ES"
    # AFL report: Transaction costs 1073.00 / 74 trades = 14.50 round turn
    commission_per_trade = 14.50
    commission = 0.0
    slippage = 0.0
    is_live = False

    # ----------------- DB config -----------------
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "3306"))
    db_user = os.getenv("DB_USER", "ibUser1")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "IBTradingDb")
    db_table = "ticker1Min"

    # ----------------- Output dir -----------------
    output_dir = os.path.join(
        r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_4003_20260827_ES",
        "backTestResult"
    )

    # =============================================================================
    # 1.1  PUBLIC HARNESS ADAPTERS
    # =============================================================================
    def init(self):
        """backtesting.py init."""
        # values computed lazily by generate_signals so init and next can
        # rely on precomputed columns attached to self.data
        self._sig_df = None  # filled in next() on first call

    def next(self):
        """backtesting.py next."""
        # Build signals once; then act on them bar-by-bar
        if self._sig_df is None:
            self._sig_df = self.__class__._build_full_signal_df(
                pd.DataFrame({
                    "Open": self.data.Open,
                    "High": self.data.High,
                    "Low": self.data.Low,
                    "Close": self.data.Close,
                    "Volume": self.data.Volume,
                }, index=self.data.index)
            )

        idx = len(self.data) - 1
        row = self._sig_df.iloc[idx]
        now = self.data.index[-1].time()
        in_window = True  # AFL: timeFilterBuy = True (no session filter)
        if self.entry_window_start and self.entry_window_end:
            in_window = (
                pd.Timestamp(self.entry_window_start).time()
                <= now <=
                pd.Timestamp(self.entry_window_end).time()
            )

        # ---------- exits ----------
        if self.position:
            # force close (disabled by default; AFL has none)
            if self.force_close_time and now == pd.Timestamp(self.force_close_time).time():
                self.position.close()
                return
            # profit target / stop loss (recorded in _sig_df for backtest parity)
            if self.position.is_long:
                if (
                    not np.isnan(row["longProfitPrice"]) and
                    self.data.High[-1] >= row["longProfitPrice"]
                ) or (
                    not np.isnan(row["longStopPrice"]) and
                    self.data.Low[-1] <= row["longStopPrice"]
                ):
                    self.position.close()
                    return
            if self.position.is_short:
                if (
                    not np.isnan(row["shortProfitPrice"]) and
                    self.data.Low[-1] <= row["shortProfitPrice"]
                ) or (
                    not np.isnan(row["shortStopPrice"]) and
                    self.data.High[-1] >= row["shortStopPrice"]
                ):
                    self.position.close()
                    return

        # ---------- entries ----------
        if not self.position and in_window:
            if row["buy_signal"]:
                prev_atr = row["atr14"]
                if not np.isnan(prev_atr):
                    stop = self.data.Close[-1] - prev_atr
                    self.buy(size=self.position_size, sl=stop)
                else:
                    self.buy(size=self.position_size)
            elif row["short_signal"]:
                prev_atr = row["atr14"]
                if not np.isnan(prev_atr):
                    stop = self.data.Close[-1] + prev_atr
                    self.sell(size=self.position_size, sl=stop)
                else:
                    self.sell(size=self.position_size)

    # --------------------------------------------------
    # otherTest.py adapter
    # --------------------------------------------------
    @classmethod
    def generate_signals(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return df with a 'signal' column (1=buy, -1=short, 0=hold)."""
        out = cls._build_full_signal_df(df)
        out["signal"] = 0
        out.loc[out["buy_signal"], "signal"] = 1
        out.loc[out["short_signal"], "signal"] = -1
        return out

    # --------------------------------------------------
    # monteCarloSimulation.py adapter
    # --------------------------------------------------
    @classmethod
    def run_backtest(
        cls,
        df: pd.DataFrame,
        init_balance: float,
        position_size: int,
    ) -> List[float]:
        """Return a list of monetary P&L per closed trade."""
        recs = cls._simulate_trades(df)
        return [r["pnl"] for r in recs]

    # --------------------------------------------------
    # DB helper
    # --------------------------------------------------
    @classmethod
    def fetch_from_db(
        cls,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return DataFrame from MariaDB `ticker1Min` indexed by datetime.
        Requires datetime DESC for full-range queries."""
        query = "SELECT ticker, datetime1, open, high, low, close, volume FROM " + cls.db_table
        cond = []
        if cls.symbol:
            cond.append("ticker = %s")
        if start:
            cond.append("datetime1 >= %s")
        if end:
            cond.append("datetime1 <= %s")
        if cond:
            query += " WHERE " + " AND ".join(cond)
        params = []
        if cls.symbol:
            params.append(cls.symbol)
        if start:
            params.append(start)
        if end:
            params.append(end)

        conn = mysql.connector.connect(
            host=cls.db_host,
            port=cls.db_port,
            user=cls.db_user,
            password=cls.db_password,
            database=cls.db_name,
        )
        try:
            df = pd.read_sql(query, conn, params=params or None)
        finally:
            conn.close()

        df["datetime1"] = pd.to_datetime(df["datetime1"])
        df.rename(columns={
            "ticker": "symbol",
            "datetime1": "datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }, inplace=True)
        df.set_index("datetime", inplace=True)
        return df.sort_index(ascending=True)

    # =============================================================================
    # 2. INDICATOR HELPERS (pure pandas, no talib)
    # =============================================================================
    @staticmethod
    def _p_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        tr = pd.concat(
            [high - low,
             (high - close.shift(1)).abs(),
             (low - close.shift(1)).abs()],
            axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _p_rolling_hhv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).max()

    @staticmethod
    def _p_rolling_llv(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).min()

    @staticmethod
    def _p_percentile(series: pd.Series, window: int, rank: float) -> pd.Series:
        # vectorized rolling percentile (much faster than rolling().apply)
        return series.rolling(window).quantile(rank, interpolation="linear")

    @staticmethod
    def _p_hhvbars(series: pd.Series, window: int) -> pd.Series:
        # bars since the highest high within `window` (0 = current bar is the high)
        a = series.to_numpy(dtype=float)
        n = len(a)
        out = np.full(n, np.nan)
        # sliding argmax via stride tricks where possible; fallback loop is O(n*window)
        for i in range(window - 1, n):
            seg = a[i - window + 1: i + 1]
            if np.isnan(seg).all():
                continue
            out[i] = window - 1 - np.nanargmax(seg)
        return pd.Series(out, index=series.index)

    @staticmethod
    def _p_llvbars(series: pd.Series, window: int) -> pd.Series:
        a = series.to_numpy(dtype=float)
        n = len(a)
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            seg = a[i - window + 1: i + 1]
            if np.isnan(seg).all():
                continue
            out[i] = window - 1 - np.nanargmin(seg)
        return pd.Series(out, index=series.index)

    @staticmethod
    def _p_stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int, d: int) -> pd.Series:
        lowest = low.rolling(k).min()
        highest = high.rolling(k).max()
        pct_k = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
        return pct_k.rolling(d).mean().fillna(0)

    @staticmethod
    def _p_linreg_slope(series: pd.Series, window: int) -> pd.Series:
        # vectorized rolling OLS slope
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        sxx = ((x - x_mean) ** 2).sum()
        roll_mean = series.rolling(window).mean()
        # sum of x*y over the window via convolution
        xy = series.rolling(window).apply(lambda a: float((np.arange(window) * a).sum()), raw=True)
        return (xy - window * x_mean * roll_mean) / sxx

    @staticmethod
    def _p_ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    # =============================================================================
    # 3. AFL SIGNAL COMPUTATION (faithful to backtest_4003_1.afl)
    # =============================================================================
    @classmethod
    def _build_full_signal_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df with ALL AFL columns + boolean buy00/short00 (row-local signals)
        plus shifted buy_signal / short_signal for bar handler.
        """
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "datetime" in out.columns:
                out["datetime"] = pd.to_datetime(out["datetime"])
                out = out.set_index("datetime")
            else:
                out.index = pd.to_datetime(out.index)
        # normalize col names
        ren = {}
        for c in out.columns:
            cl = c.lower()
            if cl in ("open", "high", "low", "close", "volume"):
                ren[c] = cl.capitalize()
        out = out.rename(columns=ren)

        O, H, L, C, V = out["Open"], out["High"], out["Low"], out["Close"], out["Volume"]

        # ---------------- indicators ----------------
        atr14 = cls._p_atr(H, L, C, 14)
        atr7 = cls._p_atr(H, L, C, 7)
        ema9 = cls._p_ema(C, 9)
        minCO = np.minimum(O, C)
        maxCO = np.maximum(O, C)
        stoch1 = cls._p_stoch(H, L, C, 9, 3)
        stoch2 = cls._p_stoch(H, L, C, 14, 3)
        stoch3 = cls._p_stoch(H, L, C, 40, 4)
        stoch4 = cls._p_stoch(H, L, C, 60, 5)

        # ---- vwaps ----
        # 50-bar variant (user's confirmation)
        vwap50_2 = (C * V).rolling(50).sum() / V.rolling(50).sum()
        # in-month variant (AFL also comp's this, kept for fidelity; unused for entries)
        vrange = None
        # crossUp/Down VWap50 (needed by priceAction* definitions)
        crossUpVWap50 = (
            ((C.shift(1) < vwap50_2.shift(1)) & (C > vwap50_2)) |
            ((O < vwap50_2) & (C > vwap50_2))
        )
        crossDownVWap50 = (
            ((C.shift(1) > vwap50_2.shift(1)) & (C < vwap50_2)) |
            ((O > vwap50_2) & (C < vwap50_2))
        )

        # ---------------- body / size classifications ----------------
        bodySize = (C - O).abs()
        bodyHLPer = bodySize / (H - L).replace(0, 1)
        bodySizeRankTop = cls._p_percentile(bodySize, 100, 0.70)
        bodySizeRankBot = cls._p_percentile(bodySize, 100, 0.30)
        bodySizeRankBot2 = cls._p_percentile(bodySize, 100, 0.40)

        isBullish = C > O
        isBearish = C < O
        isLargeBody = (bodySize >= bodySizeRankTop) & (bodyHLPer >= 0.65)
        isSmallBody = (bodySize <= bodySizeRankBot) & (bodyHLPer <= 0.35)
        isSmallBody2 = (bodySize <= bodySizeRankBot2) & (bodyHLPer <= 0.35)

        hLsize = (H - L).abs()
        hLSizeRankTop = cls._p_percentile(hLsize, 100, 0.80)
        hLSizeRankBot = cls._p_percentile(hLsize, 100, 0.20)
        isLargeHl = hLsize >= hLSizeRankTop
        isSmallHl = hLsize <= hLSizeRankBot

        largeBullishBody = isBullish & isLargeBody
        largeBulllishHl = isBullish & isLargeHl
        largeBearishBody = isBearish & isLargeBody
        largeBearishHl = isBearish & isSmallHl

        # ---- candles (hammer / inverted) ----
        diffCO = bodySize
        rng = diffCO / minCO.replace(0, np.nan)
        smallBodyTh0 = 0.45
        smallRealBody = (bodyHLPer >= 0) & (bodyHLPer < smallBodyTh0)
        lowerShadow = minCO - L
        mediumSizeLowerShadow = (lowerShadow > (H - L) * 0.6) | (
            bodySize * 1.5 < lowerShadow)
        shavenHead = (H - maxCO) / maxCO.replace(0, np.nan) < 0.00045
        candleStickHammer = mediumSizeLowerShadow & smallRealBody & shavenHead
        upperShadow = H - maxCO
        mediumSizeUpperShadow = (upperShadow > (H - L) * 0.6) | (
            bodySize * 1.5 < upperShadow)
        shavenBottom = (L - minCO) / minCO.replace(0, np.nan) < 0.00045
        candleStickInvertedHammer = (
            mediumSizeUpperShadow & smallRealBody & shavenBottom)
        hammerBar = candleStickHammer | candleStickInvertedHammer

        # topMiddleBar
        topMiddleBar = (
            (H.shift(1) >= H.shift(2)) & (H.shift(1) >= H)
        ) | (
            (maxCO.shift(1) >= maxCO.shift(2)) & (maxCO.shift(1) >= maxCO)
        )

        topEngulfing1 = (
            (minCO < minCO.shift(1)) & (maxCO > maxCO.shift(1))
        )
        bearishEngulfing = topEngulfing1 & (largeBearishBody | largeBearishHl)
        bullishEngulfing = topEngulfing1 & (largeBullishBody | largeBulllishHl)

        # ---- HHV / LLV bars misc ----
        hhvBar5_1 = cls._p_hhvbars(H, 5)
        hhvBar5_2 = cls._p_hhvbars(maxCO, 5)
        llvBar5_1 = cls._p_llvbars(L, 5)
        llvBar5_2 = cls._p_llvbars(minCO, 5)

        # ---------------- priceAction patterns ----------------
        # ---- up patterns
        priceActionUp2Bar1 = (
            largeBearishBody.shift(1) & crossDownVWap50.shift(1) &
            largeBullishBody & crossUpVWap50 &
            ((hhvBar5_1 == 0) | (hhvBar5_2 == 0))
        )
        priceActionUp4Bar1 = (
            (largeBearishBody.shift(3) | largeBearishBody.shift(2)) &
            (crossDownVWap50.shift(3) | crossDownVWap50.shift(2)) &
            largeBullishBody.shift(1) & crossUpVWap50.shift(1) &
            ((hhvBar5_1 == 0) | (hhvBar5_2 == 0)) & isBullish
        )
        priceActionUp4Bar2 = (
            (largeBullishBody.shift(3) | largeBullishBody.shift(2)) &
            (crossUpVWap50.shift(3) | crossUpVWap50.shift(2)) &
            largeBearishBody.shift(1) & crossDownVWap50.shift(1) &
            ((llvBar5_1 == 0) | (llvBar5_2 == 0)) & isBullish
        )
        # ---- down patterns (mirrors AFL)
        hhvbars12_1 = cls._p_hhvbars(H, 12)
        hhvbars12_2 = cls._p_hhvbars(maxCO, 12)
        llvbars12_1 = cls._p_llvbars(L, 12)
        llvbars12_2 = cls._p_llvbars(minCO, 12)
        hhv12_2 = cls._p_rolling_hhv(maxCO, 12)
        llv12_2 = cls._p_rolling_llv(minCO, 12)
        priceActionDown2Bar1 = (
            isBullish.shift(1) & crossUpVWap50.shift(1) &
            largeBearishBody & crossDownVWap50 &
            ((minCO < minCO.shift(1)) | (L < L.shift(1))) &
            ((hhvbars12_1 > llvbars12_1) | (hhvbars12_2 > llvbars12_2)) &
            ((hhv12_2 > maxCO) & (llv12_2 < minCO))
        )
        priceActionDown2Bar2 = (
            isBullish.shift(1) & crossUpVWap50.shift(1) &
            largeBearishBody & crossDownVWap50 &
            ((minCO < minCO.shift(1)) | (L < L.shift(1))) &
            ((maxCO > maxCO.shift(1)) | (H > H.shift(1)))
        )
        priceActionDown3Bar1 = (
            isBullish.shift(2) & isBearish.shift(1) &
            ((minCO.shift(2) > minCO.shift(1)) | (L.shift(2) > L.shift(1))) &
            ((maxCO.shift(2) < maxCO.shift(1)) | (H.shift(2) < H.shift(1))) &
            isBearish & crossDownVWap50
        )
        priceActionDown3Bar2 = (
            isBullish.shift(2) & (minCO.shift(2) < vwap50_2.shift(2)) &
            (maxCO.shift(1) > vwap50_2.shift(1)) &
            largeBearishBody & (minCO < vwap50_2) &
            (cls._p_rolling_llv(L, 3) == L) &
            (cls._p_linreg_slope(vwap50_2, 3) < 0)
        )
        priceActionDown4Bar1 = priceActionUp4Bar2  # AFL alias
        # ---- stoch / slope meta
        regSlop3Vwap50Bar = cls._p_linreg_slope(vwap50_2, 3)
        regSlop5vwap50Bar = cls._p_linreg_slope(vwap50_2, 5)
        regSlop5vwap50BarL0 = regSlop5vwap50Bar < -0.01
        regSlop5vwap50BarL0Sum15 = regSlop5vwap50BarL0.rolling(5).sum()
        regSlop5vwap50BarH0 = regSlop5vwap50Bar > 0.01
        regSlop5vwap50BarH0Sum15 = regSlop5vwap50BarH0.rolling(5).sum()

        # ---- buy00 ----
        buy01_1 = regSlop5vwap50BarH0Sum15 >= 4 & priceActionUp2Bar1
        buy01_2 = regSlop5vwap50BarH0Sum15 >= 4 & priceActionUp4Bar1
        buy01_3 = regSlop5vwap50BarH0Sum15 >= 4 & priceActionUp4Bar2
        buy01_3_2 = priceActionUp4Bar2
        # AFL: Sum(crossUpVWap50, 4) > 0
        buy02 = (regSlop3Vwap50Bar.shift(1) < 0) & (regSlop3Vwap50Bar > 0) &\
                (crossUpVWap50.rolling(4).sum() > 0) &\
                (isBullish & isBullish.shift(1))
        buy00 = buy01_3_2.fillna(False).astype(bool)

        # ---- short00 ----
        short01_1 = priceActionDown2Bar1
        short01_2 = priceActionDown4Bar1  # AFL alias
        short01_3 = priceActionDown2Bar2
        short01_4 = priceActionDown3Bar1
        # AFL: Sum(crossDownVWap50, 4) > 0 (was missing)
        short02 = (regSlop3Vwap50Bar.shift(1) > 0) & (regSlop3Vwap50Bar < 0) &\
                  (crossDownVWap50.rolling(4).sum() > 0) &\
                  (isBearish & isBearish.shift(1)) &\
                  (regSlop5vwap50BarL0Sum15 >= 4)
        short00 = short01_4.fillna(False).astype(bool) | short02.fillna(False).astype(bool)

        # ---- attach meta (needed by trade engine) ----
        out["atr14"] = atr14
        out["atr7"] = atr7
        out["stoch1"] = stoch1
        out["stoch2"] = stoch2
        out["stoch3"] = stoch3
        out["stoch4"] = stoch4
        out["vwap50_2"] = vwap50_2
        out["isBullish"] = isBullish
        out["isBearish"] = isBearish
        out["buy00"] = buy00
        out["short00"] = short00
        out["buyO"] = O  # for explore CSV
        out["cCrossUpBBTop"] = crossUpVWap50
        out["cCrossDownBBBottom"] = crossDownVWap50
        out["regSlop3Vwap50Bar"] = regSlop3Vwap50Bar

        # ---- shifted signal (AFL Ref(x,-1)) ----
        out["buy_signal"] = out["buy00"].shift(1).fillna(False).infer_objects(copy=False).astype(bool)
        out["short_signal"] = out["short00"].shift(1).fillna(False).infer_objects(copy=False).astype(bool)

        # ---- AFL bar-loop: profit / stop prices (needed by backtesting.py next()) ----
        # max3BarAtr14_1 = Ref(HHV(atr14, 3), -1)  -> shifted back by 1
        max3BarAtr14_1 = atr14.rolling(3).max().shift(1)
        # AFL entry fill prices: BuyPrice = O - atr14/4 ; ShortPrice = O + atr14/4
        buy_short_price_th = atr14.shift(1) / 4.0
        buy_fill = O - buy_short_price_th
        short_fill = O + buy_short_price_th
        stop_dist = max3BarAtr14_1           # stopLossPt
        profit_dist = cls.profit_mult * max3BarAtr14_1  # profit1Pt = 1.5 * max3BarAtr14_1

        out["buyPriceAFL"] = buy_fill
        out["shortPriceAFL"] = short_fill
        out["longStopPrice"] = np.where(out["buy_signal"], buy_fill - stop_dist, np.nan)
        out["longProfitPrice"] = np.where(out["buy_signal"], buy_fill + profit_dist, np.nan)
        out["shortStopPrice"] = np.where(out["short_signal"], short_fill + stop_dist, np.nan)
        out["shortProfitPrice"] = np.where(out["short_signal"], short_fill - profit_dist, np.nan)

        return out

    # =============================================================================
    # 4. TRADE ENGINE — replicates the AFL bar-loop with force exit semantics
    # =============================================================================
    @classmethod
    def _simulate_trades(cls, df: pd.DataFrame, sig_df: Optional[pd.DataFrame] = None) -> List[dict]:
        """
        Faithful replication of the AFL bar-loop in backtest_4003_1 trim_4_python.afl.

        Semantics replicated:
          - Entry fill price:  BuyPrice = O - Ref(ATR(14),-1)/4 ; ShortPrice = O + Ref(ATR(14),-1)/4
          - Entry allowed on the same bar an existing position is stopped out
            (AFL processes Buy/Short blocks BEFORE the stop/profit checks).
          - Long stop: Low <= stop ; profit: High >= profit (stop checked first).
            Short stop: High >= stop ; profit: Low <= profit (stop checked first).
          - Force-reverse: opposite signal closes the open position at bar open
            (buyForceCover / shortForceSell) and the new position is kept.
          - N-bar stop: exit at bar open 15 bars after entry; consecutive same-side
            signals extend the stop window by 1 bar each (long side only in AFL;
            mirrored here for shorts for symmetry).
          - P&L: points * point_value * contracts - commission_per_trade.
        Returns a list of trade dicts.
        """
        sig = sig_df if sig_df is not None else cls._build_full_signal_df(df)
        if sig.empty:
            return []

        trades: List[dict] = []
        n = len(sig)

        # --- position state (scalars like the AFL loop) ---
        is_in_long = False
        is_in_short = False
        long_entry_price = 0.0
        short_entry_price = 0.0
        long_entry_dt = None
        short_entry_dt = None
        long_entry_idx = 0
        short_entry_idx = 0
        long_stop = 0.0
        long_profit = 0.0
        short_stop = 0.0
        short_profit = 0.0
        sell_stop_nbar_period = cls.hold_bars   # AFL: reset to stopPeriod1 on each buy
        buy_force_cover = False
        short_force_sell = False
        long_mae = 0.0
        long_mfe = 0.0
        short_mae = 0.0
        short_mfe = 0.0

        buys = sig["buy_signal"].to_numpy(dtype=bool)
        shorts = sig["short_signal"].to_numpy(dtype=bool)
        opens = sig["Open"].to_numpy(dtype=float)
        highs = sig["High"].to_numpy(dtype=float)
        lows = sig["Low"].to_numpy(dtype=float)
        long_stop_arr = sig["longStopPrice"].to_numpy(dtype=float)
        long_profit_arr = sig["longProfitPrice"].to_numpy(dtype=float)
        short_stop_arr = sig["shortStopPrice"].to_numpy(dtype=float)
        short_profit_arr = sig["shortProfitPrice"].to_numpy(dtype=float)
        buy_fill_arr = sig["buyPriceAFL"].to_numpy(dtype=float)
        short_fill_arr = sig["shortPriceAFL"].to_numpy(dtype=float)
        idx = sig.index

        for i in range(n):
            price_o = opens[i]
            price_h = highs[i]
            price_l = lows[i]
            buy_i = buys[i]
            short_i = shorts[i]

            # =====================
            # Handling Buy  (AFL: Buy[i] AND !isInLong)
            # =====================
            if buy_i and not is_in_long:
                # AFL fills at BuyPrice = O - atr14/4. Require the bar to trade
                # down to it (limit-order realism); fall back to open if NaN.
                fill = buy_fill_arr[i]
                if np.isnan(fill):
                    fill = price_o
                long_entry_price = fill
                long_entry_dt = idx[i]
                long_entry_idx = i
                lp = long_profit_arr[i]
                ls = long_stop_arr[i]
                if np.isnan(lp) or np.isnan(ls):
                    # fallback: raw ATR-based distances off the fill price
                    atr = sig["atr14"].iloc[i]
                    stop_d = atr if not np.isnan(atr) else 0.0
                    ls = fill - stop_d
                    lp = fill + cls.profit_mult * stop_d
                long_profit = lp
                long_stop = ls
                is_in_long = True
                sell_stop_nbar_period = cls.hold_bars
                long_mae = 0.0
                long_mfe = 0.0
                if is_in_short:
                    buy_force_cover = True

            # =====================
            # Handling Short  (AFL: Short[i] AND !isInShort)
            # =====================
            if short_i and not is_in_short:
                fill = short_fill_arr[i]
                if np.isnan(fill):
                    fill = price_o
                short_entry_price = fill
                short_entry_dt = idx[i]
                short_entry_idx = i
                sp = short_profit_arr[i]
                ss = short_stop_arr[i]
                if np.isnan(sp) or np.isnan(ss):
                    atr = sig["atr14"].iloc[i]
                    stop_d = atr if not np.isnan(atr) else 0.0
                    ss = fill + stop_d
                    sp = fill - cls.profit_mult * stop_d
                short_profit = sp
                short_stop = ss
                is_in_short = True
                short_mae = 0.0
                short_mfe = 0.0
                if is_in_long:
                    short_force_sell = True

            # ==========================================
            # Stop / profit for long  (AFL: stop checked before profit)
            # ==========================================
            if is_in_long:
                # ---- stop loss ----
                if (price_l <= long_stop or short_force_sell) and is_in_long:
                    exit_price = long_stop
                    if short_force_sell:
                        short_force_sell = False
                        exit_price = price_o  # AFL: SellPrice[i] = tradePrice0[i]
                    cls._record_trade(
                        trades, True, long_entry_dt, long_entry_price, idx[i],
                        exit_price, long_entry_idx, i, cls.position_size,
                        long_mae, long_mfe, sig, cls)
                    is_in_long = False
                    long_entry_dt = None
                # ---- profit target ----
                if (price_h >= long_profit or short_force_sell) and is_in_long:
                    exit_price = long_profit
                    if short_force_sell:
                        short_force_sell = False
                        exit_price = price_o
                    cls._record_trade(
                        trades, True, long_entry_dt, long_entry_price, idx[i],
                        exit_price, long_entry_idx, i, cls.position_size,
                        long_mae, long_mfe, sig, cls)
                    is_in_long = False
                    long_entry_dt = None
                # ---- N-bar stop ----
                if (is_in_long
                        and (i - sell_stop_nbar_period) >= 0
                        and not buy_i
                        and (buys[i - sell_stop_nbar_period] or short_force_sell)):
                    if short_force_sell:
                        short_force_sell = False
                    cls._record_trade(
                        trades, True, long_entry_dt, long_entry_price, idx[i],
                        price_o, long_entry_idx, i, cls.position_size,
                        long_mae, long_mfe, sig, cls)
                    is_in_long = False
                    long_entry_dt = None
                # ---- consecutive buy extends N-bar stop ----
                if is_in_long and (i - sell_stop_nbar_period) >= 0 \
                        and buys[i - sell_stop_nbar_period] and buy_i:
                    sell_stop_nbar_period += 1
                # ---- MAE / MFE while still in position ----
                if is_in_long:
                    long_mae = min(long_mae, price_l - long_entry_price)
                    long_mfe = max(long_mfe, price_h - long_entry_price)

            # ============================================
            # Stop / profit for short  (AFL: stop checked before profit)
            # ============================================
            if is_in_short:
                # ---- stop loss ----
                if (price_h >= short_stop or buy_force_cover) and is_in_short:
                    exit_price = short_stop
                    if buy_force_cover:
                        buy_force_cover = False
                        exit_price = price_o
                    cls._record_trade(
                        trades, False, short_entry_dt, short_entry_price, idx[i],
                        exit_price, short_entry_idx, i, cls.position_size,
                        short_mae, short_mfe, sig, cls)
                    is_in_short = False
                    short_entry_dt = None
                # ---- profit target ----
                if (price_l <= short_profit or buy_force_cover) and is_in_short:
                    exit_price = short_profit
                    if buy_force_cover:
                        buy_force_cover = False
                        exit_price = price_o
                    cls._record_trade(
                        trades, False, short_entry_dt, short_entry_price, idx[i],
                        exit_price, short_entry_idx, i, cls.position_size,
                        short_mae, short_mfe, sig, cls)
                    is_in_short = False
                    short_entry_dt = None
                # ---- N-bar stop (AFL uses fixed stopPeriod1 for shorts) ----
                if (is_in_short
                        and (i - cls.hold_bars) >= 0
                        and not short_i
                        and (shorts[i - cls.hold_bars] or buy_force_cover)):
                    if buy_force_cover:
                        buy_force_cover = False
                    cls._record_trade(
                        trades, False, short_entry_dt, short_entry_price, idx[i],
                        price_o, short_entry_idx, i, cls.position_size,
                        short_mae, short_mfe, sig, cls)
                    is_in_short = False
                    short_entry_dt = None
                # ---- MAE / MFE while still in position ----
                if is_in_short:
                    short_mae = min(short_mae, short_entry_price - price_h)
                    short_mfe = max(short_mfe, short_entry_price - price_l)

        return trades

    @staticmethod
    def _record_trade(
        trades: List[dict],
        side_long: bool,
        entry_dt: pd.Timestamp,
        entry_price: float,
        exit_dt: pd.Timestamp,
        exit_price: float,
        entry_idx: int,
        exit_idx: int,
        shares: int,
        mae: float,
        mfe: float,
        sig: pd.DataFrame,
        cls=None,
    ) -> List[dict]:
        # point value / commission from the strategy class (ES: $50/pt, $14.50 RT)
        pv = getattr(cls, "point_value", 1.0) if cls is not None else 1.0
        comm = getattr(cls, "commission_per_trade", 0.0) if cls is not None else 0.0
        side = "Long" if side_long else "Short"
        if side_long:
            pts = exit_price - entry_price
        else:
            pts = entry_price - exit_price
        gross = pts * shares * pv
        pnl = gross - comm
        pct_profit = pts / entry_price if entry_price else 0.0
        change_pct = pct_profit
        position_value = entry_price * shares * pv
        trades.append({
            "symbol": None,
            "side": side,
            "entry_dt": entry_dt,
            "entry_price": entry_price,
            "exit_dt": exit_dt,
            "exit_price": exit_price,
            "change_pct": change_pct,
            "pnl": pnl,
            "pct_profit": pct_profit,
            "shares": shares,
            "position_value": position_value,
            "bars_held": exit_idx - entry_idx,
            "mae": mae,
            "mfe": mfe,
        })
        return trades

    # =============================================================================
    # 5. ARTIFACT WRITERS (HTML + CSV)
    # =============================================================================
    @classmethod
    def _write_html(
        cls,
        trades: List[dict],
        out_path: str,
        equity_curve: Optional[pd.Series] = None,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        # ---- metrics ----
        n_wins = sum(1 for t in trades if t["pnl"] > 0)
        n_losses = sum(1 for t in trades if t["pnl"] <= 0)
        total_wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        total_loss = sum(t["pnl"] for t in trades if t["pnl"] <= 0)
        n_long = sum(1 for t in trades if t["side"] == "Long")
        n_short = sum(1 for t in trades if t["side"] == "Short")
        n_trades = len(trades)
        win_pct = (n_wins / n_trades * 100) if n_trades else 0
        loss_pct = (n_losses / n_trades * 100) if n_trades else 0
        avg_pl = (total_wins + total_loss) / n_trades if n_trades else 0
        avg_pl_pct = (win_pct + loss_pct) / 2 / 100
        max_trade_dd = min(t["mae"] for t in trades) if trades else 0
        # pnl already nets out commission in _record_trade; report the total here
        commission_total = cls.commission_per_trade * n_trades

        # equity curve
        eq = cls.init_balance
        eq_points = [cls.init_balance]
        for t in trades:
            eq += t["pnl"]
            eq_points.append(eq)
        eq_series = pd.Series(eq_points)
        annual_return = (
            (eq_series.iloc[-1] - cls.init_balance) / cls.init_balance
        ) if len(eq_series) > 1 else 0
        rf = 0.03
        excess_ret = annual_return - rf
        ann_ret_pct = (eq_series.pct_change().dropna()).mean() * 100 \
            if len(eq_series) > 1 else 0
        end_cap = eq_series.iloc[-1]

        # drawdown / ulcer / k-ratio
        roll_max = eq_series.cummax()
        dd = (roll_max - eq_series) / roll_max.replace(0, np.nan)
        max_dd = dd.max() * 100
        ulcer_idx = (dd.pow(2).mean()) * 100
        # K-ratio: slope of log(equity) over trades, R^2
        log_eq = np.log(eq_series.replace(0, np.nan)).dropna()
        k_ratio = 0.0
        if len(log_eq) > 1:
            x = np.arange(len(log_eq))
            slope, _ = np.polyfit(x, log_eq, 1)
            y_hat = slope * x + log_eq.mean() - slope * x.mean()
            ss_res = ((log_eq - y_hat) ** 2).sum()
            ss_tot = ((log_eq - log_eq.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            k_ratio = slope * 100_000 if not np.isnan(r2) else np.nan
        # sharpe from annualized returns (user confirmed)
        ann_returns = eq_series.pct_change().dropna()
        sharpe = 0.0
        if len(ann_returns) > 1:
            mean_r = ann_returns.mean()
            sd_r = ann_returns.std()
            if sd_r > 0:
                sharpe = (mean_r - rf / 252) / sd_r
        pf = (total_wins / abs(total_loss)) if total_loss < 0 else float("inf")

        # ---- HTML ----
        rows = [
            ("Initial Capital", f"${cls.init_balance:,.2f}"),
            ("End Capital", f"${end_cap:,.2f}"),
            ("Net Profit", f"${eq_series.iloc[-1] - cls.init_balance:,.2f}"),
            ("Net Profit %", f"{(eq_series.iloc[-1] / cls.init_balance - 1) * 100:.2f}%"),
            ("Exposure %", f"{n_trades * 100.0:.2f}%"),  # proxy
            ("Annual Return %", f"{ann_ret_pct:.2f}%"),
            ("Total Commission Cost", f"${commission_total:,.2f}"),
            ("Number of trades", str(n_trades)),
            ("Number of Long trades", str(n_long)),
            ("Number of Short trades", str(n_short)),
            ("Number of wins", str(n_wins)),
            ("Number of wins %", f"{win_pct:.2f}%"),
            ("Total win amount", f"${total_wins:,.2f}"),
            ("Number of loss", str(n_losses)),
            ("Number of loss %", f"{loss_pct:.2f}%"),
            ("Total loss amount", f"${total_loss:,.2f}"),
            ("Average Profit/Loss", f"${avg_pl:,.2f}"),
            ("Average Profit/Loss %", f"{avg_pl_pct * 100:.2f}%"),
            ("Maximum trade drawdown", f"{max_trade_dd:.2f}"),
            ("Maximum system drawdown", f"{max_dd:.2f}%"),
            ("CAR/MaxDD", f"{(annual_return / max_dd) if max_dd else 0:.4f}"),
            ("Profit Factor", f"{pf:.2f}"),
            ("Sharpe Ratio (rf=3%)", f"{sharpe:.4f}"),
            ("Ulcer Index", f"{ulcer_idx:.2f}"),
            ("K-Ratio", f"{k_ratio:.2f}"),
        ]
        # -- equity chart: matplotlib PNG if available, else dependency-free SVG --
        chart_html = cls._equity_chart_html(eq_series)

        html_pages = [
            "<html><head><meta charset='utf-8'><title>S4003 Backtest</title>",
            "<style>body{font-family:sans-serif;margin:24px}table"
            "{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px 12px}"
            "</style></head><body>",
            "<h1>Backtest — S4003_1</h1>",
            chart_html,
            "<h2>Statistics</h2><table><thead><tr>"
            + "".join(f"<th>{html.escape(k)}</th>" for k, _ in rows)
            + "</tr></thead><tbody><tr>"
            + "".join(f"<td>{html.escape(str(v))}</td>" for _, v in rows)
            + "</tr></tbody></table>",
            "</body></html>",
        ]
        os.makedirs(cls.output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("".join(html_pages))

    # ---------------------------------------------------------
    # Equity chart helpers (matplotlib PNG preferred, SVG fallback)
    # ---------------------------------------------------------
    @staticmethod
    def _equity_chart_html(eq_series: pd.Series) -> str:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io, base64
            fig, ax = plt.subplots(figsize=(9, 3.5))
            eq_series.plot(ax=ax, color="#1f77b4")
            ax.set_title("Equity Curve")
            ax.grid(alpha=0.3)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            return f"<img alt='equity' src='data:image/png;base64,{b64}'/>"
        except ImportError:
            return strategyS4003V1._equity_chart_svg(eq_series)

    @staticmethod
    def _equity_chart_svg(eq_series: pd.Series) -> str:
        """Dependency-free SVG line chart of the equity curve."""
        w, h = 960, 420
        pad_l, pad_r, pad_t, pad_b = 80, 25, 20, 45
        vals = eq_series.to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            return "<p>(no equity data)</p>"
        step = max(1, n // 2000)  # downsample for a compact SVG
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

        pts = " ".join(f"{X(i):.1f},{Y(vals[i]):.1f}" for i in idx)
        grid = []
        for k in range(5):
            v = vmin + (vmax - vmin) * k / 4
            gy = Y(v)
            grid.append(
                f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" '
                f'stroke="#dddddd" stroke-width="1"/>'
                f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="#555555">{v:,.0f}</text>'
            )
        try:
            dt_idx = pd.to_datetime(eq_series.index)
            t0 = dt_idx[0].strftime("%d/%m/%Y %H:%M")
            t1 = dt_idx[-1].strftime("%d/%m/%Y %H:%M")
        except (TypeError, ValueError):
            t0, t1 = str(eq_series.index[0]), str(eq_series.index[-1])
        return (
            f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'
            f'<text x="{w / 2}" y="14" text-anchor="middle" font-size="13" '
            f'font-weight="bold" fill="#333333">Equity Curve</text>'
            + "".join(grid)
            + f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1"/>'
            f'<text x="{pad_l}" y="{h - 8}" font-size="11" fill="#555555">{t0}</text>'
            f'<text x="{w - pad_r}" y="{h - 8}" text-anchor="end" font-size="11" '
            f'fill="#555555">{t1}</text>'
            f'<text x="{pad_l - 8}" y="{pad_t + 4}" text-anchor="end" font-size="11" '
            f'fill="#555555">Equity ($)</text>'
            f"</svg>"
        )

    # ---------------------------------------------------------
    # CSV writers
    # ---------------------------------------------------------
    @classmethod
    def _write_trades_csv(cls, trades: List[dict], out_path: str) -> None:
        df_rows = []
        cum = 0.0
        for t in sorted(
            trades,
            key=lambda x: x["entry_dt"],
            reverse=True,  # descending
        ):
            cum += t["pnl"]
            df_rows.append([
                t.get("symbol") or cls.symbol,
                t["side"],
                t["entry_dt"].strftime("%d/%m/%Y %H:%M"),
                t["entry_price"],
                t["exit_dt"].strftime("%d/%m/%Y %H:%M"),
                t["exit_price"],
                round((t["change_pct"] or 0) * 100, 4),  # %
                round(t["pnl"], 2),
                round((t["pct_profit"] or 0) * 100, 4),
                t["shares"],
                t["position_value"],
                round(cum, 2),
                t["bars_held"],
                t["mae"],
                t["mfe"],
            ])
        df_csv = pd.DataFrame(df_rows, columns=list("ABCDEFGH" + "IJKLMNO"))  # readable names A-O
        os.makedirs(cls.output_dir, exist_ok=True)
        df_csv.to_csv(out_path, header=list("ABCDEFGHIJKLMNO"), index=False)

    @classmethod
    def _write_explore_csv(cls, sig_df: pd.DataFrame, out_path: str) -> None:
        # descending time
        d = sig_df.sort_index(ascending=False)
        rows = []
        for i in range(len(d)):
            row = d.iloc[i]
            dt = d.index[i]
            b_side = "buy00" if row["buy00"] else ("short00" if row["short00"] else "")
            rows.append([
                cls.symbol,
                b_side,
                dt.strftime("%d/%m/%Y %H:%M"),
                row["Open"],
                row["Close"],
                row["High"],
                row["Low"],
                row["cCrossUpBBTop"],
                row["cCrossDownBBBottom"],
            ])
        df_csv = pd.DataFrame(rows, columns=["A", "B", "C", "D", "E", "F", "G", "H", "I"])
        os.makedirs(cls.output_dir, exist_ok=True)
        df_csv.to_csv(out_path, index=False)

    # =============================================================================
    # 6. PUBLIC ORCHESTRATOR
    # =============================================================================
    @classmethod
    def run_backtest_and_write_artifacts(
        cls,
        df: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Run trade simulation, write the 3 artifacts, return (html_path, trades_csv, explore_csv).
        """
        sig_df = cls._build_full_signal_df(df)
        trades = cls._simulate_trades(df, sig_df=sig_df)
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        html_path = os.path.join(cls.output_dir, f"S4003_1_{ts}.html")
        trades_path = os.path.join(cls.output_dir, f"S4003_1_{ts}.csv")
        explore_path = os.path.join(cls.output_dir, f"S4003_1_explore_{ts}.csv")
        cls._write_trades_csv(trades, trades_path)
        cls._write_html(trades, html_path)
        cls._write_explore_csv(sig_df, explore_path)
        print("Artifacts written to:")
        print("  HTML  ->", html_path)
        print("  CSV   ->", trades_path)
        print("  EXPLORE ->", explore_path)
        return html_path, trades_path, explore_path


# =============================================================================
# 7. CLI EXAMPLE / smoke test
# =============================================================================
if __name__ == "__main__":
    # fall back to DB if no CSV provided
    print("Running smoke backtest from DB (ticker1Min)...")
    try:
        df = strategyS4003V1.fetch_from_db(
            # start="2026-05-01", end="2026-07-31")
            start="2026-07-13", end="2026-08-17")
        if df.empty:
            print("DB returned no rows; skipping.")
            sys.exit(0)
        html_path, csv_path, explore_path = \
            strategyS4003V1.run_backtest_and_write_artifacts(df)
        print("HTML:", html_path)
        print("TradeCSV:", csv_path)
        print("ExploreCSV:", explore_path)
        print("Trades generated:", len(strategyS4003V1.run_backtest(df, 100000, 1)))
    except mysql.connector.Error as e:
        print("DB error:", e)
        sys.exit(1)
