import pandas as pd
import numpy as np
import talib as ta
from backtesting import Backtest, Strategy
from pathlib import Path
from dotenv import load_dotenv
import os
import sys

class strategyIRB1000V1(Strategy):
    """
    Strategy: IRB 1000 V1
    Timeframe: 5-minute
    Period: 09:30 to 16:00 (Regular NASDAQ trading hours, NY time)
    Symbol: NVDA
    """
    # Define strategy parameters as class attributes 
    is_live = False
    position_size = 110
    
    # Note: In standard backtesting.py, slippage is usually handled in the 
    # Backtest() initialization, but we keep this for your reference.
    slippage = 0.03 

    def init(self):
        """
        Calculates all required technical indicators upfront.
        Uses self.I() to wrap indicator arrays so backtesting.py tracks them.
        """
        # Extract underlying numpy arrays from the data object
        H = self.data.High
        L = self.data.Low
        O = self.data.Open
        C = self.data.Close

        # ---------------------------------------------------------
        # 1. IRB (Inside Range Bar) Logic
        # ---------------------------------------------------------
        def calc_bullish(h, l, o, c):
            irb_range = h - l
            lower_th = l + (irb_range * 0.45)
            return np.where((c > l) & (o > l) & (c < lower_th) & (o < lower_th), 1, 0)

        def calc_bearish(h, l, o, c):
            irb_range = h - l
            upper_th = h - (irb_range * 0.45)
            return np.where((c < h) & (o < h) & (c > upper_th) & (o > upper_th), 1, 0)

        self.iRbBullish = self.I(calc_bullish, H, L, O, C)
        self.iRbBearish = self.I(calc_bearish, H, L, O, C)

        # ---------------------------------------------------------
        # 2. EMAs and Regression Slope
        # ---------------------------------------------------------
        self.ema10 = self.I(ta.EMA, C, timeperiod=10)
        self.ema20 = self.I(ta.EMA, C, timeperiod=20)
        self.regSlopema10 = self.I(ta.LINEARREG_SLOPE, self.ema10, timeperiod=3)

        # ---------------------------------------------------------
        # 3. EMA20 Differential Ranking
        # ---------------------------------------------------------
        def calc_rank(ema20, close):
            diff = np.round(np.abs(ema20 - close) * 100, 0)
            conditions = [
                (diff >= 0) & (diff < 10),
                (diff >= 10) & (diff < 21),
                (diff >= 21) & (diff < 35),
                (diff >= 35) & (diff < 66),
                (diff >= 66)
            ]
            return np.select(conditions, [1, 2, 3, 4, 5], default=1)

        self.ema20DiffCAbsRank = self.I(calc_rank, self.ema20, C)

        # ---------------------------------------------------------
        # 4. Rolling Historical Returns (Next 5 bars) & Win Rate Rank
        # ---------------------------------------------------------
        def calc_rolling_win_rate(c, ranks):
            s_c = pd.Series(c)
            rt10_hist = s_c.shift(-5) - s_c
            rt10_up = np.where(rt10_hist > 0, 1, 0)
            
            pRt = np.zeros(len(c))
            for i in range(100, len(c)):
                current_rank = ranks[i]
                start_idx = max(0, i - 100)
                end_idx = max(0, i - 5)

                window_ranks = ranks[start_idx:end_idx]
                window_ups = rt10_up[start_idx:end_idx]

                mask = (window_ranks == current_rank)
                sumRank = np.sum(mask)
                sumUp = np.sum(window_ups[mask])

                if sumRank > 0:
                    pRt[i] = (sumUp * 100.0) / sumRank
                else:
                    pRt[i] = 50.0
            return pRt

        self.pRtEma20Rankper = self.I(calc_rolling_win_rate, C, self.ema20DiffCAbsRank)

        # ---------------------------------------------------------
        # 5. Stop Loss ATR
        # ---------------------------------------------------------
        self.atr7 = self.I(ta.ATR, H, L, C, timeperiod=7)

    def next(self):
        """
        Executes step-by-step logic automatically on every bar.
        backtesting.py executes orders on the NEXT bar's open, 
        meaning index [-1] represents the currently completed bar.
        """
        # Ensure enough history exists to pull shifted/previous values
        if len(self.data) < 4:
            return

        # ---------------------------------------------------------
        # EXITS: Time Stop (5-bar hold constraint)
        # ---------------------------------------------------------
        # Backtesting.py handles Stop-Loss automatically via the 'sl' parameter.
        # We only need to manually track the time constraint here.
        for trade in self.trades:
            # calculate bars held since entry
            bars_held = len(self.data) - 1 - trade.entry_bar 
            if bars_held >= 5:
                trade.close()

        # ---------------------------------------------------------
        # ENTRIES
        # ---------------------------------------------------------
        # Only evaluate entry conditions if we are not currently in a position
        if not self.position:
            
            # Fetch values for the currently formed bar (-1) and previous bars (-2, -3)
            iRbBull_curr = self.iRbBullish[-1] == 1
            iRbBear_curr = self.iRbBearish[-1] == 1
            
            regSlope_curr = self.regSlopema10[-1]
            regSlope_prev = self.regSlopema10[-2]
            
            pRt_curr = self.pRtEma20Rankper[-1]
            pRt_prev1 = self.pRtEma20Rankper[-2]
            pRt_prev2 = self.pRtEma20Rankper[-3]
            
            rank_curr = self.ema20DiffCAbsRank[-1]
            
            close_curr = self.data.Close[-1]
            atr_curr = self.atr7[-1]

            # Signal Generation Logic
            buy00 = (
                iRbBull_curr and
                (regSlope_curr > regSlope_prev) and
                (pRt_curr > 50) and
                (rank_curr >= 3) and
                (pRt_curr >= pRt_prev1) and
                (pRt_curr >= pRt_prev2)
            )

            short00 = (
                iRbBear_curr and
                (regSlope_curr < regSlope_prev) and
                (pRt_curr < 50) and
                (rank_curr >= 3) and
                (pRt_curr <= pRt_prev1) and
                (pRt_curr <= pRt_prev2)
            )

            # Execution Logic
            if buy00:
                stop_price = close_curr - (atr_curr * 2.5)
                stop_limit_price = stop_price - 0.06
                
                # Standard buy with SL integrated directly.
                self.buy(size=self.position_size, sl=stop_limit_price)

            elif short00:
                stop_price = close_curr + (atr_curr * 2.5)
                stop_limit_price = stop_price + 0.06
                
                # Standard short with SL integrated directly.
                self.sell(size=self.position_size, sl=stop_limit_price)


if __name__ == "__main__":
    # --------------------------------------------
    current_dir = Path(__file__).resolve().parent
    dotenv_path = current_dir.parents[2] / "VS_0002_config" / ".env"
    
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"ERROR: .env file not found at {dotenv_path}")
        
    csv_excel_path = os.getenv("csvExcelPath")
    if csv_excel_path:
        print(f"Successfully retrieved csvExcelPath: {csv_excel_path}")
        env_dir = dotenv_path.parent
        absolute_csv_path = (env_dir / csv_excel_path).resolve()
    else:
        print("ERROR: csvExcelPath not found in the .env file.")

    csvFileName = Path(csv_excel_path) / "NVDA_20250101_20260430_5Min.csv"
    
    # Ensure possible parent directories are in the Python path
    candidate_roots = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    ]
    for p in candidate_roots:
        if p not in sys.path:
            sys.path.insert(0, p)

    import importlib
    backtestStrategy = None
    import_error = None
    for p in candidate_roots:
        pkg_dir = os.path.join(p, 'VS_0003_test')
        if os.path.isdir(pkg_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                mod = importlib.import_module('VS_0003_test.backtest')
                backtestStrategy = getattr(mod, 'backtestStrategy')
                break
            except Exception as e:
                import_error = e

    if backtestStrategy is None:
        try:
            from VS_0003_test.backtest import backtestStrategy
        except Exception as e:
            print(f"Import error for VS_0003_test.backtest: {e}")
            if import_error:
                print(f"Earlier import attempt error: {import_error}")
            raise

    # Run the backtest using the defined strategy class
    backtestStrategy(strategyIRB1000V1, csvFileName)