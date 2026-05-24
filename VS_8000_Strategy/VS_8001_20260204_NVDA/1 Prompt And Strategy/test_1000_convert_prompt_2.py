import numpy as np
import pandas as pd
import talib as ta

class strategyIRB1000_V1:
    """
    Strategy: IRB 1000 V1
    Timeframe: 5-minute
    Period: 09:30 to 16:00 (Regular NASDAQ trading hours, NY time)
    Symbol: NVDA
    """

    def __init__(self, is_live=False):
        """
        Initializes the strategy state.
        :param is_live: Boolean. Set to True for live trading, False for backtesting.
        """
        self.is_live = is_live
        self.position_size = 110
        self.slippage = 0.03 if is_live else 0.0

    def calculate_indicators(self, df):
        """
        Calculates all required technical indicators and custom IRB/Rank logic.
        Expects a Pandas DataFrame with ['Open', 'High', 'Low', 'Close', 'Volume'] columns.
        """
        # Ensure data consistency
        H = df['High']
        L = df['Low']
        O = df['Open']
        C = df['Close']

        # ---------------------------------------------------------
        # 1. IRB (Inside Range Bar) Logic
        # ---------------------------------------------------------
        df['iRbhLRange'] = H - L
        df['iRbUpperTh'] = H - (df['iRbhLRange'] * 0.45)
        df['iRbLowerTh'] = L + (df['iRbhLRange'] * 0.45)

        df['iRbBullish'] = np.where((C > L) & (O > L) & (C < df['iRbLowerTh']) & (O < df['iRbLowerTh']), 1, 0)
        df['iRbBearish'] = np.where((C < H) & (O < H) & (C > df['iRbUpperTh']) & (O > df['iRbUpperTh']), 1, 0)

        # ---------------------------------------------------------
        # 2. EMAs and Regression Slope
        # ---------------------------------------------------------
        df['ema10'] = ta.EMA(C, timeperiod=10)
        df['ema20'] = ta.EMA(C, timeperiod=20)
        
        # 3-bar linear regression slope of ema10
        df['regSlopema10'] = ta.LINEARREG_SLOPE(df['ema10'], timeperiod=3)

        # ---------------------------------------------------------
        # 3. EMA20 Differential Ranking
        # ---------------------------------------------------------
        df['ema20DiffCAbsRound0'] = np.round(np.abs(df['ema20'] - C) * 100, 0)

        # Divide into 5 ranks
        conditions = [
            (df['ema20DiffCAbsRound0'] >= 0) & (df['ema20DiffCAbsRound0'] < 10),
            (df['ema20DiffCAbsRound0'] >= 10) & (df['ema20DiffCAbsRound0'] < 21),
            (df['ema20DiffCAbsRound0'] >= 21) & (df['ema20DiffCAbsRound0'] < 35),
            (df['ema20DiffCAbsRound0'] >= 35) & (df['ema20DiffCAbsRound0'] < 66),
            (df['ema20DiffCAbsRound0'] >= 66)
        ]
        choices = [1, 2, 3, 4, 5]
        df['ema20DiffCAbsRank'] = np.select(conditions, choices, default=1)

        # ---------------------------------------------------------
        # 4. Rolling Historical Returns (Next 5 bars) & Win Rate Rank
        # ---------------------------------------------------------
        # For historical analysis: difference between current close and 5-bar future close
        df['rt10_hist'] = C.shift(-5) - C
        df['rt10_is_up'] = np.where(df['rt10_hist'] > 0, 1, 0)

        pRtEma20Rankper = np.zeros(len(df))
        ranks = df['ema20DiffCAbsRank'].values
        rt10_up = df['rt10_is_up'].values

        # Rolling calculation for the last 100 bars
        for i in range(100, len(df)):
            current_rank = ranks[i]
            
            # Look at past 100 bars, but offset by 5 to prevent lookahead bias (ensuring the 5-bar outcome is known)
            start_idx = max(0, i - 100)
            end_idx = max(0, i - 5)

            window_ranks = ranks[start_idx:end_idx]
            window_ups = rt10_up[start_idx:end_idx]

            mask = (window_ranks == current_rank)
            sumEma20DiffCAbsRank = np.sum(mask)             # Total occurrences of this rank
            sumEma20DiffCAbsUpRankper = np.sum(window_ups[mask]) # Occurrences where price went up

            if sumEma20DiffCAbsRank > 0:
                pRtEma20Rankper[i] = (sumEma20DiffCAbsUpRankper * 100.0) / sumEma20DiffCAbsRank
            else:
                pRtEma20Rankper[i] = 50.0 # Default fallback if rank hasn't appeared historically

        df['pRtEma20Rankper'] = pRtEma20Rankper

        # ---------------------------------------------------------
        # 5. Stop Loss ATR
        # ---------------------------------------------------------
        df['atr7'] = ta.ATR(H, L, C, timeperiod=7)

        # ---------------------------------------------------------
        # 6. Base Signal Generation (Calculated at bar close)
        # ---------------------------------------------------------
        df['buy00'] = (
            (df['iRbBullish'] == 1) &
            (df['regSlopema10'] > df['regSlopema10'].shift(1)) &
            (df['pRtEma20Rankper'] > 50) &
            (df['ema20DiffCAbsRank'] >= 3) &
            (df['pRtEma20Rankper'] >= df['pRtEma20Rankper'].shift(1)) &
            (df['pRtEma20Rankper'] >= df['pRtEma20Rankper'].shift(2))
        )

        df['short00'] = (
            (df['iRbBearish'] == 1) &
            (df['regSlopema10'] < df['regSlopema10'].shift(1)) &
            (df['pRtEma20Rankper'] < 50) &
            (df['ema20DiffCAbsRank'] >= 3) &
            (df['pRtEma20Rankper'] <= df['pRtEma20Rankper'].shift(1)) &
            (df['pRtEma20Rankper'] <= df['pRtEma20Rankper'].shift(2))
        )

        return df

    def generate_trades(self, df):
        """
        Executes the logic on the calculated dataframe, handling entries, 
        stop loss/limit limits, positions sizing, and 5-bar holds.
        """
        df = self.calculate_indicators(df)

        in_position = 0      # 1 for Long, -1 for Short
        bars_held = 0
        stop_price = 0.0
        stop_limit_price = 0.0

        # Output tracking columns
        df['Position'] = 0
        df['Action'] = ''
        df['ExecPrice'] = 0.0
        df['StopLossLimit'] = np.nan

        # Loop through the data to simulate the bar-by-bar state machine
        for i in range(1, len(df)):
            # Signals are triggered based on the completed previous bar
            prev_buy00 = df['buy00'].iloc[i-1]
            prev_short00 = df['short00'].iloc[i-1]
            prev_atr7 = df['atr7'].iloc[i-1]

            current_O = df['Open'].iloc[i]
            current_C = df['Close'].iloc[i]
            current_H = df['High'].iloc[i]
            current_L = df['Low'].iloc[i]

            # Price modeling (Live vs Backtest rules)
            if self.is_live:
                entry_long_price = current_C + self.slippage
                entry_short_price = current_C - self.slippage
                exit_long_price = current_C - self.slippage
                exit_short_price = current_C + self.slippage
                sl_base_price = current_C
            else:
                entry_long_price = current_O
                entry_short_price = current_O
                exit_long_price = current_O
                exit_short_price = current_O
                sl_base_price = current_C

            # --- PROCESS EXITS ---
            if in_position == 1:
                bars_held += 1
                # Check Stop Loss / Time Stop (Sell 5 bars after buy)
                if current_L <= stop_price or bars_held >= 5:
                    df.at[df.index[i], 'Action'] = 'SELL'
                    df.at[df.index[i], 'ExecPrice'] = exit_long_price if bars_held >= 5 else stop_limit_price
                    in_position = 0
                    bars_held = 0

            elif in_position == -1:
                bars_held += 1
                # Check Stop Loss / Time Stop (Cover 5 bars after short)
                if current_H >= stop_price or bars_held >= 5:
                    df.at[df.index[i], 'Action'] = 'COVER'
                    df.at[df.index[i], 'ExecPrice'] = exit_short_price if bars_held >= 5 else stop_limit_price
                    in_position = 0
                    bars_held = 0

            # --- PROCESS ENTRIES ---
            if in_position == 0:
                if prev_buy00:
                    in_position = 1
                    bars_held = 0
                    
                    # Calculate Stop / Stop Limit
                    stop_price = sl_base_price - (prev_atr7 * 2.5)
                    stop_limit_price = stop_price - 0.06
                    
                    df.at[df.index[i], 'Action'] = 'BUY'
                    df.at[df.index[i], 'ExecPrice'] = entry_long_price
                    df.at[df.index[i], 'StopLossLimit'] = stop_limit_price

                elif prev_short00:
                    in_position = -1
                    bars_held = 0
                    
                    # Calculate Stop / Stop Limit
                    stop_price = sl_base_price + (prev_atr7 * 2.5)
                    stop_limit_price = stop_price + 0.06
                    
                    df.at[df.index[i], 'Action'] = 'SHORT'
                    df.at[df.index[i], 'ExecPrice'] = entry_short_price
                    df.at[df.index[i], 'StopLossLimit'] = stop_limit_price

            # Update sizing
            df.at[df.index[i], 'Position'] = in_position * self.position_size

        return df