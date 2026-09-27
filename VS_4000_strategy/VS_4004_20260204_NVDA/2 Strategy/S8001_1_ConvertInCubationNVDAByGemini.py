import pandas as pd
import numpy as np
import talib as ta

def calculate_indicators(df):
    """
    Calculates the technical indicators required for the NVDA strategy.
    Expects a DataFrame with 'Open', 'High', 'Low', 'Close' columns.
    """
    # Moving Averages (EMAs & SMAs)
    df['ema3'] = ta.EMA(df['Close'], timeperiod=3)
    df['ema5'] = ta.EMA(df['Close'], timeperiod=5)
    df['ema8'] = ta.EMA(df['Close'], timeperiod=8)
    df['ema10'] = ta.EMA(df['Close'], timeperiod=10)
    df['ema20'] = ta.EMA(df['Close'], timeperiod=20)
    df['ema35'] = ta.EMA(df['Close'], timeperiod=35)
    df['ema50'] = ta.EMA(df['Close'], timeperiod=50)
    
    df['sma30'] = ta.SMA(df['Close'], timeperiod=30)
    
    # Linear Regression Slopes
    df['regSlopema10'] = ta.LINEARREG_SLOPE(df['ema10'], timeperiod=3)
    
    # ATR and RSI
    df['atr15'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=15)
    df['rsi9'] = ta.RSI(df['Close'], timeperiod=9)
    df['rsi14'] = ta.RSI(df['Close'], timeperiod=14)

    # -----------------------------------------------------
    # IRB (Inside Range Bar) Setup
    # -----------------------------------------------------
    df['iRbhLRange'] = df['High'] - df['Low']
    df['iRbUpperTh'] = df['High'] - df['iRbhLRange'] * 0.45
    df['iRbLowerTh'] = df['Low'] + df['iRbhLRange'] * 0.45
    
    # IRB Bullish / Bearish Conditions
    df['iRbBullish'] = np.where((df['Close'] > df['Low']) & (df['Open'] > df['Low']) & 
                                (df['Close'] < df['iRbLowerTh']) & (df['Open'] < df['iRbLowerTh']), 1, 0)
                                
    df['iRbBearish'] = np.where((df['Close'] < df['High']) & (df['Open'] < df['High']) & 
                                (df['Close'] > df['iRbUpperTh']) & (df['Open'] > df['iRbUpperTh']), 1, 0)
    
    # -----------------------------------------------------
    # Ranks & Differentials 
    # -----------------------------------------------------
    df['ema20DiffC'] = df['ema20'] - df['Close']
    df['ema20DiffCAbs'] = df['ema20DiffC'].abs()
    
    # PercentRank emulation (Lookback period = 100 for ranking)
    # The AFL utilizes PercentRank divided by a rankBase (20) to bin into 1 to 5
    df['ema20DiffCAbsRank'] = df['ema20DiffCAbs'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=True)
    df['ema20DiffCAbsRank'] = np.ceil(df['ema20DiffCAbsRank'] / 20).clip(lower=1)
    
    # Note: pRtEma20Rankper is part of AFL custom array math; approximating a 100-period return rank 
    df['pRtEma20'] = df['Close'].pct_change(periods=20)
    df['pRtEma20Rankper'] = df['pRtEma20'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=True)
        
    return df

def generate_signals(df):
    """
    Translates the specific Buy, Sell, Short, and Cover logic from the AFL script.
    """
    # Shift series for "Ref(..., -1)" AFL equivalents
    regSlopema10_prev = df['regSlopema10'].shift(1)
    pRtEma20Rankper_prev = df['pRtEma20Rankper'].shift(1)
    pRtEma20Rankper_prev2 = df['pRtEma20Rankper'].shift(2)
    
    # -----------------------------------------------------
    # Base Conditions
    # -----------------------------------------------------
    # buyCond1 = ema20 > ema35 AND ema8 > ema20
    df['buyCond1'] = (df['ema20'] > df['ema35']) & (df['ema8'] > df['ema20'])
    
    # shortCond1 = ema20 < ema35 AND ema8 < ema20
    df['shortCond1'] = (df['ema20'] < df['ema35']) & (df['ema8'] < df['ema20'])

    # -----------------------------------------------------
    # BUY & SHORT Execution Signals (The "00" series)
    # -----------------------------------------------------
    df['Buy'] = False
    df['Short'] = False
    df['Sell'] = False
    df['Cover'] = False
    
    # buy00 specific rules from AFL line 1510
    df['buy00'] = (
        (df['iRbBullish'] == 1) & 
        (df['regSlopema10'] > regSlopema10_prev) & 
        (df['pRtEma20Rankper'] > 50) & 
        (df['ema20DiffCAbsRank'] >= 3) & 
        (df['pRtEma20Rankper'] >= pRtEma20Rankper_prev) & 
        (df['pRtEma20Rankper'] >= pRtEma20Rankper_prev2)
    )

    # Assuming mirrored logic for short00 (Based on standard AFL pattern in code)
    df['short00'] = (
        (df['iRbBearish'] == 1) & 
        (df['regSlopema10'] < regSlopema10_prev) & 
        (df['shortCond1'] == True)
    )

    # -----------------------------------------------------
    # Position & Trade Management Loop (Iterative exactly like AFL)
    # -----------------------------------------------------
    is_in_long = False
    is_in_short = False
    
    long_entry_price = 0.0
    short_entry_price = 0.0
    
    # Example Profit / Stop-loss targets (Ex: ATR based or Fixed Points)
    stop_period = 5 # Example N-Bar stop

    for i in range(1, len(df)):
        # 1. Evaluate Buy signals
        if not is_in_long and not is_in_short and df['buy00'].iloc[i-1]:
            df.at[df.index[i], 'Buy'] = True
            is_in_long = True
            long_entry_price = df['Open'].iloc[i]  # Next day open execution
            bars_in_trade = 0
            
            # Set dynamic stops
            stop_loss_price = long_entry_price - (df['atr15'].iloc[i] * 1.5)
            profit_price = long_entry_price + (df['atr15'].iloc[i] * 3.0)

        # 2. Evaluate Short Signals
        elif not is_in_long and not is_in_short and df['short00'].iloc[i-1]:
            df.at[df.index[i], 'Short'] = True
            is_in_short = True
            short_entry_price = df['Open'].iloc[i]
            bars_in_trade = 0
            
            stop_loss_price = short_entry_price + (df['atr15'].iloc[i] * 1.5)
            profit_price = short_entry_price - (df['atr15'].iloc[i] * 3.0)

        # 3. Handle Long Exits (Sell)
        if is_in_long:
            bars_in_trade += 1
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            
            # Stop Loss OR Take Profit OR N-Bar Time Stop
            if low_price <= stop_loss_price or high_price >= profit_price or bars_in_trade >= stop_period:
                df.at[df.index[i], 'Sell'] = True
                is_in_long = False

        # 4. Handle Short Exits (Cover)
        if is_in_short:
            bars_in_trade += 1
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            
            if high_price >= stop_loss_price or low_price <= profit_price or bars_in_trade >= stop_period:
                df.at[df.index[i], 'Cover'] = True
                is_in_short = False

    return df

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    # 1. Load your OHLCV data
    # df = pd.read_csv("NVDA_1Min.csv") 
    
    # Mock Data for illustration
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 200),
        'High': np.random.uniform(105, 205, 200),
        'Low': np.random.uniform(95, 195, 200),
        'Close': np.random.uniform(100, 200, 200),
        'Volume': np.random.uniform(1000, 5000, 200)
    }, index=dates)

    # Clean mock data so High is highest, Low is lowest
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1) + 2
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1) - 2

    # 2. Run logic
    df = calculate_indicators(df)
    df = generate_signals(df)

    # 3. View Trades
    trades = df[(df['Buy'] == True) | (df['Sell'] == True) | (df['Short'] == True) | (df['Cover'] == True)]
    print(trades[['Close', 'Buy', 'Sell', 'Short', 'Cover']])