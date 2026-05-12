import pandas as pd
import numpy as np
import random
from backtesting import Backtest, Strategy

# =====================================================================
# 1. LOOK AHEAD BIAS TEST
# =====================================================================
def lookAheadBiasTest(strategy1, inFileName, init_balance, position_size):
    """
    Tests a trading strategy for look-ahead bias by comparing signals generated 
    on a full dataset vs. iteratively truncated datasets.
    """
    print("\n==========================================================")
    print("             LOOK-AHEAD BIAS DETECTION TEST               ")
    print("==========================================================")
    
    try:
        df_full = pd.read_csv(inFileName)
    except FileNotFoundError:
        print(f"Error: Could not find file {inFileName}")
        return False, pd.DataFrame()

    try:
        df_baseline = strategy1.generate_signals(df_full.copy())
    except AttributeError:
        print("Error: strategy1 must have a 'generate_signals' method.")
        return False, pd.DataFrame()

    look_ahead_indices = []
    test_size = min(len(df_full), 500)
    start_idx = len(df_full) - test_size
    if start_idx < 200: start_idx = 200

    print(f"Running truncation test from row {start_idx} to {len(df_full)}...")

    for i in range(start_idx, len(df_full)):
        df_trunc = df_full.iloc[:i+1].copy()
        df_trunc_res = strategy1.generate_signals(df_trunc)
        
        trunc_signal = df_trunc_res['signal'].iloc[-1]
        base_signal = df_baseline['signal'].iloc[i]

        if trunc_signal != base_signal:
            look_ahead_indices.append(i)

    if len(look_ahead_indices) > 0:
        isLookAhead = True
        lookAheadBar = df_baseline.iloc[look_ahead_indices].copy()
        print(f"[FAILED] LOOK-AHEAD BIAS DETECTED! Found {len(lookAheadBar)} instances.")
    else:
        isLookAhead = False
        lookAheadBar = pd.DataFrame(columns=df_baseline.columns)
        print("[PASSED] NO Look-Ahead Bias detected.")

    print("==========================================================\n")
    return isLookAhead, lookAheadBar


# =====================================================================
# 2. MONKEY TEST (RANDOMIZED TRADING)
# =====================================================================
class MonkeyStrategy(Strategy):
    """
    A pure noise strategy. Buys, Sells, or Holds completely at random.
    Inherits constraints from the original strategy's framework where possible.
    """
    stop_loss_pct = 0.02
    take_profit_pct = 0.05
    
    def init(self):
        pass

    def next(self):
        current_price = self.data.Close[-1]
        action = random.choice(['buy', 'sell', 'hold'])
        
        if action == 'buy':
            if self.position.is_short:
                self.position.close()
            if not self.position.is_long:
                sl_price = current_price * (1 - self.stop_loss_pct)
                tp_price = current_price * (1 + self.take_profit_pct)
                self.buy(sl=sl_price, tp=tp_price)
                
        elif action == 'sell':
            if self.position.is_long:
                self.position.close()
            if not self.position.is_short:
                sl_price = current_price * (1 + self.stop_loss_pct)
                tp_price = current_price * (1 - self.take_profit_pct)
                self.sell(sl=sl_price, tp=tp_price)


def monkeyTest(strategy1, inFileName, init_balance, position_size):
    """
    Runs the given strategy against a purely random 'Monkey' strategy to determine
    if the original strategy possesses actual predictive edge.
    
    Returns:
        monkeyTestBetter (bool): True if random trading beats the strategy1.
    """
    print("\n==========================================================")
    print("                    MONKEY TEST ENGINE                    ")
    print("==========================================================")
    
    # 1. Load and format data for backtesting.py standard
    try:
        df = pd.read_csv(inFileName)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
        df.set_index('datetime', inplace=True)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        df.sort_index(ascending=True, inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
        
    commission_rate = getattr(strategy1, 'commission', 0.000)

    # 2. Run Base Strategy (a)
    print("Running Base Strategy Backtest...")
    bt_base = Backtest(df, strategy1, cash=init_balance, commission=commission_rate, trade_on_close=False)
    stats_base = bt_base.run()

    # 3. Run Monkey Strategy (b)
    print("Running Monkey (Random) Strategy Backtest...")
    # Inject parameters from strategy1 into Monkey strategy if needed
    MonkeyStrategy.commission = commission_rate
    MonkeyStrategy.stop_loss_pct = getattr(strategy1, 'stop_loss_pct', 0.02)
    MonkeyStrategy.take_profit_pct = getattr(strategy1, 'take_profit_pct', 0.05)
    
    bt_monkey = Backtest(df, MonkeyStrategy, cash=init_balance, commission=commission_rate, trade_on_close=False)
    stats_monkey = bt_monkey.run()

    # 4. Extract and safely format Metrics
    # Handle NaNs (e.g., if a strategy made no trades, its Sharpe is NaN)
    def safe_get(val, default=0.0):
        return val if not pd.isna(val) else default

    base_profit = safe_get(stats_base['Equity Final [$]'] - init_balance)
    base_sharpe = safe_get(stats_base['Sharpe Ratio'])
    base_mdd = safe_get(stats_base['Max. Drawdown [%]'], -100.0) # MDD is negative %

    monkey_profit = safe_get(stats_monkey['Equity Final [$]'] - init_balance)
    monkey_sharpe = safe_get(stats_monkey['Sharpe Ratio'])
    monkey_mdd = safe_get(stats_monkey['Max. Drawdown [%]'], -100.0)

    # 5. Comparison Logic
    # Note: Max Drawdown is represented as a negative number (e.g., -15.4%). 
    # Therefore, a "better" drawdown is mathematically greater (closer to 0).
    is_profit_better = monkey_profit > base_profit
    is_sharpe_better = monkey_sharpe > base_sharpe
    is_mdd_better = monkey_mdd > base_mdd 

    # The Monkey is considered "better" if it outperforms in ALL 3 core metrics
    monkeyTestBetter = is_profit_better and is_sharpe_better and is_mdd_better

    # 6. Output Results
    print("\n--- COMPARISON RESULTS ---")
    print(f"{'Metric':<20} | {'Base Strategy':<15} | {'Monkey Strategy':<15}")
    print("-" * 55)
    print(f"{'Net Profit':<20} | ${base_profit:<14.2f} | ${monkey_profit:<14.2f}")
    print(f"{'Sharpe Ratio':<20} | {base_sharpe:<15.3f} | {monkey_sharpe:<15.3f}")
    print(f"{'Max Drawdown':<20} | {base_mdd:<14.2f}% | {monkey_mdd:<14.2f}%")
    print("-" * 55)
    
    if monkeyTestBetter:
        print("\n[VERDICT] True: The Monkey beat your strategy. You may be over-fitting or trading noise.")
    else:
        print("\n[VERDICT] False: The Base Strategy successfully outperformed random noise.")
        
    print("==========================================================\n")

    return monkeyTestBetter

# =====================================================================
# EXAMPLE EXECUTION (For isolated testing)
# =====================================================================
if __name__ == "__main__":
    # Assuming you have a dummy CSV file created from the earlier steps
    test_csv = "dummy_data.csv"
    
    # We will import the MACrossoverWithStops from your backtest.py
    # Ensure backtest.py is in the same directory or accessible via sys.path
    try:
        import sys
        import os
        # Add the directory containing backtest.py to sys.path if needed
        # sys.path.append(os.path.abspath("..\\path\\to\\backtest_folder"))
        from backtest import MACrossoverWithStops
        
        # Test the function
        result = monkeyTest(
            strategy1=MACrossoverWithStops,
            inFileName=test_csv,
            init_balance=100000.0,
            position_size=100
        )
    except ImportError:
        print("Could not import MACrossoverWithStops from backtest.py for the test execution.")