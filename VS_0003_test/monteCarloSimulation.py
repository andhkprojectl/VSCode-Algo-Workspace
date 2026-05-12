import pandas as pd
import numpy as np

class TradingStrategy:
    """
    Template for strategy1. Contains the required logic and parameters.
    """
    def __init__(self, commission=0.0, slippage=0.0, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def generate_signals(self, df):
        """
        Placeholder for Buy/Sell and Short/Cover logic.
        You will implement your specific indicator logic here.
        """
        # Example: Randomly generate 1 (Buy), -1 (Short), 0 (Hold) for demonstration
        df['signal'] = np.random.choice([1, 0, -1], size=len(df), p=[0.1, 0.8, 0.1])
        return df

    def run_backtest(self, df, init_balance, position_size):
        """
        Simulates the trades based on signals and returns a list of P&L per trade.
        """
        df = self.generate_signals(df)
        trade_pnl = []
        
        # This is a simplified backtest loop to extract Trade P&L.
        # In a real engine, you would iterate over signals, track entry price, apply 
        # stop loss/take profit, and deduct commission and slippage.
        
        # MOCK IMPLEMENTATION FOR MONTE CARLO PURPOSES:
        # Generates a realistic distribution of trade outcomes based on position size
        # Assuming avg stock price is around the 'close' mean
        avg_price = df['close'].mean()
        
        # Generate N random trades to simulate the output of the strategy
        # Profit/Loss per trade = (exit - entry) * position_size - commission - slippage
        num_mock_trades = len(df[df['signal'] != 0])
        
        for _ in range(num_mock_trades):
            # 40% win rate dummy logic
            is_win = np.random.random() < 0.40 
            if is_win:
                pnl = (avg_price * self.take_profit_pct * position_size) - self.commission - self.slippage
            else:
                pnl = -(avg_price * self.stop_loss_pct * position_size) - self.commission - self.slippage
            trade_pnl.append(pnl)
            
        return trade_pnl


def monteCarloSimulation1(strategy1, inFileName, num_runs, init_balance, position_size):
    """
    Runs a Monte Carlo simulation based on historical backtest trade results.
    """
    print(f"--- Starting Monte Carlo Simulation ({num_runs} runs) ---")
    
    # 1. Load historical data
    try:
        # Assuming standard CSV loading. Parse datetime columns.
        df = pd.read_csv(inFileName)
    except FileNotFoundError:
        print(f"Error: Could not find file {inFileName}")
        return

    # 2. Run Backtest of strategy1 to get P&L per trade
    # The output is a list of monetary profit/loss per closed trade
    trades_pnl = strategy1.run_backtest(df, init_balance, position_size)
    num_trades = len(trades_pnl)
    
    if num_trades == 0:
        print("No trades generated during backtest. Cannot run Monte Carlo.")
        return

    print(f"Backtest complete. Generated {num_trades} historical trades.")

    # 3. Monte Carlo Simulation Engine
    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    
    # Convert list to numpy array for faster random sampling
    trades_array = np.array(trades_pnl)

    for run in range(num_runs):
        # Resample trades with replacement
        simulated_trades = np.random.choice(trades_array, size=num_trades, replace=True)
        
        equity = init_balance
        peak_equity = init_balance
        max_dd = 0.0
        ruined = False
        
        for trade in simulated_trades:
            equity += trade
            
            # Update Peak and Drawdown
            if equity > peak_equity:
                peak_equity = equity
            
            # Calculate current drawdown percentage from peak
            if peak_equity > 0:
                current_dd = (peak_equity - equity) / peak_equity
                if current_dd > max_dd:
                    max_dd = current_dd
                    
            # Check for Ruin
            if equity <= 0:
                ruined = True
                equity = 0
                max_dd = 1.0 # 100% drawdown
                break
                
        if ruined:
            ruin_count += 1
            
        final_equities.append(equity)
        max_drawdowns.append(max_dd)

    # 4. Calculate Outputs
    final_equities = np.array(final_equities)
    max_drawdowns = np.array(max_drawdowns)
    
    # Calculate Returns for Confidence Interval (Total Return %)
    total_returns = ((final_equities - init_balance) / init_balance) * 100

    # Output: Final Equity Percentiles
    p99 = np.percentile(final_equities, 99)
    p95 = np.percentile(final_equities, 95)
    p90 = np.percentile(final_equities, 90)
    p50 = np.percentile(final_equities, 50)
    
    # Output: Max Drawdown Probabilities
    dd_levels = [0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    dd_probs = {}
    for dd in dd_levels:
        # Percentage of runs where max_dd >= specified level
        prob = np.mean(max_drawdowns >= dd) * 100
        dd_probs[dd * 100] = prob
        
    # Output: Probability of Ruin
    prob_ruin = (ruin_count / num_runs) * 100
    
    # Output: Confidence Level (95% CI of returns)
    ci_lower = np.percentile(total_returns, 2.5)
    ci_upper = np.percentile(total_returns, 97.5)

    # 5. Display Formatting
    print("\n==========================================================")
    print("                MONTE CARLO SIMULATION RESULTS            ")
    print("==========================================================")
    
    print("\n1. Final Equity Percentiles:")
    print(f"   99th Percentile: ${p99:,.2f}")
    print(f"   95th Percentile: ${p95:,.2f}")
    print(f"   90th Percentile: ${p90:,.2f}")
    print(f"   50th (Median)  : ${p50:,.2f}")
    
    print("\n2. Max Drawdown (MDD) Probability:")
    for level, prob in dd_probs.items():
        print(f"   There is a {prob:05.2f}% chance you will experience a {int(level)}% drawdown.")
        
    print(f"\n3. Probability of Ruin:")
    print(f"   {prob_ruin:.2f}% of simulations hit $0. ", end="")
    if prob_ruin > 1.0:
        print("(Warning: >1%, position size is likely too large!)")
    else:
        print("(Safe: <=1%)")
        
    print("\n4. Confidence Level:")
    print(f"   I am 95% confident that the total return will be between {ci_lower:.2f}% and {ci_upper:.2f}%.")
    print("==========================================================\n")

    return {
        "final_equity_percentiles": {"99": p99, "95": p95, "90": p90, "50": p50},
        "mdd_probabilities": dd_probs,
        "probability_of_ruin": prob_ruin,
        "confidence_interval_95": (ci_lower, ci_upper)
    }

# =======================================================
# Example Execution Block (For testing purposes)
# =======================================================
if __name__ == "__main__":
    # Create a dummy CSV for testing if none exists
    import os
    test_csv = "dummy_data.csv"
    if not os.path.exists(test_csv):
        dates = pd.date_range(start="2024-01-01", periods=1000, freq='H')
        df = pd.DataFrame({
            "datetime": dates.strftime("%m/%d/%Y %H:%M"),
            "date": dates.strftime("%m/%d/%Y"),
            "time": dates.strftime("%H:%M:%S"),
            "high": np.random.uniform(100, 105, 1000),
            "low": np.random.uniform(95, 100, 1000),
            "close": np.random.uniform(98, 102, 1000),
            "open": np.random.uniform(98, 102, 1000),
            "volume": np.random.randint(1000, 50000, 1000),
            "symbolName": "AAPL"
        })
        df.to_csv(test_csv, index=False)

    # Initialize the strategy parameters
    my_strategy = TradingStrategy(
        commission=1.50, 
        slippage=0.05, 
        stop_loss_pct=0.02, 
        take_profit_pct=0.04
    )

    # Run Monte Carlo
    monteCarloSimulation1(
        strategy1=my_strategy,
        inFileName=test_csv,
        num_runs=10000,
        init_balance=100000.0,
        position_size=100
    )