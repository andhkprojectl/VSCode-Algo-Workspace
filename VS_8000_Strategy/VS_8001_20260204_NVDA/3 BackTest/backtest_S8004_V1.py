"""
Backtest program for Strategy S8004 (IRB1000 V1) on NVDA 5-minute data
Generated from prompt: prompt_backtest_s80004.md
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Import strategy module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2 Strategy'))
from S8001_4_GenerateFromPromptQwen37Max import strategyIRB1000_V1


def load_data(csv_path):
    """
    Load NVDA 5-minute data from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with datetime index and OHLCV data
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Parse datetime column (column A: datetime, format MM/DD/YYYY hh24:mi)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Rename columns to match strategy expectations
    df = df.rename(columns={
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'open': 'Open',
        'volume': 'Volume',
        'symbolName': 'Symbol'
    })
    
    return df


def run_backtest(df, strategy, initial_capital=100000):
    """
    Run backtest simulation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    strategy : strategyIRB1000_V1
        Strategy instance
    initial_capital : float
        Initial capital
        
    Returns:
    --------
    dict
        Dictionary with backtest results
    """
    # Calculate indicators and generate signals
    df = strategy.calculate_indicators(df)
    df = strategy.generate_signals(df)
    
    # Initialize tracking variables
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    entry_time = None
    position_size = strategy.position_size
    
    # Trade tracking
    trades = []
    equity_curve = []
    
    # Commission per trade (assuming $0.005 per share, minimum $1)
    commission_per_share = 0.005
    min_commission = 1.0
    
    # Track statistics
    total_commission = 0
    wins = 0
    losses = 0
    total_win_amount = 0
    total_loss_amount = 0
    max_drawdown = 0
    peak_capital = initial_capital
    
    # Iterate through data
    for i in range(len(df)):
        current_time = df.index[i]
        current_close = df.iloc[i]['Close']
        
        # Check for buy signal
        if df.iloc[i]['buy_signal'] == 1 and position == 0:
            # Enter long position
            entry_price = df.iloc[i]['buy_price']
            entry_time = current_time
            entry_index = i
            position = 1
            
            # Calculate commission
            commission = max(position_size * commission_per_share, min_commission)
            total_commission += commission
            capital -= commission
        
        # Check for short signal
        elif df.iloc[i]['short_signal'] == 1 and position == 0:
            # Enter short position
            entry_price = df.iloc[i]['short_price']
            entry_time = current_time
            entry_index = i
            position = -1
            
            # Calculate commission
            commission = max(position_size * commission_per_share, min_commission)
            total_commission += commission
            capital -= commission
        
        # Check for sell signal (exit long)
        elif df.iloc[i]['sell_signal'] == 1 and position == 1:
            # Exit long position
            exit_price = df.iloc[i]['sell_price']
            exit_time = current_time
            exit_index = i
            
            # Calculate profit/loss
            pnl = (exit_price - entry_price) * position_size
            
            # Calculate commission
            commission = max(position_size * commission_per_share, min_commission)
            total_commission += commission
            
            # Update capital
            capital += pnl - commission
            
            # Calculate MAE and MFE for long trade
            trade_bars = df.iloc[entry_index:exit_index + 1]
            mae = (trade_bars['Low'].min() - entry_price) * position_size  # worst unrealized loss
            mfe = (trade_bars['High'].max() - entry_price) * position_size  # best unrealized gain
            
            # Number of bars held
            bars_held = exit_index - entry_index
            
            # Track trade
            trade = {
                'symbol': 'NVDA',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'type': 'Long',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'commission': commission * 2,  # Entry + exit
                'shares': position_size,
                'position_value': entry_price * position_size,
                'mae': mae,
                'mfe': mfe,
                'bars_held': bars_held
            }
            trades.append(trade)
            
            # Track wins/losses
            if pnl > 0:
                wins += 1
                total_win_amount += pnl
            else:
                losses += 1
                total_loss_amount += abs(pnl)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_time = None
        
        # Check for cover signal (exit short)
        elif df.iloc[i]['cover_signal'] == 1 and position == -1:
            # Exit short position
            exit_price = df.iloc[i]['cover_price']
            exit_time = current_time
            exit_index = i
            
            # Calculate profit/loss
            pnl = (entry_price - exit_price) * position_size
            
            # Calculate commission
            commission = max(position_size * commission_per_share, min_commission)
            total_commission += commission
            
            # Update capital
            capital += pnl - commission
            
            # Calculate MAE and MFE for short trade
            trade_bars = df.iloc[entry_index:exit_index + 1]
            mae = (entry_price - trade_bars['High'].max()) * position_size  # worst unrealized loss
            mfe = (entry_price - trade_bars['Low'].min()) * position_size   # best unrealized gain
            
            # Number of bars held
            bars_held = exit_index - entry_index
            
            # Track trade
            trade = {
                'symbol': 'NVDA',
                'entry_time': entry_time,
                'exit_time': exit_time,
                'type': 'Short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'commission': commission * 2,  # Entry + exit
                'shares': position_size,
                'position_value': entry_price * position_size,
                'mae': mae,
                'mfe': mfe,
                'bars_held': bars_held
            }
            trades.append(trade)
            
            # Track wins/losses
            if pnl > 0:
                wins += 1
                total_win_amount += pnl
            else:
                losses += 1
                total_loss_amount += abs(pnl)
            
            # Reset position
            position = 0
            entry_price = 0
            entry_time = None
        
        # Track equity curve
        if position == 1:
            unrealized_pnl = (current_close - entry_price) * position_size
            equity_curve.append({'datetime': current_time, 'equity': capital + unrealized_pnl})
        elif position == -1:
            unrealized_pnl = (entry_price - current_close) * position_size
            equity_curve.append({'datetime': current_time, 'equity': capital + unrealized_pnl})
        else:
            equity_curve.append({'datetime': current_time, 'equity': capital})
        
        # Track drawdown
        current_equity = equity_curve[-1]['equity']
        if current_equity > peak_capital:
            peak_capital = current_equity
        drawdown = (peak_capital - current_equity) / peak_capital
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate statistics
    equity_df = pd.DataFrame(equity_curve)
    equity_df.set_index('datetime', inplace=True)
    
    # Calculate returns
    equity_df['returns'] = equity_df['equity'].pct_change()
    
    # Calculate statistics
    total_trades = len(trades)
    long_trades = len([t for t in trades if t['type'] == 'long'])
    short_trades = len([t for t in trades if t['type'] == 'short'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    loss_rate = (losses / total_trades * 100) if total_trades > 0 else 0
    
    net_profit = capital - initial_capital
    net_profit_pct = (net_profit / initial_capital) * 100
    
    # Calculate annual return
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    annual_return = ((capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Calculate average profit/loss
    avg_profit_loss = net_profit / total_trades if total_trades > 0 else 0
    avg_profit_loss_pct = (avg_profit_loss / initial_capital) * 100
    
    # Calculate profit factor
    profit_factor = total_win_amount / total_loss_amount if total_loss_amount > 0 else 0
    
    # Calculate Sharpe ratio (annual risk-free rate of 3%)
    # Compute bars per year from actual data to correctly scale per-bar returns
    risk_free_rate_annual = 0.03
    days_in_data = (equity_df.index[-1] - equity_df.index[0]).days
    bars_per_year = len(equity_df) / days_in_data * 365.25 if days_in_data > 0 else 252 * 78
    per_bar_rf = risk_free_rate_annual / bars_per_year
    excess_returns = equity_df['returns'].dropna() - per_bar_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(bars_per_year) if excess_returns.std() > 0 else 0
    
    # Calculate Ulcer Index
    drawdowns = (equity_df['equity'] / equity_df['equity'].cummax() - 1) * 100
    ulcer_index = np.sqrt((drawdowns ** 2).mean())
    
    # Calculate CAR/MaxDD
    car_maxdd = annual_return / (max_drawdown * 100) if max_drawdown > 0 else 0
    
    # Calculate K-Ratio
    k_ratio = (equity_df['returns'].sum() / equity_df['returns'].std()) / np.sqrt(len(equity_df)) if equity_df['returns'].std() > 0 else 0
    
    # Calculate maximum trade drawdown
    max_trade_drawdown = min([t['pnl'] for t in trades]) if trades else 0
    
    stats = {
        'initial_capital': initial_capital,
        'end_capital': capital,
        'net_profit': net_profit,
        'net_profit_pct': net_profit_pct,
        'exposure_pct': 100,  # Simplified - would need more complex calculation
        'annual_return': annual_return,
        'total_commission': total_commission,
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'wins': wins,
        'win_rate': win_rate,
        'total_win_amount': total_win_amount,
        'losses': losses,
        'loss_rate': loss_rate,
        'total_loss_amount': total_loss_amount,
        'avg_profit_loss': avg_profit_loss,
        'avg_profit_loss_pct': avg_profit_loss_pct,
        'max_trade_drawdown': max_trade_drawdown,
        'max_system_drawdown': max_drawdown * 100,
        'car_maxdd': car_maxdd,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'ulcer_index': ulcer_index,
        'k_ratio': k_ratio,
        'trades': trades,
        'equity_curve': equity_df,
        'data': df  # Full dataframe with indicators for explore report
    }
    
    return stats


def generate_html_report(stats, output_path):
    """
    Generate HTML report with backtest statistics and equity chart.
    
    Parameters:
    -----------
    stats : dict
        Backtest statistics
    output_path : str
        Output HTML file path
    """
    # Prepare equity curve data for chart
    equity_df = stats['equity_curve']
    equity_dates = [d.strftime('%Y-%m-%d %H:%M') for d in equity_df.index]
    equity_values = equity_df['equity'].tolist()
    
    # Sample data if too many points (max 500 points for performance)
    if len(equity_dates) > 500:
        step = len(equity_dates) // 500
        equity_dates = equity_dates[::step]
        equity_values = equity_values[::step]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - S8004 IRB1000 V1 NVDA</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .stat-card {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }}
            .stat-label {{
                font-weight: bold;
                color: #666;
                font-size: 14px;
            }}
            .stat-value {{
                font-size: 24px;
                color: #333;
                margin-top: 5px;
            }}
            .positive {{
                color: #4CAF50;
            }}
            .negative {{
                color: #f44336;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .chart-container {{
                position: relative;
                height: 400px;
                margin-top: 30px;
                margin-bottom: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Backtest Report - S8004 IRB1000 V1 NVDA</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
            
            <h2>Performance Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Initial Capital</div>
                    <div class="stat-value">${stats['initial_capital']:,.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">End Capital</div>
                    <div class="stat-value">${stats['end_capital']:,.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Net Profit</div>
                    <div class="stat-value {'positive' if stats['net_profit'] >= 0 else 'negative'}">
                        ${stats['net_profit']:,.2f}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Net Profit %</div>
                    <div class="stat-value {'positive' if stats['net_profit_pct'] >= 0 else 'negative'}">
                        {stats['net_profit_pct']:.2f}%
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Exposure %</div>
                    <div class="stat-value">{stats['exposure_pct']:.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Annual Return %</div>
                    <div class="stat-value {'positive' if stats['annual_return'] >= 0 else 'negative'}">
                        {stats['annual_return']:.2f}%
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Commission Cost</div>
                    <div class="stat-value">${stats['total_commission']:,.2f}</div>
                </div>
            </div>
            
            <h2>Trade Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Number of Trades</div>
                    <div class="stat-value">{stats['total_trades']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Long Trades</div>
                    <div class="stat-value">{stats['long_trades']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Short Trades</div>
                    <div class="stat-value">{stats['short_trades']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Wins</div>
                    <div class="stat-value positive">{stats['wins']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Wins %</div>
                    <div class="stat-value">{stats['win_rate']:.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Win Amount</div>
                    <div class="stat-value positive">${stats['total_win_amount']:,.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Losses</div>
                    <div class="stat-value negative">{stats['losses']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Number of Losses %</div>
                    <div class="stat-value">{stats['loss_rate']:.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Loss Amount</div>
                    <div class="stat-value negative">${stats['total_loss_amount']:,.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average Profit/Loss</div>
                    <div class="stat-value {'positive' if stats['avg_profit_loss'] >= 0 else 'negative'}">
                        ${stats['avg_profit_loss']:,.2f}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Average Profit/Loss %</div>
                    <div class="stat-value {'positive' if stats['avg_profit_loss_pct'] >= 0 else 'negative'}">
                        {stats['avg_profit_loss_pct']:.2f}%
                    </div>
                </div>
            </div>
            
            <h2>Risk Metrics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Maximum Trade Drawdown</div>
                    <div class="stat-value negative">${stats['max_trade_drawdown']:,.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Maximum System Drawdown</div>
                    <div class="stat-value negative">{stats['max_system_drawdown']:.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">CAR/MaxDD</div>
                    <div class="stat-value">{stats['car_maxdd']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Profit Factor</div>
                    <div class="stat-value">{stats['profit_factor']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Sharpe Ratio</div>
                    <div class="stat-value">{stats['sharpe_ratio']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Ulcer Index</div>
                    <div class="stat-value">{stats['ulcer_index']:.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">K-Ratio</div>
                    <div class="stat-value">{stats['k_ratio']:.2f}</div>
                </div>
            </div>
        </div>
        
        <script>
            const ctx = document.getElementById('equityChart').getContext('2d');
            const equityChart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {equity_dates},
                    datasets: [{{
                        label: 'Equity ($)',
                        data: {equity_values},
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false
                        }}
                    }},
                    scales: {{
                        x: {{
                            display: true,
                            title: {{
                                display: true,
                                text: 'Date/Time'
                            }},
                            ticks: {{
                                maxTicksLimit: 20
                            }}
                        }},
                        y: {{
                            display: true,
                            title: {{
                                display: true,
                                text: 'Equity ($)'
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")


def generate_csv_report(trades, output_path):
    """
    Generate CSV report with trade details.
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
    output_path : str
        Output CSV file path
    """
    # Create DataFrame from trades
    trades_df = pd.DataFrame(trades)
    
    # Calculate additional columns
    trades_df['pct_change'] = ((trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'] * 100) * trades_df['type'].map({'Long': 1, 'Short': -1})
    trades_df['pct_profit'] = (trades_df['pnl'] / trades_df['position_value'] * 100)
    
    # Calculate cumulative profit
    trades_df['cumulative_profit'] = trades_df['pnl'].cumsum()
    
    # Format datetime columns
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time']).dt.strftime('%d/%m/%Y %H:%M')
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time']).dt.strftime('%d/%m/%Y %H:%M')
    
    # Reorder and rename columns to match requirements
    trades_df = trades_df[[
        'symbol', 'type', 'entry_time', 'entry_price', 'exit_time', 'exit_price',
        'pct_change', 'pnl', 'pct_profit', 'shares', 'position_value',
        'cumulative_profit', 'bars_held', 'mae', 'mfe'
    ]]
    
    # Sort by entry_time (column C) in descending order
    trades_df = trades_df.sort_values(by='entry_time', ascending=False)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write CSV file
    trades_df.to_csv(output_path, index=False)
    
    print(f"CSV report generated: {output_path}")


def generate_explore_csv(df, output_path):
    """
    Generate explore CSV file with all 5-minute bars from 04/22 to 04/30.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with all indicators and signals
    output_path : str
        Output CSV file path
    """
    # Filter rows from 04/22 to 04/30 (April 22 to April 30)
    start_date = pd.Timestamp('2026-04-22')
    end_date = pd.Timestamp('2026-04-30 23:59:59')
    
    # Filter dataframe for the date range
    mask = (df.index >= start_date) & (df.index <= end_date)
    filtered_df = df[mask]
    
    explore_data = []
    
    for i in range(len(filtered_df)):
        row = filtered_df.iloc[i]
        original_idx = df.index.get_loc(filtered_df.index[i])
        
        # Determine signal type
        if row['buy00'] == 1:
            signal_type = 'buy00'
        elif row['short00'] == 1:
            signal_type = 'short00'
        else:
            signal_type = 'none'
        
        # Build explore data dictionary
        row_data = {
            'symbol': 'NVDA',
            'signal_type': signal_type,
            'datetime': filtered_df.index[i].strftime('%d/%m/%Y %H:%M'),
            'open': row['Open'],
            'close': row['Close'],
            'high': row['High'],
            'low': row['Low'],
            'buy00': row['buy00'],
            'short00': row['short00'],
            'iRbBullish': row['iRbBullish'],
            'iRbBearish': row['iRbBearish'],
            'regSlopema10': row['regSlopema10'],
            'regSlopema10_shift1': df.iloc[original_idx-1]['regSlopema10'] if original_idx > 0 else np.nan,
            'rt10': row['rt10'],
            'ema20DiffCAbsRound0': row['ema20DiffCAbsRound0'],
            'ema20DiffCAbsRank': row['ema20DiffCAbsRank'],
            'pRtEma20Rankper': row['pRtEma20Rankper'],
            'pRtEma20Rankper_current': row['pRtEma20Rankper'],
            'pRtEma20Rankper_shift1': df.iloc[original_idx-1]['pRtEma20Rankper'] if original_idx > 0 else np.nan,
            'pRtEma20Rankper_shift2': df.iloc[original_idx-2]['pRtEma20Rankper'] if original_idx > 1 else np.nan
        }
        
        # Add new 5 columns for sumEma20DiffCAbsRank and sumEma20DiffCAbsUpRank
        for r in range(1, 6):
            row_data[f'sumEma20DiffCAbsRank_{r}'] = row[f'sumEma20DiffCAbsRank_{r}']
            row_data[f'sumEma20DiffCAbsUpRank_{r}'] = row[f'sumEma20DiffCAbsUpRank_{r}']
            
        explore_data.append(row_data)
    
    # Create DataFrame
    explore_df = pd.DataFrame(explore_data)
    
    # Sort by datetime (column C) in descending order
    if not explore_df.empty:
        explore_df = explore_df.sort_values(by='datetime', ascending=False)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write CSV file
    explore_df.to_csv(output_path, index=False)
    
    print(f"Explore CSV generated: {output_path} ({len(explore_df)} signals)")


def main():
    """
    Main function to run backtest.
    """
    # Configuration
    # csv only from 0930 to 1600
    # data_file = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\csvExcel\NVDA_20250101_20260430_5Min_0930_1600.csv"
    # csv include pre and post trading hour, i.e. from 0400 to 2000
    data_file = r"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_8001_20260204_NVDA\csvExcel\NVDA_20250101_20260430_5Min_.csv"
    
    # initial_capital = 100000
    initial_capital = 20000
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    
    # Output paths
    html_output = rf"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_8001_20260204_NVDA\backTestResult\S8001_4_GenerateFromPromptQwen37Max_{timestamp}.html"
    csv_output = rf"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_8001_20260204_NVDA\backTestResult\S8001_4_GenerateFromPromptQwen37Max_{timestamp}.csv"
    explore_output = rf"C:\Project\ProjectLife\VSCode Algo Workspace DataFile\VS_8001_20260204_NVDA\backTestResult\S8001_4_GenerateFromPromptQwen37Max_explore_{timestamp}.csv"
    
    print("=" * 80)
    print("Backtest: S8001_4 IRB1000 V1 Strategy on NVDA 5-Minute Data")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_data(data_file)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Initialize strategy
    print("\nInitializing strategy...")
    strategy = strategyIRB1000_V1(mode='backtest')
    
    # Run backtest
    print("\nRunning backtest...")
    stats = run_backtest(df, strategy, initial_capital)
    
    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital:        ${stats['initial_capital']:,.2f}")
    print(f"End Capital:            ${stats['end_capital']:,.2f}")
    print(f"Net Profit:             ${stats['net_profit']:,.2f} ({stats['net_profit_pct']:.2f}%)")
    print(f"Annual Return:          {stats['annual_return']:.2f}%")
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"  - Long Trades:        {stats['long_trades']}")
    print(f"  - Short Trades:       {stats['short_trades']}")
    print(f"Wins:                   {stats['wins']} ({stats['win_rate']:.2f}%)")
    print(f"Losses:                 {stats['losses']} ({stats['loss_rate']:.2f}%)")
    print(f"Profit Factor:          {stats['profit_factor']:.2f}")
    print(f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {stats['max_system_drawdown']:.2f}%")
    print(f"Total Commission:       ${stats['total_commission']:,.2f}")
    print("=" * 80)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_html_report(stats, html_output)
    generate_csv_report(stats['trades'], csv_output)
    generate_explore_csv(stats['data'], explore_output)
    
    print("\nBacktest completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
