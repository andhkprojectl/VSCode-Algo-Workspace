import pandas as pd
df  = pd.read_csv('C:/Project/ProjectLife/Feasible Study/calcuate_statistics/aa2_sort_sell_datetime.csv')
total_profit = df['Profit'].sum()
print(f"Total Profit: ${total_profit:,.2f}")