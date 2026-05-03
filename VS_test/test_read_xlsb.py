import pandas as pd

file_path = "c:/tmp/nvda_1min_dtn_data.xlsb"

df = pd.read_excel(file_path, sheet_name="nvda_1min_dtn_data", engine="pyxlsb")

print(df.head())