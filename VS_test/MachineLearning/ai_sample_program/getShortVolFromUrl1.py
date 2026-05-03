import pandas as pd
from datetime import datetime
#  import iqfeed as iq

date = datetime.now().strftime("%Y%m%d")
urls = [
    # f"http://regsho.finra.org/CNMSshvol{date}.txt",
    # f"http://regsho.finra.org/FNSQshvol{date}.txt",
    # f"http://regsho.finra.org/FNYXshvol{date}.txt"
    f"http://regsho.finra.org/FNQCshvol20251203.txt"
]

short_vol = {}
for url in urls:
    try:
        print ("1")
        df = pd.read_csv(url, sep="|", nrows=None)
        print("2")
        df = df.iloc[:-1]  # 最後一行是空
        for _, row in df.iterrows():
            sym = row['Symbol']
            short_vol[sym] = short_vol.get(sym, 0) + row['ShortVolume']
        df.to_csv("c:/tmp/get_short_vol.csv")
    except:
        print ("error")
        continue

