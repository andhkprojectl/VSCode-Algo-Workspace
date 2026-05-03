import datetime
from contextlib import nullcontext

import pandas as pd
import datetime as dt
from datetime import datetime, date, timedelta
from pyiqfeed import HistoryConn

def testConn(conn):
    try :
        if conn is None:
            conn = HistoryConn(name="History")
            conn.connect()

        start_dt = datetime(2025, 4, 1)
        end_dt = datetime(2025, 4, 30, 23, 59, 59)

        data = conn.request_bars_in_period(
            ticker="DIA",
            interval_len=1800,  # 30 minutes (1800 seconds)
            interval_type="s",  # Seconds
            bgn_prd=start_dt,
            end_prd=end_dt,
            bgn_flt=datetime.strptime("09:30", "%H:%M").time(),
            end_flt=datetime.strptime("16:00", "%H:%M").time(),
            ascend=True,  # Oldest to latest
            max_bars=None,  # Fetch all available bars
            timeout=None
        )
        print ("Output:")
        print (data)
        conn.disconnect()
    except Exception as e:
        print (e)

#main
testConn(None)