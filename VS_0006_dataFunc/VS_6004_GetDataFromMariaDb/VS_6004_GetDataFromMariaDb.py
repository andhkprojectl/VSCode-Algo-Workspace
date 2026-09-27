"""
VS_6004_GetDataFromMariaDb.py
=============================
Load 1-minute OHLCV bars from MariaDB IBTradingDb.ticker1Min.

Credentials via VS_0002_config/.env DB_* variables (repo convention).
Function moved from
VS_0006_dataFunc/VS_6006_FeatureEngineering/F6006_GeneralFeature_2_glm.py
(previously named load_data_from_db).
"""

import os

import mysql.connector
import pandas as pd


def _load_db_env():
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        for p in (Path(__file__).resolve().parents[2] / "VS_0002_config" / ".env",
                  Path.cwd() / "VS_0002_config" / ".env"):
            if p.exists():
                load_dotenv(dotenv_path=str(p), override=False)
                return
    except ImportError:
        pass


def load_1Min_data_from_db(ticker='ES', start_date=None, end_date=None) -> pd.DataFrame:
    """Load 1-min bars with columns [open, high, low, close, volume] and a
    DatetimeIndex, sorted ascending."""
    _load_db_env()
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "ibUser1"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "IBTradingDb"),
    )
    try:
        cur = conn.cursor()
        q = ("SELECT datetime1, open, high, low, close, volume FROM ticker1Min "
             "WHERE ticker = %s")
        params = [ticker]
        if start_date:
            q += " AND datetime1 >= %s"
            params.append(start_date)
        if end_date:
            q += " AND datetime1 <= %s"
            params.append(end_date)
        q += " ORDER BY datetime1 ASC"
        cur.execute(q, tuple(params))
        rows = cur.fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=['datetime1', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime1'] = pd.to_datetime(df['datetime1'])
    df = df.set_index('datetime1')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df
