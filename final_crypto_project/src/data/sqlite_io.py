import sqlite3
import pandas as pd

def load_ohlcv_from_sqlite(table_name: str, db_path: str, start: str = None, end: str = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    if start and end:
        query += f" WHERE timestamp BETWEEN '{start}' AND '{end}'"
    elif start:
        query += f" WHERE timestamp >= '{start}'"
    elif end:
        query += f" WHERE timestamp <= '{end}'"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, parse_dates=["timestamp"])

    df.set_index("timestamp", inplace=True)
    df = df.sort_index()
    cols = {c.lower(): c for c in df.columns}
    needed = ["open","high","low","close","volume"]
    for k in needed:
        if k not in cols:
            raise ValueError(f"Missing required column: {k}")
    return pd.DataFrame({k: df[cols[k]] for k in needed}, index=df.index)