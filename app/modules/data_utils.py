# app/modules/data_utils.py

from datetime import timedelta
import pandas as pd
from sqlalchemy import text
from app.modules.data_processing import ENGINE

# --- Your fixed “earliest” start date ---
MASTER_START = pd.Timestamp("2021-11-15")

# --- Per‑table offsets (if any) ---
TABLE_OFFSETS = {
    "pricing_vector":        0,
    "bond_stocks":           0,
    "us_imports_exports":    0,
    "global_imports_exports":0,
    "wpr_sliding":           0,
    "daily_pipeline":        0,
    "daily_movement":        0,
}

def load_raw(table_name: str) -> pd.DataFrame:
    """Load raw table with proper DateTime index."""
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
    # find the date column
    date_col = next(c for c in df.columns if c.lower() in ("date","index"))
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def get_pricing_end() -> pd.Timestamp:
    """Return the maximum date present in pricing_vector."""
    df_price = load_raw("pricing_vector")
    return df_price.index.max()

def load_aligned(table_name: str) -> pd.DataFrame:
    """
    1) Load the raw table
    2) Shift its index by TABLE_OFFSETS
    3) Build a master calendar from MASTER_START to pricing end
    4) Reindex & ffill/bfill
    """
    # 1) raw
    df = load_raw(table_name)

    # 2) apply any table‐specific shift
    shift = TABLE_OFFSETS.get(table_name, 0)
    if shift:
        df = df.copy()
        df.index = df.index + timedelta(days=shift)

    # 3) master calendar up to pricing last date
    master_end   = get_pricing_end()
    master_index = pd.date_range(MASTER_START, master_end, freq="D")

    # 4) reindex & fill
    df = df.reindex(master_index)
    return df.ffill().bfill()


