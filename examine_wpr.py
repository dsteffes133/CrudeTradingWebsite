# examine_wpr.py

from sqlalchemy import create_engine, text
import pandas as pd

# Adjust this path if your combined.db lives elsewhere
DB_URI = "sqlite:///app/data/combined.db"

def inspect_wpr(table_name: str, engine):
    print(f"\n=== {table_name} ===")
    # 1) Load via a Connection + raw SQL
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(f"SELECT * FROM {table_name}"),
            conn
        )

    # 2) Find the date‑like column
    date_col = next((c for c in df.columns if c.lower() in ("date","index")), None)
    if date_col is None:
        print(f"  ERROR: no date/index column found (cols={df.columns.tolist()})")
        return

    # 3) Parse and set as index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"

    # 4) Print summary
    print(f"Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"Columns    : {', '.join(df.columns)}")

    # 5) Peek
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nLast  3 rows:")
    print(df.tail(3).to_string())

if __name__ == "__main__":
    engine = create_engine(DB_URI)

    for tbl in ("wpr_truth", "wpr_sliding"):
        inspect_wpr(tbl, engine)
