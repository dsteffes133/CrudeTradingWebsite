# examine_db.py

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import pandas as pd

# Adjust this path if your combined.db lives elsewhere
DB_URI = "sqlite:///app/data/combined.db"

def main():
    engine    = create_engine(DB_URI)
    inspector = inspect(engine)

    print(f"Inspecting {DB_URI}\n")

    try:
        tables = inspector.get_table_names()
    except SQLAlchemyError as e:
        print("Error listing tables:", e)
        return

    for table in tables:
        # find any date‑like column
        try:
            cols = [col["name"] for col in inspector.get_columns(table)]
        except SQLAlchemyError as e:
            print(f"{table:20s} – could not inspect columns: {e}")
            continue

        date_col = next((c for c in cols if c.lower() in ("date","index")), None)
        if not date_col:
            print(f"{table:20s} – no date/index column found (cols={cols})")
            continue

        # use a Connection and exec_driver_sql to get the max date
        with engine.connect() as conn:
            try:
                result = conn.exec_driver_sql(
                    f"SELECT MAX([{date_col}]) AS last_date FROM [{table}]"
                )
                last = result.scalar_one()
            except SQLAlchemyError as e:
                print(f"{table:20s} – query failed: {e}")
                continue

        print(f"{table:20s} – last date = {last}")

if __name__ == "__main__":
    main()
