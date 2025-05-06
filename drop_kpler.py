import sqlite3

DB_PATH = "app/data/combined.db"  # adjust if yours is elsewhere

# List the Kpler tables you want to remove:
kpler_tables = [
    "us_imports_exports",
    "global_imports_exports",     # if you created this via Kpler
    # add any other Kpler-driven tables here
]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

for tbl in kpler_tables:
    print(f"Dropping table {tbl} if it exists…")
    cur.execute(f"DROP TABLE IF EXISTS {tbl};")

conn.commit()

# Verify remaining tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
remaining = cur.fetchall()
print("\nRemaining tables in the database:")
for name, in remaining:
    print(" -", name)

conn.close()
