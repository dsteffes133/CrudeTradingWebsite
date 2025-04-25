#!/usr/bin/env python
"""
kpler_fetch.py

Run inside the isolated kpler-env venv to update all API‑driven tables
(us_imports_exports, global_imports_exports, bond_stocks) in combined.db.
"""


import sys, os

# ensure that the repo root (one level up) is on Python’s import path
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dotenv import load_dotenv
load_dotenv()   # loads KPLER_EMAIL / KPLER_TOKEN / FRED_API_KEY

from app.modules.data_processing import (
    fetch_us_imports_exports,
    fetch_global_imports_exports,
    update_macro_table,
)

def main():
    print("🔄 Starting Kpler & FRED API pipelines…")
    fetch_us_imports_exports()
    fetch_global_imports_exports()
    update_macro_table()
    print("✅ Kpler & FRED API pipelines complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Let the subprocess call see the non-zero exit code
        print(f"❌ Error in kpler_fetch: {e}", file=sys.stderr)
        sys.exit(1)
