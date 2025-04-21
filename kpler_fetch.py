#!/usr/bin/env python
"""
kpler_fetch.py

Run inside the isolated kpler-env venv to update all API‑driven tables
(us_imports_exports, global_imports_exports, bond_stocks) in combined.db.
"""

import sys
from pathlib import Path

# make sure project root is on PYTHONPATH so we can import our code
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

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
