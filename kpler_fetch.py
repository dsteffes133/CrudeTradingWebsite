#!/usr/bin/env python
"""
Standalone script to pull ALL Kpler pipelines
and write them into data/combined.db
"""
import os, sys
from datetime import date
# ensure we add your app folder to path
sys.path.insert(0, os.path.dirname(__file__)+"/app/modules")

from app.modules.data_processing import (
    fetch_us_imports_exports,
    fetch_global_imports_exports,
    update_macro_table
)

def main():
    # these env vars should already be set in Streamlit Cloud
    # e.g. KPLER_EMAIL / KPLER_TOKEN
    # run each pipeline:
    fetch_us_imports_exports()
    fetch_global_imports_exports()
    update_macro_table()
    print("✅ Kpler tables refreshed.")

if __name__ == "__main__":
    main()