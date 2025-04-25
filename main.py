"""
Streamlit front-end for Integrated Trading Analytics System
===========================================================
• Upload latest Excel files (WPR, CRUDE SD, VectorDBPricing)
• Ingest them into the shared `combined.db` (SQLite)
• Show simple status / row counts after each action

Launch with:
    streamlit run main.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# project imports
from app.modules import data_processing as dp
from app.modules.data_processing import DB_PATH, ENGINE

st.set_page_config(page_title="Southbow Trading Analytics")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _count_rows(table: str) -> int:
    try:
        with ENGINE.connect() as conn:
            result = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}")
            return result.scalar_one()
    except Exception:
        return 0

def _success(msg: str):
    st.success(msg)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: File uploads
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📡  Upload Excel Files")
    uploaded_wpr   = st.file_uploader("Weekly Petroleum Report",   type=["xlsx","xlsm"])
    uploaded_pipe  = st.file_uploader("CRUDE SD pipeline/movements", type=["xlsm","xlsx"])
    uploaded_price = st.file_uploader("VectorDBPricing",          type=["xlsm","xlsx"])
    ingest_btn     = st.button("🔄 Ingest Selected Files")

    if ingest_btn:
        with st.spinner("Ingesting…"):
            if uploaded_wpr:
                truth, sliding = dp.import_weekly_petroleum_report(uploaded_wpr, save_sql=True)
                _success(f"WPR imported ({len(truth):,} truth rows)")
            if uploaded_pipe:
                mv, pl = dp.import_pipeline_flow_excel(uploaded_pipe, save_sql=True)
                _success(f"Pipeline imported ({pl.shape[0]:,} rows × {pl.shape[1]:,} cols)")
            if uploaded_price:
                merged = dp.import_vector_db_pricing(uploaded_price, save_sql=True)
                _success(f"Pricing vector imported ({len(merged):,} rows)")
        # Clear cache so counts update
        _count_rows.clear()

# ──────────────────────────────────────────────────────────────────────────────
# Main Area: Status & Dashboard
# ──────────────────────────────────────────────────────────────────────────────
st.title("SOUTHBOW TRADING ANALYTICS")
st.write("Use the sidebar to upload Excel files and ingest into the database.")

if st.sidebar.button("🔄 Refresh Kpler & FRED"):
    with st.spinner("Running ETL…"):
        # Call your isolated ETL script via PowerShell
        result = subprocess.run([
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "etl\\run.ps1"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            st.sidebar.success("✅ ETL complete!")
        else:
            st.sidebar.error("❌ ETL failed:")
            st.sidebar.text(result.stderr)


st.subheader("📊  Current table sizes")
status_cols = [
    ("bond_stocks",          "Macro / Market"),
    ("us_imports_exports",   "US Imports/Exports"),
    ("global_imports_exports","Global Imports/Exports"),
    ("wpr_truth",            "WPR Truth"),
    ("wpr_sliding",          "WPR Sliding"),
    ("daily_pipeline",       "Pipeline Daily"),
    ("daily_movement",       "Movement Daily"),
    ("pricing_vector",       "Pricing Vector"),
]
rows = {nice: _count_rows(tbl) for tbl, nice in status_cols}

df_status = pd.Series(rows, name="Rows").rename_axis("Table").to_frame()
st.dataframe(df_status, use_container_width=True)

st.caption(
    f"Database path: `{DB_PATH}`   |   "
    f"Last refreshed: {datetime.now():%Y-%m-%d %H:%M:%S}"
)
