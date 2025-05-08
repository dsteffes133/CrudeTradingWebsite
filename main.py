import os
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# project imports
from app.modules import data_processing as dp
from app.modules.data_processing import DB_PATH, ENGINE

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Import the FRED update function
from app.modules.data_processing import update_macro_table

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
# Sidebar: File uploads & FRED refresh
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
        _count_rows.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: FRED refresh button
    st.header("🔄 Macro / FRED Refresh")
    if st.button("Refresh FRED Data"):
        with st.spinner("Pulling data from FRED…"):
            try:
                df = update_macro_table()
                _success(f"FRED updated ({len(df):,} rows)")
            except Exception as e:
                st.error("❌ FRED update failed")
                st.text(str(e))
        _count_rows.clear()

# ──────────────────────────────────────────────────────────────────────────────
# Main Area: Status & Dashboard
# ──────────────────────────────────────────────────────────────────────────────
st.title("SOUTHBOW TRADING ANALYTICS")
st.write("Use the sidebar to upload Excel files and pull FRED data into the database.")

st.subheader("📊  Current table sizes")
status_cols = [
    ("bond_stocks",    "Macro / Market"),    # now kept, since FRED writes to bond_stocks
    ("wpr_truth",      "WPR Truth"),
    ("wpr_sliding",    "WPR Sliding"),
    ("daily_pipeline", "Pipeline Daily"),
    ("daily_movement","Movement Daily"),
    ("pricing_vector","Pricing Vector"),
]
rows = {nice: _count_rows(tbl) for tbl, nice in status_cols}

df_status = (
    pd.Series(rows, name="Rows")
      .rename_axis("Table")
      .to_frame()
)
st.dataframe(df_status, use_container_width=True)

st.caption(
    f"Database path: `{DB_PATH}`   |   "
    f"Last refreshed: {datetime.now():%Y-%m-%d %H:%M:%S}"
)
