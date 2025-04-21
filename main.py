"""
Streamlit front‑end for Integrated Trading Analytics System
===========================================================
• Upload latest Excel files (WPR, CRUDE SD, VectorDBPricing)
• Trigger API refresh (Kpler + FRED) via isolated subprocess
• Show simple status / row counts after each action

Launch with:
    streamlit run main.py
"""
import os
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

# project imports
from app.modules import data_processing as dp
from app.modules.data_processing import DB_PATH, ENGINE

load_dotenv()  # pick up KPLER_EMAIL / KPLER_TOKEN

# ──────────────────────────────────────────────────────────────────────────────
# Setup for isolated Kpler subprocess
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
INIT_SCRIPT  = PROJECT_ROOT / "init_kpler_venv.sh"
VENV_PY      = PROJECT_ROOT / "kpler-env" / "bin" / "python"
FETCH_SCRIPT = PROJECT_ROOT / "kpler_fetch.py"

def run_kpler_subprocess():
    """
    1) Bootstraps `kpler-env` venv (if not already present) via init_kpler_venv.sh
    2) Runs kpler_fetch.py inside that venv, which writes to combined.db
    """
    try:
        # only create the venv once
        if not VENV_PY.exists():
            subprocess.run([str(INIT_SCRIPT)], check=True)
        # now invoke our helper inside that dedicated venv
        subprocess.run(
            [str(VENV_PY), str(FETCH_SCRIPT)],
            check=True,
            cwd=str(PROJECT_ROOT)
        )
    except subprocess.CalledProcessError:
        st.error("⚠️ Kpler data refresh failed. Check logs in kpler_fetch.py")
        st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit helpers
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
    st.toast(msg, icon="✅")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📡  Data Pipelines")
    with st.form("full_pipeline"):
        uploaded_wpr   = st.file_uploader("WPR Excel",   type=["xlsx","xlsm"])
        uploaded_pipe  = st.file_uploader("CRUDE SD",    type=["xlsm","xlsx"])
        uploaded_price = st.file_uploader("Pricing",     type=["xlsm","xlsx"])
        run_all        = st.form_submit_button("🔄 Upload & Refresh")

    if run_all:
        with st.spinner("Ingesting files and refreshing API…"):
            # 1) Excel imports (if provided)
            if uploaded_wpr:
                dp.import_weekly_petroleum_report(uploaded_wpr, save_sql=True)
            if uploaded_pipe:
                dp.import_pipeline_flow_excel(uploaded_pipe, save_sql=True)
            if uploaded_price:
                dp.import_vector_db_pricing(uploaded_price, save_sql=True)

            # 2) Pull Kpler + FRED via isolated subprocess
            run_kpler_subprocess()

            # 3) Clear cached row‑counts so the dashboard updates
            _count_rows.clear()

        st.success("All data ingested and refreshed ✅")

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.title("SOUTHBOW TRADING ANALYTICS")
st.markdown("Select a task in the sidebar →")

# --- Handle individual uploads -----------------------------------------------
if 'uploaded_wpr' in locals() and uploaded_wpr is not None:
    with st.spinner("Importing Weekly Petroleum Report …"):
        truth, sliding = dp.import_weekly_petroleum_report(uploaded_wpr)
    _success(f"WPR imported ({len(truth):,} truth rows)")

if 'uploaded_pipe' in locals() and uploaded_pipe is not None:
    with st.spinner("Importing CRUDE SD pipeline / movements …"):
        mv, pl = dp.import_pipeline_flow_excel(uploaded_pipe)
    _success("Pipeline + Movements imported ✔")

if 'uploaded_price' in locals() and uploaded_price is not None:
    with st.spinner("Importing VectorDBPricing.xlsm …"):
        merged = dp.import_vector_db_pricing(uploaded_price)
    _success("Pricing vector imported ✔")

# --- Status dashboard --------------------------------------------------------
st.subheader("📊  Current table sizes")
status_cols = [
    ("bond_stocks", "Macro / Market"),
    ("us_imports_exports", "US Imports / Exports"),
    ("global_imports_exports", "Global Imports / Exports"),
    ("wpr_truth", "WPR Truth"),
    ("wpr_sliding", "WPR Sliding"),
    ("daily_pipeline", "Pipeline Daily"),
    ("daily_movement", "Movement Daily"),
    ("pricing_vector", "Pricing Vector"),
]
rows = {nice: _count_rows(tbl) for tbl, nice in status_cols}
st.dataframe(pd.Series(rows, name="rows").rename_axis("Table"))

st.caption(
    f"Database path: `{DB_PATH}`  •  "
    f"Last refreshed: {datetime.now():%Y-%m-%d %H:%M:%S}"
)
