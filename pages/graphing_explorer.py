# pages/02_data_explorer.py

import streamlit as st
import pandas as pd
from app.modules.data_processing import ENGINE

# —————————————————————————————————————————————————————————————
# 1) Caching helper to load each SQL table on demand
# —————————————————————————————————————————————————————————————
from sqlalchemy import text

@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(
            text(f"SELECT * FROM {table_name}"),
            conn
        )

    # find the first column that looks like a date
    for col in df.columns:
        if col.lower() in ("date", "index"):
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            return df.sort_index()

    # fallback: assume the index was written without a name
    raise KeyError(f"No date-like column found in {table_name}: {df.columns.tolist()}")

# —————————————————————————————————————————————————————————————
# 2) Map human‑readable categories to table names
# —————————————————————————————————————————————————————————————
CATEGORY_TO_TABLE = {
    "Macro & Market":       "bond_stocks",
    "US Imports / Exports": "us_imports_exports",
    "Global IE":            "global_imports_exports",
    "WPR Sliding":            "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
    # (omit pipeline flows if you want)
}

# —————————————————————————————————————————————————————————————
# 3) Sidebar UI for category + series selection
# —————————————————————————————————————————————————————————————
st.sidebar.header("📈 Data Explorer")

category = st.sidebar.selectbox(
    "1️⃣ Select data category",
    list(CATEGORY_TO_TABLE.keys())
)

table_name = CATEGORY_TO_TABLE[category]
df = load_table(table_name)

# Pre‑define your three KPI series (adjust names exactly as in your DB)
kpi_defaults = [
    'WCS Cushing weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD',  # use the exact column suffixes your DB has, e.g. "WCS…_rm1"
    'WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD',
    'WTI Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD'
]
# ensure defaults exist
defaults = [c for c in kpi_defaults if c in df.columns]

# Multi‑select for series to plot
to_plot = st.sidebar.multiselect(
    "2️⃣ Pick series to plot",
    options=df.columns.tolist(),
    default=defaults
)

# —————————————————————————————————————————————————————————————
# 4) Main area: render the chart
# —————————————————————————————————————————————————————————————
st.title("📊 Data Explorer")

if not to_plot:
    st.info("Select at least one series from the sidebar to see a chart.")
else:
    st.line_chart(df[to_plot])
    st.caption(f"Showing `{table_name}` from {df.index.min().date()} to {df.index.max().date()}")
