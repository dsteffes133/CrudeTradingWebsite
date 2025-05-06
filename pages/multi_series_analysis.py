# pages/02_data_explorer.py

import streamlit as st
import pandas as pd
from sqlalchemy import text

from app.modules.data_processing import ENGINE

# —————————————————————————————————————————————————————————————
# 1) Caching helper to load each SQL table on demand
# —————————————————————————————————————————————————————————————
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
            return df.set_index(col).sort_index()
    raise KeyError(f"No date-like column found in {table_name}: {df.columns.tolist()}")

# —————————————————————————————————————————————————————————————
# 2) Which tables exist
# —————————————————————————————————————————————————————————————
CATEGORY_TO_TABLE = {
    "Macro & Market":       "bond_stocks",
    "WPR Sliding":          "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
    "Daily Pipeline":      "daily_pipeline",
    "Daily Movement":      "daily_movement",
}

# —————————————————————————————————————————————————————————————
# 3) Sidebar: choose which tables to include
# —————————————————————————————————————————————————————————————
st.sidebar.header("📈 Data Explorer")

# Let user pick one or more tables to draw from
tables_chosen = st.sidebar.multiselect(
    "1️⃣ Select table(s)",
    options=list(CATEGORY_TO_TABLE.keys()),
    default=list(CATEGORY_TO_TABLE.keys())[:1],  # default to first table
)

# Load each chosen table
loaded_dfs: dict[str, pd.DataFrame] = {
    tbl: load_table(CATEGORY_TO_TABLE[tbl])
    for tbl in tables_chosen
}

# Build a single flat list of "Table: Series" for selection
series_labels: list[str] = []
for tbl, df in loaded_dfs.items():
    for col in df.columns:
        series_labels.append(f"{tbl} ▶ {col}")

# —————————————————————————————————————————————————————————————
# 4) Sidebar: pick any series from that combined list
# —————————————————————————————————————————————————————————————
to_plot = st.sidebar.multiselect(
    "2️⃣ Pick series to plot",
    options=series_labels,
    # you could set some defaults here if you like:
    default=[],
)

# —————————————————————————————————————————————————————————————
# 5) Main area: assemble and render the chart
# —————————————————————————————————————————————————————————————
st.title("📊 Data Explorer")

if not to_plot:
    st.info("Select at least one series from the sidebar to see a chart.")
    st.stop()

# Build a DataFrame that contains each selected series under its own column
chart_df = pd.DataFrame()

for label in to_plot:
    tbl, col = label.split(" ▶ ", 1)
    df = loaded_dfs[tbl]
    # Pull out that column and rename it to the short series name
    chart_df[col] = df[col]

# Drop any rows where all selected series are NaN, then plot
chart_df = chart_df.dropna(how="all")

if chart_df.empty:
    st.error("No overlapping dates found among your selections.")
else:
    st.line_chart(chart_df)

    # Show covered date range
    start, end = chart_df.index.min().date(), chart_df.index.max().date()
    st.caption(f"Showing data from {start} to {end}")

