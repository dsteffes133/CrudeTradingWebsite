# pages/seasonal_and_dist.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sqlalchemy import text

from app.modules.data_processing import ENGINE

# —————————————————————————————————————————————————————————————
# Helper to load & index any SQL table
# —————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
    # find and set the date-like column as the index
    for col in df.columns:
        if col.lower() in ("date", "index"):
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    raise KeyError(f"No date-like column found in {table_name}")

# —————————————————————————————————————————————————————————————
st.sidebar.header("📊 Seasonal & Distribution Explorer")

# 1️⃣ Pick your table
TABLES = {
    "Macro & Market":       "bond_stocks",
    "US Imports/Exports":   "us_imports_exports",
    "Global IE":            "global_imports_exports",
    "WPR Sliding":          "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
}
category = st.sidebar.selectbox("Select table", list(TABLES.keys()))
table_name = TABLES[category]

# 2️⃣ Load and pick your series
df = load_table(table_name)
series = st.sidebar.selectbox("Select series", df.columns.tolist())
ts = df[series].dropna()

if ts.empty:
    st.error("No data available for that series.")
    st.stop()

st.title("📈 Seasonal & Distribution Explorer")
st.markdown(f"**Series:** `{series}`  •  **Table:** `{table_name}`")

# —————————————————————————————————————————————————————————————
# 1) Interactive Seasonal Plot
# —————————————————————————————————————————————————————————————
season = ts.to_frame("value")
season["year"]  = season.index.year
season["month"] = season.index.month

# Long form for Altair
df_long = (
    season
    .reset_index()
    .rename(columns={"index": "date"})
)

# Compute monthly min/max (for band)
band_df = (
    df_long
    .groupby("month")["value"]
    .agg(min_v="min", max_v="max")
    .reset_index()
)

base = (
    alt.Chart(df_long)
    .encode(
        x=alt.X("month:O", title="Month"),
        y=alt.Y("value:Q", title=series),
        color=alt.Color("year:O", title="Year"),
        tooltip=[
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("month:O", title="Month"),
            alt.Tooltip("value:Q", title=series, format=".2f")
        ],
    )
)

# shaded min-max band
band = (
    alt.Chart(band_df)
    .mark_area(color="lightgray", opacity=0.3)
    .encode(
        x="month:O",
        y="min_v:Q",
        y2="max_v:Q"
    )
)

lines = base.mark_line(point=True)

seasonal_chart = (
    (band + lines)
    .properties(width="container", height=350, title="Seasonal Plot by Year")
    .interactive()
)

st.altair_chart(seasonal_chart, use_container_width=True)

# —————————————————————————————————————————————————————————————
# 2) Interactive Distribution Plot
# —————————————————————————————————————————————————————————————
latest = ts.iloc[-1]

hist = (
    alt.Chart(ts.rename("value").reset_index().rename(columns={"index": "date"}))
    .mark_bar(color="steelblue")
    .encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=40), title=series),
        y=alt.Y("count()", title="Frequency"),
        tooltip=[
            alt.Tooltip("count()", title="Count"),
            alt.Tooltip("value:Q", title=series, format=".2f")
        ]
    )
)

rule = (
    alt.Chart(pd.DataFrame({"value": [latest]}))
    .mark_rule(color="red", strokeWidth=2)
    .encode(
        x="value:Q",
        tooltip=[alt.Tooltip("value:Q", title="Current Value", format=".2f")]
    )
)

dist_chart = (
    (hist + rule)
    .properties(width="container", height=300, title="Historical Distribution")
    .interactive()
)

st.altair_chart(dist_chart, use_container_width=True)
