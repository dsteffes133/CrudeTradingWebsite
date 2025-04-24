# pages/seasonal_and_dist.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import text

from app.modules.data_processing import ENGINE

# —————————————————————————————————————————————————————————————
# Helper to load & index any table
@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
    # find first date‐like column and set as index
    for col in df.columns:
        if col.lower() in ("date", "index"):
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    raise KeyError(f"No date column in {table_name}")

# —————————————————————————————————————————————————————————————
st.sidebar.header("📊 Seasonal & Distribution Explorer")

# pick table
TABLES = {
    "Macro & Market":       "bond_stocks",
    "US Imports/Exports":   "us_imports_exports",
    "Global IE":            "global_imports_exports",
    "WPR Sliding":          "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
}
cat = st.sidebar.selectbox("1️⃣ Select table", list(TABLES.keys()))
table_name = TABLES[cat]

# load it
df = load_table(table_name)

# pick series
series = st.sidebar.selectbox("2️⃣ Select series", df.columns.tolist())
ts = df[series].dropna()

if ts.empty:
    st.error("No data for that series.")
    st.stop()

st.title("📈 Seasonal & Distribution Explorer")
st.markdown(f"**Series:** `{series}` from **{table_name}**")

# —————————————————————————————————————————————————————————————
# 1) Seasonal plot
# —————————————————————————————————————————————————————————————
# Prepare a pivot: index=month (1–12), columns=year, values=series
season = ts.copy().to_frame("v")
season["year"]  = season.index.year
season["month"] = season.index.month
pivot = (
    season
    .pivot_table(index="month", columns="year", values="v")
    .sort_index()
)

# compute min/max band
min_band = pivot.min(axis=1)
max_band = pivot.max(axis=1)

fig1, ax1 = plt.subplots(figsize=(8,4))
# plot each year
for yr in pivot.columns:
    ax1.plot(pivot.index, pivot[yr], label=str(yr), alpha=0.6)
# shaded band
ax1.fill_between(
    pivot.index, min_band, max_band,
    color="gray", alpha=0.2,
    label="Min–Max range"
)
ax1.set_xticks(range(1,13))
ax1.set_xlabel("Month")
ax1.set_ylabel(series)
ax1.set_title("Seasonal Plot (by Year)")
ax1.legend(ncol=2, fontsize="small", loc="upper left")

st.pyplot(fig1)

# —————————————————————————————————————————————————————————————
# 2) Distribution + current value
# —————————————————————————————————————————————————————————————
latest = ts.iloc[-1]

fig2, ax2 = plt.subplots(figsize=(8,3))
ax2.hist(ts, bins=30, color="skyblue", edgecolor="white")
ax2.axvline(latest, color="red", linewidth=2, label=f"Current: {latest:.2f}")
ax2.set_xlabel(series)
ax2.set_ylabel("Frequency")
ax2.set_title("Historical Distribution")
ax2.legend()

st.pyplot(fig2)
