# pages/correlation_scatter.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sqlalchemy import text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from app.modules.data_processing import ENGINE

# —————————————————————————————————————————————————————————————
# 1) Helper to load & index any SQL table
# —————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)
    # find & set the date-like column as index
    for col in df.columns:
        if col.lower() in ("date", "index"):
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    raise KeyError(f"No date-like column in {table_name}")

# —————————————————————————————————————————————————————————————
st.sidebar.header("🔍 Scatter & Regression")

# 2) Let the user pick any two tables and series
TABLES = {
    "Macro & Market":       "bond_stocks",
    "WPR Sliding":          "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
    "Daily Pipeline":       "daily_pipeline",
    "Daily Movement":       "daily_movement",
}

table1_label = st.sidebar.selectbox("1️⃣ Table for X-axis", list(TABLES.keys()), index=0)
table2_label = st.sidebar.selectbox("2️⃣ Table for Y-axis", list(TABLES.keys()), index=1)

df1 = load_table(TABLES[table1_label])
df2 = load_table(TABLES[table2_label])

col1 = st.sidebar.selectbox("3️⃣ X-variable", df1.columns.tolist(), key="xcol")
col2 = st.sidebar.selectbox("4️⃣ Y-variable", df2.columns.tolist(), key="ycol")

# 3) Align the two series on their common dates
x = df1[col1].rename("x")
y = df2[col2].rename("y")
df = pd.concat([x, y], axis=1).dropna()

if df.empty:
    st.error("No overlapping dates between your two selections.")
    st.stop()

st.title("📈 Scatter & Regression Explorer")
st.markdown(f"""
**X:** `{table1_label} → {col1}`  
**Y:** `{table2_label} → {col2}`
""")

# —————————————————————————————————————————————————————————————
# 4) Interactive scatter + regression line via Altair
# —————————————————————————————————————————————————————————————
# Prepare for Altair
plot_df = df.reset_index().rename(columns={"index": "date"})

base = alt.Chart(plot_df).encode(
    x=alt.X("x:Q", title=col1),
    y=alt.Y("y:Q", title=col2),
    tooltip=[
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("x:Q", title=col1, format=".2f"),
        alt.Tooltip("y:Q", title=col2, format=".2f"),
    ],
)

points = base.mark_circle(size=60, opacity=0.6, color="steelblue")

# regression line
reg_line = (
    base
    .transform_regression("x", "y", method="linear")
    .mark_line(color="red", strokeWidth=2)
)

st.altair_chart(
    (points + reg_line)
    .properties(width="container", height=400, title="Scatter with Linear Regression")
    .interactive(),
    use_container_width=True
)

# —————————————————————————————————————————————————————————————
# 5) Compute & display regression statistics
# —————————————————————————————————————————————————————————————
# Pearson correlation
pearson_r = df["x"].corr(df["y"])

# Fit a sklearn linear model
model = LinearRegression()
model.fit(df[["x"]], df["y"])
y_pred = model.predict(df[["x"]])
r2 = r2_score(df["y"], y_pred)
slope = float(model.coef_[0])
intercept = float(model.intercept_)

st.subheader("📊 Regression Metrics")
metrics = {
    "Pearson ρ":            f"{pearson_r:.3f}",
    "R² (sklearn)":         f"{r2:.3f}",
    "Slope (β₁)":           f"{slope:.4f}",
    "Intercept (β₀)":       f"{intercept:.4f}",
}
st.table(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]))
