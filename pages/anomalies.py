import streamlit as st
import pandas as pd
from sqlalchemy import text
from app.modules.data_processing import ENGINE
from app.modules.anomaly_detection import (
    detect_anomalies_zscore,
    detect_anomalies_iforest
)

# —————————————————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    """Load any SQL table into a DataFrame indexed by its date column."""
    with ENGINE.connect() as conn:
        df = pd.read_sql_query(text(f"SELECT * FROM {table_name}"), conn)

    # find the date-like column
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "index")),
        None
    )
    if date_col is None:
        st.error(f"No date/index column in {table_name}: {df.columns.tolist()}")
        return df  # bail out, we’ll hit problems downstream

    # parse and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df

# —————————————————————————————————————————————————————————————————————————
st.sidebar.header("🔍 Anomaly Dashboard")

# 1) choose your table
category = st.sidebar.selectbox("Data category", [
    "Macro & Market", "US Imports / Exports",
    "Global IE", "WPR Sliding", "Pricing Vector"
])
table_map = {
    "Macro & Market":       "bond_stocks",
    "US Imports / Exports": "us_imports_exports",
    "Global IE":            "global_imports_exports",
    "WPR Sliding":          "wpr_sliding",
    "Pricing Vector":       "pricing_vector",
}
df = load_table(table_map[category])

# 2) sanity‑check
if df.empty:
    st.warning(f"`{table_map[category]}` is empty or failed to load.")
    st.stop()


# ───────────────────────────────────────────────────
# Alert banner: any z‑score anomalies (>2σ) in last 7 days?
# ───────────────────────────────────────────────────
alert_vars = []
for col in df.columns:
    # run a quick z‑score detection with window=30, threshold=2.0
    ser = df[col].dropna()
    res = detect_anomalies_zscore(ser, window=30, threshold=2.0)
    # look at the last 7 days
    cutoff = res.index.max() - pd.Timedelta(days=7)
    recent = res.loc[res.index >= cutoff, "is_anomaly"]
    if recent.any():
        alert_vars.append(col)

if alert_vars:
    st.warning(
        "⚠️ The following variables had z‑score anomalies (>2σ) in the last 7 days:\n\n"
        + ", ".join(alert_vars)
    )

# 3) pick your series
series_name = st.sidebar.selectbox("Select series", df.columns.tolist())
series = df[series_name].dropna()

# 4) choose method
method = st.sidebar.radio("Method", ["Z‑score", "IsolationForest"])
if method == "Z‑score":
    window    = st.sidebar.slider("Rolling window",  5, 90, 30)
    threshold = st.sidebar.slider("Z threshold",    1.0, 5.0, 3.0)
    result_df = detect_anomalies_zscore(series, window, threshold)
else:
    contamination = st.sidebar.slider(
        "Contamination", 0.001, 0.1, 0.01, step=0.001
    )
    # detect_iforest expects a DataFrame
    result_df = detect_anomalies_iforest(df[[series_name]], contamination)

# 6) build chart DataFrame
if method == "Z‑score":
    val_col = "value"
else:
    val_col = series_name

if val_col not in result_df.columns:
    st.error(f"Column `{val_col}` not found in result_df")
    st.stop()

chart_df = pd.DataFrame({
    "value":   result_df[val_col],
    "anomaly": result_df[val_col].where(result_df["is_anomaly"])
}, index=result_df.index)

# —————————————————————————————————————————————————————————————————————————
st.title("📈 Anomaly Dashboard")
st.subheader(f"{series_name} – {method}")

st.line_chart(chart_df)

# 7) tabulate anomalies
anoms = result_df[result_df["is_anomaly"]]
if not anoms.empty:
    if method == "Z‑score":
        st.dataframe(anoms[["value","z_score"]])
    else:
        st.dataframe(anoms[[series_name]])
else:
    st.info("No anomalies detected.")

