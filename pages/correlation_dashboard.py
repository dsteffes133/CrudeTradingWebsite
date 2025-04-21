# pages/correlation_dashboard.py

import streamlit as st
import pandas as pd

from app.modules.data_utils import load_aligned

# —————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def load_all_data() -> pd.DataFrame:
    """
    Load & align every relevant table, prefix columns by table name,
    and join them into a single DataFrame on the master date index.
    """
    tables = {
        "Macro & Market":        ("bond_stocks",           "macro__"),
        "US Imports / Exports":  ("us_imports_exports",    "usie__"),
        "Global IE":             ("global_imports_exports","gie__"),
        "WPR Sliding":           ("wpr_sliding",           "wpr__"),
        "Pricing Vector":        ("pricing_vector",        "price__"),
        "Pipeline Daily":        ("daily_pipeline",        "pipe__"),
        "Movement Daily":        ("daily_movement",        "mov__"),
    }
    dfs = []
    for display_name, (tbl, prefix) in tables.items():
        df = load_aligned(tbl).add_prefix(prefix)
        dfs.append(df)
    return pd.concat(dfs, axis=1)

# —————————————————————————————————————————————————————————————
st.sidebar.header("📊 Correlation Explorer")

# 1️⃣ Load the combined DataFrame once
df_all = load_all_data()

# 2️⃣ Pick a data category
CATEGORY_TO_PREFIX = {
    "Macro & Market":        "macro__",
    "US Imports / Exports":  "usie__",
    "Global IE":             "gie__",
    "WPR Sliding":           "wpr__",
    "Pricing Vector":        "price__",
    "Pipeline Daily":        "pipe__",
    "Movement Daily":        "mov__",
}
category = st.sidebar.selectbox(
    "Select data category",
    list(CATEGORY_TO_PREFIX.keys())
)
prefix = CATEGORY_TO_PREFIX[category]

# 3️⃣ Find all numeric columns in that category
numeric_cols = [
    col for col in df_all.columns
    if col.startswith(prefix) and pd.api.types.is_numeric_dtype(df_all[col])
]
if not numeric_cols:
    st.error(f"No numeric columns found for category {category}")
    st.stop()

# 4️⃣ Let the user pick their target variable
target = st.sidebar.selectbox("Select target variable", sorted(numeric_cols))

# 5️⃣ Compute one numeric‑only Pearson correlation matrix, pull out our target
corr_full = (
    df_all
      .corr(method="pearson", numeric_only=True)
      [target]
      .drop(target)
)

# 6️⃣ Pick the top 15 drivers by absolute correlation
top15    = corr_full.abs().nlargest(15).index
top_corr = corr_full.loc[top15]

# —————————————————————————————————————————————————————————————
st.title("📊 Correlation Explorer")
st.markdown(
    f"Showing **15** series most correlated with **`{target}`** "
    f"across **all tables**"
)

# — Table & Download Button —
st.subheader("Top 15 Correlated Variables")
table = (
    top_corr
      .rename_axis("variable")
      .reset_index(name="correlation")
      .assign(correlation=lambda df: df["correlation"].round(3))
)
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False)
st.download_button(
    label="Download correlations CSV",
    data=csv,
    file_name=f"{target}_top15_correlations.csv",
    mime="text/csv"
)

