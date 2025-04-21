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
    # concatenate side by side; they share the same MASTER_INDEX
    return pd.concat(dfs, axis=1)

# —————————————————————————————————————————————————————————————
st.sidebar.header("📊 Correlation Explorer")

# 1️⃣ Load the combined DataFrame once
df_all = load_all_data()

# 2️⃣ Step 1: pick a data category
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

# 3️⃣ Step 2: pick a variable within that category
candidates = [col for col in df_all.columns if col.startswith(prefix)]
if not candidates:
    st.error(f"No columns found for category {category}")
    st.stop()

# 1) pull out only the float/int columns
numeric_cols = df_all.select_dtypes(include="number").columns.tolist()

# 2) restrict the sidebar picker to those columns
target = st.sidebar.selectbox("Select target variable", sorted(numeric_cols))

# 3) compute corr on just that subset
corr_full = (
    df_all[numeric_cols]
      .corr(method="pearson")
      [target]
      .drop(labels=[target])
)

# 4️⃣ Compute correlations of target vs. all others
corr_full = df_all.corr(method="pearson")[target].drop(labels=[target])

# 5️⃣ Pick the top 15 by absolute correlation
top15 = corr_full.abs().nlargest(15).index
top_corr = corr_full.loc[top15]

# —————————————————————————————————————————————————————————————
st.title("📊 Correlation Explorer")
st.markdown(
    f"Showing **15** series most correlated with **`{target}`** "
    f"across **all tables**"
)

# — 6) Table & download —
st.subheader("Top 15 Correlated Variables")
table = (
    top_corr
    .rename_axis("variable")
    .reset_index(name="correlation")
    .assign(correlation=lambda d: d["correlation"].round(3))
)
st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False)
st.download_button(
    label="Download correlations CSV",
    data=csv,
    file_name=f"{target}_top15_correlations.csv",
    mime="text/csv"
)

