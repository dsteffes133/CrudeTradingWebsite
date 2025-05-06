# pages/nearest_neighbors.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from app.modules.vector_db import (
    build_series_index, query_series_index,
    build_table_index,  query_table_index
)
from app.modules.data_utils import load_aligned

st.sidebar.header("🔍 Nearest Neighbors")

mode = st.sidebar.radio("Mode", ["Series", "Table"])
lookback = st.sidebar.slider("Lookback window (days)", 5, 90, 30)
table = st.sidebar.selectbox("Select table", [
    "pricing_vector","bond_stocks","wpr_sliding",
    "daily_pipeline","daily_movement"
])
query_date = st.sidebar.date_input("Query date")
k = st.sidebar.slider("k (neighbors)", 1, 10, 5)
distance_threshold = st.sidebar.slider(
    "Alert threshold (distance)", 0.0, 2.0, 0.5, step=0.01
)

if mode == "Series":
    cols = load_aligned(table).columns.tolist()
    series_name = st.sidebar.selectbox("Select series", cols)

    @st.cache_resource
    def get_series_idx(tbl, col, lb):
        return build_series_index(tbl, col, lb)

    index, dates = get_series_idx(table, series_name, lookback)

    if st.sidebar.button("Find Nearest Neighbors"):
        df_all = load_aligned(table)[series_name].fillna(method="ffill")
        with st.spinner("Searching…"):
            nn = query_series_index(
                index, dates, table, series_name,
                pd.Timestamp(query_date), lookback, k
            )
        neigh_dates, distances = zip(*nn)

        # Sparkline
        fig, ax = plt.subplots(figsize=(8,4))
        for d in neigh_dates:
            win = df_all[d - pd.Timedelta(days=lookback-1):d]
            ax.plot(win.index, win.values, color="gray", alpha=0.4)
        qwin = df_all[
            pd.Timestamp(query_date) - pd.Timedelta(days=lookback-1)
            :pd.Timestamp(query_date)
        ]
        ax.plot(qwin.index, qwin.values, color="blue", linewidth=2)
        ax.set_title(f"{series_name}: last {lookback} days vs. neighbors")
        st.pyplot(fig)

        # Momentum percentile
        hist_mom = (df_all.diff(lookback) / df_all.shift(lookback)).dropna()
        perc = int(
            hist_mom.rank(pct=True)
            .loc[pd.Timestamp(query_date)]
            * 100
        )
        st.info(f"30‑day momentum is at the {perc}th percentile of history.")

        # Results table
        df_nn = pd.DataFrame({"date": neigh_dates, "distance": distances})
        st.dataframe(df_nn, use_container_width=True)

else:  # Table mode
    @st.cache_resource
    def get_table_idx(tbl, lb):
        return build_table_index(tbl, lb)

    index, dates, X = get_table_idx(table, lookback)

    if st.sidebar.button("Find Nearest Neighbors"):
        df_all = load_aligned(table)
        with st.spinner("Searching…"):
            nn = query_table_index(
                index, dates, table,
                pd.Timestamp(query_date), lookback, k
            )
        neigh_dates, distances = zip(*nn)

        # PCA scatter
        rows = [dates.index(pd.Timestamp(query_date))] + \
               [dates.index(d) for d in neigh_dates]
        pts = PCA(n_components=2).fit_transform(X[rows])
        fig2, ax2 = plt.subplots()
        for pt, lbl in zip(pts, ["Query"] + [str(d.date()) for d in neigh_dates]):
            ax2.scatter(
                pt[0], pt[1],
                color="red" if lbl == "Query" else "blue",
                label=lbl
            )
        ax2.legend()
        ax2.set_title("PCA of lookback-window vectors")
        st.pyplot(fig2)

        # Heatmap of deviations
        today_vals = df_all.loc[pd.Timestamp(query_date)]
        neigh_vals = pd.DataFrame({d: df_all.loc[d] for d in neigh_dates}).T
        means = neigh_vals.mean()
        stds  = neigh_vals.std().replace(0, 1)
        zs    = ((today_vals - means) / stds).to_frame("z_score")
        fig3, ax3 = plt.subplots(figsize=(6, len(zs) / 4))
        im = ax3.imshow(zs, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        ax3.set_yticks(np.arange(len(zs))); ax3.set_yticklabels(zs.index)
        ax3.set_xticks([0]); ax3.set_xticklabels(["z_score"])
        fig3.colorbar(im, ax=ax3)
        ax3.set_title("Column‑wise deviation vs. neighbor avg")
        st.pyplot(fig3)

        # Alert
        min_dist = min(distances)
        if min_dist < distance_threshold:
            st.success(f"High‑confidence match (distance {min_dist:.3f})")
        else:
            st.warning(f"No close match (distance {min_dist:.3f})")

        # Results table
        df_nn = pd.DataFrame({"date": neigh_dates, "distance": distances})
        st.dataframe(df_nn, use_container_width=True)


