# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.modules.data_utils import load_aligned
from app.modules.vmd_models import (
    decompose_vmd,
    prepare_huber_data, train_huber,
    prepare_lstm_data, train_lstm
)

st.set_page_config(page_title="VMD + Huber / LSTM", layout="wide")
st.title("🔬 VMD Decomposition & Modeling")

# ─── Sidebar: select table & series ───────────────────────────────────────────
TABLES = {
    "Macro / Market":        "bond_stocks",
    "US Imports / Exports":  "us_imports_exports",
    "Global IE":             "global_imports_exports",
    "WPR Sliding":           "wpr_sliding",
    "Pricing Vector":        "pricing_vector",
    "Pipeline Daily":        "daily_pipeline",
    "Movement Daily":        "daily_movement",
}
table_label = st.sidebar.selectbox("Table", list(TABLES.keys()))
tbl = TABLES[table_label]

# load aligned numeric dataframe
df = load_aligned(tbl)
series = st.sidebar.selectbox("Series", df.select_dtypes(include="number").columns.tolist())

# ─── Sidebar: VMD & forecast settings ─────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("VMD parameters")
alpha = st.sidebar.number_input("α (bandwidth)", value=2000.0, step=100.0)
K     = st.sidebar.slider("Number of modes (K)", 2, 10, 5)
tol   = st.sidebar.number_input("Tolerance", value=1e-7, format="%.0e")

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast horizon")
lookback   = st.sidebar.number_input("Lookback window (days)", 5, 90, 30)
horizon    = st.sidebar.number_input("Forecast ahead (days)", 1, 30, 1)
split_date = st.sidebar.date_input("Train/Test split date", value=pd.to_datetime("2023-01-01"))

st.sidebar.markdown("---")
st.sidebar.subheader("Models to run")
do_huber = st.sidebar.checkbox("Huber Regressor", value=True)
do_lstm  = st.sidebar.checkbox("LSTM", value=True)

run_btn = st.sidebar.button("▶ Run models")

# ─── Main ─────────────────────────────────────────────────────────────────────
if run_btn:
    # 1️⃣ VMD decomposition on your chosen series
    signal = df[series].ffill()
    comps  = decompose_vmd(signal, alpha=alpha, K=K, tol=tol, tau=0.0, DC=0, init=1)

    st.markdown("### VMD Components (first 5 modes)")
    st.line_chart(comps.iloc[:, :5])

    # --- Huber Regressor pipeline ---------------------------------------------
    if do_huber:
        st.markdown("## 🤖 Huber Regressor")

        X_tr, y_tr, X_te, y_te = prepare_huber_data(
            table=tbl,
            series=series,
            lookback=int(lookback),
            horizon=int(horizon),
            split_date=str(split_date),
            vmd_kwargs=dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=tol)
        )
        hub = train_huber(X_tr, y_tr)
        y_pred_te = hub.predict(X_te)

        # Backtest: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(y_te.index, y_te, label="Actual")
        ax.plot(y_te.index, y_pred_te, label="Predicted")
        ax.set_title("Huber: Test Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # Residuals & distribution
        resid = y_te - y_pred_te
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(resid.index, resid); ax.axhline(0, color="k")
            ax.set_title("Residuals Over Time")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.hist(resid, bins=30, edgecolor="k")
            ax.set_title("Error Distribution")
            st.pyplot(fig)

    # --- LSTM pipeline ---------------------------------------------------------
    if do_lstm:
        st.markdown("## 🤖 LSTM Model")

        X_tr, y_tr, X_te, y_te = prepare_lstm_data(
            table=tbl,
            series=series,
            lookback=int(lookback),
            horizon=int(horizon),
            split_date=str(split_date),
            vmd_kwargs=dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=tol)
        )
        model = train_lstm(
            X_train=X_tr, y_train=y_tr,
            lookback=int(lookback), K=comps.shape[1],
            units=32, epochs=50, batch_size=16
        )

        # Training history (if available)
        if hasattr(model, "history"):
            hist = model.history.history
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(hist["loss"],  label="train loss")
            ax.plot(hist["val_loss"], label="val loss")
            ax.set_title("LSTM Training History")
            st.pyplot(fig)

        # Predict & backtest
        y_pred_te = model.predict(X_te).flatten()
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(y_te.index, y_te, label="Actual")
        ax.plot(y_te.index, y_pred_te, label="Predicted")
        ax.set_title("LSTM: Test Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # Residuals & distribution
        resid_l = y_te - y_pred_te
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.plot(resid_l.index, resid_l); ax.axhline(0, color="k")
            ax.set_title("Residuals Over Time")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.hist(resid_l, bins=30, edgecolor="k")
            ax.set_title("Error Distribution")
            st.pyplot(fig)

    st.success("✅ Models trained and plots rendered correctly, good work David!")
