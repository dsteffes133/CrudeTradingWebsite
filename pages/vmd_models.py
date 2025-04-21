# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.modules.vmd_models import (
    decompose_vmd,
    prepare_huber_data, train_huber,
    prepare_lstm_data, train_lstm
)

st.set_page_config(page_title="VMD + Regression/LSTM", layout="wide")
st.title("🔬 VMD Decomposition & Modeling")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Model Configuration")

# 1) Pick your table & series
TABLE = st.sidebar.selectbox("Data Table", [
    "bond_stocks", "us_imports_exports", "global_imports_exports"
])
SERIES = st.sidebar.text_input("Column name in table", value="VIX (Volatility)")

# 2) VMD params
st.sidebar.subheader("VMD settings")
alpha = st.sidebar.number_input("alpha", value=2000.0)
K     = st.sidebar.slider("Number of modes (K)", 2, 10, 5)
tol   = st.sidebar.number_input("tolerance", value=1e-7, format="%.0e")

# 3) Forecast settings
st.sidebar.subheader("Forecast settings")
lookback   = st.sidebar.number_input("Lookback days", 30, 90, 30)
horizon    = st.sidebar.number_input("Horizon days ahead", 1, 30, 1)
split_date = st.sidebar.date_input("Train/Test split date", value=pd.to_datetime("2023-01-01"))

# 4) Choose models
st.sidebar.subheader("Which models?")
do_huber = st.sidebar.checkbox("Huber Regressor", value=True)
do_lstm  = st.sidebar.checkbox("LSTM", value=True)
run_btn  = st.sidebar.button("▶ Run VMD + Models")

# ── Main ────────────────────────────────────────────────────────────────────
if run_btn:
    # 1) Load & decompose
    df = decompose_vmd(
        series=pd.read_sql_table(TABLE, con=st.experimental_get_query_params().get('engine', None))[SERIES],
        alpha=alpha, K=K, tol=tol
    )
    st.markdown(f"### VMD Components (first 5 modes)")
    st.line_chart(df.iloc[:, :5])

    if do_huber:
        st.markdown("## 🤖 Huber Regressor")
        X_tr, y_tr, X_te, y_te = prepare_huber_data(
            TABLE, SERIES,
            lookback=int(lookback), horizon=int(horizon),
            split_date=str(split_date), vmd_kwargs=dict(alpha=alpha, K=K, tol=tol, tau=0, DC=0, init=1)
        )
        hub = train_huber(X_tr, y_tr)
        y_pred_te = hub.predict(X_te)
        y_pred_tr = hub.predict(X_tr)

        # 1) Backtest: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(y_te.index, y_te, label="Actual")
        ax.plot(y_te.index, y_pred_te, label="Predicted")
        ax.set_title("Huber: Test Set Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # 2) Residuals over time
        resid = y_te - y_pred_te
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(resid.index, resid, color="tab:orange")
        ax.axhline(0, color="k", lw=1)
        ax.set_title("Huber Residuals (Test)")
        st.pyplot(fig)

        # 3) Error distribution
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(resid, bins=30, edgecolor="k")
        ax.set_title("Huber Error Distribution")
        st.pyplot(fig)

        # 4) Scatter Pred vs Actual
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_te, y_pred_te, alpha=0.6)
        ax.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], 'r--')
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs. Actual (Huber)")
        st.pyplot(fig)

    if do_lstm:
        st.markdown("## 🤖 LSTM Model")
        X_tr, y_tr, X_te, y_te = prepare_lstm_data(
            TABLE, SERIES,
            lookback=int(lookback), horizon=int(horizon),
            split_date=str(split_date), vmd_kwargs=dict(alpha=alpha, K=K, tol=tol, tau=0, DC=0, init=1)
        )
        model = train_lstm(X_tr, y_tr, lookback=int(lookback), K=K, units=32, epochs=50)
        y_pred_te = model.predict(X_te).flatten()

        # 1) Training history (if available)
        if hasattr(model, 'history'):
            hist = model.history.history
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(hist['loss'], label="train loss")
            ax.plot(hist['val_loss'], label="val loss")
            ax.set_title("LSTM Training History")
            ax.legend()
            st.pyplot(fig)

        # 2) Backtest: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(y_te.index, y_te, label="Actual")
        ax.plot(y_te.index, y_pred_te, label="Predicted")
        ax.set_title("LSTM: Test Set Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        # 3) Residuals
        resid_l = y_te - y_pred_te
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(resid_l.index, resid_l)
        ax.axhline(0, color="k", lw=1)
        ax.set_title("LSTM Residuals (Test)")
        st.pyplot(fig)

        # 4) Error distribution
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(resid_l, bins=30, edgecolor="k")
        ax.set_title("LSTM Error Distribution")
        st.pyplot(fig)

        # 5) Modes + prediction overlay
        comps = decompose_vmd(
            series=pd.read_sql_table(TABLE, con=st.experimental_get_query_params().get('engine', None))[SERIES],
            alpha=alpha, K=K, tol=tol
        )
        # sum of selected modes vs. actual
        recon = comps.sum(axis=1)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(recon.index, recon, label="Reconstruction")
        ax.plot(y_te.index, y_te, label="Actual", alpha=0.6)
        ax.set_title("Reconstructed Signal vs Actual")
        ax.legend()
        st.pyplot(fig)

    st.success("✅ Models trained and plots rendered.")
