# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.modules.vmd_models import (
    prepare_huber_data, train_huber,
    prepare_lstm_data,  train_lstm
)
from app.modules.data_utils import load_aligned

st.sidebar.header("🔬 VMD + Huber / LSTM")

# ─── User inputs ───────────────────────────────────────────────────────────────
TABLES = {
    "Macro / Market":        "bond_stocks",
    "US Imports/Exports":    "us_imports_exports",
    "Global IE":             "global_imports_exports",
}
table_display = st.sidebar.selectbox("Table", list(TABLES.keys()))
TABLE         = TABLES[table_display]

# Load numeric series to choose from
df_tmp = load_aligned(TABLE).select_dtypes("number")
SERIES = st.sidebar.selectbox("Series", df_tmp.columns.tolist())

lookback   = st.sidebar.slider("Lookback days", 5, 60, 30)
horizon    = st.sidebar.slider("Horizon days",  1, 14,  1)
split_date = st.sidebar.date_input("Train/Test split", pd.to_datetime("2023-01-01"))

alpha = st.sidebar.slider("VMD α", 100.0, 5000.0, 2000.0)

if st.sidebar.button("Run VMD models"):
    vmd_kwargs = dict(alpha=alpha, tau=0.0, K=5, DC=0, init=1, tol=1e-7)

    # ─── H U B E R ───────────────────────────────────────────────────────────────
    st.subheader("1) Huber Regressor on VMD‑mode vectors")
    X_tr, y_tr, X_te, y_te = prepare_huber_data(
        TABLE, SERIES,
        lookback=int(lookback),
        horizon=int(horizon),
        split_date=str(split_date),
        vmd_kwargs=vmd_kwargs
    )
    hub = train_huber(X_tr, y_tr)
    # evaluate
    y_pred_tr = hub.predict(X_tr)
    y_pred_te = hub.predict(X_te)
    mae_tr = np.mean(np.abs(y_tr - y_pred_tr))
    mae_te = np.mean(np.abs(y_te - y_pred_te))
    rmse_tr = np.sqrt(np.mean((y_tr - y_pred_tr)**2))
    rmse_te = np.sqrt(np.mean((y_te - y_pred_te)**2))

    st.write({
        "Train MAE": mae_tr, "Train RMSE": rmse_tr,
        "Test  MAE": mae_te, "Test  RMSE": rmse_te
    })

    fig, ax = plt.subplots()
    ax.plot(y_te.index, y_te,    label="Actual")
    ax.plot(y_te.index, y_pred_te, label="Huber Pred")
    ax.set_title("Huber: Test Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)


    # ─── L S T M ────────────────────────────────────────────────────────────────
    st.subheader("2) LSTM on VMD‑mode vectors")
    X_tr, y_tr, X_te, y_te = prepare_lstm_data(
        TABLE, SERIES,
        lookback=int(lookback),
        horizon=int(horizon),
        split_date=str(split_date),
        vmd_kwargs=vmd_kwargs
    )

    timesteps, features = X_tr.shape[1], X_tr.shape[2]
    lstm = train_lstm(X_tr, y_tr, timesteps, features,
                      units=32, epochs=50, batch_size=16)

    y_pred_te = lstm.predict(X_te).ravel()
    mae_te    = np.mean(np.abs(y_te - y_pred_te))
    rmse_te   = np.sqrt(np.mean((y_te - y_pred_te)**2))

    st.write({"Test MAE": mae_te, "Test RMSE": rmse_te})

    fig, ax = plt.subplots()
    ax.plot(y_te.index, y_te,    label="Actual")
    ax.plot(y_te.index, y_pred_te, label="LSTM Pred")
    ax.set_title("LSTM: Test Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Models trained and plots rendered correctly, good work David!")
