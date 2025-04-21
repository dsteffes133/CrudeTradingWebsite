# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from app.modules.vmd_models import (
    prepare_huber_data, train_huber,
    prepare_lstm_data, train_lstm
)
from app.modules.data_utils import load_aligned
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.sidebar.header("🔬 VMD + ML Models")

# 1) Select table & series
TABLES = {
    "Macro / Market":        "bond_stocks",
    "US Imports / Exports":  "us_imports_exports",
    "Global IE":             "global_imports_exports",
    "WPR Sliding":           "wpr_sliding",
    "Pricing Vector":        "pricing_vector",
    "Pipeline Daily":        "daily_pipeline",
    "Movement Daily":        "daily_movement",
}
table_name = st.sidebar.selectbox("Table", list(TABLES.keys()))
tbl = TABLES[table_name]
df = load_aligned(tbl)

series = st.sidebar.selectbox("Series", df.select_dtypes("number").columns.tolist())

# 2) VMD params
st.sidebar.subheader("VMD settings")
alpha = st.sidebar.number_input("alpha", 100.0, 1e4, value=2000.0)
K     = st.sidebar.slider("Number of modes K", 2, 10, value=5)
init  = st.sidebar.radio("init", [0,1,2], index=1)
tol   = st.sidebar.number_input("tol", 1e-9, 1e-3, value=1e-7, format="%.0e")
vmd_kwargs = dict(alpha=alpha, K=K, init=init, tol=tol)

# 3) ML settings
st.sidebar.subheader("Data / ML settings")
lookback   = st.sidebar.number_input("Lookback days", 5, 60, value=30)
horizon    = st.sidebar.number_input("Horizon (days ahead)", 1, 30, value=7)
split_date = st.sidebar.date_input("Train/test split date", datetime(2023,1,1))

model_type = st.sidebar.radio("Model", ["Huber", "LSTM"])
if model_type == "Huber":
    hub_params = {
        "epsilon": st.sidebar.number_input("Huber epsilon", 1.1, 2.0, 1.35),
        "alpha":   st.sidebar.number_input("Huber alpha",   0.0001, 1.0, 0.0001, format="%.4f"),
        "max_iter":st.sidebar.number_input("max_iter", 100, 10000, 100)
    }
else:
    units      = st.sidebar.number_input("LSTM units", 8, 128, 32)
    epochs     = st.sidebar.number_input("Epochs", 10, 200, 50)
    batch_size = st.sidebar.number_input("Batch size", 8, 128, 16)

if st.sidebar.button("▶️ Train & Evaluate"):
    if model_type == "Huber":
        X_tr, y_tr, X_te, y_te = prepare_huber_data(
            tbl, series, lookback, horizon, split_date.isoformat(), vmd_kwargs
        )
        model = train_huber(X_tr, y_tr, **hub_params)

        # predict
        y_pred_tr = model.predict(X_tr)
        y_pred_te = model.predict(X_te)

    else:  # LSTM
        X_tr, y_tr, X_te, y_te = prepare_lstm_data(
            tbl, series, lookback, horizon, split_date.isoformat(), vmd_kwargs
        )
        n_features = X_tr.shape[2]
        model = train_lstm(X_tr, y_tr, lookback, n_features, units, epochs, batch_size)

        y_pred_tr = model.predict(X_tr).flatten()
        y_pred_te = model.predict(X_te).flatten()

    # metrics
    def metrics(y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    train_mae, train_rmse = metrics(y_tr, y_pred_tr)
    test_mae,  test_rmse  = metrics(y_te, y_pred_te)

    st.subheader(f"{model_type} Results for `{series}`")
    st.write(f"**Train** → MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}")
    st.write(f"**Test**  → MAE: {test_mae:.3f},  RMSE: {test_rmse:.3f}")

    # optional plot for LSTM
    if model_type == "LSTM":
        fig, ax = plt.subplots()
        ax.plot(y_te.index, y_te, label="Actual")
        ax.plot(y_te.index, y_pred_te, label="Predicted")
        ax.set_title("Test Set: Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)
