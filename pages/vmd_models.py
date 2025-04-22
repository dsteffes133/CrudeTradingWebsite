# pages/vmd_models.py

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from app.modules.vmd_models import (
    prepare_vmd_ml_data,
    train_huber,
    train_lstm
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("📈 VMD + Huber vs. LSTM Models")

# three series to model
OPTIONS = {
    "WTI Crude Oil": (
        "bond_stocks", "WTI Crude Oil"
    ),
    "WCS Houston": (
        "pricing_vector", 
        "WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"
    ),
    "Cushing‑OK Stocks": (
        "wpr_sliding", "EIA CUSHING- OK CRUDE EXCL SPR STK"
    ),
}

series_choice = st.sidebar.selectbox("Choose series", list(OPTIONS.keys()))
table, column = OPTIONS[series_choice]

lookback = st.sidebar.slider("Lookback days", 10, 60, 30)
horizon  = st.sidebar.number_input("Horizon (days ahead)", 1, 7, 1)
split_pct= st.sidebar.slider("Train fraction", 50, 95, 85) / 100.0

# VMD parameters
alpha = st.sidebar.number_input("VMD alpha", 500.0, 5000.0, 2000.0)
K     = st.sidebar.number_input("VMD modes K", 2, 10, 5)
vmd_args = dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=1e-7)

if st.sidebar.button("Run models"):
    with st.spinner("Preparing data & training…"):
        X_tr, y_tr, X_te, y_te = prepare_vmd_ml_data(
            table, column,
            lookback=lookback,
            horizon=horizon,
            split_frac=split_pct,
            vmd_kwargs=vmd_args
        )
        D = X_tr.shape[2]

        # —— Huber —— 
        hub = train_huber(X_tr, y_tr)
        # predict
        n_tr = len(y_tr)
        Xf_tr = X_tr.reshape(n_tr, -1)
        n_te = len(y_te)
        Xf_te = X_te.reshape(n_te, -1)
        y_pred_tr = hub.predict(Xf_tr)
        y_pred_te = hub.predict(Xf_te)
        mae_tr = mean_absolute_error(y_tr, y_pred_tr)
        rmse_tr= np.sqrt(mean_squared_error(y_tr, y_pred_tr))
        mae_te = mean_absolute_error(y_te, y_pred_te)
        rmse_te= np.sqrt(mean_squared_error(y_te, y_pred_te))

        st.subheader("🔷 HuberRegressor Results")
        st.write(f"Train MAE / RMSE: {mae_tr:.3f} / {rmse_tr:.3f}")
        st.write(f"Test  MAE / RMSE: {mae_te:.3f} / {rmse_te:.3f}")
        fig, ax = plt.subplots()
        ax.plot(y_te, label="Actual")
        ax.plot(y_pred_te, label="Predicted")
        ax.set_title("Huber: Test Actual vs. Predicted")
        ax.legend()
        st.pyplot(fig)

        # —— LSTM —— 
        lstm = train_lstm(
            X_tr, y_tr,
            lookback=lookback,
            num_features=D,
            units=32, epochs=50, batch_size=16
        )
        y_pred_tr2 = lstm.predict(X_tr).flatten()
        y_pred_te2 = lstm.predict(X_te).flatten()
        mae_tr2 = mean_absolute_error(y_tr, y_pred_tr2)
        rmse_tr2= np.sqrt(mean_squared_error(y_tr, y_pred_tr2))
        mae_te2 = mean_absolute_error(y_te, y_pred_te2)
        rmse_te2= np.sqrt(mean_squared_error(y_te, y_pred_te2))

        st.subheader("🟢 LSTM Results")
        st.write(f"Train MAE / RMSE: {mae_tr2:.3f} / {rmse_tr2:.3f}")
        st.write(f"Test  MAE / RMSE: {mae_te2:.3f} / {rmse_te2:.3f}")
        fig2, ax2 = plt.subplots()
        ax2.plot(y_te,    label="Actual")
        ax2.plot(y_pred_te2, label="LSTM Pred")
        ax2.set_title("LSTM: Test Actual vs. Predicted")
        ax2.legend()
        st.pyplot(fig2)


    st.success("✅ Models trained and plots rendered correctly, good work David!")
