# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.modules.data_utils import load_aligned
from app.modules.vmd_models import prepare_vmd_ml_data, train_huber, forecast_vmd
import config

# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Calibration")
coverage_target = st.sidebar.slider("Calibration coverage", 0.80, 0.99, 0.95, 0.01)

st.sidebar.header("📊 Series Selection")
TABLES = {
    "WTI Crude (FRED)": ("bond_stocks", "WTI Crude Oil"),
    # add more series here…
}
series_label = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_col = TABLES[series_label]

split_pct = st.sidebar.slider("Train %", 50, 90, 85)
run_backtest = st.sidebar.button("▶️ Run backtest")
run_forecast = st.sidebar.button("🔮 Run forecast")

# ──────────────────────────────────────────────────────────────────────────────
df_full = load_aligned(tbl)[series_col].ffill().dropna()
series = df_full
split_ix = int(len(series) * (split_pct / 100))
split_date = series.index[split_ix]

st.title("📈 VMD → Huber Modeling")
st.markdown(f"split at {split_date:%Y-%m-%d}")

# ──────────────────────────────────────────────────────────────────────────────
if run_backtest:
    X_tr, y_tr, idx_tr, X_te, y_te, idx_te = prepare_vmd_ml_data(
        tbl, series_col, split_frac=split_pct/100
    )
    model = train_huber(X_tr, y_tr)
    preds_te = model.predict(X_te.reshape(len(X_te), -1))

    y_true = pd.Series(y_te, index=idx_te)
    y_pred = pd.Series(preds_te, index=idx_te)
    mae  = (y_true - y_pred).abs().mean()
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    st.write({"MAE": f"{mae:.3f}", "RMSE": f"{rmse:.3f}"})

    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true, label="Actual", color="black")
    ax.plot(y_pred.index, y_pred, label="Predicted", color="blue")
    ax.set_title("Backtest Results")
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Backtest complete!")

# ──────────────────────────────────────────────────────────────────────────────
if run_forecast:
    # backtest part for calibration
    X_tr, y_tr, idx_tr, X_te, y_te, idx_te = prepare_vmd_ml_data(
        tbl, series_col, split_frac=split_pct/100
    )
    model = train_huber(X_tr, y_tr)
    preds_te = model.predict(X_te.reshape(len(X_te), -1))

    y_true = pd.Series(y_te, index=idx_te)
    y_pred = pd.Series(preds_te, index=idx_te)
    sigma = (y_true - y_pred).std()

    # forecast future
    future = forecast_vmd(series)

    # combine
    y_pred_full = pd.concat([y_pred, future])

    # build ± band
    resid = (y_true - y_pred).dropna()
    sigma = resid.std()
    alpha_band = 1.0
    lo_full = y_pred_full - alpha_band * sigma
    hi_full = y_pred_full + alpha_band * sigma

    # plot
    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true, label="Actual (Backtest)", color="black")
    ax.plot(y_pred.index, y_pred, label="Predicted (Backtest)", color="blue")
    ax.plot(future.index, future, label="Forecast", linestyle="--", color="red")
    ax.fill_between(
        y_pred_full.index, lo_full, hi_full,
        alpha=0.2, label=f"±{alpha_band}σ"
    )
    ax.set_title(f"Backtest & {config.HORIZON}-Day Forecast")
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Forecast complete!")
