# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from app.modules.data_utils import load_aligned
from app.modules.vmd_models import prepare_vmd_ml_data, train_huber, forecast_vmd

# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Calibration")
coverage_target = st.sidebar.slider(
    "Calibration coverage (backtest & forecast bands)",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01,
)

st.sidebar.header("📊 Series & Splits")
TABLES = {
    "WTI Crude (FRED)": ("bond_stocks", "WTI Crude Oil"),
    # add more here…
}
series_label    = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_col = TABLES[series_label]
split_pct       = st.sidebar.slider("Train %", 50, 90, 85)

st.sidebar.header("🔮 Forecast Settings")
forecast_horizon = st.sidebar.slider(
    "Forecast length (days)",
    min_value=1,
    max_value=30,
    value=7,
)

run_backtest = st.sidebar.button("▶️ Run backtest")
run_forecast = st.sidebar.button("🔮 Run forecast")

# ──────────────────────────────────────────────────────────────────────────────
# load & split
series   = load_aligned(tbl)[series_col].ffill().dropna()
split_ix = int(len(series) * (split_pct / 100))
split_date = series.index[split_ix]

st.title(f"📈 VMD → Huber Modeling (Forecast = {forecast_horizon} days)")
st.markdown(f"Train/test split at **{split_date:%Y-%m-%d}**")

# ──────────────────────────────────────────────────────────────────────────────
if run_backtest:
    X_tr, y_tr, idx_tr, X_te, y_te, idx_te = prepare_vmd_ml_data(
        tbl, series_col, split_frac=split_pct/100
    )
    model    = train_huber(X_tr, y_tr)
    preds_te = model.predict(X_te.reshape(len(X_te), -1))

    y_true = pd.Series(y_te, index=idx_te)
    y_pred = pd.Series(preds_te, index=idx_te)

    # metrics
    mae  = (y_true - y_pred).abs().mean()
    rmse = np.sqrt(((y_true - y_pred)**2).mean())
    st.write({"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}"})

    # calibration band α from normal quantile
    alpha_bt = norm.ppf((1 + coverage_target) / 2)
    sigma_bt = (y_true - y_pred).std()
    lo_bt    = y_pred - alpha_bt * sigma_bt
    hi_bt    = y_pred + alpha_bt * sigma_bt

    # plot
    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true, label="Actual", color="black")
    ax.plot(y_pred.index, y_pred, label="Predicted", color="blue")
    ax.fill_between(
        y_pred.index, lo_bt, hi_bt,
        color="blue", alpha=0.2,
        label=f"±{coverage_target*100:.0f}% band"
    )
    ax.set_title("Backtest Results with ±band")
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Backtest complete!")

# ──────────────────────────────────────────────────────────────────────────────
if run_forecast:
    # re-run backtest to calibrate
    X_tr, y_tr, idx_tr, X_te, y_te, idx_te = prepare_vmd_ml_data(
        tbl, series_col, split_frac=split_pct/100
    )
    model    = train_huber(X_tr, y_tr)
    preds_te = model.predict(X_te.reshape(len(X_te), -1))

    y_true = pd.Series(y_te, index=idx_te)
    y_pred = pd.Series(preds_te, index=idx_te)

    # calibration band
    alpha_bt = norm.ppf((1 + coverage_target) / 2)
    sigma_bt = (y_true - y_pred).std()

    # forecast
    future = forecast_vmd(series, horizon=forecast_horizon)

    # full series
    idx_full    = y_pred.index.append(future.index)
    y_pred_full = pd.concat([y_pred, future])
    lo_full     = y_pred_full - alpha_bt * sigma_bt
    hi_full     = y_pred_full + alpha_bt * sigma_bt

    # plot backtest + forecast + band
    fig, ax = plt.subplots()
    ax.plot(y_true.index, y_true, label="Actual (Backtest)", color="black")
    ax.plot(y_pred.index, y_pred, label="Predicted (Backtest)", color="blue")
    ax.plot(future.index, future, linestyle="--", label="Forecast", color="red")
    ax.fill_between(
        idx_full, lo_full, hi_full,
        alpha=0.2, label=f"±{coverage_target*100:.0f}% band"
    )
    ax.set_title(f"Backtest & {forecast_horizon}-Day Forecast with ±band")
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Forecast complete!")

