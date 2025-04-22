import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vmdpy import VMD
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.modules.data_utils import load_aligned
from app.modules.vmd_models import (
    decompose_vmd,
    prepare_vmd_ml_data,
    train_huber,
    train_lstm
)

# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Configuration")
model_type      = st.sidebar.selectbox("Model Type", ["Huber", "LSTM"])
coverage_target = st.sidebar.slider("Calibration target coverage", 0.80, 0.99, 0.95, 0.01)

# series picker
TABLES = {
    "WTI Crude (FRED)": (
        "bond_stocks", "WTI Crude Oil"
    ),
    "WCS Houston": (
        "pricing_vector",
        "WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"
    ),
    "Cushing Inventory": (
        "wpr_sliding", "EIA CUSHING- OK CRUDE EXCL SPR STK"
    )
}
table_label   = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_name = TABLES[table_label]

# lookback/horizon/split
lookback  = st.sidebar.number_input("Lookback days", min_value=5,  value=30)
horizon   = st.sidebar.number_input("Forecast horizon (days)", min_value=1, value=1)
split_pct = st.sidebar.slider("Train %", 50, 90, 85)
df_full   = load_aligned(tbl)[series_name].ffill().dropna()
split_ix  = int(len(df_full) * (split_pct / 100))
split_date = df_full.index[split_ix]

# VMD params
alpha = st.sidebar.number_input("VMD α",     min_value=0.0, value=2000.0)
K     = st.sidebar.number_input("VMD modes", min_value=1,   value=5)
tol   = st.sidebar.number_input("VMD tol",   value=1e-7,    format="%.1e")

# run button
run = st.sidebar.button("▶️ Run models")
if not run:
    st.write("Adjust parameters and click ▶️ Run models to start.")
    st.stop()

# page header
st.title("📈 VMD → Huber / LSTM Modeling")
st.markdown(f"**{model_type}** on `{series_name}` • split at {split_date:%Y-%m-%d}")

series = df_full

# ──────────────────────────────────────────────────────────────────────────────
# 1) Prepare & train on raw features
X_tr, y_tr, idx_tr, X_te, y_te, idx_te = prepare_vmd_ml_data(
    tbl, series_name,
    lookback=lookback,
    horizon=horizon,
    split_frac=split_pct/100,
    vmd_kwargs=dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=tol)
)

if model_type == "Huber":
    raw_model   = train_huber(X_tr, y_tr)
    # flatten test set for predict
    X_te_flat   = X_te.reshape(len(X_te), -1)
    preds_raw   = raw_model.predict(X_te_flat)
else:
    raw_model   = train_lstm(
        X_tr, y_tr,
        lookback=lookback,
        num_features=X_tr.shape[2],
        units=32, epochs=50, batch_size=16
    )
    preds_raw   = raw_model.predict(X_te).flatten()

# wrap into Series
y_true_raw   = pd.Series(y_te,    index=idx_te)
y_pred_raw   = pd.Series(preds_raw, index=idx_te)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Raw backtest metrics & plot
mae_raw  = (y_true_raw - y_pred_raw).abs().mean()
rmse_raw = np.sqrt(((y_true_raw - y_pred_raw)**2).mean())
st.write({"Raw MAE": f"{mae_raw:.3f}", "Raw RMSE": f"{rmse_raw:.3f}"})

fig, ax = plt.subplots()
ax.plot(y_true_raw.index,      y_true_raw,   label="Actual")
ax.plot(y_pred_raw.index,      y_pred_raw,   label="Predicted")
ax.set_title("Backtest (Raw Features)")
ax.legend()
st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Calibration helper
def calibrate_band(y_true: pd.Series, y_pred: pd.Series, coverage_target: float):
    resid  = (y_true - y_pred).dropna()
    sigma  = resid.std()
    alphas = np.linspace(0.5, 3.0, 26)
    recs   = []
    for α in alphas:
        lo  = y_pred - α * sigma
        hi  = y_pred + α * sigma
        cov = ((y_true >= lo) & (y_true <= hi)).mean()
        w   = (hi - lo).mean()
        recs.append((α, cov, w))
    df = pd.DataFrame(recs, columns=["alpha","coverage","avg_width"])
    cands = df[df.coverage >= coverage_target]
    α_opt = (
        cands.sort_values("avg_width").iloc[0].alpha
        if not cands.empty
        else df.sort_values("coverage", ascending=False).iloc[0].alpha
    )
    return df, α_opt

# 4) Raw calibration band plots
df_calib_raw, α_opt_raw = calibrate_band(y_true_raw, y_pred_raw, coverage_target)

fig, ax = plt.subplots(); ax2 = ax.twinx()
ax.plot(df_calib_raw.alpha,  df_calib_raw.coverage,  label="Coverage")
ax2.plot(df_calib_raw.alpha, df_calib_raw.avg_width, color="gray", label="Avg Width")
ax.axvline(α_opt_raw, color="red", linestyle="--", label=f"opt α={α_opt_raw:.2f}")
ax.set_xlabel("α"); ax.set_ylabel("Coverage"); ax2.set_ylabel("Avg Width")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
st.pyplot(fig)

fig, ax = plt.subplots()
lo = y_pred_raw - α_opt_raw * (y_true_raw - y_pred_raw).std()
hi = y_pred_raw + α_opt_raw * (y_true_raw - y_pred_raw).std()
ax.plot(y_true_raw.index, y_true_raw,   label="Actual")
ax.plot(y_pred_raw.index, y_pred_raw,   label="Predicted")
ax.fill_between(y_pred_raw.index, lo, hi, color="orange", alpha=0.3,
                label=f"±{α_opt_raw:.2f}σ")
ax.set_title("Backtest w/ Raw Range Band")
ax.legend()
st.pyplot(fig)


st.success("✅ Models trained and plots rendered correctly, good work David!")
