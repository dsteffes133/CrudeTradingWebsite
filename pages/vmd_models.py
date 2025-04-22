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
from app.modules.ml_utils import extract_aggregated_features
from app.modules.vmd_models import train_huber, train_lstm, prepare_vmd_ml_data

# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Configuration")
model_type      = st.sidebar.selectbox("Model Type", ["Huber", "LSTM"])
coverage_target = st.sidebar.slider("Calibration target coverage", 0.80, 0.99, 0.95, 0.01)

@st.cache_data(show_spinner=False)
def decompose_vmd(series: pd.Series, alpha: float, K: int, tol: float) -> pd.DataFrame:
    # note: tau=0.0, DC=0, init=1 are hard‑coded here
    u, _, _ = VMD(series.values, alpha, 0.0, K, 0, 1, tol)
    return pd.DataFrame(
        u.T,
        index=series.index,
        columns=[f"mode_{i+1}" for i in range(u.shape[0])]
    )

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
table_label  = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_name = TABLES[table_label]

lookback  = st.sidebar.number_input("Lookback days", min_value=5,  value=30)
horizon   = st.sidebar.number_input("Forecast horizon (days)", min_value=1, value=1)
split_pct = st.sidebar.slider("Train %", 50, 90, 85)
df_full   = load_aligned(tbl)[series_name].ffill().dropna()
split_ix  = int(len(df_full) * (split_pct / 100))
split_date = df_full.index[split_ix]

alpha = st.sidebar.number_input("VMD α",     min_value=0.0, value=2000.0)
K     = st.sidebar.number_input("VMD modes", min_value=1,   value=5)
tol   = st.sidebar.number_input("VMD tol",   value=1e-7,    format="%.1e")

run = st.sidebar.button("▶️ Run models")
if not run:
    st.write("Adjust parameters and click ▶️ Run models to start.")
    st.stop()

st.title("📈 VMD → Huber / LSTM Modeling")
st.markdown(f"**{model_type}** on `{series_name}` • split at {split_date:%Y-%m-%d}")

series = df_full

# 1) Prepare & train on raw features
X_tr, y_tr, X_te, y_te = None, None, None, None
if model_type == "Huber":
    # flatten‑feature version
    comps = decompose_vmd(series, alpha, K, tol)
    # reuse our helper on the stacked VMD+engineered features
    X_tr, y_tr, X_te, y_te = prepare_vmd_ml_data(
        tbl, series_name,
        lookback=lookback, horizon=horizon,
        split_frac=split_pct/100,
        vmd_kwargs=dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=tol)
    )
    raw_model  = train_huber(X_tr, y_tr)
    y_pred_raw = pd.Series(raw_model.predict(
        X_te.reshape(len(X_te), -1)
    ), index=y_te.index)

else:
    # 3D‑input version
    X_tr, y_tr, X_te, y_te = prepare_vmd_ml_data(
        tbl, series_name,
        lookback=lookback, horizon=horizon,
        split_frac=split_pct/100,
        vmd_kwargs=dict(alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=tol)
    )
    raw_model  = train_lstm(
        X_tr, y_tr,
        lookback=lookback,
        num_features=X_tr.shape[2],
        units=32, epochs=50, batch_size=16
    )
    y_pred_raw = pd.Series(
        raw_model.predict(X_te).flatten(),
        index=y_te.index
    )

# 2) Raw backtest metrics & plot
mae_raw  = (y_te - y_pred_raw).abs().mean()
rmse_raw = np.sqrt(((y_te - y_pred_raw)**2).mean())
st.write({"Raw MAE": f"{mae_raw:.3f}", "Raw RMSE": f"{rmse_raw:.3f}"})

fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,       label="Actual")
ax.plot(y_te.index, y_pred_raw, label="Predicted")
ax.set_title("Backtest (Raw Features)")
ax.legend()
st.pyplot(fig)

# 3) Raw calibration band
def calibrate_band(y_true, y_pred, coverage_target):
    resid = (y_true - y_pred).dropna()
    sigma = resid.std()
    alphas = np.linspace(0.5, 3.0, 26)
    recs = []
    for α in alphas:
        lo = y_pred - α*sigma
        hi = y_pred + α*sigma
        cov = ((y_true>=lo)&(y_true<=hi)).mean()
        w   = (hi-lo).mean()
        recs.append((α, cov, w))
    df = pd.DataFrame(recs, columns=["alpha","coverage","avg_width"])
    cands = df[df.coverage>=coverage_target]
    return df, (
        cands.sort_values("avg_width").iloc[0].alpha
        if not cands.empty
        else df.sort_values("coverage", ascending=False).iloc[0].alpha
    )

df_calib_raw, α_opt_raw = calibrate_band(y_te, y_pred_raw, coverage_target)
fig, ax = plt.subplots(); ax2 = ax.twinx()
ax.plot(df_calib_raw.alpha,   df_calib_raw.coverage,   label="Coverage")
ax2.plot(df_calib_raw.alpha,  df_calib_raw.avg_width,  color="gray", label="Avg Width")
ax.axvline(α_opt_raw, color="red", linestyle="--", label=f"opt α={α_opt_raw:.2f}")
ax.set_xlabel("α"); ax.set_ylabel("Coverage"); ax2.set_ylabel("Avg Width")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
st.pyplot(fig)

fig, ax = plt.subplots()
lo = y_pred_raw - α_opt_raw * (y_te - y_pred_raw).std()
hi = y_pred_raw + α_opt_raw * (y_te - y_pred_raw).std()
ax.plot(y_te.index, y_te,       label="Actual")
ax.plot(y_te.index, y_pred_raw, label="Predicted")
ax.fill_between(y_te.index, lo, hi, color="orange", alpha=0.3, label=f"±{α_opt_raw:.2f}σ")
ax.set_title("Backtest w/ Raw Range Band")
ax.legend()
st.pyplot(fig)

# 4) Flatten & scale both branches
arr_tr = np.asarray(X_tr)
if arr_tr.ndim == 3:
    n_tr, L, D = arr_tr.shape
    flat_tr = arr_tr.reshape(n_tr, L*D)
else:
    n_tr, D = arr_tr.shape
    flat_tr = arr_tr

scaler_X   = StandardScaler().fit(flat_tr)
flat_tr_s  = scaler_X.transform(flat_tr)
X_tr_s     = (flat_tr_s.reshape(n_tr, L, D)
              if arr_tr.ndim == 3 else flat_tr_s)

arr_te = np.asarray(X_te)
if arr_te.ndim == 3:
    n_te = arr_te.shape[0]
    flat_te = arr_te.reshape(n_te, L*D)
else:
    n_te, _ = arr_te.shape
    flat_te = arr_te

flat_te_s  = scaler_X.transform(flat_te)
X_te_s     = (flat_te_s.reshape(n_te, L, D)
              if arr_te.ndim == 3 else flat_te_s)

# scale y
scaler_y = StandardScaler().fit(y_tr.reshape(-1,1))
y_tr_s   = scaler_y.transform(y_tr.reshape(-1,1)).flatten()
y_te_s   = scaler_y.transform(y_te.reshape(-1,1)).flatten()

# 5) Retrain on scaled data
if model_type == "Huber":
    scaled_model = train_huber(X_tr_s, y_tr_s)
    y_pred_s     = scaled_model.predict(
        flat_te_s if arr_te.ndim==2 else X_te_s.reshape(n_te, -1)
    )
else:
    scaled_model = train_lstm(
        X_tr_s, y_tr_s,
        lookback=lookback,
        num_features=D,
        units=32, epochs=50, batch_size=16
    )
    y_pred_s     = scaled_model.predict(X_te_s).flatten()

y_pred_scaled = pd.Series(
    scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).flatten(),
    index=y_te.index
)

# 6) Scaled backtest & plots
mae_s  = (y_te - y_pred_scaled).abs().mean()
rmse_s = np.sqrt(((y_te - y_pred_scaled)**2).mean())
st.write({"Scaled MAE": f"{mae_s:.3f}", "Scaled RMSE": f"{rmse_s:.3f}"})

fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,           label="Actual")
ax.plot(y_te.index, y_pred_scaled,  label="Predicted")
ax.set_title("Backtest (Scaled Features)")
ax.legend()
st.pyplot(fig)

df_calib_s, α_opt_s = calibrate_band(y_te, y_pred_scaled, coverage_target)
fig, ax = plt.subplots(); ax2 = ax.twinx()
ax.plot(df_calib_s.alpha,    df_calib_s.coverage,  label="Coverage")
ax2.plot(df_calib_s.alpha,   df_calib_s.avg_width, label="Avg Width", color="gray")
ax.axvline(α_opt_s, color="red", linestyle="--", label=f"opt α={α_opt_s:.2f}")
ax.set_xlabel("α"); ax.set_ylabel("Coverage"); ax2.set_ylabel("Avg Width")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
st.pyplot(fig)

fig, ax = plt.subplots()
lo_s = y_pred_scaled - α_opt_s * (y_te - y_pred_scaled).std()
hi_s = y_pred_scaled + α_opt_s * (y_te - y_pred_scaled).std()
ax.plot(y_te.index, y_te,           label="Actual")
ax.plot(y_te.index, y_pred_scaled,  label="Predicted")
ax.fill_between(y_te.index, lo_s, hi_s, color="green", alpha=0.3, label=f"±{α_opt_s:.2f}σ")
ax.set_title("Backtest w/ Scaled Range Band")
ax.legend()
st.pyplot(fig)


st.success("✅ Models trained and plots rendered correctly, good work David!")
