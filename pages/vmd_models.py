# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.modules.data_utils import load_aligned
from app.modules.ml_utils import extract_aggregated_features
from app.modules.vmd_models import train_huber, train_lstm

# —————————————————————————————————————————————————————————————
st.sidebar.header("⚙️ Model Configuration")
model_type      = st.sidebar.selectbox("Model Type", ["Huber", "LSTM"])
coverage_target = st.sidebar.slider("Calibration target coverage", 0.80, 0.99, 0.95, 0.01)

# 1) pick your series
TABLES = {
    "WTI Crude (FRED)": (
        "bond_stocks", "WTI Crude Oil"
    ),
    "WCS Houston (Pricing)": (
        "pricing_vector",
        "WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"
    ),
    "Cushing Inventory": (
        "wpr_sliding", "EIA CUSHING- OK CRUDE EXCL SPR STK"
    )
}
table_label  = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_name = TABLES[table_label]

# 2) lookback / horizon / split
lookback   = st.sidebar.number_input("Lookback days", min_value=5, value=30)
horizon    = st.sidebar.number_input("Forecast horizon (days ahead)", min_value=1, value=1)
split_pct  = st.sidebar.slider("Train %", 50, 90, 85)
df_full    = load_aligned(tbl)[series_name].ffill().dropna()
split_ix   = int(len(df_full) * (split_pct / 100))
split_date = df_full.index[split_ix]

# 3) VMD params
alpha = st.sidebar.number_input("VMD α",    min_value=0.0, value=2000.0)
K     = st.sidebar.number_input("VMD modes", min_value=1,   value=5)
tol   = st.sidebar.number_input("VMD tol",  value=1e-7,      format="%.1e")

# 4) Run button
run = st.sidebar.button("▶️ Run models")
if not run:
    st.write("Adjust parameters and click ▶️ Run models to start.")
    st.stop()

# —————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def decompose_vmd(series: pd.Series, alpha: float, K: int, tol: float) -> pd.DataFrame:
    u, _, _ = VMD(series.values, alpha, 0.0, K, 0, 1, tol)
    return pd.DataFrame(
        u.T,
        index=series.index,
        columns=[f"mode_{i+1}" for i in range(u.shape[0])]
    )

def prepare_huber(series, lookback, horizon, split_date, alpha, K, tol):
    full  = series
    comps = decompose_vmd(full, alpha, K, tol)
    feats = []
    for col in comps.columns:
        f = extract_aggregated_features(comps[col], lookback)
        f.columns = [f"{col}__{c}" for c in f.columns]
        feats.append(f)
    X = pd.concat(feats, axis=1)
    y = full.shift(-horizon).reindex(X.index).dropna()
    df = X.join(y.rename("target")).dropna()
    train = df[df.index <= split_date]
    test  = df[df.index  > split_date]
    return train.drop(columns="target"), train["target"], test.drop(columns="target"), test["target"]

def prepare_lstm(series, lookback, horizon, split_date, alpha, K, tol):
    full  = series
    comps = decompose_vmd(full, alpha, K, tol)
    X, y, idxs = [], [], []
    for i in range(lookback-1, len(full)-horizon):
        X.append(comps.iloc[i-lookback+1:i+1].values)
        y.append(full.iloc[i+horizon])
        idxs.append(full.index[i])
    X = np.stack(X)
    y = np.array(y)
    idxs = pd.to_datetime(idxs)
    train_mask = idxs <= split_date
    return X[train_mask], y[train_mask], X[~train_mask], y[~train_mask], idxs[train_mask], idxs[~train_mask]

def calibrate_band(y_true: pd.Series, y_pred: pd.Series, coverage_target: float):
    resid = (y_true - y_pred).dropna()
    sigma = resid.std()
    alphas = np.linspace(0.5, 3.0, 26)
    recs = []
    for α in alphas:
        lo  = y_pred - α * sigma
        hi  = y_pred + α * sigma
        cov = ((y_true >= lo) & (y_true <= hi)).mean()
        w   = (hi - lo).mean()
        recs.append((α, cov, w))
    df = pd.DataFrame(recs, columns=["alpha","coverage","avg_width"])
    cands = df[df.coverage >= coverage_target]
    α_opt = (cands.sort_values("avg_width").iloc[0].alpha
             if not cands.empty
             else df.sort_values("coverage", ascending=False).iloc[0].alpha)
    return df, α_opt

# —————————————————————————————————————————————————————————————
st.title("📈 VMD → Huber / LSTM Modeling")
st.markdown(f"**{model_type}** on `{series_name}` • split at {split_date:%Y-%m-%d}")

series = df_full

# 5) Fit & predict on raw features
if model_type == "Huber":
    X_tr, y_tr, X_te, y_te = prepare_huber(series, lookback, horizon, split_date, alpha, K, tol)
    raw_model    = train_huber(X_tr, y_tr)
    y_pred_raw   = pd.Series(raw_model.predict(X_te), index=y_te.index)
else:
    X_tr, y_tr, X_te, y_te, idx_tr, idx_te = prepare_lstm(series, lookback, horizon, split_date, alpha, K, tol)
    raw_model    = train_lstm(X_tr, y_tr, lookback=lookback, num_features=X_tr.shape[2], units=32, epochs=50, batch_size=16)
    y_pred_raw   = pd.Series(raw_model.predict(X_te).flatten(), index=idx_te)

# 6) Raw backtest metrics & plot
mae_raw  = (y_te - y_pred_raw).abs().mean()
rmse_raw = np.sqrt(((y_te - y_pred_raw)**2).mean())
st.write({"Raw MAE": f"{mae_raw:.3f}", "Raw RMSE": f"{rmse_raw:.3f}"})

fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,        label="Actual")
ax.plot(y_te.index, y_pred_raw,  label="Predicted")
ax.set_title("Backtest (Raw Features)")
ax.legend()
st.pyplot(fig)

# 7) Raw calibration & band
df_calib_raw, α_opt_raw = calibrate_band(y_te, y_pred_raw, coverage_target)
st.subheader("Raw Calibration: Coverage vs α")
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(df_calib_raw.alpha,   df_calib_raw.coverage,   label="Coverage")
ax2.plot(df_calib_raw.alpha,  df_calib_raw.avg_width,  color="gray", label="Avg Width")
ax.axvline(α_opt_raw, color="red", linestyle="--", label=f"opt α={α_opt_raw:.2f}")
ax.set_xlabel("α"); ax.set_ylabel("Coverage")
ax2.set_ylabel("Avg Width")
ax.legend(loc="upper left"); ax2.legend(loc="upper right")
st.pyplot(fig)

fig, ax = plt.subplots()
lo_raw = y_pred_raw - α_opt_raw * (y_te - y_pred_raw).std()
hi_raw = y_pred_raw + α_opt_raw * (y_te - y_pred_raw).std()
ax.plot(y_te.index, y_te,       label="Actual")
ax.plot(y_te.index, y_pred_raw, label="Predicted")
ax.fill_between(y_te.index, lo_raw, hi_raw, color="orange", alpha=0.3, label=f"±{α_opt_raw:.2f}σ")
ax.set_title("Backtest w/ Raw Range Band")
ax.legend()
st.pyplot(fig)

# —————————————————————————————————————————————————————————————
# 8) Always scale & re‑train on scaled features

# flatten & scale X
n_tr, L, D = X_tr.shape
scaler_X    = StandardScaler()
X_tr_flat   = X_tr.reshape(n_tr, L * D)
X_tr_s_flat = scaler_X.fit_transform(X_tr_flat)
X_tr_s      = X_tr_s_flat.reshape(n_tr, L, D)

n_te        = X_te.shape[0]
X_te_flat   = X_te.reshape(n_te, L * D)
X_te_s_flat = scaler_X.transform(X_te_flat)
X_te_s      = X_te_s_flat.reshape(n_te, L, D)

# scale y
scaler_y = StandardScaler()
y_tr_s   = scaler_y.fit_transform(y_tr.values.reshape(-1,1)).flatten()
y_te_s   = scaler_y.transform(y_te.values.reshape(-1,1)).flatten()

# re‑train on scaled
if model_type == "Huber":
    hub_scaled  = train_huber(X_tr_s.reshape(n_tr, L * D), y_tr_s)
    y_pred_s    = hub_scaled.predict(X_te_s.reshape(n_te, L * D))
else:
    lstm_scaled = train_lstm(X_tr_s, y_tr_s, lookback=lookback, num_features=D, units=32, epochs=50, batch_size=16)
    y_pred_s    = lstm_scaled.predict(X_te_s).flatten()

# invert y‑scale
y_pred_scaled = pd.Series(
    scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).flatten(),
    index=y_te.index if model_type=="Huber" else idx_te
)

# 9) Scaled backtest metrics & plot
mae_scaled  = (y_te - y_pred_scaled).abs().mean()
rmse_scaled = np.sqrt(((y_te - y_pred_scaled)**2).mean())
st.write({"Scaled MAE": f"{mae_scaled:.3f}", "Scaled RMSE": f"{rmse_scaled:.3f}"})

fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,          label="Actual")
ax.plot(y_te.index, y_pred_scaled, label="Predicted")
ax.set_title("Backtest (Scaled Features)")
ax.legend()
st.pyplot(fig)

# 10) Scaled calibration & band
df_calib_s, α_opt_s = calibrate_band(y_te, y_pred_scaled, coverage_target)
st.subheader("Scaled Calibration: Coverage vs α")
fig, ax = plt.subplots()
ax3 = ax.twinx()
ax.plot(df_calib_s.alpha,     df_calib_s.coverage,  label="Coverage")
ax3.plot(df_calib_s.alpha,    df_calib_s.avg_width, color="gray", label="Avg Width")
ax.axvline(α_opt_s, color="red", linestyle="--", label=f"opt α={α_opt_s:.2f}")
ax.set_xlabel("α"); ax.set_ylabel("Coverage")
ax3.set_ylabel("Avg Width")
ax.legend(loc="upper left"); ax3.legend(loc="upper right")
st.pyplot(fig)

fig, ax = plt.subplots()
lo_s = y_pred_scaled - α_opt_s * (y_te - y_pred_scaled).std()
hi_s = y_pred_scaled + α_opt_s * (y_te - y_pred_scaled).std()
ax.plot(y_te.index, y_te,            label="Actual")
ax.plot(y_te.index, y_pred_scaled,   label="Predicted")
ax.fill_between(y_te.index, lo_s, hi_s, color="green", alpha=0.3, label=f"±{α_opt_s:.2f}σ")
ax.set_title("Backtest w/ Scaled Range Band")
ax.legend()
st.pyplot(fig)


st.success("✅ Models trained and plots rendered correctly, good work David!")
