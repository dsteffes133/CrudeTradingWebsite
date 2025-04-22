# pages/vmd_models.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.modules.data_utils import load_aligned
from app.modules.ml_utils import extract_aggregated_features

# —————————————————————————————————————————————————————————————
st.sidebar.header("⚙️ Model Configuration")
model_type = st.sidebar.selectbox("Model Type", ["Huber", "LSTM"])
coverage_target = st.sidebar.slider("Calibration target coverage", 0.80, 0.99, 0.95, 0.01)

# 1) pick your series
TABLES = {
    "WTI Crude (FRED)":        ("bond_stocks", "WTI Crude Oil"),
    "WCS Houston (Pricing)":   ("pricing_vector",
        "WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"),
    "Cushing Inventory":       ("wpr_sliding", "EIA CUSHING- OK CRUDE EXCL SPR STK")
}
table_name = st.sidebar.selectbox("Series to model", list(TABLES.keys()))
tbl, series_name = TABLES[table_name]

# 2) lookback / horizon / split
lookback   = st.sidebar.number_input("Lookback days", min_value=5, value=30)
horizon    = st.sidebar.number_input("Forecast horizon (days ahead)", min_value=1, value=1)
split_pct  = st.sidebar.slider("Train %", 50, 90, 85)
# we'll compute split_date from pct:
df_full = load_aligned(tbl)[series_name].ffill().dropna()
split_ix = int(len(df_full) * (split_pct/100))
split_date = df_full.index[split_ix]

# 3) VMD params
alpha = st.sidebar.number_input("VMD α", min_value=0.0, value=2000.0)
K     = st.sidebar.number_input("VMD modes K", min_value=1, value=5)
tol   = st.sidebar.number_input("VMD tol", value=1e-7, format="%.1e")

# —————————————————————————————————————————————————————————————
@st.cache_data(show_spinner=False)
def decompose_vmd(series, alpha, K, tol):
    u, _, _ = VMD(series.values, alpha, 0.0, K, 0, 1, tol)
    return pd.DataFrame(u.T,
                        index=series.index,
                        columns=[f"mode_{i+1}" for i in range(u.shape[0])])

def prepare_huber(series, lookback, horizon, split_date, alpha, K, tol):
    full = series
    comps = decompose_vmd(full, alpha, K, tol)
    # for each mode build rolling features
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
    return (
    train.drop(columns="target"),
    train["target"],
    test.drop(columns="target"),
    test["target"],
)


def prepare_lstm(series, lookback, horizon, split_date, alpha, K, tol):
    full = series
    comps = decompose_vmd(full, alpha, K, tol)
    # prepare sliding windows of raw modes
    X, y, idxs = [], [], []
    for i in range(lookback-1, len(full)-horizon):
        X.append(comps.iloc[i-lookback+1:i+1].values)
        y.append(full.iloc[i+horizon])
        idxs.append(full.index[i])
    X, y = np.stack(X), np.array(y)
    idxs = pd.to_datetime(idxs)
    mask = idxs <= split_date
    return X[mask], y[mask], X[~mask], y[~mask], idxs[mask], idxs[~mask]

def calibrate_band(y_true, y_pred, coverage_target):
    resid = (y_true - y_pred).dropna()
    sigma = resid.std()
    alphas = np.linspace(0.5, 3.0, 26)
    records = []
    for α in alphas:
        lo = y_pred - α*sigma
        hi = y_pred + α*sigma
        cov = ((y_true>=lo)&(y_true<=hi)).mean()
        w   = (hi-lo).mean()
        records.append((α, cov, w))
    df = pd.DataFrame(records, columns=["alpha","coverage","avg_width"])
    cands = df[df.coverage>=coverage_target]
    if len(cands)==0:
        α_opt = df.sort_values("coverage", ascending=False).iloc[0].alpha
    else:
        α_opt = cands.sort_values("avg_width").iloc[0].alpha
    return df, α_opt

# —————————————————————————————————————————————————————————————
st.title("📈 VMD → Huber / LSTM Modeling")
st.markdown(f"**{model_type}** on `{series_name}`  •  split at {split_date:%Y-%m-%d}")

series = df_full

# 4) Prepare & train
if model_type=="Huber":
    X_tr, y_tr, X_te, y_te = prepare_huber(series, lookback, horizon, split_date,
                                           alpha, K, tol)
    model = HuberRegressor().fit(X_tr, y_tr)
    y_pred_te = pd.Series(model.predict(X_te), index=y_te.index)
else:
    X_tr, y_tr, X_te, y_te, idx_tr, idx_te = prepare_lstm(
        series, lookback, horizon, split_date, alpha, K, tol
    )
    # build simple LSTM
    m = Sequential([Input(shape=(lookback, K)), LSTM(32), Dense(1)])
    m.compile("adam","mse")
    m.fit(X_tr, y_tr, validation_split=0.1,
          epochs=50, batch_size=16,
          callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
          verbose=0)
    y_pred_te = pd.Series(m.predict(X_te).flatten(), index=idx_te)

# 5) report backtest error
mae = (y_te - y_pred_te).abs().mean()
rmse = np.sqrt(((y_te-y_pred_te)**2).mean())
st.write({"MAE":f"{mae:.3f}", "RMSE":f"{rmse:.3f}"})

# 6) plot raw backtest
fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,    label="Actual")
ax.plot(y_te.index, y_pred_te,label="Predicted")
ax.set_title("Backtest Actual vs Pred")
ax.legend()
st.pyplot(fig)

# 7) range-band calibration
df_calib, α_opt = calibrate_band(y_te, y_pred_te, coverage_target)
st.subheader("Calibration: coverage vs α")
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(df_calib.alpha, df_calib.coverage, label="Coverage")
ax2.plot(df_calib.alpha, df_calib.avg_width, color="gray", label="Avg Width")
ax.axvline(α_opt, color="red", linestyle="--", label=f"opt α={α_opt:.2f}")
ax.set_xlabel("α")
ax.set_ylabel("Coverage")
ax2.set_ylabel("Avg width")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
st.pyplot(fig)

# 8) show backtest with band
sigma = (y_te - y_pred_te).std()
lo = y_pred_te - α_opt*sigma
hi = y_pred_te + α_opt*sigma
fig, ax = plt.subplots()
ax.plot(y_te.index, y_te,    label="Actual")
ax.plot(y_te.index, y_pred_te,label="Predicted")
ax.fill_between(y_te.index, lo, hi, color="orange", alpha=0.3,
                label=f"±{α_opt:.2f}σ")
ax.set_title("Backtest w/ Optimized Range Band")
ax.legend()
st.pyplot(fig)



st.success("✅ Models trained and plots rendered correctly, good work David!")
