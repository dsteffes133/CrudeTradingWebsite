# app/modules/vmd_models.py

import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.modules.data_utils import load_aligned

# ──────────────────────────────────────────────────────────────────────────────
# FAISS‑style feature vector builder
# ──────────────────────────────────────────────────────────────────────────────
def _make_vector(
    series: pd.Series,
    lookback: int = 30,
    quantiles: list[float] = (0.1, 0.5, 0.9),
) -> np.ndarray:
    vals = series.values[-lookback:]
    if len(vals) < lookback:
        pad = np.full(lookback - len(vals), vals[0])
        vals = np.concatenate([pad, vals])

    # 1) normalize raw window
    mu, sd = vals.mean(), vals.std()
    den = sd if sd != 0 else 1.0
    raw = (vals - mu) / den
    feats = list(raw)

    # 2) rolling mean/std for 7,14,30
    for w in (7, 14, 30):
        win = vals[-w:]
        feats += [win.mean(), win.std()]

    # 3) quantiles
    feats += [float(np.quantile(vals, q)) for q in quantiles]

    # 4) momentum for 1,7,30
    for w in (1, 7, 30):
        prev = vals[-w] if w <= len(vals) else vals[0]
        den = prev if prev != 0 else 1.0
        feats.append((vals[-1] - prev) / den)

    # 5) acceleration
    if lookback >= 3:
        m1 = (vals[-1] - vals[-2]) / (vals[-2] if vals[-2] != 0 else 1.0)
        m2 = (vals[-2] - vals[-3]) / (vals[-3] if vals[-3] != 0 else 1.0)
        feats.append(m1 - m2)
    else:
        feats.append(0.0)

    # 6) slope & curvature
    x = np.arange(lookback)
    feats.append(np.polyfit(x, vals, 1)[0])
    feats.append(np.diff(vals, 2).mean() if lookback >= 3 else 0.0)

    # 7) extremes & drawdown
    vmin, vmax = float(vals.min()), float(vals.max())
    feats += [vmin, vmax, vmax - vmin]
    peak = np.maximum.accumulate(vals)[-1]
    feats.append((vals[-1] - peak) / (peak if peak != 0 else 1.0))

    # 8) FFT bins 1–3
    mags = np.abs(np.fft.rfft(vals))
    feats += [float(mags[b]) if b < len(mags) else 0.0 for b in (1, 2, 3)]

    vec = np.array(feats, dtype="float32")
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# ──────────────────────────────────────────────────────────────────────────────
# 1) VMD decomposition
# ──────────────────────────────────────────────────────────────────────────────
def decompose_vmd(
    series: pd.Series,
    alpha: float = 2000.0,
    tau: float   = 0.0,
    K: int       = 5,
    DC: int      = 0,
    init: int    = 1,
    tol: float   = 1e-7
) -> pd.DataFrame:
    u, _, _ = VMD(series.values, alpha, tau, K, DC, init, tol)
    return pd.DataFrame(u.T,
        index=series.index,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2) Prepare Huber data: flatten each mode-vector
# ──────────────────────────────────────────────────────────────────────────────
def prepare_huber_data(
    table: str,
    series: str,
    lookback: int,
    horizon: int,
    split_date: str,
    vmd_kwargs: dict
):
    full  = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)
    dates = comps.index

    X_vecs = []
    for i, dt in enumerate(dates[lookback-1: len(full)-horizon]):
        row_vecs = [
            _make_vector(comps[col].iloc[i:i+lookback], lookback)
            for col in comps.columns
        ]
        X_vecs.append(np.concatenate(row_vecs))

    X = pd.DataFrame(
        X_vecs,
        index=dates[lookback-1: len(full)-horizon]
    )
    y = full.shift(-horizon).loc[X.index].rename("target")
    df = X.join(y).dropna()

    cutoff = pd.to_datetime(split_date)
    train = df[df.index <= cutoff]
    test  = df[df.index >  cutoff]

    return train.drop(columns="target"), train["target"], \
           test .drop(columns="target"), test["target"]

# ──────────────────────────────────────────────────────────────────────────────
# 3) Train Huber
# ──────────────────────────────────────────────────────────────────────────────
def train_huber(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **hub_kwargs
):
    hub = HuberRegressor(**hub_kwargs)
    hub.fit(X_train, y_train)
    return hub

# ──────────────────────────────────────────────────────────────────────────────
# 4) Prepare LSTM data: sequence of mode‑vectors
# ──────────────────────────────────────────────────────────────────────────────
def prepare_lstm_data(
    table: str,
    series: str,
    lookback: int,
    horizon: int,
    split_date: str,
    vmd_kwargs: dict
):
    full  = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)
    dates = comps.index
    K     = comps.shape[1]

    X, y, idxs = [], [], []
    for i in range(lookback-1, len(full)-horizon):
        # build sequence of length K, each a vector of length M
        mode_vecs = [
            _make_vector(comps[col].iloc[i-lookback+1:i+1], lookback)
            for col in comps.columns
        ]
        X.append(np.stack(mode_vecs))       # shape (K, M)
        y.append(full.iloc[i+horizon])
        idxs.append(dates[i])

    X     = np.stack(X)                    # (N, K, M)
    y     = np.array(y)
    index = pd.to_datetime(idxs)

    cutoff = pd.to_datetime(split_date)
    mask   = index <= cutoff

    return X[mask], y[mask], X[~mask], y[~mask]

# ──────────────────────────────────────────────────────────────────────────────
# 5) Train LSTM
# ──────────────────────────────────────────────────────────────────────────────
def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    timesteps: int,
    features: int,
    units: int = 32,
    epochs: int = 50,
    batch_size: int = 16
):
    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(units),
        Dense(1)
    ])
    model.compile("adam", "mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    return model
