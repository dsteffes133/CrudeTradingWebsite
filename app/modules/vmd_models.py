# app/modules/vmd_models.py

import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from app.modules.ml_utils import extract_aggregated_features
from app.modules.data_utils import load_aligned

def decompose_vmd(
    series: pd.Series,
    alpha: float = 2000.0,
    tau: float   = 0.0,
    K: int       = 5,
    DC: int      = 0,
    init: int    = 1,
    tol: float   = 1e-7
) -> pd.DataFrame:
    u, u_hat, omega = VMD(
        series.values, alpha, tau, K, DC, init, tol
    )
    df = pd.DataFrame(
        u.T,
        index=series.index,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )
    return df

def prepare_vmd_ml_data(
    table: str,
    series: str,
    lookback: int = 30,
    horizon: int  = 1,
    split_frac: float = 0.85,
    vmd_kwargs: dict = dict(alpha=2000.0, tau=0.0, K=5, DC=0, init=1, tol=1e-7)
):
    # 1) load & ffill
    full = load_aligned(table)[series].ffill()
    # 2) VMD modes
    comps = decompose_vmd(full, **vmd_kwargs)
    # 3) engineered features on raw series
    feats = extract_aggregated_features(full, lookback)
    # align
    data = pd.concat([comps, feats], axis=1).dropna()
    dates = data.index
    n = len(data) - horizon - lookback + 1

    X, y = [], []
    for i in range(lookback-1, lookback-1 + n):
        window = data.iloc[i-lookback+1: i+1].values  # (lookback, K+F)
        X.append(window)
        y.append(float(full.iloc[i+horizon]))
    X = np.stack(X)    # (N, lookback, D)
    y = np.array(y)    # (N,)

    # train/test split
    split_i = int(len(X) * split_frac)
    X_tr, X_te = X[:split_i], X[split_i:]
    y_tr, y_te = y[:split_i], y[split_i:]

    return X_tr, y_tr, X_te, y_te

def train_huber(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **hub_kwargs
) -> HuberRegressor:
    # flatten
    n, L, D = X_train.shape
    Xf = X_train.reshape(n, L*D)
    hub = HuberRegressor(**hub_kwargs)
    hub.fit(Xf, y_train)
    return hub

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int,
    num_features: int,
    units: int = 32,
    epochs: int = 50,
    batch_size: int = 16
):
    model = Sequential([
        Input(shape=(lookback, num_features)),
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
