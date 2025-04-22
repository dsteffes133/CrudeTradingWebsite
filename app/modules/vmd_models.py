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
    """Run VMD and return K mode components as a DataFrame."""
    u, _, _ = VMD(series.values, alpha, tau, K, DC, init, tol)
    return pd.DataFrame(
        u.T,
        index=series.index,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )

def prepare_vmd_ml_data(
    table: str,
    series: str,
    lookback: int = 30,
    horizon: int  = 1,
    split_frac: float = 0.85,
    vmd_kwargs: dict = dict(alpha=2000.0, tau=0.0, K=5, DC=0, init=1, tol=1e-7)
):
    """
    Build X/y arrays *and* their corresponding index arrays for both Huber
    (flattened) and LSTM (3D) training:
      • X: shape (N, lookback, D) where D = K + feature_count
      • y: shape (N,)
      • idxs: list of pd.Timestamp of length N
    """
    # 1) load & forward‑fill
    full = load_aligned(table)[series].ffill()
    # 2) VMD decomposition
    comps = decompose_vmd(full, **vmd_kwargs)
    # 3) engineered features on raw series
    feats = extract_aggregated_features(full, lookback)
    # 4) align and drop NaNs
    data = pd.concat([comps, feats], axis=1).dropna()
    # we will predict full.shift(-horizon) for each window ending at i
    dates = data.index[lookback-1 : len(data)-horizon]
    n = len(dates)

    X, y, idxs = [], [], []
    for i, date in enumerate(dates, start=lookback-1):
        window = data.iloc[i - lookback + 1 : i + 1].values  # (lookback, D)
        X.append(window)
        y.append(float(full.iloc[i + horizon]))
        idxs.append(full.index[i])

    X = np.stack(X)  # (N, lookback, D)
    y = np.array(y)  # (N,)

    # train/test split by fraction
    split_i = int(n * split_frac)
    return (
        X[:split_i], y[:split_i], np.array(idxs[:split_i]),
        X[split_i:], y[split_i:], np.array(idxs[split_i:])
    )

def train_huber(X_train, y_train, **hub_kwargs) -> HuberRegressor:
    """
    Train a HuberRegressor.
    Accepts either 3D (n, lookback, D) → flattens to (n, lookback*D)
    or 2D (n, D) → uses directly.
    """
    arr = np.asarray(X_train)
    if arr.ndim == 3:
        n, L, D = arr.shape
        Xf = arr.reshape(n, L * D)
    elif arr.ndim == 2:
        Xf = arr
    else:
        raise ValueError(f"train_huber got array with ndim={arr.ndim}")

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
    """
    Build & train a simple LSTM on 3D input (n, lookback, num_features).
    """
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

