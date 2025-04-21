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
    """
    Run VMD on `series` and return a DataFrame of K intrinsic mode components,
    indexed the same as `series`.
    """
    u, u_hat, omega = VMD(
        series.values, alpha, tau, K, DC, init, tol
    )
    # u is shape (K, T)
    comp_df = pd.DataFrame(
        u.T,
        index=series.index,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )
    return comp_df

def prepare_huber_data(
    table: str,
    series: str,
    lookback: int,
    horizon: int,
    split_date: str,
    vmd_kwargs: dict
):
    """
    Decompose `series` via VMD, extract features for each mode,
    stack them, and return X_train, y_train, X_test, y_test.
    """
    full = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)

    # extract features for each mode
    feat_list = []
    for col in comps.columns:
        # uses same lookback features as RF
        f = extract_aggregated_features(
            table=None,  # we bypass table logic below
            series=col,
            lookback=lookback
        ) if False else None  # placeholder
        # instead, we need a helper that builds features from comps[col] directly:
        # for brevity, we just reuse extract_aggregated_features by temporarily
        # injecting comps[col] into a DataFrame, but you should instead call
        # a variant that takes a pd.Series.
        raise NotImplementedError("Extract features from comps[col] here")

    # once you have feat_list of DataFrames, do:
    X = pd.concat(feat_list, axis=1)
    y = full.shift(-horizon).loc[X.index].dropna()
    df = X.join(y.rename("target")).dropna()

    cutoff = pd.to_datetime(split_date)
    train = df[df.index <= cutoff]
    test  = df[df.index > cutoff]

    return (
        train.drop(columns="target"),
        train["target"],
        test.drop(columns="target"),
        test["target"]
    )

def train_huber(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **hub_kwargs
) -> HuberRegressor:
    """Train a HuberRegressor on (X_train, y_train)."""
    hub = HuberRegressor(**hub_kwargs)
    hub.fit(X_train, y_train)
    return hub

def prepare_lstm_data(
    table: str,
    series: str,
    lookback: int,
    horizon: int,
    split_date: str,
    vmd_kwargs: dict
):
    """
    Decompose `series` via VMD, then construct X (shape [N, lookback, K])
    and y (shape [N,]) for LSTM. Returns X_train, y_train, X_test, y_test.
    """
    full = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)
    K = comps.shape[1]

    X, y = [], []
    idxs = []
    for i in range(lookback-1, len(full)-horizon):
        window = comps.iloc[i-lookback+1:i+1].values  # (lookback, K)
        X.append(window)
        y.append(full.iloc[i+horizon])
        idxs.append(full.index[i])

    X = np.stack(X)  # (N, lookback, K)
    y = np.array(y)
    index = pd.to_datetime(idxs)

    # split
    cutoff = pd.to_datetime(split_date)
    mask = index <= cutoff
    X_train, X_test = X[mask], X[~mask]
    y_train, y_test = y[mask], y[~mask]

    return (X_train, y_train, X_test, y_test)

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int,
    K: int,
    units: int = 32,
    epochs: int = 50,
    batch_size: int = 16
):
    """
    Build & train a simple LSTM:
      Input(shape=(lookback, K)) → LSTM(units) → Dense(1)
    Returns the trained Keras model.
    """
    model = Sequential([
        Input(shape=(lookback, K)),
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
