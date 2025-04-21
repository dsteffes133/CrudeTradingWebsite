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
    Decompose `series` via VMD, extract aggregated features for each mode,
    and return X_train, y_train, X_test, y_test for a HuberRegressor.
    """
    # 1️⃣ load the full series & decompose
    full  = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)

    # 2️⃣ for each mode, build the same rolling/window features
    feat_dfs = []
    for mode in comps.columns:
        f = extract_aggregated_features(comps[mode], lookback)
        f.columns = [f"{mode}__{c}" for c in f.columns]
        feat_dfs.append(f)

    # 3️⃣ concatenate all mode‐features horizontally
    X_full = pd.concat(feat_dfs, axis=1)

    # 4️⃣ build target shifted by horizon and align
    y_full = full.shift(-horizon).reindex(X_full.index)
    df     = pd.concat([X_full, y_full.rename("target")], axis=1).dropna()

    # 5️⃣ split by date
    cutoff = pd.to_datetime(split_date)
    train  = df[df.index <= cutoff]
    test   = df[df.index >  cutoff]

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
    Decompose `series` via VMD, extract aggregated features for each mode,
    stack them as a sequence for LSTM, and return X_train/y_train/X_test/y_test.
    """
    full  = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)
    K     = comps.shape[1]

    # build per‐mode feature DataFrames
    feat_dfs = []
    for mode in comps.columns:
        f = extract_aggregated_features(comps[mode], lookback)
        f.columns = [f"{mode}__{c}" for c in f.columns]
        feat_dfs.append(f)

    # merge into one DataFrame, index aligned
    X_df = pd.concat(feat_dfs, axis=1).dropna()
    y_df = full.shift(-horizon).reindex(X_df.index).dropna()
    common_idx = X_df.index.intersection(y_df.index)
    X_df, y_df = X_df.loc[common_idx], y_df.loc[common_idx]

    # now reshape each row into a (lookback, K) window
    modes = [col.split("__")[0] for col in X_df.columns]
    unique_modes = sorted(set(modes), key=lambda m: int(m.split("_")[-1]))
    # collect feature‐blocks per time step
    X_windows = []
    y_vals    = []
    idxs      = []
    for i in range(lookback - 1, len(X_df)):
        win = X_df.iloc[i - lookback + 1 : i + 1].values.reshape(lookback, -1)
        X_windows.append(win)
        y_vals.append(y_df.iloc[i])
        idxs.append(common_idx[i])

    X = np.stack(X_windows)  # shape (N, lookback, features)
    y = np.array(y_vals)
    idx = pd.to_datetime(idxs)

    # split
    cutoff = pd.to_datetime(split_date)
    mask   = idx <= cutoff
    return X[mask], y[mask], X[~mask], y[~mask]

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int,
    n_features: int,
    units: int = 32,
    epochs: int = 50,
    batch_size: int = 16
):
    """
    Build & train a simple LSTM:
      Input(shape=(lookback, n_features)) → LSTM(units) → Dense(1)
    """
    model = Sequential([
        Input(shape=(lookback, n_features)),
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
