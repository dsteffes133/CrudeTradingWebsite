# app/modules/vmd_models.py

import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor

from app.modules.ml_utils import extract_aggregated_features
from app.modules.data_utils import load_aligned
import config

def decompose_vmd(series: pd.Series) -> pd.DataFrame:
    u, _, _ = VMD(series.values, **config.VMD_KWARGS)
    n_pts = u.shape[1]
    idx   = series.index[:n_pts]
    modes = pd.DataFrame(
        u.T,
        index=idx,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )
    return modes

def prepare_vmd_ml_data(table: str, series_col: str, split_frac: float = 0.85):
    full = load_aligned(table)[series_col].ffill().dropna()
    comps = decompose_vmd(full)
    feats = extract_aggregated_features(full, config.LOOKBACK)
    data = pd.concat([comps, feats], axis=1).dropna()

    X, y, idxs = [], [], []
    for i in range(config.LOOKBACK - 1, len(data) - 1):
        window = data.iloc[i - config.LOOKBACK + 1 : i + 1].values
        X.append(window)
        y.append(float(full.iloc[i + 1]))
        idxs.append(full.index[i])

    X = np.stack(X)
    y = np.array(y)
    split_i = int(len(X) * split_frac)
    return (
        X[:split_i], y[:split_i], np.array(idxs[:split_i]),
        X[split_i:], y[split_i:], np.array(idxs[split_i:])
    )

def train_huber(X_train, y_train) -> HuberRegressor:
    arr = np.asarray(X_train)
    if arr.ndim == 3:
        n, L, D = arr.shape
        Xf = arr.reshape(n, L * D)
    else:
        Xf = arr

    hub = HuberRegressor(**config.HUBER_KWARGS)
    hub.fit(Xf, y_train)
    return hub

def forecast_vmd(series: pd.Series) -> pd.Series:
    lookback = config.LOOKBACK
    horizon  = config.HORIZON

    full  = series.ffill().dropna()
    comps = decompose_vmd(full)
    feats = extract_aggregated_features(full, lookback)
    data  = pd.concat([comps, feats], axis=1).dropna()

    # Build full training arrays
    X_all, y_all = [], []
    for i in range(lookback - 1, len(data) - 1):
        window = data.iloc[i - lookback + 1 : i + 1].values
        X_all.append(window)
        y_all.append(float(full.iloc[i + 1]))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)

    # Train on all history with Huber
    model = train_huber(X_all, y_all)
    predict_one = lambda w: model.predict(w.reshape(1, -1))[0]

    # Recursive forecasting
    future_vals = []
    temp_full = full.copy()
    for _ in range(horizon):
        comps_roll = decompose_vmd(temp_full)
        feats_roll = extract_aggregated_features(temp_full, lookback)
        window = pd.concat([comps_roll, feats_roll], axis=1) \
                   .dropna() \
                   .iloc[-lookback:].values

        yhat = predict_one(window)
        future_vals.append(yhat)
        next_date = temp_full.index[-1] + pd.Timedelta(days=1)
        temp_full.loc[next_date] = yhat

    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(future_vals, index=idx, name='VMD Forecast')

