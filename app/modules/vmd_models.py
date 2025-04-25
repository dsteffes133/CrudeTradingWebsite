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

def decompose_vmd(series: pd.Series, alpha: float, tau: float,
                  K: int, DC: bool, init: int, tol: float) -> pd.DataFrame:
    u, _, _ = VMD(series.values, alpha, tau, K, DC, init, tol)
    n_pts = u.shape[1]
    idx   = series.index[:n_pts]
    modes = pd.DataFrame(
        u.T,
        index=idx,
        columns=[f"vmd_mode_{i+1}" for i in range(u.shape[0])]
    )
    return modes

def prepare_vmd_ml_data(table: str, series: str,
    lookback: int = 30, horizon: int  = 1, split_frac: float = 0.85,
    vmd_kwargs: dict = dict(alpha=2000.0, tau=0.0, K=5, DC=0, init=1, tol=1e-7)
):
    full = load_aligned(table)[series].ffill()
    comps = decompose_vmd(full, **vmd_kwargs)
    feats = extract_aggregated_features(full, lookback)
    data = pd.concat([comps, feats], axis=1).dropna()

    dates = data.index[lookback-1 : len(data)-horizon]
    n = len(dates)
    X, y, idxs = [], [], []
    for i, date in enumerate(dates, start=lookback-1):
        window = data.iloc[i-lookback+1 : i+1].values
        X.append(window)
        y.append(float(full.iloc[i + horizon]))
        idxs.append(full.index[i])

    X = np.stack(X)
    y = np.array(y)
    split_i = int(n * split_frac)
    return (
        X[:split_i], y[:split_i], np.array(idxs[:split_i]),
        X[split_i:], y[split_i:], np.array(idxs[split_i:])
    )

def train_huber(X_train, y_train, **hub_kwargs) -> HuberRegressor:
    arr = np.asarray(X_train)
    if arr.ndim == 3:
        n, L, D = arr.shape
        Xf = arr.reshape(n, L * D)
    else:
        Xf = arr
    hub = HuberRegressor(**hub_kwargs)
    hub.fit(Xf, y_train)
    return hub

def train_lstm(X_train, y_train, lookback, num_features,
               units: int = 32, epochs: int = 50, batch_size: int = 16):
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

def forecast_vmd(series: pd.Series,
                 lookback: int,
                 horizon: int,
                 model_type: str,
                 vmd_kwargs: dict,
                 lstm_kwargs: dict
) -> pd.Series:
    """
    Recursively forecast `horizon` days ahead using VMD + either Huber or LSTM.
    Returns a pd.Series indexed by date with the forecasts.
    """
    full = series.ffill().dropna()
    comps = decompose_vmd(full, **vmd_kwargs)
    feats = extract_aggregated_features(full, lookback)
    data = pd.concat([comps, feats], axis=1).dropna()

    # build full X/y for training on all history
    X_all, y_all, _, _, _, _ = prepare_vmd_ml_data(
        table=None, series=None,  # we bypass table/series here
        lookback=lookback, horizon=1, split_frac=1.0,
        vmd_kwargs=vmd_kwargs
    )
    # note: you may prefer to refactor prepare_vmd_ml_data to accept a Series directly.

    # train on entire history
    if model_type == "Huber":
        model = train_huber(X_all, y_all)
        pred_fn = lambda X: model.predict(X.reshape(1, -1))[0]
    else:
        model = train_lstm(
            X_all, y_all,
            lookback, X_all.shape[2], **lstm_kwargs
        )
        pred_fn = lambda X: model.predict(X.reshape(1, X.shape[1], X.shape[2]))[0,0]

    future = []
    temp_full = full.copy()

    for _ in range(horizon):
        # recompute components & features on rolling window
        comps_roll = decompose_vmd(temp_full, **vmd_kwargs)
        feats_roll = extract_aggregated_features(temp_full, lookback)
        window = pd.concat([comps_roll, feats_roll], axis=1).dropna().iloc[-lookback:].values
        yhat = pred_fn(window)
        future.append(yhat)
        next_date = temp_full.index[-1] + pd.Timedelta(days=1)
        temp_full.loc[next_date] = yhat

    idx = pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=horizon)
    return pd.Series(future, index=idx, name="Forecast")
