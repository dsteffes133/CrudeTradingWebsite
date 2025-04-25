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

def forecast_vmd(
    series: pd.Series,
    lookback: int,
    horizon: int,
    model_type: str,
    vmd_kwargs: dict,
    lstm_kwargs: dict
) -> pd.Series:
    """
    Recursively forecast `horizon` days ahead using VMD + either Huber or LSTM,
    WITHOUT touching the DB. Returns a pd.Series of forecasts.
    """
    # 1) Prepare the history entirely in-memory
    full = series.ffill().dropna()

    # 2) Decompose & feature-engineer full history
    comps = decompose_vmd(full, **vmd_kwargs)
    feats = extract_aggregated_features(full, lookback)
    data = pd.concat([comps, feats], axis=1).dropna()

    # 3) Build X_all, y_all for single-step training
    X_all, y_all = [], []
    # we predict next-day value, so y at i is full.iloc[i+1]
    for i in range(lookback-1, len(data)-1):
        window = data.iloc[i-lookback+1 : i+1].values  # (lookback, D)
        X_all.append(window)
        y_all.append(float(full.iloc[i+1]))
    X_all = np.stack(X_all)
    y_all = np.array(y_all)

    # 4) Train on all history
    if model_type == "Huber":
        model = train_huber(X_all, y_all)
        predict_one = lambda w: model.predict(w.reshape(1, -1))[0]
    else:
        # X_all.shape == (N, lookback, D)
        model = train_lstm(
            X_all, y_all,
            lookback=lookback,
            num_features=X_all.shape[2],
            **lstm_kwargs
        )
        predict_one = lambda w: model.predict(w.reshape(1, lookback, w.shape[1]))[0, 0]

    # 5) Recursive forecasting
    future_vals = []
    temp_full = full.copy()

    for _ in range(horizon):
        comps_roll = decompose_vmd(temp_full, **vmd_kwargs)
        feats_roll = extract_aggregated_features(temp_full, lookback)
        window = pd.concat([comps_roll, feats_roll], axis=1).dropna().iloc[-lookback:].values

        yhat = predict_one(window)
        future_vals.append(yhat)

        # append to series for next iteration
        next_date = temp_full.index[-1] + pd.Timedelta(days=1)
        temp_full.loc[next_date] = yhat

    # return as a new Series
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(future_vals, index=idx, name="VMD Forecast")

