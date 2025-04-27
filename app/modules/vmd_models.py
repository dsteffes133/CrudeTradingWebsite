# app/modules/vmd_models.py

import numpy as np
import pandas as pd
from vmdpy import VMD
from sklearn.linear_model import HuberRegressor

from app.modules.ml_utils import extract_aggregated_features
from app.modules.data_utils import load_aligned
import config


def decompose_vmd(series: pd.Series) -> pd.DataFrame:
    """
    Run VMD with the tuned hyper-parameters from config.
    """
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
    """
    Build train/test arrays for a one-step-ahead Huber/LSTM backtest.
    """
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
    """
    Train a HuberRegressor on flattened VMD+feature windows.
    """
    arr = np.asarray(X_train)
    if arr.ndim == 3:
        n, L, D = arr.shape
        Xf = arr.reshape(n, L * D)
    else:
        Xf = arr

    hub = HuberRegressor(**config.HUBER_KWARGS)
    hub.fit(Xf, y_train)
    return hub


def train_lstm(X_train, y_train):
    """
    Train a simple LSTM on the VMD+feature windows.
    (TF imports deferred so module import doesn’t fail if TF is broken.)
    """
    # deferred imports
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping

    lookback = config.LOOKBACK
    num_features = X_train.shape[2]

    model = Sequential([
        Input(shape=(lookback, num_features)),
        LSTM(config.LSTM_KWARGS['units']),
        Dense(1)
    ])
    model.compile('adam', 'mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=config.LSTM_KWARGS['epochs'],
        batch_size=config.LSTM_KWARGS['batch_size'],
        callbacks=[es],
        verbose=0
    )
    return model


def forecast_vmd(series: pd.Series) -> pd.Series:
    """
    Recursively forecast `config.HORIZON` days ahead using VMD + Huber or LSTM.
    """
    lookback   = config.LOOKBACK
    horizon    = config.HORIZON
    model_type = config.MODEL_TYPE

    full = series.ffill().dropna()
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

    # Train on all history
    if model_type == 'Huber':
        model = train_huber(X_all, y_all)
        predict_one = lambda w: model.predict(w.reshape(1, -1))[0]
    else:
        # deferred TF imports happen inside train_lstm
        model = train_lstm(X_all, y_all)
        predict_one = lambda w: model.predict(
            w.reshape(1, lookback, w.shape[1])
        )[0, 0]

    # Recursive forecasting
    future_vals = []
    temp_full = full.copy()
    for _ in range(horizon):
        comps_roll = decompose_vmd(temp_full)
        feats_roll = extract_aggregated_features(temp_full, lookback)
        window = pd.concat([comps_roll, feats_roll], axis=1)\
                   .dropna()\
                   .iloc[-lookback:].values

        yhat = predict_one(window)
        future_vals.append(yhat)

        next_date = temp_full.index[-1] + pd.Timedelta(days=1)
        temp_full.loc[next_date] = yhat

    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    return pd.Series(future_vals, index=idx, name='VMD Forecast')

