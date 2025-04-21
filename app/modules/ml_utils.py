# app/modules/ml_utils.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, List

from app.modules.data_utils import load_aligned

def extract_aggregated_features(
    series: pd.Series | pd.DataFrame,
    lookback: int = 30
) -> pd.DataFrame:
    """
    Build aggregated features for a single series over a rolling window.
    If a 1‑column DataFrame is passed, it will be converted to a Series.
    Returns a DataFrame indexed from series.index[lookback-1:] with columns:
      roll7_mean, roll7_std, roll14_mean, roll14_std,
      roll30_mean, roll30_std, q10, q50, q90,
      mom1, mom7, mom30, accel,
      slope, curvature,
      vmin, vmax, vrange, drawdown,
      fft1, fft2, fft3
    """
    # --- ensure we have a Series ---
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError(
                f"extract_aggregated_features expected a single series, "
                f"got DataFrame with columns {series.columns.tolist()}"
            )

    # 1) prepare a float array, forward‐filled
    vals = series.fillna(method="ffill").to_numpy(dtype=float)
    dates = series.index
    feats = []

    # 2) slide the window
    for i in range(lookback - 1, len(vals)):
        window = vals[i - lookback + 1 : i + 1]  # 1D float array

        # rolling stats
        row = {
            "roll7_mean":  float(np.mean(window[-7:])),
            "roll7_std":   float(np.std(window[-7:], ddof=0)),
            "roll14_mean": float(np.mean(window[-14:])),
            "roll14_std":  float(np.std(window[-14:], ddof=0)),
            "roll30_mean": float(np.mean(window)),
            "roll30_std":  float(np.std(window, ddof=0)),
            "q10":         float(np.quantile(window, 0.1)),
            "q50":         float(np.quantile(window, 0.5)),
            "q90":         float(np.quantile(window, 0.9)),
        }

        # momentum features
        prev1  = float(window[-2])
        den1   = prev1 if prev1 != 0.0 else 1.0
        row["mom1"]  = float((window[-1] - prev1) / den1)

        prev7  = float(window[-7])
        den7   = prev7 if prev7 != 0.0 else 1.0
        row["mom7"]  = float((window[-1] - prev7) / den7)

        prev30 = float(window[0])
        den30  = prev30 if prev30 != 0.0 else 1.0
        row["mom30"] = float((window[-1] - prev30) / den30)

        # acceleration
        if lookback >= 3:
            prev2 = float(window[-3])
            den2  = prev2 if prev2 != 0.0 else 1.0
            m1 = (window[-1] - window[-2]) / den1
            m2 = (window[-2] - prev2)      / den2
            row["accel"] = float(m1 - m2)
        else:
            row["accel"] = 0.0

        # trend slope & curvature
        x = np.arange(lookback, dtype=float)
        row["slope"]     = float(np.polyfit(x, window, 1)[0])
        row["curvature"] = float(np.diff(window, 2).mean()) if lookback >= 3 else 0.0

        # extremes & drawdown
        vmin, vmax = float(window.min()), float(window.max())
        row["vmin"]    = vmin
        row["vmax"]    = vmax
        row["vrange"]  = vmax - vmin
        denom = vmax if vmax != 0.0 else 1.0
        row["drawdown"] = float((window[-1] - vmax) / denom)

        # spectral features
        mags = np.abs(np.fft.rfft(window))
        row["fft1"] = float(mags[1]) if len(mags) > 1 else 0.0
        row["fft2"] = float(mags[2]) if len(mags) > 2 else 0.0
        row["fft3"] = float(mags[3]) if len(mags) > 3 else 0.0

        feats.append(row)

    return pd.DataFrame(feats, index=dates[lookback - 1 :])


def prepare_ml_data(
    responder_table: str,
    responder_col: str,
    predictor_tables: List[str],
    lookback: int,
    horizon: int,
    split_date: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build X/y for ML:
      • y: responder_col from responder_table at t+horizon
      • X: all numeric series from predictor_tables, lagged 1d,
            with aggregated features over lookback
    """
    # load & merge predictors
    dfs = [load_aligned(tbl) for tbl in predictor_tables]
    preds = pd.concat(dfs, axis=1)

    # ⚠️ REMOVE duplicate columns so preds[col] is always a Series:
    preds = preds.loc[:, ~preds.columns.duplicated()]

    # 2) Keep only numeric, drop responder if present, lag by 1 day
    preds = preds.select_dtypes(include=[np.number]).astype(float)
    if responder_col in preds.columns:
        preds = preds.drop(columns=[responder_col])
    preds = preds.shift(1).fillna(method="bfill")


    # build target
    full   = load_aligned(responder_table)[responder_col]
    y_full = full.shift(-horizon)

    # feature‑engineer each predictor
    feature_dfs = []
    for col in preds.columns:
        feat = extract_aggregated_features(preds[col], lookback)
        feat.columns = [f"{col}__{c}" for c in feat.columns]
        feature_dfs.append(feat)

    X = pd.concat(feature_dfs, axis=1)

    # align and drop NaNs
    data = X.join(y_full.rename("target")).dropna()

    # train/test split
    cutoff = pd.to_datetime(split_date)
    train  = data[data.index <= cutoff]
    test   = data[data.index > cutoff]

    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test,  y_test  = test.drop(columns="target"),  test["target"]

    return X_train, y_train, X_test, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **rf_kwargs
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on X_train/y_train."""
    model = RandomForestRegressor(**rf_kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """Compute MAE & RMSE on (X, y)."""
    preds = model.predict(X)
    mae   = mean_absolute_error(y, preds)
    mse   = mean_squared_error(y, preds)
    rmse  = np.sqrt(mse)
    return {"mae": mae, "rmse": rmse}
