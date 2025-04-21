# app/modules/anomaly_detection.py

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies_zscore(
    series: pd.Series,
    window: int = 30,
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Compute rolling mean & std, then z = (x - mean)/std.
    Returns a DataFrame with columns [value, z_score, is_anomaly].
    """
    roll_mean = series.rolling(window, min_periods=1).mean()
    roll_std  = series.rolling(window, min_periods=1).std().fillna(0)
    z_scores  = (series - roll_mean) / roll_std.replace(0, np.nan)
    
    df = pd.DataFrame({
        "value": series,
        "z_score": z_scores,
    })
    df["is_anomaly"] = df["z_score"].abs() > threshold
    return df

def detect_anomalies_iforest(
    df: pd.DataFrame,
    contamination: float = 0.01,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Fit IsolationForest on the raw values (and any other numeric cols).
    Returns the original df with an extra 'is_anomaly' column.
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    # work on copy
    X = df.select_dtypes(include=[np.number]).fillna(0)
    preds = model.fit_predict(X)   # -1 = outlier, 1 = inlier
    df_out = df.copy()
    df_out["is_anomaly"] = preds == -1
    return df_out
