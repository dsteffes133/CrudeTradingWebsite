# app/modules/vector_db.py

from typing import List, Tuple
import numpy as np
import pandas as pd
import faiss

from app.modules.data_utils import load_aligned

def _make_vector(
    series: pd.Series,
    lookback: int = 30,
    quantiles: List[float] = [0.1, 0.5, 0.9],
) -> np.ndarray:
    """Build a normalized feature vector over the last `lookback` days."""
    vals = series.values[-lookback:]
    if len(vals) < lookback:
        pad = np.full(lookback - len(vals), vals[0])
        vals = np.concatenate([pad, vals])

    # raw normalization
    mean_val, std_val = vals.mean(), vals.std()
    den = std_val if std_val != 0 else 1.0
    raw = (vals - mean_val) / den
    feats = raw.tolist()

    # rolling mean/std
    for w in (7, 14, 30):
        win = vals[-w:]
        feats += [win.mean(), win.std()]

    # quantiles
    for q in quantiles:
        feats.append(np.quantile(vals, q))

    # momentum
    for w in (1, 7, 30):
        prev = vals[-w] if w <= len(vals) else vals[0]
        den = prev if prev != 0 else 1.0
        feats.append((vals[-1] - prev) / den)

    # acceleration
    if lookback >= 3:
        m1 = (vals[-1] - vals[-2]) / (vals[-2] if vals[-2] != 0 else 1.0)
        m2 = (vals[-2] - vals[-3]) / (vals[-3] if vals[-3] != 0 else 1.0)
        feats.append(m1 - m2)
    else:
        feats.append(0.0)

    # slope & curvature
    x = np.arange(lookback)
    feats.append(np.polyfit(x, vals, 1)[0])
    feats.append(np.diff(vals, 2).mean() if lookback >= 3 else 0.0)

    # extremes & drawdown
    vmin, vmax = vals.min(), vals.max()
    feats += [vmin, vmax, vmax - vmin]
    peak = np.maximum.accumulate(vals)[-1]
    den = peak if peak != 0 else 1.0
    feats.append((vals[-1] - peak) / den)

    # FFT bins 1–3
    mags = np.abs(np.fft.rfft(vals))
    for b in (1, 2, 3):
        feats.append(mags[b] if b < len(mags) else 0.0)

    vec = np.array(feats, dtype="float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

# ──────────────────────────────────────────────────────────────────────────────
# SERIES‑MODE INDEX
# ──────────────────────────────────────────────────────────────────────────────

def build_series_index(
    table_name: str,
    series_name: str,
    lookback: int = 30
) -> Tuple[faiss.IndexFlatL2, List[pd.Timestamp]]:
    """
    Index every lookback‑window for a single series.
    Returns (index, list_of_dates).
    """
    df = load_aligned(table_name)[series_name].fillna(method="ffill")
    dates = list(df.index[lookback-1:])
    vectors = [
        _make_vector(df.iloc[i-lookback+1:i+1], lookback)
        for i in range(lookback-1, len(df))
    ]
    X = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, dates

def query_series_index(
    index: faiss.IndexFlatL2,
    dates: List[pd.Timestamp],
    table_name: str,
    series_name: str,
    query_date: pd.Timestamp,
    lookback: int = 30,
    k: int = 5
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Query the series‑mode index to find the k most similar dates.
    """
    df = load_aligned(table_name)[series_name].fillna(method="ffill")
    if query_date not in df.index:
        raise KeyError(f"{query_date} not in index for {series_name}")
    i = df.index.get_loc(query_date)
    if i < lookback-1:
        raise ValueError("Not enough history for lookback on this date")
    vec = _make_vector(df.iloc[i-lookback+1:i+1], lookback).reshape(1, -1)
    D, I = index.search(vec, k+1)
    out = []
    for dist, idx in zip(D[0], I[0]):
        if dates[idx] != query_date:
            out.append((dates[idx], float(dist)))
        if len(out) >= k:
            break
    return out

# ──────────────────────────────────────────────────────────────────────────────
# TABLE‑MODE INDEX
# ──────────────────────────────────────────────────────────────────────────────

def build_table_index(
    table_name: str,
    lookback: int = 30
) -> Tuple[faiss.IndexFlatL2, List[pd.Timestamp], np.ndarray]:
    """
    Index each lookback‑window across all columns.
    Returns (index, list_of_dates, feature_matrix).
    """
    df = load_aligned(table_name)
    dates = list(df.index[lookback-1:])
    vectors = []
    for i in range(lookback-1, len(df)):
        window = df.iloc[i-lookback+1:i+1]
        block = [
            _make_vector(window[col].fillna(method="ffill"), lookback)
            for col in df.columns
        ]
        vectors.append(np.concatenate(block))
    X = np.stack(vectors).astype("float32")
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, dates, X

def query_table_index(
    index: faiss.IndexFlatL2,
    dates: List[pd.Timestamp],
    table_name: str,
    query_date: pd.Timestamp,
    lookback: int = 30,
    k: int = 5
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Query the table‑mode index to find the k most similar dates.
    """
    df = load_aligned(table_name)
    if query_date not in df.index:
        raise KeyError(f"{query_date} not in index for {table_name}")
    i = df.index.get_loc(query_date)
    if i < lookback-1:
        raise ValueError("Not enough history for lookback on this date")
    window = df.iloc[i-lookback+1:i+1]
    block = [
        _make_vector(window[col].fillna(method="ffill"), lookback)
        for col in df.columns
    ]
    vec = np.concatenate(block).reshape(1, -1)
    D, I = index.search(vec, k+1)
    out = []
    for dist, idx in zip(D[0], I[0]):
        if dates[idx] != query_date:
            out.append((dates[idx], float(dist)))
        if len(out) >= k:
            break
    return out

