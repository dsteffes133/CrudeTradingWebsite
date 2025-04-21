# pages/forecasting.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from app.modules.ml_utils import (
    prepare_ml_data,
    train_random_forest,
    evaluate_model
)

st.sidebar.header("🤖 Random Forest Forecasting")

# 1️⃣ Responder series & their source table
RESPONDERS = {
    "WCS Houston (Pricing Vector)": (
        "pricing_vector",
        "WCS Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"
    ),
    "WTI Houston (Pricing Vector)": (
        "pricing_vector",
        "WTI Houston weighted average month 1, Houston close, diff index, USD/bl, fip, FILLED FORWARD"
    ),
    "WTI Cushing (FRED)": (
        "bond_stocks",
        "WTI Crude Oil"
    ),
}

responder_label = st.sidebar.selectbox(
    "Responder variable", list(RESPONDERS.keys())
)
responder_table, responder_col = RESPONDERS[responder_label]

# 2️⃣ Predictor tables (all those you want to include)
PREDICTOR_TABLES = [
    "bond_stocks", "pricing_vector",
    "us_imports_exports", "global_imports_exports",
    "wpr_sliding", "daily_pipeline", "daily_movement"
]

# 3️⃣ Forecast settings
horizon    = st.sidebar.selectbox("Forecast horizon (days)", [1, 7, 30])
lookback   = st.sidebar.slider("Feature lookback (days)", 7, 60, 30)
split_date = st.sidebar.date_input(
    "Train/test split date",
    value=pd.to_datetime("2024-01-01")
)

# 4️⃣ RF hyperparameters
st.sidebar.subheader("RF Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, step=50)
max_depth     = st.sidebar.slider("max_depth", 5, 30, 10, step=5)

# 5️⃣ Action buttons
train_btn    = st.sidebar.button("▶️ Train & Evaluate")
forecast_btn = st.sidebar.button("🔮 Forecast Forward")

st.title("📈 Random Forest Forecasting")

if train_btn:
    with st.spinner("Preparing data…"):
        X_train, y_train, X_test, y_test = prepare_ml_data(
            responder_table=responder_table,
            responder_col=responder_col,
            predictor_tables=PREDICTOR_TABLES,
            lookback=lookback,
            horizon=horizon,
            split_date=str(split_date)
        )

    st.subheader("Training Random Forest")
    with st.spinner("Training…"):
        rf = train_random_forest(
            X_train, y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    train_metrics = evaluate_model(rf, X_train, y_train)
    test_metrics  = evaluate_model(rf, X_test,  y_test)

    st.markdown("**Metrics**")
    st.table(pd.DataFrame({
        "Train": train_metrics,
        "Test":  test_metrics
    }))

    preds = rf.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test.index, y_test.values, label="Actual")
    ax.plot(y_test.index, preds,     label="Predicted")
    ax.set_title(f"Backtest ({horizon}‑day horizon)")
    ax.legend()
    st.pyplot(fig)

    full_feats = pd.concat([X_train, X_test])
    st.session_state["rf_model"]      = rf
    st.session_state["last_features"] = full_feats.iloc[-1]

if forecast_btn:
    if "rf_model" not in st.session_state:
        st.error("Please train first, then Forecast Forward.")
    else:
        rf         = st.session_state["rf_model"]
        last_feats = st.session_state["last_features"]

        point = rf.predict(last_feats.values.reshape(1, -1))[0]
        tree_preds = np.array([
            est.predict(last_feats.values.reshape(1, -1))[0]
            for est in rf.estimators_
        ])
        lower, upper = np.percentile(tree_preds, [10, 90])

        fc_date = last_feats.name + timedelta(days=horizon)
        st.subheader(f"📅 Forecast for {fc_date.date()}")
        st.metric("Prediction", f"{point:.2f}")
        st.write(f"10–90% band: [{lower:.2f}, {upper:.2f}]")

        # Plot last history + forecast
        hist_date = last_feats.name
        hist_val  = rf.predict(last_feats.values.reshape(1, -1))[0]  # or load from responder series
        ser = pd.Series([hist_val, point], index=[hist_date, fc_date])
        st.line_chart(ser, height=300)
