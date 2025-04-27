# config.py

# 1) VMD decomposition settings (from Huber-only tuning)
VMD_KWARGS = {
    "alpha": 1500.0,
    "tau":     0.0,
    "K":       5,       # tuned best
    "DC":      0,
    "init":    1,
    "tol":     1e-8,    # tuned best
}

# 2) Model choice and hyper-parameters
MODEL_TYPE   = "Huber"   # still using Huber
HUBER_KWARGS = {
    "epsilon": 1.35,
    "max_iter": 1000      # raised from default 100
}

# You can leave LSTM settings here for future use
LSTM_KWARGS = {
    "units":      64,
    "epochs":     50,
    "batch_size": 16,
}

# 3) Window lengths
LOOKBACK = 30
HORIZON  = 7
