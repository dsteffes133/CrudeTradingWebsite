from app.modules.vmd_models import prepare_vmd_ml_data, train_huber
from app.modules.data_utils    import load_aligned
import config, numpy as np

# load the series once
series_df = load_aligned("bond_stocks")["WTI Crude Oil"].ffill().dropna()

best = {"rmse": np.inf, "params": None}

grid = [
    {"alpha": α, "tau": 0.0, "K": k, "DC": 0, "init": 1, "tol": tol}
    for α in [1000,1500,2000]
    for k in [3,5,7]
    for tol in [1e-8,1e-7,1e-6]
]

for vmd_kw in grid:
    # override the global settings
    config.VMD_KWARGS = vmd_kw

    # build train/test
    X_tr, y_tr, _, X_te, y_te, _ = prepare_vmd_ml_data(
        "bond_stocks", "WTI Crude Oil", split_frac=0.8
    )

    # fit & eval Huber
    model = train_huber(X_tr, y_tr)
    preds = model.predict(X_te.reshape(len(X_te), -1))
    rmse = np.sqrt(((y_te - preds) ** 2).mean())

    if rmse < best["rmse"]:
        best["rmse"], best["params"] = rmse, vmd_kw

print("Best RMSE:", best["rmse"])
print("Best VMD settings:", best["params"])

