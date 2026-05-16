import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "vu_pa_history_features.csv"

# Load — first row is column names, second row is the ticker label (not data)
df = pd.read_csv(DATA_PATH, skiprows=[1])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Cast numeric columns (some may have been read as object due to the skipped ticker row)
num_cols = [c for c in df.columns if c != "date"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

print(f"Loaded {len(df)} rows, date range: {df['date'].min().date()} → {df['date'].max().date()}")

# --- Target columns -----------------------------------------------------------
df["y_next_ret_20d"]  = df["adj_close"].shift(-20)  / df["adj_close"] - 1
df["y_next_ret_60d"]  = df["adj_close"].shift(-60)  / df["adj_close"] - 1
df["y_next_ret_252d"] = df["adj_close"].shift(-252) / df["adj_close"] - 1

df = df.dropna(subset=["y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d"])
df = df.reset_index(drop=True)
print(f"After dropping NaN targets: {len(df)} rows")

# --- Features -----------------------------------------------------------------
FEATURE_COLS = [
    "adj_close", "close", "high", "low", "open", "volume",
    "ret_1d", "ret_1d_lag1", "ret_1d_lag2", "ret_1d_lag5",
    "ret_5d", "ret_10d",
    "ma_5", "ma_10", "ma_20", "ma_50",
    "vol_10", "vol_20", "vol_mean_10", "vol_mean_20",
]
TARGETS = ["y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d"]

X = df[FEATURE_COLS]
split = int(len(df) * 0.80)

print(f"\nTrain rows: {split}  |  Test rows: {len(df) - split}")
print(f"Train dates: {df['date'].iloc[0].date()} → {df['date'].iloc[split-1].date()}")
print(f"Test  dates: {df['date'].iloc[split].date()} → {df['date'].iloc[-1].date()}")

# --- Train & evaluate ---------------------------------------------------------
results = {}
for target in TARGETS:
    y = df[target]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2  = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    results[target] = {"R2": r2, "MAE": mae}
    print(f"\n{'='*50}")
    print(f"Target : {target}")
    print(f"  Test R²  : {r2:.4f}")
    print(f"  Test MAE : {mae:.6f}  ({mae*100:.4f}%)")

print("\n" + "="*50)
print("Summary")
print("="*50)
print(f"{'Target':<20} {'R²':>10} {'MAE':>12}")
print("-"*44)
for t, v in results.items():
    print(f"{t:<20} {v['R2']:>10.4f} {v['MAE']:>12.6f}")
