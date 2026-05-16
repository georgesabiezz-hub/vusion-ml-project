import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = "vu_pa_history_features.csv"

# Load — first row is column names, second row is the ticker label (not data)
df = pd.read_csv(DATA_PATH, skiprows=[1])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

num_cols = [c for c in df.columns if c != "date"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

print(f"Loaded {len(df)} rows, date range: {df['date'].min().date()} → {df['date'].max().date()}")

# =============================================================================
# NEW TECHNICAL INDICATORS
# =============================================================================

# --- EMA helper ---------------------------------------------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Exponential Moving Averages on adj_close
df["ema_10"] = ema(df["adj_close"], 10)
df["ema_20"] = ema(df["adj_close"], 20)
df["ema_50"] = ema(df["adj_close"], 50)

# MACD (12/26/9)
_ema12 = ema(df["adj_close"], 12)
_ema26 = ema(df["adj_close"], 26)
df["macd_line"]   = _ema12 - _ema26
df["macd_signal"] = ema(df["macd_line"], 9)
df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

# RSI-14
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = rsi(df["adj_close"], 14)

# Bollinger Bands (20-day SMA ± 2σ)
_bb_mid  = df["adj_close"].rolling(20).mean()
_bb_std  = df["adj_close"].rolling(20).std(ddof=0)
_bb_upper = _bb_mid + 2 * _bb_std
_bb_lower = _bb_mid - 2 * _bb_std
_bb_range = (_bb_upper - _bb_lower).replace(0, np.nan)

df["bb_mid_20"]   = _bb_mid
df["bb_upper_20"] = _bb_upper
df["bb_lower_20"] = _bb_lower
df["bb_width_20"] = _bb_range / _bb_mid
df["bb_pos_20"]   = (df["adj_close"] - _bb_mid) / _bb_range

# Intraday price-action features
df["range_pct"]      = (df["high"] - df["low"]) / df["close"]
df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

# Volume-momentum features
df["vol_change_1d"] = df["volume"] / df["volume"].shift(1) - 1
_vol_std_20         = df["volume"].rolling(20).std(ddof=0)
df["vol_zscore_20"] = (df["volume"] - df["vol_mean_20"]) / _vol_std_20.replace(0, np.nan)

# =============================================================================
# TARGET COLUMNS
# =============================================================================
df["y_next_ret_20d"]  = df["adj_close"].shift(-20)  / df["adj_close"] - 1
df["y_next_ret_60d"]  = df["adj_close"].shift(-60)  / df["adj_close"] - 1
df["y_next_ret_252d"] = df["adj_close"].shift(-252) / df["adj_close"] - 1

TARGETS = ["y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d"]

# Drop NaNs from both indicators and targets
df = df.dropna(subset=TARGETS + [
    "ema_10", "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_width_20", "bb_pos_20",
    "range_pct", "close_open_pct", "gap_pct",
    "vol_change_1d", "vol_zscore_20",
])
df = df.reset_index(drop=True)
print(f"After dropping NaN targets/indicators: {len(df)} rows")

# =============================================================================
# FEATURES
# =============================================================================
ORIGINAL_FEATURES = [
    "adj_close", "close", "high", "low", "open", "volume",
    "ret_1d", "ret_1d_lag1", "ret_1d_lag2", "ret_1d_lag5",
    "ret_5d", "ret_10d",
    "ma_5", "ma_10", "ma_20", "ma_50",
    "vol_10", "vol_20", "vol_mean_10", "vol_mean_20",
]
NEW_FEATURES = [
    "ema_10", "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_mid_20", "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pos_20",
    "range_pct", "close_open_pct", "gap_pct",
    "vol_change_1d", "vol_zscore_20",
]

FEATURE_COLS_V1 = ORIGINAL_FEATURES
FEATURE_COLS_V2 = ORIGINAL_FEATURES + NEW_FEATURES

split = int(len(df) * 0.80)
print(f"\nTrain rows: {split}  |  Test rows: {len(df) - split}")
print(f"Train dates: {df['date'].iloc[0].date()} → {df['date'].iloc[split-1].date()}")
print(f"Test  dates: {df['date'].iloc[split].date()} → {df['date'].iloc[-1].date()}")

# =============================================================================
# TRAIN & EVALUATE — both feature sets
# =============================================================================
def train_eval(feature_cols, label):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    results = {}
    for target in TARGETS:
        y = df[target]
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[target] = {
            "R2":  r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
        }
    return results

print("\n" + "="*60)
print("Training v1 (original features) …")
results_v1 = train_eval(FEATURE_COLS_V1, "v1")

print("Training v2 (original + new technical indicators) …")
results_v2 = train_eval(FEATURE_COLS_V2, "v2")

# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("\n" + "="*70)
print(f"{'Target':<20} {'v1 R²':>8} {'v2 R²':>8} {'ΔR²':>8}  {'v1 MAE':>10} {'v2 MAE':>10} {'ΔMAE':>10}")
print("-"*70)
for t in TARGETS:
    r2_v1,  mae_v1  = results_v1[t]["R2"],  results_v1[t]["MAE"]
    r2_v2,  mae_v2  = results_v2[t]["R2"],  results_v2[t]["MAE"]
    delta_r2  = r2_v2  - r2_v1
    delta_mae = mae_v2 - mae_v1
    r2_arrow  = "▲" if delta_r2  > 0 else "▼"
    mae_arrow = "▼" if delta_mae < 0 else "▲"   # lower MAE is better
    print(
        f"{t:<20} {r2_v1:>8.4f} {r2_v2:>8.4f} {r2_arrow}{abs(delta_r2):>7.4f}"
        f"  {mae_v1:>10.6f} {mae_v2:>10.6f} {mae_arrow}{abs(delta_mae):>9.6f}"
    )

print("\n(▲/▼ for R²: higher is better  |  ▲/▼ for MAE: lower is better)")
print(f"\nFeature count: v1={len(FEATURE_COLS_V1)}  →  v2={len(FEATURE_COLS_V2)}"
      f"  (+{len(NEW_FEATURES)} new indicators)")
