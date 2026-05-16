import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)

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

# =============================================================================
# CLASSIFICATION — 20-day up/down label
# =============================================================================
print("\n\n" + "="*70)
print("CLASSIFICATION: Will adj_close be higher in 20 trading days?")
print("="*70)

df["label_20d_up"] = (df["y_next_ret_20d"] > 0).astype(int)

X_cls = df[FEATURE_COLS_V2].replace([np.inf, -np.inf], np.nan).fillna(0)
y_cls = df["label_20d_up"]

X_tr, X_te = X_cls.iloc[:split], X_cls.iloc[split:]
y_tr, y_te = y_cls.iloc[:split], y_cls.iloc[split:]

# Class distribution
pos_train = y_tr.mean()
pos_test  = y_te.mean()
majority_class = int(pos_test >= 0.5)
baseline_acc   = max(pos_test, 1 - pos_test)
print(f"\nLabel distribution  —  train: {pos_train:.1%} up  |  test: {pos_test:.1%} up")
print(f"Majority-class baseline accuracy (always predict {majority_class}): {baseline_acc:.4f}")

# --- Logistic Regression (needs scaled features) ---
scaler    = StandardScaler()
X_tr_sc   = scaler.fit_transform(X_tr)
X_te_sc   = scaler.transform(X_te)

lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr.fit(X_tr_sc, y_tr)
lr_pred      = lr.predict(X_te_sc)
lr_prob      = lr.predict_proba(X_te_sc)[:, 1]

# --- Random Forest Classifier ---
rf_cls = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_cls.fit(X_tr, y_tr)
rf_pred = rf_cls.predict(X_te)
rf_prob = rf_cls.predict_proba(X_te)[:, 1]

# --- Metrics helper ---
def cls_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }

lr_metrics = cls_metrics(y_te, lr_pred, lr_prob)
rf_metrics = cls_metrics(y_te, rf_pred, rf_prob)

# --- Print detailed report ---
for name, metrics, preds in [
    ("Logistic Regression (C=0.1)", lr_metrics, lr_pred),
    ("Random Forest Classifier (200 trees)", rf_metrics, rf_pred),
]:
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

# --- Side-by-side comparison ---
print("\n\n" + "="*70)
print(f"{'Metric':<14} {'Baseline':>10} {'LogReg':>10} {'RF Clf':>10}")
print("-"*46)
metrics_list = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
baseline_vals = {"Accuracy": baseline_acc, "Precision": "—", "Recall": "—",
                 "F1": "—", "ROC-AUC": 0.5}
for m in metrics_list:
    bv = baseline_vals[m]
    bv_str = f"{bv:.4f}" if isinstance(bv, float) else f"{'—':>10}"
    print(f"{m:<14} {bv_str:>10} {lr_metrics[m]:>10.4f} {rf_metrics[m]:>10.4f}")

# --- Verdict ---
print("\n" + "="*70)
print("VERDICT")
print("="*70)
beats_baseline_lr = lr_metrics["Accuracy"] > baseline_acc and lr_metrics["ROC-AUC"] > 0.5
beats_baseline_rf = rf_metrics["Accuracy"] > baseline_acc and rf_metrics["ROC-AUC"] > 0.5

for name, m, beats in [("LogReg", lr_metrics, beats_baseline_lr),
                        ("RF Clf", rf_metrics, beats_baseline_rf)]:
    gap_acc = m["Accuracy"] - baseline_acc
    gap_auc = m["ROC-AUC"] - 0.5
    verdict = "BEATS" if beats else "FAILS TO BEAT"
    print(f"  {name}: {verdict} baseline  "
          f"(Accuracy {gap_acc:+.4f} vs baseline, AUC {gap_auc:+.4f} vs 0.5)")
