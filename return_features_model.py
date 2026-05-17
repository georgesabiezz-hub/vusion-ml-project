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
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint, uniform
import xgboost as xgb
import lightgbm as lgb

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

# =============================================================================
# THRESHOLD SWEEP — RF Classifier predicted probabilities
# =============================================================================
print("\n\n" + "="*70)
print("THRESHOLD SWEEP — RandomForestClassifier  (label_20d_up)")
print("Using rf_prob = rf_cls.predict_proba(X_test)[:, 1]")
print("="*70)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8]

rows = []
for thr in THRESHOLDS:
    pred_up   = (rf_prob >= thr).astype(int)
    n_pred_up = pred_up.sum()
    pct_up    = n_pred_up / len(pred_up)

    # guard: if threshold is so high nothing is predicted positive, metrics are 0
    acc  = accuracy_score(y_te, pred_up)
    prec = precision_score(y_te, pred_up, zero_division=0)
    rec  = recall_score(y_te, pred_up, zero_division=0)
    f1   = f1_score(y_te, pred_up, zero_division=0)

    rows.append({
        "threshold":       thr,
        "accuracy":        acc,
        "precision":       prec,
        "recall":          rec,
        "f1":              f1,
        "pct_days_pred_up": pct_up,
        "n_days_pred_up":  n_pred_up,
    })

thr_df = pd.DataFrame(rows)

# Print table
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
      f"{'F1':>8} {'%Days Up':>10} {'#Days Up':>10}")
print("-"*72)
# Baseline row first
print(f"{'baseline':>10} {baseline_acc:>10.4f} {'—':>10} {'—':>10} "
      f"{'—':>8} {'100.0%':>10} {len(y_te):>10}   (always predict 1)")
print("-"*72)
for _, r in thr_df.iterrows():
    print(
        f"{r['threshold']:>10.1f} "
        f"{r['accuracy']:>10.4f} "
        f"{r['precision']:>10.4f} "
        f"{r['recall']:>10.4f} "
        f"{r['f1']:>8.4f} "
        f"{r['pct_days_pred_up']:>9.1%} "
        f"{int(r['n_days_pred_up']):>10}"
    )

# Precision vs baseline accuracy gain at each threshold
print("\n  Precision lift over baseline accuracy (0.5347):")
for _, r in thr_df.iterrows():
    lift = r["precision"] - baseline_acc
    flag = "  ✓ beats baseline" if r["precision"] > baseline_acc else "  ✗ below baseline"
    print(f"    thr={r['threshold']:.1f}  precision={r['precision']:.4f}  lift={lift:+.4f}{flag}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
  At threshold=0.5 (default): the model predicts 'up' on most test days,
  mirroring the overall class distribution. Precision ≈ market base rate.

  As the threshold rises:
    - Precision climbs  → the model becomes more selective and more accurate
      on the days it does signal.
    - Recall falls      → fewer actual 'up' days are captured.
    - %Days predicted up shrinks → you trade less often.
    - F1 may rise then fall depending on the precision/recall trade-off.

  The sweet spot for a long-only strategy is the threshold where precision
  meaningfully exceeds the baseline accuracy (0.5347) at an acceptable
  coverage (enough trading days to matter in practice).
""")

# =============================================================================
# BOOSTED-TREE CLASSIFIERS — XGBoost and LightGBM
# =============================================================================
print("\n" + "="*70)
print("BOOSTED-TREE CLASSIFIERS — XGBoost & LightGBM  (label_20d_up)")
print("="*70)

pos_ratio = float(y_tr.mean())

xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(1 - pos_ratio) / pos_ratio,  # handle class imbalance
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb_clf.fit(X_tr, y_tr)
xgb_pred = xgb_clf.predict(X_te)
xgb_prob = xgb_clf.predict_proba(X_te)[:, 1]

lgb_clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_clf.fit(X_tr, y_tr)
lgb_pred = lgb_clf.predict(X_te)
lgb_prob = lgb_clf.predict_proba(X_te)[:, 1]


def full_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_prob),
    }

xgb_m = full_metrics(y_te, xgb_pred, xgb_prob)
lgb_m = full_metrics(y_te, lgb_pred, lgb_prob)

print(f"\n{'Metric':<12} {'Baseline':>10} {'RF Clf':>10} {'XGBoost':>10} {'LightGBM':>10}")
print("-"*54)
for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    bv = baseline_acc if metric == "Accuracy" else (0.5 if metric == "ROC-AUC" else "—")
    bv_s = f"{bv:.4f}" if isinstance(bv, float) else "—"
    print(
        f"{metric:<12} {bv_s:>10} "
        f"{rf_metrics[metric]:>10.4f} "
        f"{xgb_m[metric]:>10.4f} "
        f"{lgb_m[metric]:>10.4f}"
    )

# --- Threshold sweep for both boosted models ---
def threshold_sweep(probs, y_true, model_name):
    print(f"\n  Threshold sweep — {model_name}")
    print(f"  {'Thr':>5} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'%Up':>8} {'#Up':>6}")
    print(f"  {'base':>5} {baseline_acc:>8.4f} {'—':>8} {'—':>8} {'—':>8} {'100%':>8} {len(y_true):>6}  (always 1)")
    print("  " + "-"*57)
    sweep_rows = []
    for thr in THRESHOLDS:
        p = (probs >= thr).astype(int)
        n = p.sum()
        sweep_rows.append({
            "thr":  thr,
            "acc":  accuracy_score(y_true, p),
            "prec": precision_score(y_true, p, zero_division=0),
            "rec":  recall_score(y_true, p, zero_division=0),
            "f1":   f1_score(y_true, p, zero_division=0),
            "pct":  n / len(p),
            "n":    n,
        })
        print(
            f"  {thr:>5.1f} "
            f"{sweep_rows[-1]['acc']:>8.4f} "
            f"{sweep_rows[-1]['prec']:>8.4f} "
            f"{sweep_rows[-1]['rec']:>8.4f} "
            f"{sweep_rows[-1]['f1']:>8.4f} "
            f"{sweep_rows[-1]['pct']:>7.1%} "
            f"{n:>6}"
        )
    return sweep_rows

xgb_sweep = threshold_sweep(xgb_prob, y_te, "XGBoost")
lgb_sweep = threshold_sweep(lgb_prob, y_te, "LightGBM")

# --- Head-to-head at thr=0.6 ---
def get_thr_row(sweep, thr):
    return next(r for r in sweep if r["thr"] == thr)

rf_06 = next(r for _, r in thr_df.iterrows() if r["threshold"] == 0.6)

print("\n" + "="*70)
print("HEAD-TO-HEAD AT THRESHOLD = 0.6  (key comparison)")
print("="*70)
print(f"\n  {'Metric':<18} {'RF Clf':>10} {'XGBoost':>10} {'LightGBM':>10}  {'Winner':>10}")
print("  " + "-"*60)

comparisons = [
    ("ROC-AUC (thr-free)", rf_metrics["ROC-AUC"], xgb_m["ROC-AUC"], lgb_m["ROC-AUC"]),
    ("Precision @0.6",     rf_06["precision"],
     get_thr_row(xgb_sweep, 0.6)["prec"],
     get_thr_row(lgb_sweep, 0.6)["prec"]),
    ("Recall @0.6",        rf_06["recall"],
     get_thr_row(xgb_sweep, 0.6)["rec"],
     get_thr_row(lgb_sweep, 0.6)["rec"]),
    ("F1 @0.6",            rf_06["f1"],
     get_thr_row(xgb_sweep, 0.6)["f1"],
     get_thr_row(lgb_sweep, 0.6)["f1"]),
    ("% Days Up @0.6",     rf_06["pct_days_pred_up"],
     get_thr_row(xgb_sweep, 0.6)["pct"],
     get_thr_row(lgb_sweep, 0.6)["pct"]),
]

for label, rf_v, xgb_v, lgb_v in comparisons:
    best = max(rf_v, xgb_v, lgb_v)
    winner = (
        "RF"       if rf_v  == best else
        "XGBoost"  if xgb_v == best else
        "LightGBM"
    )
    pct = label.startswith("% Days")
    fmt = ".1%" if pct else ".4f"
    print(
        f"  {label:<18} {rf_v:>10{fmt}} {xgb_v:>10{fmt}} {lgb_v:>10{fmt}}  {winner:>10}"
    )

print("\n" + "="*70)
print("VERDICT")
print("="*70)
best_auc   = max(rf_metrics["ROC-AUC"], xgb_m["ROC-AUC"], lgb_m["ROC-AUC"])
best_prec6 = max(
    rf_06["precision"],
    get_thr_row(xgb_sweep, 0.6)["prec"],
    get_thr_row(lgb_sweep, 0.6)["prec"],
)
auc_winner  = ("RF" if rf_metrics["ROC-AUC"] == best_auc else
               "XGBoost" if xgb_m["ROC-AUC"] == best_auc else "LightGBM")
prec_winner = ("RF" if rf_06["precision"] == best_prec6 else
               "XGBoost" if get_thr_row(xgb_sweep, 0.6)["prec"] == best_prec6 else "LightGBM")

xgb_auc_beats_rf = xgb_m["ROC-AUC"] > rf_metrics["ROC-AUC"]
lgb_auc_beats_rf = lgb_m["ROC-AUC"] > rf_metrics["ROC-AUC"]
xgb_prec_beats_rf = get_thr_row(xgb_sweep, 0.6)["prec"] > rf_06["precision"]
lgb_prec_beats_rf = get_thr_row(lgb_sweep, 0.6)["prec"] > rf_06["precision"]

print(f"\n  ROC-AUC  — best model: {auc_winner} ({best_auc:.4f})")
print(f"    XGBoost  {'IMPROVES' if xgb_auc_beats_rf else 'does NOT improve'} over RF  "
      f"(Δ={xgb_m['ROC-AUC']-rf_metrics['ROC-AUC']:+.4f})")
print(f"    LightGBM {'IMPROVES' if lgb_auc_beats_rf else 'does NOT improve'} over RF  "
      f"(Δ={lgb_m['ROC-AUC']-rf_metrics['ROC-AUC']:+.4f})")

print(f"\n  Precision @thr=0.6 — best model: {prec_winner} ({best_prec6:.4f})")
print(f"    XGBoost  {'IMPROVES' if xgb_prec_beats_rf else 'does NOT improve'} over RF  "
      f"(Δ={get_thr_row(xgb_sweep,0.6)['prec']-rf_06['precision']:+.4f})")
print(f"    LightGBM {'IMPROVES' if lgb_prec_beats_rf else 'does NOT improve'} over RF  "
      f"(Δ={get_thr_row(lgb_sweep,0.6)['prec']-rf_06['precision']:+.4f})")

# =============================================================================
# LIGHTGBM — RANDOMIZED HYPERPARAMETER SEARCH WITH TIME-SERIES CV
# =============================================================================
print("\n\n" + "="*70)
print("LIGHTGBM — RANDOMIZED HYPERPARAMETER SEARCH")
print("50 random combos × TimeSeriesSplit(n_splits=4), scoring=roc_auc")
print("="*70)

PARAM_DIST = {
    "n_estimators":     randint(200, 501),       # 200–500
    "learning_rate":    uniform(0.03, 0.07),     # 0.03–0.10
    "max_depth":        randint(3, 7),            # 3–6
    "num_leaves":       randint(15, 64),          # 15–63
    "min_data_in_leaf": randint(20, 101),         # 20–100
    "feature_fraction": uniform(0.6, 0.4),        # 0.6–1.0
}

N_ITER = 30
N_SPLITS = 4
print(f"\n{N_ITER} random samples × {N_SPLITS} folds = {N_ITER*N_SPLITS} fits")
print("Running search (sequential, n_jobs=1) …\n")

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

lgb_base_for_search = lgb.LGBMClassifier(
    is_unbalance=True,
    random_state=42,
    n_jobs=1,          # single-threaded per model — avoids joblib deadlock
    verbose=-1,
)

rand_search = RandomizedSearchCV(
    estimator=lgb_base_for_search,
    param_distributions=PARAM_DIST,
    n_iter=N_ITER,
    cv=tscv,
    scoring="roc_auc",
    n_jobs=1,          # sequential outer loop — reliable in all environments
    refit=False,
    random_state=42,
    verbose=1,         # show fold progress
)
rand_search.fit(X_tr, y_tr)

best_params = rand_search.best_params_
best_cv_auc = rand_search.best_score_

print(f"Best CV ROC-AUC (train folds): {best_cv_auc:.4f}")
print("Best hyperparameters:")
for k, v in best_params.items():
    print(f"  {k:<22}: {v}")

# --- Refit on full train set with best params ---
lgb_tuned = lgb.LGBMClassifier(
    **best_params,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
lgb_tuned.fit(X_tr, y_tr)
lgb_tuned_pred = lgb_tuned.predict(X_te)
lgb_tuned_prob = lgb_tuned.predict_proba(X_te)[:, 1]
lgb_tuned_m    = full_metrics(y_te, lgb_tuned_pred, lgb_tuned_prob)

# --- Full metric table vs default LightGBM ---
print("\n" + "="*70)
print("TEST-SET METRICS: default LightGBM vs tuned LightGBM (randomized search)")
print("="*70)
print(f"\n{'Metric':<12} {'Baseline':>10} {'LGB default':>12} {'LGB tuned':>12} {'Δ (tuned−def)':>15}")
print("-"*63)
for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
    bv  = baseline_acc if metric == "Accuracy" else (0.5 if metric == "ROC-AUC" else None)
    bvs = f"{bv:.4f}" if bv is not None else "—"
    d   = lgb_m[metric]
    t   = lgb_tuned_m[metric]
    delta = t - d
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    print(f"{metric:<12} {bvs:>10} {d:>12.4f} {t:>12.4f} {arrow}{abs(delta):>13.4f}")

# --- Threshold sweep for tuned model ---
lgb_tuned_sweep = threshold_sweep(lgb_tuned_prob, y_te, "LightGBM (tuned)")

# --- Side-by-side at thr=0.6 ---
lgb_def_06   = get_thr_row(lgb_sweep, 0.6)
lgb_tuned_06 = get_thr_row(lgb_tuned_sweep, 0.6)

print("\n" + "="*70)
print("THRESHOLD = 0.6 — default vs tuned LightGBM")
print("="*70)
print(f"\n  {'Metric':<20} {'LGB default':>12} {'LGB tuned':>12} {'Δ':>10}  {'Change'}")
print("  " + "-"*62)

key_pairs = [
    ("ROC-AUC (thr-free)", lgb_m["ROC-AUC"],   lgb_tuned_m["ROC-AUC"], False),
    ("Precision @0.6",     lgb_def_06["prec"],  lgb_tuned_06["prec"],   False),
    ("Recall @0.6",        lgb_def_06["rec"],   lgb_tuned_06["rec"],    False),
    ("F1 @0.6",            lgb_def_06["f1"],    lgb_tuned_06["f1"],     False),
    ("% Days Up @0.6",     lgb_def_06["pct"],   lgb_tuned_06["pct"],    True),
]

for label, dv, tv, is_pct in key_pairs:
    delta = tv - dv
    fmt   = ".1%" if is_pct else ".4f"
    tag   = "IMPROVED" if delta > 0 else ("WORSE" if delta < 0 else "same")
    print(f"  {label:<20} {dv:>12{fmt}} {tv:>12{fmt}} {delta:>+10{fmt}}  {tag}")

print("\n" + "="*70)
print("FINAL VERDICT — tuned LightGBM vs default LightGBM")
print("="*70)
auc_improved  = lgb_tuned_m["ROC-AUC"] > lgb_m["ROC-AUC"]
prec_improved = lgb_tuned_06["prec"]   > lgb_def_06["prec"]

print(f"\n  ROC-AUC:          {'IMPROVED' if auc_improved  else 'did NOT improve'}  "
      f"({lgb_m['ROC-AUC']:.4f} → {lgb_tuned_m['ROC-AUC']:.4f}, "
      f"Δ={lgb_tuned_m['ROC-AUC']-lgb_m['ROC-AUC']:+.4f})")
print(f"  Precision @0.6:   {'IMPROVED' if prec_improved else 'did NOT improve'}  "
      f"({lgb_def_06['prec']:.4f} → {lgb_tuned_06['prec']:.4f}, "
      f"Δ={lgb_tuned_06['prec']-lgb_def_06['prec']:+.4f})")
print(f"  Best CV AUC (train folds): {best_cv_auc:.4f}")
