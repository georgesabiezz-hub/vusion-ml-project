"""
news_feature_model.py
Adds a news_count feature from hand-collected VU.PA / SES-imagotag /
VusionGroup press releases, retrains the 20-day regression and
classification models, and compares results against the v2 baseline.

News collected from:
  - actusnews.com (official French newswire for VusionGroup)
  - businesswire.com
  - Web search snippets (Yahoo Finance, vusion.com)
Data covers the full price history (2015-03-12 → 2026-05-12).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, roc_auc_score,
)

DATA_PATH = "vu_pa_history_features.csv"

# =============================================================================
# HAND-COLLECTED NEWS EVENTS  (date, headline)
# Sources: actusnews.com, businesswire.com, prnewswire.com, vusion.com
# =============================================================================
RAW_NEWS = [
    # 2015
    ("2015-03-30", "Store Electronic Systems 2014 Annual Results"),
    ("2015-07-17", "SES: an E-Commerce Awards 2015 finalist"),
    ("2015-12-02", "SES wins the largest ESL project ever in the market (~USD 105m contract)"),
    # 2016
    ("2016-07-28", "SES-imagotag First half 2016 sales: EUR 85m (+113%)"),
    # 2017  (no press release dates found in search for H1/FY 2016 results)
    # 2018
    ("2018-02-02", "SES-imagotag 2017 Preliminary Results"),
    ("2018-02-14", "SES-imagotag signs partnership with Panasonic for retail digitalisation in Europe"),
    ("2018-03-08", "SES-imagotag 2017 Full Year Results"),
    ("2018-08-01", "SES-imagotag H1 2018 Sales"),
    # 2019
    ("2019-03-13", "SES-imagotag 2018 Full Year Results"),
    ("2019-09-19", "SES-imagotag Results H1 2019"),
    # 2020
    ("2020-07-27", "SES-imagotag H1 2020 Sales"),
    # 2021
    ("2021-03-31", "SES-imagotag 2020 Full Year Results: Strong improvement in financial results"),
    ("2021-04-30", "SES-imagotag Annual Report 2020 available"),
    ("2021-11-03", "SES-imagotag partnership announcement (Sephora digitalisation)"),
    # 2022
    ("2022-01-27", "SES-imagotag Record sales growth in 2021 to EUR 423m"),
    ("2022-04-01", "SES-imagotag press release (Q1 2022 update)"),
    ("2022-07-25", "SES-imagotag H1 2022 Sales at record levels"),
    ("2022-09-08", "SES-imagotag Strong earnings growth in H1 2022"),
    # 2023
    ("2023-01-04", "SES-imagotag Half-year statement on liquidity contract"),
    ("2023-01-10", "SES-imagotag to make Strategic Acquisition of a Data Company"),
    ("2023-03-08", "SES-imagotag FY 2022 Results: Strong operating and financial performance"),
    ("2023-04-27", "SES-imagotag announces VUSION platform roll-out contract in Walmart US stores"),
    ("2023-04-27", "SES-imagotag convenes shareholders meeting to issue warrants to Walmart"),
    ("2023-06-26", "SES-imagotag Response to the Gotham City Research short-seller report"),
    ("2023-07-03", "SES-imagotag Further Expansion in the U.S. — largest convenience store chain"),
    ("2023-07-04", "SES-imagotag Half-year statement on liquidity contract"),
    ("2023-07-18", "The Global Leading Furniture Retailer selects SES-imagotag VUSION platform"),
    ("2023-07-27", "SES-imagotag H1 2023 Sales"),
    ("2023-09-11", "SES-imagotag H1 2023 Results: Strong earnings growth and positive cash flow"),
    # 2024
    ("2024-01-10", "SES-imagotag becomes VusionGroup"),
    ("2024-03-27", "VusionGroup FY 2023 Results: Strong Profitability Improvement"),
    ("2024-04-25", "VusionGroup Q1 2024 Sales: revenue in line with guidance"),
    ("2024-04-30", "VusionGroup secures contract amendment for accelerated Walmart EdgeSense deployment"),
    ("2024-09-12", "VusionGroup H1 2024 Results: Strong Profitability Improvement and Free Cash Flow"),
    ("2024-10-28", "VusionGroup Q3 2024 Sales"),
    ("2024-12-11", "Coop Alleanza 3.0 renforce la digitalisation with VusionGroup"),
    ("2024-12-17", "VusionGroup and The Fresh Market to Revolutionize Retail with Vusion 360 roll-out"),
    # 2025
    ("2025-01-11", "VusionGroup at NRF2025: Making a Positive Impact by Putting Technology at Service of Physical Commerce"),
    ("2025-03-27", "VusionGroup 2024 Annual Results: EUR 1B Revenue, 25% Growth"),
    ("2025-05-02", "VusionGroup Universal Registration Document 2024 published"),
]

news_df = pd.DataFrame(RAW_NEWS, columns=["date", "headline"])
news_df["date"] = pd.to_datetime(news_df["date"])
news_count_df = (
    news_df.groupby("date")
    .size()
    .reset_index(name="news_count")
)
print(f"News events collected : {len(news_df)}")
print(f"Unique news dates     : {len(news_count_df)}")
print(f"Date range of news    : {news_df['date'].min().date()} → {news_df['date'].max().date()}")

# =============================================================================
# LOAD PRICE DATA & BUILD FEATURES  (same pipeline as return_features_model.py)
# =============================================================================
df = pd.read_csv(DATA_PATH, skiprows=[1])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
num_cols = [c for c in df.columns if c != "date"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# --- EMA ---
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

df["ema_10"] = ema(df["adj_close"], 10)
df["ema_20"] = ema(df["adj_close"], 20)
df["ema_50"] = ema(df["adj_close"], 50)

# --- MACD ---
_ema12 = ema(df["adj_close"], 12)
_ema26 = ema(df["adj_close"], 26)
df["macd_line"]   = _ema12 - _ema26
df["macd_signal"] = ema(df["macd_line"], 9)
df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

# --- RSI ---
def rsi(series, period=14):
    delta    = series.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = rsi(df["adj_close"], 14)

# --- Bollinger Bands ---
_bb_mid   = df["adj_close"].rolling(20).mean()
_bb_std   = df["adj_close"].rolling(20).std(ddof=0)
_bb_upper = _bb_mid + 2 * _bb_std
_bb_lower = _bb_mid - 2 * _bb_std
_bb_range = (_bb_upper - _bb_lower).replace(0, np.nan)
df["bb_mid_20"]   = _bb_mid
df["bb_upper_20"] = _bb_upper
df["bb_lower_20"] = _bb_lower
df["bb_width_20"] = _bb_range / _bb_mid
df["bb_pos_20"]   = (df["adj_close"] - _bb_mid) / _bb_range

# --- Intraday price-action ---
df["range_pct"]      = (df["high"] - df["low"]) / df["close"]
df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

# --- Volume momentum ---
df["vol_change_1d"] = df["volume"] / df["volume"].shift(1) - 1
_vol_std_20         = df["volume"].rolling(20).std(ddof=0)
df["vol_zscore_20"] = (df["volume"] - df["vol_mean_20"]) / _vol_std_20.replace(0, np.nan)

# --- Targets ---
df["y_next_ret_20d"]  = df["adj_close"].shift(-20) / df["adj_close"] - 1
df["y_next_ret_60d"]  = df["adj_close"].shift(-60) / df["adj_close"] - 1
df["y_next_ret_252d"] = df["adj_close"].shift(-252) / df["adj_close"] - 1

TARGETS = ["y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d"]

df = df.dropna(subset=TARGETS + [
    "ema_10", "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist", "rsi_14",
    "bb_width_20", "bb_pos_20",
    "range_pct", "close_open_pct", "gap_pct",
    "vol_change_1d", "vol_zscore_20",
])
df = df.reset_index(drop=True)

# =============================================================================
# MERGE NEWS COUNT  (left join on date; 0 for trading days with no news)
# =============================================================================
df = df.merge(news_count_df, on="date", how="left")
df["news_count"] = df["news_count"].fillna(0).astype(int)

print(f"\nPrice rows after indicator NaN drop : {len(df)}")
print(f"Days with news_count > 0            : {(df['news_count'] > 0).sum()}")
print(f"Max news_count on a single day      : {df['news_count'].max()}")
print(f"news_count distribution:\n{df['news_count'].value_counts().sort_index().to_string()}")

# =============================================================================
# FEATURE SETS
# =============================================================================
BASE_37 = [
    "adj_close", "close", "high", "low", "open", "volume",
    "ret_1d", "ret_1d_lag1", "ret_1d_lag2", "ret_1d_lag5",
    "ret_5d", "ret_10d",
    "ma_5", "ma_10", "ma_20", "ma_50",
    "vol_10", "vol_20", "vol_mean_10", "vol_mean_20",
    "ema_10", "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist",
    "rsi_14",
    "bb_mid_20", "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pos_20",
    "range_pct", "close_open_pct", "gap_pct",
    "vol_change_1d", "vol_zscore_20",
]
PLUS_NEWS = BASE_37 + ["news_count"]

split = int(len(df) * 0.80)
print(f"\nTrain rows : {split}  ({df['date'].iloc[0].date()} → {df['date'].iloc[split-1].date()})")
print(f"Test rows  : {len(df)-split}  ({df['date'].iloc[split].date()} → {df['date'].iloc[-1].date()}")

# =============================================================================
# HELPERS
# =============================================================================
def clean(X):
    return X.replace([np.inf, -np.inf], np.nan).fillna(0)

def reg_metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred)

def cls_metrics(y_true, y_pred, y_prob):
    return accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_prob)

# =============================================================================
# 1. REGRESSION: y_next_ret_20d
# =============================================================================
print("\n" + "="*70)
print("REGRESSION — y_next_ret_20d (RandomForestRegressor, 200 trees)")
print("="*70)

reg_results = {}
for label, cols in [("v2 (37 features)", BASE_37), ("v3 (+news_count)", PLUS_NEWS)]:
    X = clean(df[cols])
    y = df["y_next_ret_20d"]
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    r2, mae = reg_metrics(y_te, preds)
    reg_results[label] = {"R2": r2, "MAE": mae}
    print(f"  {label:<22}  R²={r2:+.4f}   MAE={mae:.6f} ({mae*100:.4f}%)")

r2_delta  = reg_results["v3 (+news_count)"]["R2"]  - reg_results["v2 (37 features)"]["R2"]
mae_delta = reg_results["v3 (+news_count)"]["MAE"] - reg_results["v2 (37 features)"]["MAE"]
print(f"\n  Δ (v3 − v2)           R²={r2_delta:+.4f}   MAE={mae_delta:+.6f}")

# =============================================================================
# 2. CLASSIFICATION: label_20d_up
# =============================================================================
print("\n" + "="*70)
print("CLASSIFICATION — label_20d_up (LogReg + RandomForestClassifier)")
print("="*70)

df["label_20d_up"] = (df["y_next_ret_20d"] > 0).astype(int)
y_cls = df["label_20d_up"]
pos_test = y_cls.iloc[split:].mean()
baseline_acc = max(pos_test, 1 - pos_test)
print(f"\n  Test set positive rate : {pos_test:.1%}  |  Majority-class baseline acc : {baseline_acc:.4f}")

cls_results = {}
for label, cols in [("v2 (37 features)", BASE_37), ("v3 (+news_count)", PLUS_NEWS)]:
    X_raw = clean(df[cols])
    X_tr_raw, X_te_raw = X_raw.iloc[:split], X_raw.iloc[split:]
    y_tr, y_te = y_cls.iloc[:split], y_cls.iloc[split:]

    # --- Logistic Regression ---
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_tr)
    lr_pred = lr.predict(X_te_sc)
    lr_prob = lr.predict_proba(X_te_sc)[:, 1]
    lr_acc, lr_auc = cls_metrics(y_te, lr_pred, lr_prob)

    # --- Random Forest Classifier ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr_raw, y_tr)
    rf_pred = rf.predict(X_te_raw)
    rf_prob = rf.predict_proba(X_te_raw)[:, 1]
    rf_acc, rf_auc = cls_metrics(y_te, rf_pred, rf_prob)

    cls_results[label] = {
        "LR_acc": lr_acc, "LR_auc": lr_auc,
        "RF_acc": rf_acc, "RF_auc": rf_auc,
    }
    print(f"\n  {label}")
    print(f"    LogReg  — Accuracy={lr_acc:.4f}  ROC-AUC={lr_auc:.4f}")
    print(f"    RF Clf  — Accuracy={rf_acc:.4f}  ROC-AUC={rf_auc:.4f}")

# =============================================================================
# FINAL COMPARISON TABLE
# =============================================================================
print("\n\n" + "="*70)
print("FINAL COMPARISON SUMMARY")
print("="*70)

v2 = cls_results["v2 (37 features)"]
v3 = cls_results["v3 (+news_count)"]

print(f"\n{'':30} {'v2 (37)':>10} {'v3 (+news)':>10} {'Δ':>8}")
print("-"*62)

# Regression
r2_v2  = reg_results["v2 (37 features)"]["R2"]
r2_v3  = reg_results["v3 (+news_count)"]["R2"]
mae_v2 = reg_results["v2 (37 features)"]["MAE"]
mae_v3 = reg_results["v3 (+news_count)"]["MAE"]
print(f"{'Regression R²  (20d)':<30} {r2_v2:>10.4f} {r2_v3:>10.4f} {r2_v3-r2_v2:>+8.4f}")
print(f"{'Regression MAE (20d)':<30} {mae_v2:>10.6f} {mae_v3:>10.6f} {mae_v3-mae_v2:>+8.6f}")

# Classification
print(f"{'LogReg Accuracy':<30} {v2['LR_acc']:>10.4f} {v3['LR_acc']:>10.4f} {v3['LR_acc']-v2['LR_acc']:>+8.4f}")
print(f"{'LogReg ROC-AUC':<30} {v2['LR_auc']:>10.4f} {v3['LR_auc']:>10.4f} {v3['LR_auc']-v2['LR_auc']:>+8.4f}")
print(f"{'RF Clf Accuracy':<30} {v2['RF_acc']:>10.4f} {v3['RF_acc']:>10.4f} {v3['RF_acc']-v2['RF_acc']:>+8.4f}")
print(f"{'RF Clf ROC-AUC':<30} {v2['RF_auc']:>10.4f} {v3['RF_auc']:>10.4f} {v3['RF_auc']-v2['RF_auc']:>+8.4f}")
print(f"\n  Baseline accuracy (majority-class) = {baseline_acc:.4f}")
print(f"  Baseline ROC-AUC (random)          = 0.5000")

# Verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)
reg_improved  = r2_v3 > r2_v2 and mae_v3 < mae_v2
lr_improved   = v3["LR_acc"] > v2["LR_acc"] or v3["LR_auc"] > v2["LR_auc"]
rf_improved   = v3["RF_acc"] > v2["RF_acc"] or v3["RF_auc"] > v2["RF_auc"]

print(f"  Regression (RF): news_count {'IMPROVED' if reg_improved else 'DID NOT improve'} R²/MAE")
print(f"  LogReg cls     : news_count {'IMPROVED' if lr_improved else 'DID NOT improve'} Accuracy/AUC")
print(f"  RF Classifier  : news_count {'IMPROVED' if rf_improved else 'DID NOT improve'} Accuracy/AUC")

any_gain = reg_improved or lr_improved or rf_improved
print(f"\n  Overall: news_count {'provides marginal signal' if any_gain else 'does NOT improve any model'}.")
print( "  Note: with only ~40 dated press releases across 10 years of daily data,")
print( "  news_count equals 1 on <2% of trading days and 0 almost everywhere,")
print( "  severely limiting its predictive power as a raw count feature.")
