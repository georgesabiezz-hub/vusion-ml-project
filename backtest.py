"""
backtest.py
Rebuilds the full feature pipeline, trains the tuned LightGBM classifier,
saves the test-set predictions to CSV, then runs the equity-curve backtest.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import lightgbm as lgb

DATA_PATH = "vu_pa_history_features.csv"

# =============================================================================
# 1. LOAD & FEATURE ENGINEERING  (same as return_features_model.py)
# =============================================================================
df = pd.read_csv(DATA_PATH, skiprows=[1])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
num_cols = [c for c in df.columns if c != "date"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

def ema(s, span): return s.ewm(span=span, adjust=False).mean()

df["ema_10"] = ema(df["adj_close"], 10)
df["ema_20"] = ema(df["adj_close"], 20)
df["ema_50"] = ema(df["adj_close"], 50)

_e12 = ema(df["adj_close"], 12); _e26 = ema(df["adj_close"], 26)
df["macd_line"]   = _e12 - _e26
df["macd_signal"] = ema(df["macd_line"], 9)
df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

def rsi(s, p=14):
    d = s.diff()
    ag = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))

df["rsi_14"] = rsi(df["adj_close"])

_bm = df["adj_close"].rolling(20).mean()
_bs = df["adj_close"].rolling(20).std(ddof=0)
_br = (_bm + 2*_bs - (_bm - 2*_bs)).replace(0, np.nan)
df["bb_mid_20"]   = _bm
df["bb_upper_20"] = _bm + 2*_bs
df["bb_lower_20"] = _bm - 2*_bs
df["bb_width_20"] = _br / _bm
df["bb_pos_20"]   = (df["adj_close"] - _bm) / _br

df["range_pct"]      = (df["high"] - df["low"]) / df["close"]
df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
df["vol_change_1d"]  = df["volume"] / df["volume"].shift(1) - 1
_vs20 = df["volume"].rolling(20).std(ddof=0)
df["vol_zscore_20"]  = (df["volume"] - df["vol_mean_20"]) / _vs20.replace(0, np.nan)

df["y_next_ret_20d"]  = df["adj_close"].shift(-20) / df["adj_close"] - 1
df["y_next_ret_60d"]  = df["adj_close"].shift(-60) / df["adj_close"] - 1
df["y_next_ret_252d"] = df["adj_close"].shift(-252) / df["adj_close"] - 1

df = df.dropna(subset=[
    "y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d",
    "ema_10", "ema_20", "ema_50",
    "macd_line", "macd_signal", "macd_hist", "rsi_14",
    "bb_width_20", "bb_pos_20",
    "range_pct", "close_open_pct", "gap_pct",
    "vol_change_1d", "vol_zscore_20",
]).reset_index(drop=True)

df["label_20d_up"] = (df["y_next_ret_20d"] > 0).astype(int)

FEATURE_COLS = [
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

split = int(len(df) * 0.80)
X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["label_20d_up"]
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

# =============================================================================
# 2. TRAIN TUNED LIGHTGBM  (best params from randomized search)
# =============================================================================
BEST_PARAMS = {
    "n_estimators":     254,
    "learning_rate":    0.03634245041729746,
    "max_depth":        4,
    "num_leaves":       17,
    "min_data_in_leaf": 70,
    "feature_fraction": 0.6931085361721216,
}

print("Training tuned LightGBM on train set …")
lgb_tuned = lgb.LGBMClassifier(
    **BEST_PARAMS,
    is_unbalance=True,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)
lgb_tuned.fit(X_tr, y_tr)

# Probabilities for the entire dataset (train + test)
prob_all = lgb_tuned.predict_proba(X)[:, 1]

# =============================================================================
# 3. BUILD & SAVE THE PREDICTIONS CSV
# =============================================================================
out = pd.DataFrame({
    "Date":          df["date"],
    "Close":         df["close"],
    "adj_close":     df["adj_close"],
    "prob_up_tuned": prob_all,
    "label_20d_up":  df["label_20d_up"],
    "is_test":       ([0] * split + [1] * (len(df) - split)),
})
out.to_csv("vu_pa_history_features_with_preds.csv", index=False)
print(f"Saved vu_pa_history_features_with_preds.csv  ({len(out)} rows)")

# =============================================================================
# 4. BACKTEST — user's logic, adapted for column names + file saving
# =============================================================================
df_bt = pd.read_csv("vu_pa_history_features_with_preds.csv", parse_dates=["Date"])
df_bt = df_bt.sort_values("Date")

test = df_bt[df_bt["is_test"] == 1].copy().reset_index(drop=True)

THRESHOLD = 0.6
test["position"] = (test["prob_up_tuned"] >= THRESHOLD).astype(int)
test["ret_stock"]    = test["Close"].pct_change().fillna(0.0)
test["ret_strategy"] = (test["position"].shift(1).fillna(0) * test["ret_stock"])
test["equity_stock"]    = (1 + test["ret_stock"]).cumprod()
test["equity_strategy"] = (1 + test["ret_strategy"]).cumprod()

final_stock    = test["equity_stock"].iloc[-1]
final_strategy = test["equity_strategy"].iloc[-1]

# --- Additional stats ---
n_days       = len(test)
n_invested   = test["position"].sum()
pct_invested = n_invested / n_days

# Annualised return (252 trading days)
ann_stock    = final_stock    ** (252 / n_days) - 1
ann_strategy = final_strategy ** (252 / n_days) - 1

# Max drawdown helper
def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return dd.min()

mdd_stock    = max_drawdown(test["equity_stock"])
mdd_strategy = max_drawdown(test["equity_strategy"])

# Sharpe (annualised, rf=0)
sharpe_stock    = (test["ret_stock"].mean()    / test["ret_stock"].std())    * (252**0.5)
sharpe_strategy = (test["ret_strategy"].mean() / test["ret_strategy"].std()) * (252**0.5)

print("\n" + "="*55)
print(f"BACKTEST RESULTS  (test set: {test['Date'].iloc[0].date()} → {test['Date'].iloc[-1].date()})")
print("="*55)
print(f"{'Metric':<28} {'Buy&Hold':>10} {'Strategy':>10}")
print("-"*50)
print(f"{'Total return':<28} {final_stock-1:>9.1%} {final_strategy-1:>9.1%}")
print(f"{'Final equity multiple':<28} {final_stock:>10.3f} {final_strategy:>10.3f}")
print(f"{'Ann. return (approx)':<28} {ann_stock:>9.1%} {ann_strategy:>9.1%}")
print(f"{'Max drawdown':<28} {mdd_stock:>9.1%} {mdd_strategy:>9.1%}")
print(f"{'Sharpe ratio (rf=0)':<28} {sharpe_stock:>10.3f} {sharpe_strategy:>10.3f}")
print(f"{'Days invested':<28} {'N/A':>10} {n_invested:>10} ({pct_invested:.0%})")
print(f"{'Threshold':<28} {'—':>10} {THRESHOLD:>10.1f}")

# =============================================================================
# 5. PLOT
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={"height_ratios": [3, 1]})

# Top panel: equity curves
ax1.plot(test["Date"], test["equity_stock"],    label="Buy & Hold VU.PA",
         color="#1f77b4", linewidth=1.8)
ax1.plot(test["Date"], test["equity_strategy"], label=f"LightGBM strategy (thr≥{THRESHOLD})",
         color="#d62728", linewidth=1.8)
ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
ax1.set_ylabel("Equity (starting at 1.0)", fontsize=11)
ax1.set_title("VU.PA — Tuned LightGBM Strategy vs Buy & Hold (Test Set)", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Bottom panel: position (0/1) as a shaded area
ax2.fill_between(test["Date"], test["position"], step="pre",
                 alpha=0.5, color="#2ca02c", label="In market (position=1)")
ax2.set_ylabel("Position", fontsize=10)
ax2.set_xlabel("Date", fontsize=11)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["Cash", "Long"])
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_equity_curve.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → backtest_equity_curve.png")
