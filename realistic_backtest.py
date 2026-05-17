"""
realistic_backtest.py

Part 1 — Transaction-cost-adjusted backtest at threshold 0.70
Part 2 — Walk-forward validation: expanding train window, fixed test blocks
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
THRESHOLD   = 0.70
COST_BPS    = 0.001          # 0.10% one-way transaction cost
BEST_PARAMS = {              # from randomised search
    "n_estimators":     254,
    "learning_rate":    0.03634245041729746,
    "max_depth":        4,
    "num_leaves":       17,
    "min_data_in_leaf": 70,
    "feature_fraction": 0.6931085361721216,
}
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

def max_drawdown(equity: pd.Series) -> float:
    return ((equity - equity.cummax()) / equity.cummax()).min()

def sharpe(daily_rets: pd.Series) -> float:
    s = daily_rets.std()
    return (daily_rets.mean() / s * np.sqrt(252)) if s > 0 else 0.0

def equity_metrics(daily_rets: pd.Series, position: pd.Series) -> dict:
    eq = (1 + daily_rets).cumprod()
    return {
        "total_return":  round(eq.iloc[-1] - 1, 4),
        "final_equity":  round(eq.iloc[-1],     4),
        "max_drawdown":  round(max_drawdown(eq), 4),
        "sharpe":        round(sharpe(daily_rets), 3),
        "days_invested": int(position.sum()),
        "invested_pct":  round(position.mean(), 4),
    }

# ---------------------------------------------------------------------------
# Full feature pipeline (same as return_features_model.py)
# ---------------------------------------------------------------------------
def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    def ema(s, n): return s.ewm(span=n, adjust=False).mean()

    df["ema_10"] = ema(df["adj_close"], 10)
    df["ema_20"] = ema(df["adj_close"], 20)
    df["ema_50"] = ema(df["adj_close"], 50)

    e12 = ema(df["adj_close"], 12); e26 = ema(df["adj_close"], 26)
    df["macd_line"]   = e12 - e26
    df["macd_signal"] = ema(df["macd_line"], 9)
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]

    delta = df["adj_close"].diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + ag / al.replace(0, np.nan))

    bm = df["adj_close"].rolling(20).mean()
    bs = df["adj_close"].rolling(20).std(ddof=0)
    br = (4 * bs).replace(0, np.nan)
    df["bb_mid_20"]   = bm
    df["bb_upper_20"] = bm + 2*bs
    df["bb_lower_20"] = bm - 2*bs
    df["bb_width_20"] = br / bm
    df["bb_pos_20"]   = (df["adj_close"] - bm) / br

    df["range_pct"]      = (df["high"] - df["low"]) / df["close"]
    df["close_open_pct"] = (df["close"] - df["open"]) / df["open"]
    df["gap_pct"]        = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["vol_change_1d"]  = df["volume"] / df["volume"].shift(1) - 1
    vs20 = df["volume"].rolling(20).std(ddof=0)
    df["vol_zscore_20"]  = (df["volume"] - df["vol_mean_20"]) / vs20.replace(0, np.nan)

    df["y_next_ret_20d"]  = df["adj_close"].shift(-20) / df["adj_close"] - 1
    df["y_next_ret_60d"]  = df["adj_close"].shift(-60) / df["adj_close"] - 1
    df["y_next_ret_252d"] = df["adj_close"].shift(-252) / df["adj_close"] - 1
    df["label_20d_up"]    = (df["y_next_ret_20d"] > 0).astype(int)

    drop_cols = ["y_next_ret_20d", "y_next_ret_60d", "y_next_ret_252d",
                 "label_20d_up"] + FEATURE_COLS
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    return df

def train_lgbm(X_tr, y_tr):
    m = lgb.LGBMClassifier(**BEST_PARAMS, is_unbalance=True,
                           random_state=42, n_jobs=1, verbose=-1)
    m.fit(X_tr, y_tr)
    return m

def apply_strategy(close: pd.Series, probs: np.ndarray,
                   thr: float, cost: float) -> pd.Series:
    """
    Return daily strategy returns after transaction costs.
    cost is applied (subtracted) on every day the position changes.
    """
    position   = pd.Series((probs >= thr).astype(int), index=close.index)
    ret_stock  = close.pct_change().fillna(0.0)
    ret_strat  = position.shift(1).fillna(0) * ret_stock
    # apply cost on change days
    trades     = position.diff().abs().fillna(0)
    ret_strat -= trades * cost
    return ret_strat, position

# ===========================================================================
# LOAD & PREPARE FULL DATASET
# ===========================================================================
raw = pd.read_csv("vu_pa_history_features.csv", skiprows=[1])
raw["date"] = pd.to_datetime(raw["date"])
raw = raw.sort_values("date").reset_index(drop=True)
num_cols = [c for c in raw.columns if c != "date"]
raw[num_cols] = raw[num_cols].apply(pd.to_numeric, errors="coerce")

df = build_features(raw)
print(f"Full dataset: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}"
      f"  ({len(df)} rows)\n")

# ===========================================================================
# PART 1 — TRANSACTION-COST BACKTEST (original 80/20 split, thr=0.70)
# ===========================================================================
print("=" * 62)
print("PART 1 — Backtest with 0.10% transaction cost  (thr=0.70)")
print("=" * 62)

split    = int(len(df) * 0.80)
X_all    = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_all    = df["label_20d_up"]
X_tr, X_te = X_all.iloc[:split], X_all.iloc[split:]
y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]

model_p1 = train_lgbm(X_tr, y_tr)
test      = df.iloc[split:].copy().reset_index(drop=True)
probs_te  = model_p1.predict_proba(X_te)[:, 1]

# --- no-cost baseline (for comparison) ---
ret_nc, pos_nc = apply_strategy(test["close"], probs_te, THRESHOLD, cost=0.0)
m_nc = equity_metrics(ret_nc, pos_nc)

# --- with cost ---
ret_c, pos_c = apply_strategy(test["close"], probs_te, THRESHOLD, cost=COST_BPS)
m_c = equity_metrics(ret_c, pos_c)

# --- buy & hold ---
bh_ret = test["close"].pct_change().fillna(0)
m_bh   = equity_metrics(bh_ret, pd.Series(np.ones(len(test))))

n_trades = int(pos_c.diff().abs().fillna(0).sum())

print(f"\n{'Metric':<22} {'Buy&Hold':>12} {'No cost':>12} {'0.10% cost':>12}")
print("-" * 60)
for k, label in [("total_return","Total return"),
                 ("max_drawdown","Max drawdown"),
                 ("sharpe",      "Sharpe ratio"),
                 ("invested_pct","% days invested")]:
    fmt = ".1%" if k in ("total_return","max_drawdown","invested_pct") else ".3f"
    print(f"{label:<22} {m_bh[k]:>12{fmt}} {m_nc[k]:>12{fmt}} {m_c[k]:>12{fmt}}")
print(f"{'Num trades':<22} {'—':>12} {n_trades:>12} {n_trades:>12}")
cost_drag = m_nc["total_return"] - m_c["total_return"]
print(f"\n  Total cost drag: {cost_drag:.2%}  ({n_trades} round-trip signals × ~0.10%)")

# --- equity curve plot ---
eq_bh  = (1 + bh_ret).cumprod()
eq_nc  = (1 + ret_nc).cumprod()
eq_c   = (1 + ret_c).cumprod()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(test["date"], eq_bh, label=f"Buy & Hold ({m_bh['total_return']:+.1%})",
         color="steelblue", lw=1.8)
ax1.plot(test["date"], eq_nc, label=f"Strategy, no cost ({m_nc['total_return']:+.1%})",
         color="forestgreen", lw=1.6, linestyle="--")
ax1.plot(test["date"], eq_c,  label=f"Strategy, 0.10% cost ({m_c['total_return']:+.1%})",
         color="tomato", lw=1.8)
ax1.axhline(1, color="grey", lw=0.8, linestyle=":")
ax1.set_ylabel("Equity (start = 1.0)"); ax1.set_title(
    f"VU.PA — LightGBM Strategy (thr={THRESHOLD}) with 0.10% Transaction Cost",
    fontsize=12, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.fill_between(test["date"], pos_c, step="pre",
                 color="tomato", alpha=0.45, label="Long position")
ax2.set_ylabel("Position"); ax2.set_xlabel("Date")
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["Cash", "Long"])
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_with_costs.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved → backtest_with_costs.png")

# ===========================================================================
# PART 2 — WALK-FORWARD VALIDATION
# ===========================================================================
print("\n\n" + "=" * 62)
print("PART 2 — Walk-Forward Validation")
print(f"  Expanding train window | test block = 250 rows (~1 yr) | thr={THRESHOLD}")
print("=" * 62)

TEST_BLOCK   = 250
MIN_TRAIN    = 800    # minimum rows needed before first test window
wf_rows      = []
window_num   = 0

train_end = MIN_TRAIN
while train_end + TEST_BLOCK <= len(df):
    test_start = train_end
    test_end   = min(train_end + TEST_BLOCK, len(df))

    X_wf_tr = X_all.iloc[:train_end]
    y_wf_tr = y_all.iloc[:train_end]
    X_wf_te = X_all.iloc[test_start:test_end]
    df_wf_te = df.iloc[test_start:test_end].copy().reset_index(drop=True)

    wf_model = train_lgbm(X_wf_tr, y_wf_tr)
    wf_probs = wf_model.predict_proba(X_wf_te)[:, 1]

    ret_wf, pos_wf = apply_strategy(df_wf_te["close"], wf_probs,
                                    THRESHOLD, cost=COST_BPS)
    m_wf = equity_metrics(ret_wf, pos_wf)

    bh_wf = df_wf_te["close"].pct_change().fillna(0)
    m_bh_wf = equity_metrics(bh_wf, pd.Series(np.ones(len(df_wf_te))))

    window_num += 1
    wf_rows.append({
        "window":         window_num,
        "train_rows":     train_end,
        "train_start":    df["date"].iloc[0].date(),
        "train_end":      df["date"].iloc[train_end - 1].date(),
        "test_start":     df["date"].iloc[test_start].date(),
        "test_end":       df["date"].iloc[test_end - 1].date(),
        "test_rows":      test_end - test_start,
        "total_return":   m_wf["total_return"],
        "max_drawdown":   m_wf["max_drawdown"],
        "sharpe":         m_wf["sharpe"],
        "days_invested":  m_wf["days_invested"],
        "invested_pct":   m_wf["invested_pct"],
        "bh_return":      m_bh_wf["total_return"],
        "excess_return":  m_wf["total_return"] - m_bh_wf["total_return"],
    })

    train_end += TEST_BLOCK   # expand training window

wf_df = pd.DataFrame(wf_rows)

# --- Print table ---
display_cols = ["window", "train_end", "test_start", "test_end",
                "total_return", "bh_return", "excess_return",
                "max_drawdown", "sharpe", "invested_pct"]
fmt_pct = {"total_return", "bh_return", "excess_return", "max_drawdown", "invested_pct"}

print()
header = (f"{'Win':>3}  {'Train end':>11}  {'Test start':>11}  {'Test end':>11}"
          f"  {'Ret':>7}  {'B&H':>7}  {'Excess':>7}"
          f"  {'MDD':>7}  {'Sharpe':>6}  {'%Inv':>5}")
print(header)
print("-" * len(header))
for _, r in wf_df.iterrows():
    print(f"{int(r['window']):>3}  {str(r['train_end']):>11}  "
          f"{str(r['test_start']):>11}  {str(r['test_end']):>11}"
          f"  {r['total_return']:>+7.1%}  {r['bh_return']:>+7.1%}  "
          f"{r['excess_return']:>+7.1%}"
          f"  {r['max_drawdown']:>7.1%}  {r['sharpe']:>6.3f}"
          f"  {r['invested_pct']:>4.0%}")

# --- Summary stats ---
print("\n" + "-" * len(header))
for agg_label, fn in [("Mean  ", wf_df.mean), ("Median", wf_df.median)]:
    agg = fn(numeric_only=True)
    print(f"{agg_label}                               "
          f"  {agg['total_return']:>+7.1%}  {agg['bh_return']:>+7.1%}"
          f"  {agg['excess_return']:>+7.1%}"
          f"  {agg['max_drawdown']:>7.1%}  {agg['sharpe']:>6.3f}"
          f"  {agg['invested_pct']:>4.0%}")

wins = (wf_df["excess_return"] > 0).sum()
print(f"\n  Strategy beat B&H in {wins}/{len(wf_df)} windows")

# --- Walk-forward plot ---
fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
fig2.suptitle(f"Walk-Forward Results by Window  (thr={THRESHOLD}, cost=0.10%)",
              fontsize=12, fontweight="bold")

wins_idx = wf_df["window"]

for ax, col, title, ylabel in [
    (axes[0], "total_return",  "Total Return",   "Return"),
    (axes[1], "max_drawdown",  "Max Drawdown",   "Drawdown"),
    (axes[2], "sharpe",        "Sharpe Ratio",   "Sharpe"),
]:
    vals = wf_df[col].values * (100 if col != "sharpe" else 1)
    bh   = (wf_df["bh_return"].values * 100) if col == "total_return" else None
    colors = ["forestgreen" if v >= 0 else "tomato" for v in vals]
    axes_twin = ax if col != "total_return" else ax
    ax.bar(wins_idx, vals, color=colors, edgecolor="white", alpha=0.85)
    if bh is not None:
        ax.bar(wins_idx, bh, color="steelblue", alpha=0.4,
               edgecolor="white", label="B&H")
        ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Window"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(wins_idx)
    unit = "%" if col != "sharpe" else ""
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}{unit}"))

plt.tight_layout()
plt.savefig("walkforward_results.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved → walkforward_results.png")
