"""
full_pipeline.py
================
End-to-end ML pipeline for VU.PA with market-context features.

Step 1  – Data download (yfinance)  ← falls back to local CSV if blocked
Step 2  – Base feature table
Step 3  – Feature engineering (no look-ahead)
          A. VU.PA technicals
          B. Market & sector context
          C. Liquidity & risk
          D. Regime flags
Step 4  – Target column  (20-day forward return / binary label)
Step 5  – Train / test split + LightGBM
Step 6  – Backtest: equity curve, threshold sweep, walk-forward
"""

# ── stdlib / third-party ──────────────────────────────────────────────────────
import warnings, sys
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — DATA DOWNLOAD")
print("=" * 65)

# Tickers
TICKER   = "VU.PA"
MKT_TK   = "^STOXX50E"   # EURO STOXX 50  (broad European market)
FR_TK    = "^FCHI"        # CAC 40          (French market)
SEC_TK   = "EXV3.DE"      # iShares STOXX Europe 600 Technology ETF

START    = "2014-01-01"
END      = "2026-01-01"
LOCAL_CSV = "vu_pa_history_features.csv"

def _try_yfinance():
    """Return (vu_ohlcv, mkt_close, fr_close, sec_close) or raise."""
    import yfinance as yf
    kw = dict(start=START, end=END, progress=False, auto_adjust=True)
    vu  = yf.download(TICKER,  **kw)
    mkt = yf.download(MKT_TK,  **kw)[["Close"]].rename(columns={"Close": "mkt_close"})
    fr  = yf.download(FR_TK,   **kw)[["Close"]].rename(columns={"Close": "fr_close"})
    sec = yf.download(SEC_TK,  **kw)[["Close"]].rename(columns={"Close": "sec_close"})
    if len(vu) < 100:
        raise RuntimeError("VU.PA returned no data — network blocked")
    # Flatten multi-index if present
    if isinstance(vu.columns, pd.MultiIndex):
        vu.columns = vu.columns.get_level_values(0)
    vu.index = pd.to_datetime(vu.index)
    mkt.index = pd.to_datetime(mkt.index); fr.index = pd.to_datetime(fr.index)
    sec.index = pd.to_datetime(sec.index)
    return vu, mkt, fr, sec

def _load_local_fallback():
    """
    Load VU.PA from the pre-built CSV and synthesise plausible index proxies.
    Synthetic series share broad VU.PA market direction but are smoothed and
    rescaled to realistic European index volatility (~0.8–1.3 % σ/day).
    Returns a single base DataFrame to avoid any index-alignment issues.
    """
    print("  [!] yfinance blocked — loading local CSV + building synthetic indices")
    raw = pd.read_csv(LOCAL_CSV, skiprows=[1])
    raw["Date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    num_cols = [c for c in raw.columns if c not in ("date", "Date")]
    raw[num_cols] = raw[num_cols].apply(pd.to_numeric, errors="coerce")

    # Build VU.PA OHLCV with a clean integer RangeIndex first; Date goes in as a column
    vu = pd.DataFrame({
        "Open":      raw["open"].values,
        "High":      raw["high"].values,
        "Low":       raw["low"].values,
        "Close":     raw["close"].values,
        "Adj Close": raw["adj_close"].values,
        "Volume":    raw["volume"].values,
    })
    vu.index = pd.DatetimeIndex(raw["Date"].values, name="Date")

    ret_vu = vu["Adj Close"].pct_change().fillna(0).values   # plain numpy array
    rng    = np.random.default_rng(42)
    idx    = vu.index   # shared DatetimeIndex

    def synth_index(ret_arr, smooth, vol_target, noise_scale, start_val=1000.0):
        s   = pd.Series(ret_arr, index=idx)
        sm  = s.ewm(span=smooth, adjust=False).mean().values
        ns  = rng.normal(0, noise_scale, len(sm))
        ir  = sm + ns
        std = ir.std()
        if std > 0:
            ir = ir * (vol_target / std)
        price = np.cumprod(1 + ir) * start_val
        return pd.Series(price, index=idx)

    vu["mkt_close"] = synth_index(ret_vu, smooth=15, vol_target=0.008, noise_scale=0.003)
    vu["fr_close"]  = synth_index(ret_vu, smooth=10, vol_target=0.010, noise_scale=0.004)
    vu["sec_close"] = synth_index(ret_vu, smooth=5,  vol_target=0.013, noise_scale=0.005)

    # Return as (vu_ohlcv_with_indices, dummy mkt/fr/sec already in vu)
    return vu, None, None, None

# Try live download; fall back to local
try:
    vu, mkt, fr, sec = _try_yfinance()
    DATA_SOURCE = "yfinance (live)"
except Exception as e:
    print(f"  yfinance unavailable ({e})")
    vu, mkt, fr, sec = _load_local_fallback()
    DATA_SOURCE = "local CSV + synthetic indices"

print(f"  Source : {DATA_SOURCE}")
print(f"  VU.PA  : {vu.index[0].date()} → {vu.index[-1].date()}  ({len(vu)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BASE FEATURE TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — BASE FEATURE TABLE")
print("=" * 65)

# Align all index series to VU.PA trading days (forward-fill gaps)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
IDX_COLS   = ["mkt_close", "fr_close", "sec_close"]

if mkt is None:
    # Fallback path: synthetic index columns already inside `vu`
    base = vu[OHLCV_COLS + IDX_COLS].copy()
else:
    # Live-download path: join separate index DataFrames onto VU.PA days
    base = vu[OHLCV_COLS].copy()
    base.index = pd.DatetimeIndex(base.index, name="Date")
    for src, col in [(mkt, "mkt_close"), (fr, "fr_close"), (sec, "sec_close")]:
        src.index = pd.DatetimeIndex(src.index, name="Date")
        base = base.join(src[[col]], how="left")
        base[col] = base[col].ffill()

base = base.dropna(subset=["Close", "mkt_close", "fr_close", "sec_close"])
base = base.sort_index()
print(f"  Base table: {len(base)} rows × {len(base.columns)} columns")
print(f"  Columns: {list(base.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING  (strictly no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — FEATURE ENGINEERING")
print("=" * 65)

df = base.copy()
C  = df["Adj Close"]   # adjusted close — used for all return/MA features
Cl = df["Close"]       # raw close       — used for volume-€ and display

# ── helpers ──────────────────────────────────────────────────────────────────
def ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def sma(s, n):  return s.rolling(n).mean()
def roll_std(s, n): return s.rolling(n).std(ddof=0)
def roll_ret(s, n): return s / s.shift(n) - 1   # n-day return (past only)

# ── A. VU.PA TECHNICALS ──────────────────────────────────────────────────────
# All MA/EMA features are expressed as RATIOS to current price (price / MA - 1)
# so they are stationary and don't encode the absolute time period.
print("  A. VU.PA technicals …")

# Returns
df["ret_1d"]  = C.pct_change()
df["ret_5d"]  = roll_ret(C, 5)
df["ret_20d"] = roll_ret(C, 20)
df["ret_60d"] = roll_ret(C, 60)

# Return lags
df["ret_1d_lag1"] = df["ret_1d"].shift(1)
df["ret_1d_lag2"] = df["ret_1d"].shift(2)
df["ret_1d_lag5"] = df["ret_1d"].shift(5)

# Price distance from MAs (ratio form — stationary, no time-period leakage)
_ma5   = sma(C, 5);   _ma20 = sma(C, 20)
_ma50  = sma(C, 50);  _ma200 = sma(C, 200)
_ema10 = ema(C, 10);  _ema20 = ema(C, 20); _ema50 = ema(C, 50)

df["dist_ma5"]    = C / _ma5   - 1
df["dist_ma20"]   = C / _ma20  - 1
df["dist_ma50"]   = C / _ma50  - 1
df["dist_ma200"]  = C / _ma200 - 1
df["dist_ema10"]  = C / _ema10 - 1
df["dist_ema20"]  = C / _ema20 - 1
df["dist_ema50"]  = C / _ema50 - 1

# MA cross ratios (short MA / long MA - 1)
df["ma5_vs_ma20"]   = _ma5  / _ma20  - 1
df["ma20_vs_ma50"]  = _ma20 / _ma50  - 1
df["ma50_vs_ma200"] = _ma50 / _ma200 - 1

# RSI-14
_d  = C.diff()
_ag = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
_al = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
df["rsi_14"] = 100 - 100 / (1 + _ag / _al.replace(0, np.nan))

# MACD — normalised by price so it's scale-invariant
_e12 = ema(C, 12); _e26 = ema(C, 26)
_macd_raw    = _e12 - _e26
_signal_raw  = ema(_macd_raw, 9)
df["macd_norm"]   = _macd_raw   / C   # MACD line / price
df["signal_norm"] = _signal_raw / C   # signal line / price
df["macd_hist_norm"] = (_macd_raw - _signal_raw) / C

# Bollinger Bands (already ratio-based)
_bm = sma(C, 20); _bs = roll_std(C, 20)
_br = (4 * _bs).replace(0, np.nan)
df["bb_width_20"] = _br / _bm
df["bb_pos_20"]   = (C - _bm) / _br

# Intraday price action
df["range_pct"]      = (df["High"] - df["Low"]) / Cl
df["close_open_pct"] = (Cl - df["Open"]) / df["Open"]
df["gap_pct"]        = (df["Open"] - Cl.shift(1)) / Cl.shift(1)

# ── B. MARKET & SECTOR CONTEXT ───────────────────────────────────────────────
print("  B. Market & sector context …")

for col, tag in [("mkt_close","mkt"), ("fr_close","fr"), ("sec_close","sec")]:
    s = df[col]
    df[f"ret_1d_{tag}"]  = s.pct_change()
    df[f"ret_5d_{tag}"]  = roll_ret(s, 5)
    df[f"ret_20d_{tag}"] = roll_ret(s, 20)
    df[f"ret_60d_{tag}"] = roll_ret(s, 60)

# Relative performance: VU.PA minus market / sector
for n, tag in [(5,"5d"), (20,"20d"), (60,"60d")]:
    df[f"rel_vs_mkt_{tag}"] = roll_ret(C, n) - roll_ret(df["mkt_close"], n)
    df[f"rel_vs_sec_{tag}"] = roll_ret(C, n) - roll_ret(df["sec_close"], n)
    df[f"rel_vs_fr_{tag}"]  = roll_ret(C, n) - roll_ret(df["fr_close"], n)

# Beta proxy: 60-day correlation of VU.PA daily returns with market returns
df["corr_60d_mkt"] = (df["ret_1d"].rolling(60)
                       .corr(df["ret_1d_mkt"]))

# ── C. LIQUIDITY & RISK ───────────────────────────────────────────────────────
print("  C. Liquidity & risk …")

df["vol_20d_avg"]    = df["Volume"].rolling(20).mean()
df["euro_vol_20d"]   = (Cl * df["Volume"]).rolling(20).mean()
df["vu_vol_20d"]     = roll_std(df["ret_1d"], 20)       # VU.PA 20d return vol
df["mkt_vol_20d"]    = roll_std(df["ret_1d_mkt"], 20)   # market 20d return vol
df["vol_change_1d"]  = df["Volume"] / df["Volume"].shift(1) - 1
df["vol_zscore_20"]  = ((df["Volume"] - df["vol_20d_avg"])
                         / roll_std(df["Volume"], 20).replace(0, np.nan))

# ── D. REGIME FLAGS ───────────────────────────────────────────────────────────
print("  D. Regime flags …")

df["stock_above_200dma"]    = (df["dist_ma200"] > 0).astype(int)

mkt_rolling_max             = df["mkt_close"].cummax()
df["market_drawdown_gt20"]  = ((df["mkt_close"] / mkt_rolling_max - 1) < -0.20).astype(int)

mkt_vol_median              = df["mkt_vol_20d"].expanding().median()
df["high_vol_regime"]       = (df["mkt_vol_20d"] > mkt_vol_median).astype(int)

# (dist_ma50 already captures close vs 50dma in ratio form)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TARGET COLUMN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — TARGET COLUMN")
print("=" * 65)

df["forward_ret_20"] = C.shift(-20) / C - 1
df["target_up_20"]   = (df["forward_ret_20"] > 0).astype(int)

# Final feature list — everything except raw prices, targets, and date
EXCLUDE = {"Open","High","Low","Close","Adj Close","Volume",
           "mkt_close","fr_close","sec_close",
           "forward_ret_20","target_up_20"}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE]

# Drop rows with any NaN in features or target (indicator warm-up + 20d lookahead)
keep_cols = FEATURE_COLS + ["target_up_20", "forward_ret_20", "Close"]
df_model  = df[keep_cols].dropna().copy()
df_model.index = pd.to_datetime(df_model.index)

print(f"  Rows after NaN drop : {len(df_model)}")
print(f"  Feature count       : {len(FEATURE_COLS)}")
print(f"  Date range          : {df_model.index[0].date()} → {df_model.index[-1].date()}")
print(f"  Target balance      : {df_model['target_up_20'].mean():.1%} positive (up days)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN / TEST SPLIT + LIGHTGBM
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — MODEL TRAINING")
print("=" * 65)

# Time-based split: last 2 years as test
TEST_CUTOFF = df_model.index.max() - pd.DateOffset(years=2)
train_mask  = df_model.index <= TEST_CUTOFF
test_mask   = df_model.index >  TEST_CUTOFF

X_tr  = df_model.loc[train_mask, FEATURE_COLS].replace([np.inf,-np.inf], np.nan).fillna(0)
y_tr  = df_model.loc[train_mask, "target_up_20"]
X_te  = df_model.loc[test_mask,  FEATURE_COLS].replace([np.inf,-np.inf], np.nan).fillna(0)
y_te  = df_model.loc[test_mask,  "target_up_20"]

print(f"  Train: {X_tr.index[0].date()} → {X_tr.index[-1].date()}  ({len(X_tr)} rows)")
print(f"  Test : {X_te.index[0].date()} → {X_te.index[-1].date()}  ({len(X_te)} rows)")

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=20,         # shallow — reduces overfitting on small dataset
    min_data_in_leaf=50,   # require more samples per leaf
    feature_fraction=0.7,
    subsample=0.8,
    reg_alpha=0.1,         # L1 regularisation
    reg_lambda=1.0,        # L2 regularisation
    is_unbalance=True,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)
model.fit(X_tr, y_tr)

prob_te       = model.predict_proba(X_te)[:, 1]
prob_tr       = model.predict_proba(X_tr)[:, 1]
train_auc     = roc_auc_score(y_tr, prob_tr)
test_auc      = roc_auc_score(y_te, prob_te)
baseline_acc  = float(y_te.mean())
baseline_acc  = max(baseline_acc, 1 - baseline_acc)

print(f"  Train ROC-AUC : {train_auc:.4f}")
print(f"  Test  ROC-AUC : {test_auc:.4f}")
print(f"  Majority-class baseline acc : {baseline_acc:.4f}")

# Feature importance (top 15)
fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\n  Top 15 features by importance:")
for name, val in fi.head(15).items():
    print(f"    {name:<30} {val:>6.0f}")

# Attach predictions to test slice
test_df = df_model.loc[test_mask].copy()
test_df["prob_up_tuned"] = prob_te

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — BACKTEST & REPORTING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — BACKTEST & REPORTING")
print("=" * 65)

THRESHOLD  = 0.70
COST_BPS   = 0.001    # 0.10% per position change

# ── shared helpers ────────────────────────────────────────────────────────────
def max_dd(eq):
    return ((eq - eq.cummax()) / eq.cummax()).min()

def sharpe(r):
    return (r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

def run_strategy(close_s, probs, thr, cost=0.0):
    pos       = pd.Series((probs >= thr).astype(float), index=close_s.index)
    ret_stock = close_s.pct_change().fillna(0.0)
    ret_strat = pos.shift(1).fillna(0) * ret_stock
    trades    = pos.diff().abs().fillna(0)
    ret_strat -= trades * cost
    eq        = (1 + ret_strat).cumprod()
    return {
        "ret_strat":    ret_strat,
        "equity":       eq,
        "position":     pos,
        "total_return": eq.iloc[-1] - 1,
        "max_drawdown": max_dd(eq),
        "sharpe":       sharpe(ret_strat),
        "days_invested":int(pos.sum()),
        "invested_pct": pos.mean(),
        "n_trades":     int(trades.sum()),
    }

def bh_stats(close_s):
    r  = close_s.pct_change().fillna(0)
    eq = (1 + r).cumprod()
    return {"equity": eq, "ret": r,
            "total_return": eq.iloc[-1]-1,
            "max_drawdown": max_dd(eq),
            "sharpe": sharpe(r)}

# ── 6A. EQUITY CURVE WITH COSTS ───────────────────────────────────────────────
print("\n  6A. Equity curve (thr=0.70, cost=0.10%) …")

s    = run_strategy(test_df["Close"], prob_te, THRESHOLD, COST_BPS)
s_nc = run_strategy(test_df["Close"], prob_te, THRESHOLD, cost=0.0)
bh   = bh_stats(test_df["Close"])

print(f"\n  {'Metric':<22} {'Buy&Hold':>10} {'No cost':>10} {'0.10% cost':>10}")
print("  " + "-" * 55)
for k, label in [("total_return","Total return"),("max_drawdown","Max drawdown"),
                  ("sharpe","Sharpe ratio")]:
    fmt = ".1%" if k != "sharpe" else ".3f"
    bh_v  = bh[k]
    nc_v  = s_nc[k]
    c_v   = s[k]
    print(f"  {label:<22} {bh_v:>10{fmt}} {nc_v:>10{fmt}} {c_v:>10{fmt}}")
print(f"  {'Num trades':<22} {'—':>10} {s['n_trades']:>10} {s['n_trades']:>10}")

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                 gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(test_df.index, bh["equity"],   label=f"Buy & Hold ({bh['total_return']:+.1%})",
         color="steelblue", lw=1.8)
ax1.plot(test_df.index, s_nc["equity"], label=f"Strategy no cost ({s_nc['total_return']:+.1%})",
         color="forestgreen", lw=1.5, linestyle="--")
ax1.plot(test_df.index, s["equity"],    label=f"Strategy 0.10% cost ({s['total_return']:+.1%})",
         color="tomato", lw=1.8)
ax1.axhline(1, color="grey", lw=0.7, linestyle=":"); ax1.grid(alpha=0.3)
ax1.set_ylabel("Equity (start=1.0)")
ax1.set_title(f"VU.PA — LightGBM Strategy vs Buy & Hold  [{DATA_SOURCE}]",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax2.fill_between(test_df.index, s["position"], step="pre",
                 color="tomato", alpha=0.45, label="Long")
ax2.set_yticks([0,1]); ax2.set_yticklabels(["Cash","Long"])
ax2.set_ylabel("Position"); ax2.set_xlabel("Date")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("equity_curve_v2.png", dpi=150, bbox_inches="tight")
print("  → equity_curve_v2.png")

# ── 6B. THRESHOLD ROBUSTNESS SWEEP ───────────────────────────────────────────
print("\n  6B. Threshold robustness sweep (0.50 → 0.80) …")

thr_vals = np.arange(0.50, 0.81, 0.05)
thr_rows = []
for thr in thr_vals:
    m = run_strategy(test_df["Close"], prob_te, thr, COST_BPS)
    thr_rows.append({"threshold": thr,
                     "total_return": m["total_return"],
                     "max_drawdown": m["max_drawdown"],
                     "sharpe":       m["sharpe"],
                     "invested_pct": m["invested_pct"],
                     "n_trades":     m["n_trades"]})
thr_df = pd.DataFrame(thr_rows)

print(f"\n  {'Thr':>5} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8} "
      f"{'%Inv':>7} {'#Trades':>8}")
print("  " + "-"*50)
print(f"  {'B&H':>5} {bh['total_return']:>9.1%} {bh['max_drawdown']:>9.1%} "
      f"{bh['sharpe']:>8.3f} {'100%':>7}  {'—':>8}")
print("  " + "-"*50)
for _, r in thr_df.iterrows():
    print(f"  {r['threshold']:>5.2f} {r['total_return']:>9.1%} "
          f"{r['max_drawdown']:>9.1%} {r['sharpe']:>8.3f} "
          f"{r['invested_pct']:>7.1%} {int(r['n_trades']):>8}")

# 4-panel plot
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
fig2.suptitle(f"Threshold Robustness  [{DATA_SOURCE}]",
              fontsize=12, fontweight="bold")
panels = [
    (axes[0,0], "total_return",  "Total Return (%)",       True),
    (axes[0,1], "max_drawdown",  "Max Drawdown (%)",       True),
    (axes[1,0], "sharpe",        "Sharpe Ratio",           False),
    (axes[1,1], "invested_pct",  "% Days Invested",        True),
]
bh_ref = {"total_return": bh["total_return"]*100, "max_drawdown": bh["max_drawdown"]*100,
          "sharpe": bh["sharpe"], "invested_pct": 100.0}
for ax, col, ylabel, as_pct in panels:
    vals = thr_df[col].values * (100 if as_pct else 1)
    ax.bar(thr_df["threshold"], vals, width=0.03,
           color=["forestgreen" if v >= 0 else "tomato" for v in vals],
           edgecolor="white", alpha=0.85)
    ax.axhline(bh_ref[col], color="steelblue", linestyle="--", lw=1.2,
               label=f"B&H {bh_ref[col]:.1f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel(ylabel)
    ax.set_title(ylabel); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(thr_df["threshold"])
    ax.set_xticklabels([f"{t:.2f}" for t in thr_df["threshold"]], fontsize=8)
plt.tight_layout()
plt.savefig("threshold_robustness_v2.png", dpi=150, bbox_inches="tight")
print("  → threshold_robustness_v2.png")

# ── 6C. WALK-FORWARD VALIDATION ───────────────────────────────────────────────
print("\n  6C. Walk-forward validation …")

TEST_BLOCK = 250
MIN_TRAIN  = 600
X_all = df_model[FEATURE_COLS].replace([np.inf,-np.inf], np.nan).fillna(0)
y_all = df_model["target_up_20"]
close_all = df_model["Close"]
dates_all = df_model.index

wf_rows = []
train_end = MIN_TRAIN
wnum = 0
while train_end + TEST_BLOCK <= len(df_model):
    ts  = train_end
    te  = min(train_end + TEST_BLOCK, len(df_model))
    wnum += 1

    wf_m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        num_leaves=20, min_data_in_leaf=50, feature_fraction=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        is_unbalance=True, random_state=42, n_jobs=1, verbose=-1,
    )
    wf_m.fit(X_all.iloc[:ts], y_all.iloc[:ts])
    wf_probs = wf_m.predict_proba(X_all.iloc[ts:te])[:, 1]
    wf_close = close_all.iloc[ts:te]

    strat = run_strategy(wf_close, wf_probs, THRESHOLD, COST_BPS)
    bh_w  = bh_stats(wf_close)

    wf_rows.append({
        "window":       wnum,
        "train_end":    dates_all[ts-1].date(),
        "test_start":   dates_all[ts].date(),
        "test_end":     dates_all[te-1].date(),
        "strat_ret":    strat["total_return"],
        "bh_ret":       bh_w["total_return"],
        "excess":       strat["total_return"] - bh_w["total_return"],
        "max_drawdown": strat["max_drawdown"],
        "sharpe":       strat["sharpe"],
        "invested_pct": strat["invested_pct"],
    })
    train_end += TEST_BLOCK

wf_df = pd.DataFrame(wf_rows)
hdr = (f"  {'W':>2}  {'Train end':>11}  {'Test start':>11}  {'Test end':>11}"
       f"  {'Ret':>7}  {'B&H':>7}  {'Excess':>7}  {'MDD':>7}  {'Sharpe':>6}  {'%Inv':>5}")
print(hdr); print("  " + "-"*(len(hdr)-2))
for _, r in wf_df.iterrows():
    print(f"  {int(r['window']):>2}  {str(r['train_end']):>11}  "
          f"{str(r['test_start']):>11}  {str(r['test_end']):>11}"
          f"  {r['strat_ret']:>+7.1%}  {r['bh_ret']:>+7.1%}"
          f"  {r['excess']:>+7.1%}  {r['max_drawdown']:>7.1%}"
          f"  {r['sharpe']:>6.3f}  {r['invested_pct']:>4.0%}")

print("  " + "-"*(len(hdr)-2))
for lab, fn in [("Mean  ", wf_df.mean), ("Median", wf_df.median)]:
    a = fn(numeric_only=True)
    print(f"  {lab}                             "
          f"  {a['strat_ret']:>+7.1%}  {a['bh_ret']:>+7.1%}"
          f"  {a['excess']:>+7.1%}  {a['max_drawdown']:>7.1%}"
          f"  {a['sharpe']:>6.3f}  {a['invested_pct']:>4.0%}")

wins = (wf_df["excess"] > 0).sum()
print(f"\n  Strategy beat B&H in {wins}/{len(wf_df)} windows")

# Walk-forward 3-panel plot
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
fig3.suptitle(f"Walk-Forward Results (thr={THRESHOLD}, cost=0.10%)  [{DATA_SOURCE}]",
              fontsize=11, fontweight="bold")
wf_wins = wf_df["window"]
for ax, col, title, is_pct in [
    (axes3[0], "strat_ret",    "Total Return",  True),
    (axes3[1], "max_drawdown", "Max Drawdown",  True),
    (axes3[2], "sharpe",       "Sharpe Ratio",  False),
]:
    vals = wf_df[col].values * (100 if is_pct else 1)
    bh_v = (wf_df["bh_ret"].values * 100) if col == "strat_ret" else None
    colors = ["forestgreen" if v >= 0 else "tomato" for v in vals]
    ax.bar(wf_wins, vals, color=colors, edgecolor="white", alpha=0.85)
    if bh_v is not None:
        ax.bar(wf_wins, bh_v, color="steelblue", alpha=0.35,
               edgecolor="white", label="B&H"); ax.legend(fontsize=8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Window"); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(wf_wins)
    unit = "%" if is_pct else ""
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}{unit}"))
plt.tight_layout()
plt.savefig("walkforward_v2.png", dpi=150, bbox_inches="tight")
print("  → walkforward_v2.png")

# ── 6D. SAVE FULL FEATURE + PREDICTION CSV ────────────────────────────────────
out = df_model.copy()
out["prob_up_tuned"] = np.nan
out.loc[test_mask, "prob_up_tuned"] = prob_te
out["is_test"] = test_mask.astype(int)
out["data_source"] = DATA_SOURCE
out.to_csv("vu_pa_full_pipeline.csv")
print(f"\n  → vu_pa_full_pipeline.csv  ({len(out)} rows, {len(out.columns)} cols)")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)
print(f"  Features      : {len(FEATURE_COLS)}")
print(f"  Test AUC      : {test_auc:.4f}")
print(f"  Data source   : {DATA_SOURCE}")
print(f"  Outputs       : equity_curve_v2.png  threshold_robustness_v2.png")
print(f"                  walkforward_v2.png   vu_pa_full_pipeline.csv")
