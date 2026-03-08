"""
Primetrade.ai – Trader Performance vs Market Sentiment
Data Science Intern Assignment – Round 0
Author: Candidate
"""

# ─────────────────────────────────────────────────────────
# 0. Imports & Config
# ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# ── visual style ──────────────────────────────────────────
COLORS = {
    "fear":        "#e05c5c",
    "greed":       "#4caf7d",
    "neutral":     "#a0a0a0",
    "extreme_fear":"#b71c1c",
    "extreme_greed":"#1b5e20",
    "blue":        "#3b7dd8",
    "orange":      "#f5a623",
    "purple":      "#7b52ab",
    "bg":          "#f9f9f9",
}

plt.rcParams.update({
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor":   COLORS["bg"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

OUT = "charts"
os.makedirs(OUT, exist_ok=True)

def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  ✓  saved  {OUT}/{name}.png")


# ─────────────────────────────────────────────────────────
# PART A – Data Preparation
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART A  –  DATA PREPARATION")
print("="*60)

# ── A1. Load raw data ─────────────────────────────────────
trades_raw = pd.read_csv("data/historical_data.csv", low_memory=False)
fg_raw     = pd.read_csv("data/fear_greed_index.csv")

print(f"\nTrades   : {trades_raw.shape[0]:,} rows × {trades_raw.shape[1]} cols")
print(f"Fear/Greed: {fg_raw.shape[0]:,} rows × {fg_raw.shape[1]} cols")

# ── A2. Fear / Greed cleaning ─────────────────────────────
fg = fg_raw.copy()
fg["date"] = pd.to_datetime(fg["date"]).dt.date

# collapse 5-class into 3 for cleaner analysis
sentiment_map = {
    "Extreme Fear": "Fear",
    "Fear":         "Fear",
    "Neutral":      "Neutral",
    "Greed":        "Greed",
    "Extreme Greed":"Greed",
}
fg["sentiment_3"]  = fg["classification"].map(sentiment_map)
fg["sentiment_bin"] = fg["classification"].apply(
    lambda x: "Fear" if "Fear" in x else ("Greed" if "Greed" in x else "Neutral")
)
fg.drop_duplicates(subset="date", keep="last", inplace=True)
print(f"\nFear/Greed after dedup : {fg.shape[0]} rows")
print("Sentiment distribution:\n", fg["classification"].value_counts().to_string())

# ── A3. Trades cleaning ───────────────────────────────────
trades = trades_raw.copy()

# parse date
trades["date"] = pd.to_datetime(
    trades["Timestamp IST"], format="%d-%m-%Y %H:%M", dayfirst=True
).dt.date

# numeric coercion
for col in ["Execution Price", "Size Tokens", "Size USD",
            "Closed PnL", "Fee", "Start Position"]:
    trades[col] = pd.to_numeric(trades[col], errors="coerce")

# duplicates
dup_before = trades.duplicated().sum()
trades.drop_duplicates(inplace=True)
print(f"\nTrades duplicates removed : {dup_before}")
print(f"Missing values per column:\n{trades.isnull().sum()[trades.isnull().sum()>0].to_string()}")

# infer leverage (proxy) — only where Start Position != 0
trades["leverage_proxy"] = np.where(
    trades["Start Position"].abs() > 0,
    (trades["Size USD"] / trades["Start Position"].abs()).clip(0, 125),
    np.nan,
)

# trade direction flag
trades["is_long"] = trades["Side"].str.upper() == "BUY"

# closed trade flag (non-zero PnL → closing trade)
trades["is_closing"] = trades["Closed PnL"] != 0

print(f"\nTrades date range  : {trades['date'].min()} → {trades['date'].max()}")
print(f"Unique accounts    : {trades['Account'].nunique()}")
print(f"Unique coins       : {trades['Coin'].nunique()}")
print(f"Closing trades (PnL≠0) : {trades['is_closing'].sum():,}")

# ── A4. Build daily metrics per account ──────────────────
daily = (
    trades.groupby(["Account", "date"])
    .agg(
        trade_count    = ("Closed PnL",      "size"),
        total_pnl      = ("Closed PnL",      "sum"),
        mean_pnl       = ("Closed PnL",      "mean"),
        win_count      = ("Closed PnL",      lambda x: (x > 0).sum()),
        loss_count     = ("Closed PnL",      lambda x: (x < 0).sum()),
        total_vol_usd  = ("Size USD",         "sum"),
        mean_size_usd  = ("Size USD",         "mean"),
        long_count     = ("is_long",          "sum"),
        short_count    = ("is_long",          lambda x: (~x).sum()),
        mean_leverage  = ("leverage_proxy",   "mean"),
        max_leverage   = ("leverage_proxy",   "max"),
    )
    .reset_index()
)

daily["win_rate"]    = daily["win_count"]  / (daily["win_count"] + daily["loss_count"]).replace(0, np.nan)
daily["ls_ratio"]   = daily["long_count"] / (daily["long_count"] + daily["short_count"]).replace(0, np.nan)
daily["profit_flag"] = (daily["total_pnl"] > 0).astype(int)

# ── A5. Merge with sentiment ──────────────────────────────
fg["date"] = pd.to_datetime(fg["date"]).dt.date
daily["date"] = pd.to_datetime(daily["date"]).dt.date

merged = daily.merge(
    fg[["date", "classification", "sentiment_bin", "sentiment_3", "value"]],
    on="date", how="inner"
)

print(f"\nMerged dataset : {merged.shape[0]:,} rows")
print("Sentiment distribution in merged data:\n",
      merged["sentiment_bin"].value_counts().to_string())


# ─────────────────────────────────────────────────────────
# PART B – Analysis
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PART B  –  ANALYSIS")
print("="*60)

# ── colour palette for sentiment ─────────────────────────
SENT_COLORS = {
    "Fear":    COLORS["fear"],
    "Neutral": COLORS["neutral"],
    "Greed":   COLORS["greed"],
}

# ════════════════════════════════════════════════════════
# CHART 1  –  Mean PnL | Fear vs Greed days (bar)
# ════════════════════════════════════════════════════════
print("\n[B1] PnL vs Sentiment")
pnl_sent = (
    merged.groupby("sentiment_bin")["total_pnl"]
    .agg(["mean", "median", "std", "count"])
    .reindex(["Fear", "Neutral", "Greed"])
)
print(pnl_sent.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# mean PnL bar
ax = axes[0]
bars = ax.bar(
    pnl_sent.index, pnl_sent["mean"],
    color=[SENT_COLORS[s] for s in pnl_sent.index],
    edgecolor="white", linewidth=1.2, width=0.55
)
ax.axhline(0, color="#555", linewidth=0.9, linestyle="--")
for bar, (_, row) in zip(bars, pnl_sent.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (1 if row["mean"] >= 0 else -3),
            f"${row['mean']:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Mean Daily PnL by Market Sentiment")
ax.set_ylabel("Mean PnL (USD)")
ax.set_xlabel("Sentiment")

# median PnL bar
ax = axes[1]
bars2 = ax.bar(
    pnl_sent.index, pnl_sent["median"],
    color=[SENT_COLORS[s] for s in pnl_sent.index],
    edgecolor="white", linewidth=1.2, width=0.55
)
ax.axhline(0, color="#555", linewidth=0.9, linestyle="--")
for bar, (_, row) in zip(bars2, pnl_sent.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2,
            row["median"] + (0.3 if row["median"] >= 0 else -1),
            f"${row['median']:,.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Median Daily PnL by Market Sentiment")
ax.set_ylabel("Median PnL (USD)")
ax.set_xlabel("Sentiment")

save(fig, "01_pnl_by_sentiment")

# ════════════════════════════════════════════════════════
# CHART 2  –  Win Rate | Sentiment (violin + box combo)
# ════════════════════════════════════════════════════════
print("\n[B2] Win Rate vs Sentiment")
wr_sent = merged.groupby("sentiment_bin")["win_rate"].describe()
print(wr_sent.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
order = ["Fear", "Neutral", "Greed"]
pal   = {s: SENT_COLORS[s] for s in order}
sns.violinplot(
    data=merged.dropna(subset=["win_rate"]),
    x="sentiment_bin", y="win_rate",
    palette=pal, order=order, inner="box",
    linewidth=1.2, saturation=0.8, ax=ax
)
ax.set_title("Win Rate Distribution: Fear vs Neutral vs Greed Days")
ax.set_ylabel("Daily Win Rate")
ax.set_xlabel("Sentiment")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
save(fig, "02_winrate_sentiment_violin")

# ════════════════════════════════════════════════════════
# CHART 3  –  Leverage behavior by Sentiment
# ════════════════════════════════════════════════════════
print("\n[B3] Leverage vs Sentiment")
lev_sent = merged.groupby("sentiment_bin")["mean_leverage"].describe()
print(lev_sent.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Box plot
ax = axes[0]
sns.boxplot(
    data=merged.dropna(subset=["mean_leverage"]),
    x="sentiment_bin", y="mean_leverage",
    palette=pal, order=order,
    width=0.45, linewidth=1.2,
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    ax=ax
)
ax.set_title("Leverage Distribution by Sentiment")
ax.set_ylabel("Mean Leverage (proxy)")
ax.set_xlabel("Sentiment")

# Mean leverage bar with CI
lev_agg = (
    merged.groupby("sentiment_bin")["mean_leverage"]
    .agg(["mean", "sem"])
    .reindex(order)
)
ax = axes[1]
bars = ax.bar(
    lev_agg.index, lev_agg["mean"],
    color=[SENT_COLORS[s] for s in lev_agg.index],
    edgecolor="white", width=0.5
)
ax.errorbar(
    range(len(lev_agg)), lev_agg["mean"],
    yerr=lev_agg["sem"] * 1.96,
    fmt="none", color="#333", capsize=5, linewidth=1.5
)
for bar, val in zip(bars, lev_agg["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.05, f"{val:.2f}x",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Mean Leverage by Sentiment (±95% CI)")
ax.set_ylabel("Mean Leverage")
ax.set_xlabel("Sentiment")

save(fig, "03_leverage_sentiment")

# ════════════════════════════════════════════════════════
# CHART 4  –  Trade Frequency & Volume by Sentiment
# ════════════════════════════════════════════════════════
print("\n[B4] Trade Frequency vs Sentiment")
freq_vol = merged.groupby("sentiment_bin").agg(
    mean_trades  = ("trade_count",   "mean"),
    mean_vol_usd = ("total_vol_usd", "mean"),
).reindex(order)
print(freq_vol.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, label, title in zip(
    axes,
    ["mean_trades", "mean_vol_usd"],
    ["Avg # Trades per Day", "Avg Volume (USD)"],
    ["Trade Frequency by Sentiment", "Daily Trading Volume by Sentiment"],
):
    bars = ax.bar(
        freq_vol.index, freq_vol[col],
        color=[SENT_COLORS[s] for s in freq_vol.index],
        edgecolor="white", width=0.5
    )
    for bar, val in zip(bars, freq_vol[col]):
        fmt = f"{val:,.0f}" if "vol" in col else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01,
                fmt, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(title)
    ax.set_ylabel(label)
    ax.set_xlabel("Sentiment")

save(fig, "04_frequency_volume_sentiment")

# ════════════════════════════════════════════════════════
# CHART 5  –  Long/Short Ratio by Sentiment
# ════════════════════════════════════════════════════════
print("\n[B5] Long/Short Ratio vs Sentiment")
ls_sent = merged.groupby("sentiment_bin")["ls_ratio"].mean().reindex(order)
print(ls_sent.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    ls_sent.index, ls_sent.values,
    color=[SENT_COLORS[s] for s in ls_sent.index],
    edgecolor="white", width=0.5
)
ax.axhline(0.5, color="#333", linewidth=1.1, linestyle="--", label="50% (neutral)")
for bar, val in zip(bars, ls_sent.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.003, f"{val:.1%}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_title("Avg Long Ratio by Market Sentiment\n(>50% = more long trades)")
ax.set_ylabel("Long Ratio (Longs / Total Trades)")
ax.set_xlabel("Sentiment")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.legend()
save(fig, "05_longshort_ratio_sentiment")

# ════════════════════════════════════════════════════════
# CHART 6  –  PnL rolling trend with sentiment shading
# ════════════════════════════════════════════════════════
print("\n[B6] Rolling PnL timeline with sentiment background")

daily_agg = (
    merged.groupby(["date", "sentiment_bin"])
    .agg(total_pnl=("total_pnl", "sum"))
    .reset_index()
)
daily_agg["date"] = pd.to_datetime(daily_agg["date"])
daily_agg = daily_agg.sort_values("date")
daily_agg["rolling_pnl"] = daily_agg["total_pnl"].rolling(7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(15, 5))

prev_date = daily_agg["date"].min()
prev_sent = daily_agg.iloc[0]["sentiment_bin"]
for _, row in daily_agg.iterrows():
    if row["sentiment_bin"] != prev_sent or _ == len(daily_agg)-1:
        ax.axvspan(prev_date, row["date"],
                   alpha=0.18,
                   color=SENT_COLORS.get(prev_sent, COLORS["neutral"]))
        prev_date = row["date"]
        prev_sent = row["sentiment_bin"]

ax.plot(daily_agg["date"], daily_agg["rolling_pnl"],
        color=COLORS["blue"], linewidth=1.8, label="7-day rolling avg PnL")
ax.axhline(0, color="#555", linewidth=0.9, linestyle="--")

patches = [mpatches.Patch(color=SENT_COLORS[s], alpha=0.5, label=s)
           for s in ["Fear", "Neutral", "Greed"]]
patches.append(plt.Line2D([0], [0], color=COLORS["blue"], linewidth=2, label="7d rolling PnL"))
ax.legend(handles=patches, loc="upper left", fontsize=9)
ax.set_title("7-Day Rolling Aggregate PnL (Shaded by Sentiment)")
ax.set_xlabel("Date")
ax.set_ylabel("Total PnL (USD)")
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b '%y"))
fig.autofmt_xdate()
save(fig, "06_rolling_pnl_timeline")

# ════════════════════════════════════════════════════════
# PART B3 – Trader Segmentation
# ════════════════════════════════════════════════════════
print("\n" + "-"*50)
print("TRADER SEGMENTATION")
print("-"*50)

# ── Segment 1: Leverage Tier ─────────────────────────────
acc_stats = (
    merged.groupby("Account")
    .agg(
        mean_lev     = ("mean_leverage",  "mean"),
        total_pnl    = ("total_pnl",      "sum"),
        mean_pnl     = ("total_pnl",      "mean"),
        trade_days   = ("date",           "nunique"),
        mean_wr      = ("win_rate",       "mean"),
        mean_trades  = ("trade_count",    "mean"),
        mean_vol_usd = ("total_vol_usd",  "mean"),
    )
    .reset_index()
)
acc_stats.dropna(subset=["mean_lev"], inplace=True)

lev_q = acc_stats["mean_lev"].quantile([0.33, 0.67])
acc_stats["lev_segment"] = pd.cut(
    acc_stats["mean_lev"],
    bins=[-np.inf, lev_q.iloc[0], lev_q.iloc[1], np.inf],
    labels=["Low Lev", "Mid Lev", "High Lev"]
)

# ── Segment 2: Trade Frequency ────────────────────────────
freq_q = acc_stats["trade_days"].quantile([0.33, 0.67])
acc_stats["freq_segment"] = pd.cut(
    acc_stats["trade_days"],
    bins=[-np.inf, freq_q.iloc[0], freq_q.iloc[1], np.inf],
    labels=["Rare", "Moderate", "Frequent"]
)

# ── Segment 3: Consistency (std of daily PnL per account) ─
acc_pnl_std = merged.groupby("Account")["total_pnl"].std().reset_index()
acc_pnl_std.columns = ["Account", "pnl_std"]
acc_stats = acc_stats.merge(acc_pnl_std, on="Account", how="left")
acc_stats["consistency_score"] = acc_stats["mean_pnl"] / (acc_stats["pnl_std"] + 1e-9)

cs_q = acc_stats["consistency_score"].quantile([0.33, 0.67])
acc_stats["consist_segment"] = pd.cut(
    acc_stats["consistency_score"],
    bins=[-np.inf, cs_q.iloc[0], cs_q.iloc[1], np.inf],
    labels=["Inconsistent", "Moderate", "Consistent"]
)

print("\nSegment summary (Leverage):\n",
      acc_stats.groupby("lev_segment")[["total_pnl","mean_wr","mean_lev"]].mean().round(2).to_string())
print("\nSegment summary (Frequency):\n",
      acc_stats.groupby("freq_segment")[["total_pnl","mean_wr","trade_days"]].mean().round(2).to_string())
print("\nSegment summary (Consistency):\n",
      acc_stats.groupby("consist_segment")[["total_pnl","mean_wr","pnl_std"]].mean().round(2).to_string())

# ════════════════════════════════════════════════════════
# CHART 7 – Segment Performance Heatmap
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, seg, title in zip(
    axes,
    ["lev_segment", "freq_segment", "consist_segment"],
    ["Leverage Tier", "Trade Frequency", "Consistency"],
):
    sub = acc_stats.groupby(seg)[["total_pnl", "mean_wr"]].mean()
    sub.columns = ["Total PnL", "Win Rate"]
    sub_norm = (sub - sub.min()) / (sub.max() - sub.min() + 1e-9)
    sns.heatmap(
        sub_norm.T, annot=sub.T.round(2), fmt="g",
        cmap="RdYlGn", linewidths=0.5,
        annot_kws={"size": 10}, ax=ax
    )
    ax.set_title(f"Perf by {title}")
    ax.set_ylabel("")

save(fig, "07_segment_heatmap")

# ════════════════════════════════════════════════════════
# CHART 8 – Cross-segment: Lev Tier × Sentiment PnL
# ════════════════════════════════════════════════════════
merged2 = merged.merge(
    acc_stats[["Account", "lev_segment", "freq_segment", "consist_segment"]],
    on="Account", how="left"
)

cross = (
    merged2.groupby(["sentiment_bin", "lev_segment"])["total_pnl"]
    .mean()
    .unstack("lev_segment")
    .reindex(index=order)
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cross.index))
width = 0.25
seg_colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
for i, col in enumerate(cross.columns):
    bars = ax.bar(x + i*width, cross[col], width,
                  label=str(col), color=seg_colors[i], edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels(cross.index)
ax.axhline(0, color="#555", linewidth=0.9, linestyle="--")
ax.set_title("Mean Daily PnL: Leverage Segment × Market Sentiment")
ax.set_ylabel("Mean PnL (USD)")
ax.set_xlabel("Sentiment")
ax.legend(title="Leverage Tier")
save(fig, "08_lev_segment_x_sentiment")

# ════════════════════════════════════════════════════════
# CHART 9 – Frequent vs Rare traders in Fear/Greed
# ════════════════════════════════════════════════════════
cross2 = (
    merged2.groupby(["sentiment_bin", "freq_segment"])["total_pnl"]
    .mean()
    .unstack("freq_segment")
    .reindex(index=order)
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cross2.index))
for i, col in enumerate(cross2.columns):
    bars = ax.bar(x + i*width, cross2[col], width,
                  label=str(col), color=seg_colors[i], edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels(cross2.index)
ax.axhline(0, color="#555", linewidth=0.9, linestyle="--")
ax.set_title("Mean Daily PnL: Frequency Segment × Market Sentiment")
ax.set_ylabel("Mean PnL (USD)")
ax.set_xlabel("Sentiment")
ax.legend(title="Frequency Tier")
save(fig, "09_freq_segment_x_sentiment")


# ─────────────────────────────────────────────────────────
# BONUS – Clustering & Predictive Model
# ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BONUS  –  CLUSTERING + PREDICTIVE MODEL")
print("="*60)

# ════════════════════════════════════════════════════════
# BONUS 1  –  K-Means Clustering of Trader Archetypes
# ════════════════════════════════════════════════════════
print("\n[Bonus-1] K-Means Clustering")

cluster_features = ["mean_lev", "mean_wr", "trade_days",
                    "mean_vol_usd", "total_pnl", "mean_pnl"]
cluster_df = acc_stats.dropna(subset=cluster_features).copy()

scaler = StandardScaler()
X_clust = scaler.fit_transform(cluster_df[cluster_features])

# elbow method
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(K_range, inertias, marker="o", color=COLORS["blue"], linewidth=2)
ax.set_title("Elbow Method – Optimal K")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")

# fit with K=4
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_df["cluster"] = km_final.fit_predict(X_clust)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clust)
cluster_df["pca1"] = X_pca[:, 0]
cluster_df["pca2"] = X_pca[:, 1]

CLUSTER_COLORS = ["#3b7dd8", "#e05c5c", "#4caf7d", "#f5a623"]
ax = axes[1]
for c in sorted(cluster_df["cluster"].unique()):
    sub = cluster_df[cluster_df["cluster"] == c]
    ax.scatter(sub["pca1"], sub["pca2"],
               color=CLUSTER_COLORS[c], label=f"Cluster {c}",
               alpha=0.75, s=80, edgecolors="white", linewidths=0.5)
ax.set_title("Trader Archetypes – PCA Projection (K=4)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
ax.legend()
save(fig, "10_clustering_pca")

# cluster profile
cluster_profile = cluster_df.groupby("cluster")[cluster_features].mean().round(3)
print("\nCluster Profiles:\n", cluster_profile.to_string())

# label archetypes
ARCHETYPE_LABELS = {
    0: "Passive Low-Volume",
    1: "High-Leverage Aggressive",
    2: "Consistent Winners",
    3: "Frequent Mid-Leverage",
}
# sort by total_pnl descending and assign labels accordingly
sorted_idx = cluster_profile["total_pnl"].sort_values(ascending=False).index.tolist()
auto_labels = {}
label_list = ["Top Earners", "High-Risk Speculators", "Steady Accumulators", "Low-Activity Traders"]
for i, idx in enumerate(sorted_idx):
    auto_labels[idx] = label_list[i]

cluster_df["archetype"] = cluster_df["cluster"].map(auto_labels)
print("\nArchetype distribution:\n", cluster_df["archetype"].value_counts().to_string())

# ════════════════════════════════════════════════════════
# CHART 11  –  Cluster profile radar / bar
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)
profile_norm.index = [auto_labels[i] for i in profile_norm.index]
profile_norm[["mean_lev","mean_wr","trade_days","mean_vol_usd"]].plot(
    kind="bar", ax=ax,
    color=[COLORS["blue"], COLORS["greed"], COLORS["orange"], COLORS["purple"]],
    edgecolor="white"
)
ax.set_title("Normalised Cluster Profiles (4 Trader Archetypes)")
ax.set_ylabel("Normalised Score")
ax.set_xlabel("Archetype")
ax.legend(["Leverage", "Win Rate", "Active Days", "Volume"], bbox_to_anchor=(1, 1))
plt.xticks(rotation=15)
save(fig, "11_cluster_profiles")

# ════════════════════════════════════════════════════════
# BONUS 2  –  Predictive Model: next-day profitable?
# ════════════════════════════════════════════════════════
print("\n[Bonus-2] Predictive Model")

# Features: today's metrics + sentiment indicator
feature_cols = [
    "trade_count", "total_pnl", "win_rate", "mean_leverage",
    "ls_ratio", "total_vol_usd", "value"          # value = FG index score
]
target_col = "profit_flag"

model_df = merged[feature_cols + [target_col]].dropna()
X = model_df[feature_cols]
y = model_df[target_col]

print(f"Model dataset: {X.shape[0]:,} samples | class balance: {y.mean():.2%} profitable")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_train)
X_te_s = sc.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    cv_sc = cross_val_score(model, X_tr_s, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_tr_s, y_train)
    test_sc = cross_val_score(model, X_te_s, y_test, cv=cv, scoring="roc_auc").mean()
    results[name] = {"cv_auc": cv_sc.mean(), "cv_std": cv_sc.std(), "test_auc": test_sc}
    print(f"  {name:25s}  CV AUC: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}  |  Test AUC: {test_sc:.4f}")

best_name = max(results, key=lambda n: results[n]["cv_auc"])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# ════════════════════════════════════════════════════════
# CHART 12  –  Feature Importance + Confusion Matrix
# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# feature importance (RF)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
ax = axes[0]
importances.plot(kind="barh", ax=ax,
                 color=COLORS["blue"], edgecolor="white")
ax.set_title("Feature Importances – Random Forest")
ax.set_xlabel("Importance Score")

# confusion matrix of best model
y_pred = best_model.predict(X_te_s)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Unprofitable", "Profitable"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title(f"Confusion Matrix – {best_name}")

save(fig, "12_model_feature_importance")

# model score summary bar
fig, ax = plt.subplots(figsize=(9, 4))
names  = list(results.keys())
cv_auc = [results[n]["cv_auc"]  for n in names]
te_auc = [results[n]["test_auc"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.2, cv_auc, 0.35, label="CV AUC",   color=COLORS["blue"],  edgecolor="white")
ax.bar(x + 0.2, te_auc, 0.35, label="Test AUC", color=COLORS["greed"], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10)
ax.set_ylim(0.45, 1.0)
ax.set_title("Model Comparison – ROC-AUC")
ax.set_ylabel("AUC Score")
ax.legend()
save(fig, "13_model_comparison")

# ════════════════════════════════════════════════════════
# SUMMARY TABLE  –  Key stats for write-up
# ════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KEY STATISTICS SUMMARY")
print("="*60)
summary = merged.groupby("sentiment_bin").agg(
    days          = ("date",          "nunique"),
    acc_days      = ("Account",       "count"),
    mean_pnl      = ("total_pnl",     "mean"),
    median_pnl    = ("total_pnl",     "median"),
    mean_wr       = ("win_rate",      "mean"),
    mean_trades   = ("trade_count",   "mean"),
    mean_lev      = ("mean_leverage", "mean"),
    long_ratio    = ("ls_ratio",      "mean"),
).reindex(order).round(3)
print(summary.to_string())

print("\n✅  All charts saved to:", OUT)
print("✅  Analysis complete.\n")
