# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

---

## Overview

This project investigates how Bitcoin market sentiment (Fear/Greed Index) relates to trader behaviour and profitability on Hyperliquid. The analysis spans two years of real trade data (May 2023 – May 2025) across 32 unique trader accounts and 246 coins.

---

## Project Structure

```
primetrade_assignment/
│
├── data/
│   ├── historical_data.csv      # Hyperliquid trade records (211,224 rows)
│   └── fear_greed_index.csv     # Bitcoin Fear & Greed Index (2,644 rows)
│
├── charts/                      # All output visualisations (13 PNGs)
│   ├── 01_pnl_by_sentiment.png
│   ├── 02_winrate_sentiment_violin.png
│   ├── 03_leverage_sentiment.png
│   ├── 04_frequency_volume_sentiment.png
│   ├── 05_longshort_ratio_sentiment.png
│   ├── 06_rolling_pnl_timeline.png
│   ├── 07_segment_heatmap.png
│   ├── 08_lev_segment_x_sentiment.png
│   ├── 09_freq_segment_x_sentiment.png
│   ├── 10_clustering_pca.png
│   ├── 11_cluster_profiles.png
│   ├── 12_model_feature_importance.png
│   └── 13_model_comparison.png
│
├── trader_sentiment_analysis.ipynb   # Main analysis notebook
├── analysis.py                        # Standalone runnable script
└── README.md
```

---

## Setup & How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Python 3.9+ recommended.

### Run Notebook

```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

### Run Script

```bash
python analysis.py
```

All charts are written to `charts/` automatically.

---

## Datasets

| Dataset | Rows | Columns | Date Range |
|---------|------|---------|------------|
| Hyperliquid Historical Trades | 211,224 | 16 | May 2023 – May 2025 |
| Bitcoin Fear & Greed Index | 2,644 | 4 | Feb 2018 – May 2025 |

**Key trade fields used:** Account, Coin, Size USD, Side, Closed PnL, Start Position, Timestamp IST  
**Sentiment fields used:** date, classification (5-class), value (0–100 score)

---

## Methodology

### Data Preparation
- Parsed IST timestamps (`dd-mm-yyyy HH:MM`) and aligned both datasets at **daily granularity**
- Collapsed the 5-class Fear/Greed index into 3 bins: **Fear** (Extreme Fear + Fear), **Neutral**, **Greed** (Greed + Extreme Greed)
- Engineered a **leverage proxy** = Size USD / |Start Position| clipped at 125×
- Built **daily account-level aggregates**: total PnL, win rate, trade count, volume, long/short ratio, mean leverage
- Merged datasets on date → **2,340 account-day records** across 790 Fear days, 1,174 Greed days, 376 Neutral days

### Analysis (Part B)
- Compared PnL (mean, median), win rate, leverage, trade frequency, volume, and long/short ratio across Fear / Neutral / Greed days
- Used violin plots, box plots, bar charts with confidence intervals, and a rolling PnL timeline
- Segmented traders into **3 axes**: leverage tier (Low / Mid / High), frequency tier (Rare / Moderate / Frequent), consistency (Inconsistent / Moderate / Consistent)
- Cross-tabulated segment × sentiment to find differential performance patterns

### Bonus
- **K-Means clustering (K=4)** on per-account features to derive behavioural archetypes, visualised with PCA
- **Predictive model** (Logistic Regression, Random Forest, Gradient Boosting) to predict next-day profitability using same-day behaviour + sentiment features; evaluated via 5-fold stratified cross-validation and ROC-AUC

---

## Key Insights

### Insight 1 — Fear Days = High Variance, Not High Performance
Mean PnL on Fear days ($5,185) is higher than Greed days ($4,144), but **median PnL on Fear days ($123) is less than half that of Greed days ($265)**. A handful of outsized winning trades skew the mean upward. The typical trader does meaningfully better on Greed days. Fear-day returns follow a right-skewed distribution — high upside, but also fatter downside tails.

### Insight 2 — Traders Over-Trade During Fear (and Don't Get Paid For It)
Fear days see **37% more trades** (avg 105 vs 77 on Greed days) and **115% more volume** ($757k vs $352k). Win rates are virtually identical across sentiment conditions (~84–86%). This means the elevated activity on Fear days is not translating into better outcomes — traders are reacting to volatility rather than executing on conviction. After fees, the incremental trades likely subtract value.

### Insight 3 — Leverage Doesn't Move With Sentiment (But It Should)
Average leverage across all sentiment conditions sits in a narrow 31–33× band with overlapping 95% confidence intervals. Traders are **not de-levering during fearful conditions**, despite elevated volatility and less predictable price action. This is a systematic risk management gap: the time when leverage should be cut most is the time it is maintained most stubbornly.

### Insight 4 — Mid-Leverage Traders Are the Sweet Spot
Segmenting accounts by average leverage reveals that **Mid-Leverage traders (approx. 10–35×) generate the highest total PnL ($405k)**, outperforming both cautious Low-Leverage ($281k) and aggressive High-Leverage ($283k) traders. High-leverage traders have the best win rates (89%) but blow-up events cap their cumulative returns.

### Insight 5 — Consistent Winners Cap Their Own Upside
The Consistent segment (highest risk-adjusted PnL ratio) achieves a 91% win rate but middling total PnL. Their low PnL standard deviation ($14.6k vs $36k for Moderate traders) suggests they exit winners too early and manage loss size well — but leave significant profit on the table on trending days. Consistency is valuable for survival but needs to be balanced with conviction sizing on high-quality setups.

---

## Actionable Strategy Recommendations

### Rule 1 — "Fear Days: Trade Less, Not More"
**Recommendation:** During Fear sentiment periods, reduce trade frequency by 30–40% and apply stricter entry filters (e.g., only execute trades with a risk/reward ≥ 2:1).  

**Why:** The data shows traders execute 37% more trades during Fear but median PnL is less than half that of Greed periods. Additional trades add fee drag and emotional noise without improving outcomes. Selectivity beats activity in stressed markets.

**Best for:** Frequent traders (top 33% by active days). Rare traders are already naturally selective.

---

### Rule 2 — "Greed Days: Enforce a Leverage Ceiling for Aggressive Traders"
**Recommendation:** When market sentiment registers Greed and a trader's 7-day average leverage exceeds 40×, cap new position leverage at 25× until a Fear or Neutral signal re-appears.  

**Why:** High-leverage traders underperform mid-leverage peers on Greed days specifically. Euphoric markets create a false sense of low risk — correlations between assets compress, volatility looks low, and then reversals are sudden and large. Mid-leverage traders extract the most upside during Greed while avoiding catastrophic drawdowns.

**Best for:** High-leverage segment (top 33%). Low/mid-leverage traders are already within the safe zone.

---

### Rule 3 — "Use Sentiment as a Long/Short Tilt Filter"
**Recommendation:** Add a weak directional prior — tilt toward longs on confirmed Fear days; tilt toward shorts on prolonged Greed days (>5 consecutive days). Do not override stop-losses or position sizing — use this as an entry priority filter only.  

**Why:** The 52.2% long bias on Fear days (vs 47.2% on Greed days) correlates with higher mean PnL on Fear days — buying during pessimism has historically worked on this dataset. Conversely, increased short activity on Greed days aligns with the steadier median returns in that regime, suggesting those who fade euphoria do well.

**Best for:** All segments as a lightweight overlay.

---

## Predictive Model Results

| Model | CV ROC-AUC | Test ROC-AUC |
|-------|-----------|--------------|
| Logistic Regression | 0.987 | 0.988 |
| **Random Forest** | **1.000** | **1.000** |
| Gradient Boosting | 1.000 | 0.987 |

Random Forest achieves near-perfect classification with the top features being `total_pnl` (same-day), `win_rate`, and `trade_count`. The fear/greed `value` score contributes modestly, confirming that behaviour features dominate over raw sentiment for predicting profitability — sentiment is more useful as a context modifier than a standalone signal.

---

## Clustering — Trader Archetypes

| Archetype | Avg Leverage | Win Rate | Active Days | Total PnL |
|-----------|-------------|---------|-------------|-----------|
| Top Earners | 25× | 86% | 191 | $563,894 |
| High-Risk Speculators | 25× | 86% | 30 | $1,126,939 |
| Steady Accumulators | 18× | 72% | 51 | $100,956 |
| Low-Activity Traders | 36× | 94% | 33 | $165,082 |

The highest absolute PnL cluster ("High-Risk Speculators") trades infrequently but with large positions — consistent with a small number of highly concentrated, high-conviction bets. "Top Earners" are high-frequency accumulators with stable leverage. "Low-Activity Traders" have the best win rate but smallest total contribution.
