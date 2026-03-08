# Trader Performance vs Market Sentiment — Summary Write-up
**Primetrade.ai · Data Science Intern Assignment**

---

## Methodology

Two datasets were combined: 211,224 Hyperliquid trade records (May 2023 – May 2025) across 32 accounts, and the Bitcoin Fear/Greed Index. Trades were aggregated to a **daily account-level** granularity, producing metrics like total PnL, win rate, trade count, volume, long/short ratio, and a leverage proxy (Size USD / |Start Position|). The Fear/Greed Index was collapsed from 5 classes into 3 (Fear, Neutral, Greed) and joined on date, yielding 2,340 account-day records for analysis.

Traders were segmented along three axes — leverage tier, trade frequency, and PnL consistency — and performance was cross-tabulated against sentiment. A K-Means clustering (K=4) was run on per-account features to derive behavioural archetypes. A Random Forest classifier was trained to predict next-day profitability using same-day behaviour features plus the sentiment score (ROC-AUC: 1.00 on test set).

---

## Key Insights

**1. Fear days produce high-variance, not high-quality returns.**
Mean PnL is highest on Fear days ($5,185) but median PnL is only $123 — versus $265 on Greed days. A handful of large wins skew the mean. The typical trader earns more *consistently* during Greed; Fear-day returns have a fat downside tail that the mean hides.

**2. Traders over-trade during Fear without better outcomes.**
Fear days average 105 trades and $757k in volume per account — 37% more trades and 115% more volume than Greed days — yet win rates are virtually identical (~84–86%) across all sentiment conditions. Elevated activity during volatile periods adds fee drag without improving edge.

**3. Leverage stays flat regardless of market fear.**
Average leverage hovers at ~31–33× across Fear, Neutral, and Greed days with overlapping 95% confidence intervals. Traders are not de-risking when they should be. This is the single clearest risk management gap in the data.

---

## Strategy Recommendations

**Rule 1 — Fear days: trade less, not more.**
Cap daily trade entries to ~65% of a trader's typical frequency during Fear periods, and require a minimum 2:1 risk/reward before entry. The data shows no win-rate improvement from increased activity on Fear days — selectivity is the edge, not volume.

**Rule 2 — Greed days: enforce a leverage ceiling for aggressive traders.**
When sentiment is Greed and a trader's 7-day average leverage exceeds 40×, cap new position leverage at 25×. Mid-leverage traders ($405k total PnL) outperform high-leverage traders ($283k) over the full period — the difference compounds through avoided blowup events, which cluster in euphoric markets.

**Rule 3 — Use sentiment as a directional tilt filter.**
On Fear days, modestly prioritise long setups (52% long bias on Fear days correlates with the elevated mean PnL). On prolonged Greed streaks (5+ consecutive days), prioritise short setups. Apply as an entry filter overlay only — do not override position sizing or stop-losses.
