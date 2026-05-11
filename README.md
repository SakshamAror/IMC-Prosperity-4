# IMC Prosperity 4

This is **Team TRITONS'** work on the algorithmic trading portions of IMC Prosperity 4 — a global competition with 18,000+ teams. We finished **#122 globally**. This repo contains our final algorithms, exploration notebooks, and backtest logs for Rounds 3–5 of the competition, which introduced progressively more complex products and market dynamics each round.

---

## Table of Contents
- [Rounds 3 & 4 — Mean Reversion + Options Market Making](#rounds-3--4)
- [Round 5 — Multi-Strategy Across 50 Products](#round-5)

---

## Rounds 3 & 4

Two products and their options were available: **HYDROGEL PACK** (spot ~10,000) and **VELVETFRUIT EXTRACT** (spot ~5,248), along with VEV call options at strikes 4000–6500.

### Spot Assets: Mean Reversion Market Making

Both spot products exhibited strong mean-reverting behavior around stable fair values. The strategy maintained a fixed fair value (FV) per product calibrated from historical data, then:

- Computed the deviation of the current mid-price from FV
- Set a target inventory proportional to `-deviation` (lean long when cheap, lean short when expensive)
- Posted three order layers: aggressive taker orders when misprice exceeded a threshold, passive inventory-skewed quotes, and penny-jump fills with remaining capacity
- Round 4 added realized volatility scaling — the quote spread and position target scaled with a rolling vol ratio to adapt to regime changes

<p align="center">
  <img src="Images/r34_spot_mr.png" width="800" alt="VELVETFRUIT and HYDROGEL mean reversion"/>
</p>

*Mid-prices for both assets across 3 days, with fixed fair value reference lines. The prices oscillate tightly around FV, confirming the mean-reversion regime.*

---

### Options: Theta-Decayed Fair Values

Ten call options on VELVETFRUIT at strikes 4000–5500 were market-made using Black-Scholes calibrated fair values. The key insight was that each option's price decays predictably as expiry approaches.

**Round 3** used static BS fair values at TTE = 5 days, treating each option independently as a mean-reversion target around its BS price.

**Round 4** modeled the decay explicitly. For OTM strikes (5100–5500), the fair value followed:

$$FV(K, t) = C_K \cdot \ln(TTE)$$

where $C_K$ is a per-strike coefficient fitted from historical option prices and TTE decrements by 1 each day at end-of-day. This captured the accelerating theta decay of OTM options as expiry nears.

<p align="center">
  <img src="Images/r34_options_decay.png" width="900" alt="Option prices across days by strike"/>
</p>

*Option prices by strike across 3 days. Deep ITM options (low strikes) hold intrinsic value; OTM options decay toward zero — consistent with Black-Scholes theta.*

<p align="center">
  <img src="Images/r34_options_theta_fit.png" width="800" alt="Log-TTE fit vs actual prices"/>
</p>

*C·ln(TTE) model fit (dashed lines) vs actual observed option prices for OTM strikes. Bot trade prices are overlaid as scatter points, confirming the model captures the decay curve accurately.*

<p align="center">
  <img src="Images/r34_iv_smile.png" width="650" alt="Implied volatility smile"/>
</p>

*Implied volatility smile computed from median option prices across all strikes, shown per day. The smile is stable across days, validating the BS framework as the fair value anchor.*

Each option was traded with the same inventory-skewed MR approach as the spot assets — quoting around its BS fair value with aggression when the market price deviated beyond a strike-specific threshold.

---

### OTM Lottery: VEV 6000/6500 (Round 4 only)

A secondary bot (Mark 22) periodically dumped deep OTM options (VEV 6000/6500) at price 0. Another bot (Mark 01) rested bids to absorb them. The strategy raced Mark 01 by posting bids at the same or lower level and then offered inventory back at a floor ask price — capturing the spread on each dump cycle.

---

## Round 5

Round 5 introduced **50 new products** across 10 families of 5, each with a position limit of 50. The algorithm used a modular sector architecture — each product family is an independent strategy class with its own logic and state.

<p align="center">
  <img src="Images/r5_category_correlation.png" width="750" alt="Category correlation heatmap"/>
</p>

*Return correlation between all 10 product categories. Low cross-category correlation confirmed each family required an independent strategy.*

---

### PEBBLES: XL/XS Pair Mean Reversion

Exploration revealed a structural basket constraint:

```
PEBBLES_XS + PEBBLES_S + PEBBLES_M + PEBBLES_L + PEBBLES_XL = 50,000  (std ≈ 2.8)
```

<p align="center">
  <img src="Images/r5_pebbles_basket.png" width="850" alt="Pebbles basket decomposition"/>
</p>

*The basket sum stays pinned at 50,000 throughout the round. Per-pebble deviations from their equal share of 10,000 are mean-reverting.*

This constraint implies that XL and XS are negatively correlated — when XL rises above its fair share, XS must fall to compensate. The strategy exploited the XL−XS spread directly: when the spread deviated beyond a threshold, it bought the cheap leg and sold the expensive leg. S, M, and L were handled with standalone passive market making.

<p align="center">
  <img src="Images/r5_pebbles_zscore.png" width="750" alt="Pebbles z-score MR signal"/>
</p>

*Rolling z-scores of per-pebble deviations confirm strong mean reversion, motivating the pair trade.*

---

### Trending Products: Trend-Hold + Market Making

Several families had persistent intra-day and multi-day directional trends. Rather than tracking a lagging EMA fair value, the strategy held a fixed directional position on these products while providing passive liquidity around it.

**Galaxy Sounds** — all 5 products trended upward consistently across days:

<p align="center">
  <img src="Images/r5_galaxy_prices.png" width="800" alt="Galaxy Sounds price series"/>
</p>

*All 5 Galaxy Sounds products across all days, showing persistent upward trends with day-over-day base shifts of ~1,000.*

Each Galaxy Sounds product was assigned a hold size (e.g., +8 lots on BLACK\_HOLES) and the algorithm continuously worked toward that target while market-making around it to capture spread.

**Sleep Pods** — five products split into two structural clusters (Cotton/Polyester/Suede and Lamb Wool/Nylon):

<p align="center">
  <img src="Images/r5_sleep_prices.png" width="800" alt="Sleep Pod price series"/>
</p>

*Absolute mid-prices for all Sleep Pod variants across days.*

<p align="center">
  <img src="Images/r5_sleep_trend_heatmap.png" width="650" alt="Sleep pod trend direction heatmap"/>
</p>

*OLS slope per product per day. Consistent sign and magnitude confirmed structural trends, not noise.*

<p align="center">
  <img src="Images/r5_sleep_clusters.png" width="650" alt="Sleep pod cluster structure"/>
</p>

*Two clusters emerged from the data: Cluster A (Cotton/Polyester/Suede) and Cluster B (Lamb Wool/Nylon), each trending as a group.*

---

### Directional Sector: UV Visors + Translators

UV Visors and several Translators showed large, consistent cumulative drifts across days (e.g., UV\_VISOR\_AMBER −28.7%, UV\_VISOR\_MAGENTA +15.3%, UV\_VISOR\_RED +17.2%). The strategy held fixed directional positions based on these observed biases, supplemented by a live bot-behavior learning signal that tracked which counterparties were profitable buyers vs sellers and adjusted sizing accordingly.

---

### Cross-Cluster Relative Value

Four products — UV\_VISOR\_RED, ROBOT\_DISHES, SNACKPACK\_RASPBERRY, SNACKPACK\_STRAWBERRY — were grouped into a cross-family cluster. The strategy held each product long (+5 lots baseline) and overlaid a mean-reversion signal: when any product deviated more than a threshold from the cluster mean, it faded the deviation with an additional ±5 lot MR position.

---

### Robot Sector: Pair Mean Reversion

ROBOT\_DISHES and ROBOT\_IRONING are negatively correlated — when one rises relative to the other, they tend to converge. The strategy traded the pair: when the spread exceeded a threshold, it went long the laggard and short the leader (both at the position limit), then reversed on mean reversion. ROBOT\_LAUNDRY was handled independently with passive market making.

---

### Snackpack: Trend-Hold + Spread Overlay

- **STRAWBERRY**: clean uptrend → hold +10 directionally
- **CHOCOLATE**: downtrend → hold −10 directionally
- **PISTACHIO**: EMA market making around a stable fair value
- **RASPBERRY / VANILLA**: handled by the Cross-Cluster sector above

### Remaining Products: Passive Market Making

Translators (remaining), Panels, Microchips, and non-trending Oxygen Shakes used passive penny-jump market making — quote at `best_bid + 1` / `best_ask − 1` when spread ≥ 3, join the book when spread = 2, skip when spread = 1. Products with observed directional tendencies were selectively disabled.

---

## Repository Structure

```
IMC Prosperity 4/
├── Data/                   — Raw price and trade CSVs (ROUND_3, ROUND_4, ROUND_5)
├── Final Algorithms/       — Submitted algorithms for each round
│   ├── round3_final.py
│   ├── round4_final.py
│   └── round5_final.py
├── Images/                 — Charts used in this README
├── old models/             — Exploration notebooks and prior algorithm iterations
└── round5_claudia.py       — Current working algorithm
```
