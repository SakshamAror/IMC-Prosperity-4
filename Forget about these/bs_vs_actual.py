"""
Plot BS model (sigma=0.24) vs actual option prices across all available days.
TTE at start of each day:
  R3 day 0 -> 8, R3 day 1 -> 7, R3 day 2 -> 6, R4 day 3 -> 5
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import log, sqrt, exp, erf
from pathlib import Path

BASE = Path("/Users/saksh/Documents/GitHub/IMC Prosperity 4")
SIGMA = 0.24   # default for 5000/5100
PER_STRIKE_SIGMA = {
    5000: 0.24,
    5100: 0.24,
    5200: 0.2009,
    5300: 0.2032,
    5400: 0.1911,
    5500: 0.2062,
}
TRADING_DAYS_PER_YEAR = 252

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S, K, T, sigma=SIGMA):
    if T <= 0.0: return max(S - K, 0.0)
    d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * norm_cdf(d2)

# ── Load data ─────────────────────────────────────────────────────────────────
DAYS = [
    (BASE / "ROUND_3" / "prices_round_3_day_0.csv", 8),
    (BASE / "ROUND_3" / "prices_round_3_day_1.csv", 7),
    (BASE / "ROUND_3" / "prices_round_3_day_2.csv", 6),
    (BASE / "ROUND_4" / "prices_round_4_day_3.csv", 5),
]

frames = []
for path, tte_start in DAYS:
    df = pd.read_csv(path, sep=";")
    df["tte_start"] = tte_start
    df["file"] = path.name
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
OPT_COLS = {K: f"VEV_{K}" for K in STRIKES}

# Spot price per (file, timestamp)
spot = (data[data["product"] == "VELVETFRUIT_EXTRACT"]
        [["file", "timestamp", "mid_price"]]
        .rename(columns={"mid_price": "spot"}))

# Compute BS price for each option row
def process_strike(K):
    sym = f"VEV_{K}"
    rows = data[data["product"] == sym][["file", "timestamp", "mid_price", "tte_start"]].copy()
    rows = rows.merge(spot, on=["file", "timestamp"], how="left")
    rows = rows.dropna(subset=["spot", "mid_price"])
    rows["mid_price"] = pd.to_numeric(rows["mid_price"], errors="coerce")
    rows["spot"]      = pd.to_numeric(rows["spot"],      errors="coerce")
    rows["T"] = (rows["tte_start"] - rows["timestamp"] / 1_000_000) / TRADING_DAYS_PER_YEAR
    sigma_k = PER_STRIKE_SIGMA.get(K, SIGMA)
    rows["bs"] = rows.apply(lambda r: bs_call(r["spot"], K, max(r["T"], 1e-6), sigma_k), axis=1)
    # Add a continuous x-axis: day_offset = (8 - tte_start) + timestamp/1e6
    rows["x"] = (8 - rows["tte_start"]) + rows["timestamp"] / 1_000_000
    return rows

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=False)
axes = axes.flatten()

day_labels = {8: "R3d0", 7: "R3d1", 6: "R3d2", 5: "R4d3"}
colors_actual = "#2563eb"
colors_bs     = "#dc2626"

for ax, K in zip(axes, STRIKES):
    rows = process_strike(K)
    if rows.empty:
        ax.set_title(f"VEV_{K} — no data")
        continue

    # Plot per day with thin lines
    for tte_start, grp in rows.groupby("tte_start", sort=False):
        grp = grp.sort_values("x")
        label_a = f"Actual" if tte_start == rows["tte_start"].iloc[0] else ""
        label_b = f"BS σ={PER_STRIKE_SIGMA.get(K, SIGMA)}" if tte_start == rows["tte_start"].iloc[0] else ""
        ax.plot(grp["x"], grp["mid_price"], color=colors_actual, lw=0.8,
                alpha=0.7, label=label_a)
        ax.plot(grp["x"], grp["bs"],       color=colors_bs,     lw=0.8,
                alpha=0.7, linestyle="--", label=label_b)

    # Day boundary lines
    for tte_start in [8, 7, 6, 5]:
        x_boundary = 8 - tte_start
        ax.axvline(x_boundary, color="gray", lw=0.6, linestyle=":")
        ax.text(x_boundary + 0.02, ax.get_ylim()[1] * 0.97,
                day_labels[tte_start], fontsize=7, color="gray", va="top")

    ax.set_title(f"VEV_{K}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (days elapsed)")
    ax.set_ylabel("Price")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

fig.suptitle(f"BS Model (per-strike σ) vs Actual Option Prices — R3 days 0-2 + R4 day 3",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

out = BASE / "bs_vs_actual_otm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
