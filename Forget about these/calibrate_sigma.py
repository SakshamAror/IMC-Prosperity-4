"""
Calibrate per-strike implied vol (sigma) for VEV_5200–5500
using historical data from R3 days 0-2 + R4 day 3.
Finds sigma that minimises MSE between BS(spot, K, T, sigma) and actual mid price.
"""
import pandas as pd
import numpy as np
from math import log, sqrt, erf
from pathlib import Path
from scipy.optimize import minimize_scalar

BASE = Path("/Users/saksh/Documents/GitHub/IMC Prosperity 4")
TRADING_DAYS = 252

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T <= 0: return max(S - K, 0.0)
    d1 = (log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    return S * norm_cdf(d1) - K * norm_cdf(d1 - sigma * sqrt(T))

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
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
data["mid_price"] = pd.to_numeric(data["mid_price"], errors="coerce")

spot_df = (data[data["product"] == "VELVETFRUIT_EXTRACT"]
           [["tte_start", "timestamp", "mid_price"]]
           .rename(columns={"mid_price": "spot"}))
spot_df["spot"] = pd.to_numeric(spot_df["spot"], errors="coerce")

STRIKES = [5200, 5300, 5400, 5500]
print(f"{'Strike':<8} {'sigma':>8}  {'BS_mean':>10}  {'Actual_mean':>12}  {'RMSE':>8}")
print("-" * 55)

calibrated = {}
for K in STRIKES:
    rows = data[data["product"] == f"VEV_{K}"][["tte_start", "timestamp", "mid_price"]].copy()
    rows = rows.merge(spot_df, on=["tte_start", "timestamp"], how="inner")
    rows = rows.dropna()
    rows["T"] = (rows["tte_start"] - rows["timestamp"] / 1_000_000) / TRADING_DAYS
    rows = rows[rows["T"] > 1e-4]

    S = rows["spot"].values
    T = rows["T"].values
    actual = rows["mid_price"].values

    def mse(sigma):
        bs = np.array([bs_call(s, K, t, sigma) for s, t in zip(S, T)])
        return np.mean((bs - actual) ** 2)

    result = minimize_scalar(mse, bounds=(0.15, 2.0), method="bounded")
    sigma_opt = result.x
    calibrated[K] = round(sigma_opt, 4)

    bs_vals = np.array([bs_call(s, K, t, sigma_opt) for s, t in zip(S, T)])
    rmse = np.sqrt(np.mean((bs_vals - actual) ** 2))
    print(f"{K:<8} {sigma_opt:>8.4f}  {bs_vals.mean():>10.2f}  {actual.mean():>12.2f}  {rmse:>8.2f}")

print()
print("# Paste into round4_EMA.py:")
print(f"OPT_BS_SIGMA = {calibrated}")
