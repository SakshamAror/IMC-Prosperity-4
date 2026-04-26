"""
Calibrate the VEV options volatility smile from historical data.
Train: days 1 + 2.  Validate: day 0.

Outputs polynomial coefficients for use in the trader.
Run with:  .venv/bin/python calibrate_smile.py
"""
import csv, math
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Black-Scholes helpers ────────────────────────────────────────────────────

def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(S, K, T, sigma):
    """Black-Scholes call price.  r = 0."""
    if sigma <= 0 or T <= 0:
        return max(0.0, S - K)
    sv  = sigma * math.sqrt(T)
    d1  = math.log(S / K) / sv + 0.5 * sv
    d2  = d1 - sv
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_vega(S, K, T, sigma):
    """dC/dσ."""
    if sigma <= 0 or T <= 0:
        return 0.0
    sv = sigma * math.sqrt(T)
    d1 = math.log(S / K) / sv + 0.5 * sv
    return S * _npdf(d1) * math.sqrt(T)

def implied_vol(S, K, T, price, tol=1e-7, max_iter=100):
    """Newton-Raphson implied vol starting from σ=0.20.  Returns None on failure."""
    intrinsic = max(0.0, S - K)
    if price <= intrinsic + 1e-6 or T <= 1e-8:
        return None

    # Start at 20% — works for all our strikes (all near σ≈0.20)
    sigma = 0.20

    for _ in range(max_iter):
        p = bs_call(S, K, T, sigma)
        v = bs_vega(S, K, T, sigma)
        if v < 1e-14:
            break
        delta  = (p - price) / v
        sigma -= delta
        sigma  = max(1e-4, min(sigma, 5.0))
        if abs(delta) < tol:
            return sigma if 0.01 < sigma < 3.0 else None

    return sigma if 0.01 < sigma < 3.0 else None

# ── Data loading ─────────────────────────────────────────────────────────────

DATA_DIR   = Path("backtests/round3")
# TTE (days) at the START of each historical day
TTE_DAY0   = {0: 8, 1: 7, 2: 6}
STRIKES    = [5000, 5100, 5200, 5300, 5400, 5500]   # liquid NTM/OTM
SKIP_STRIKES = {4000, 4500}                          # deep ITM — IV unreliable

def load_day(day_num):
    """Return dict: timestamp -> {product: mid_price}."""
    path = DATA_DIR / f"prices_round_3_day_{day_num}.csv"
    ticks = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            ticks[int(row["timestamp"])][row["product"]] = float(row["mid_price"])
    return ticks

# ── Compute implied vols ─────────────────────────────────────────────────────

def collect_ivs(days):
    """Collect (log_moneyness, iv, K, day) tuples for given days."""
    records = []
    for day in days:
        ticks    = load_day(day)
        tte_base = TTE_DAY0[day]

        for ts in sorted(ticks):
            tick = ticks[ts]
            S    = tick.get("VELVETFRUIT_EXTRACT")
            if S is None or S <= 0:
                continue

            T_days  = tte_base - ts / 1_000_000
            T_years = T_days   / 252.0
            if T_years <= 1e-6:
                continue

            for K in STRIKES:
                price = tick.get(f"VEV_{K}")
                if price is None:
                    continue

                intrinsic = max(0.0, S - K)
                # Skip if price is at or below intrinsic (no time value → IV unstable)
                if price <= intrinsic + 0.1:
                    continue

                iv = implied_vol(S, K, T_years, price)
                if iv is None:
                    continue

                # Sanity: reject extreme vols
                if not (0.05 < iv < 2.0):
                    continue

                m = math.log(K / S)   # log-moneyness  (>0 for OTM calls)
                records.append((m, iv, K, day))

    return records

# ── Fit smile polynomial ──────────────────────────────────────────────────────

def fit_smile(records, degree=3):
    """Fit σ = poly(m) and return numpy poly coefficients (highest degree first)."""
    ms     = np.array([r[0] for r in records])
    sigmas = np.array([r[1] for r in records])
    coeffs = np.polyfit(ms, sigmas, degree)
    return coeffs, ms, sigmas

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot

# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Collecting IV observations (train: days 1+2)...")
train_records = collect_ivs([1, 2])
print(f"  {len(train_records):,} observations")

print("\nPer-strike breakdown:")
for K in STRIKES:
    kr = [r for r in train_records if r[2] == K]
    ivs = [r[1] for r in kr]
    if ivs:
        print(f"  VEV_{K}: n={len(ivs):5d}  "
              f"median_σ={np.median(ivs):.4f}  "
              f"mean_σ={np.mean(ivs):.4f}  "
              f"std_σ={np.std(ivs):.4f}")

# ── Fit degree 2 and 3 ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
for deg in [2, 3]:
    coeffs, ms, sigmas = fit_smile(train_records, degree=deg)
    pred   = np.polyval(coeffs, ms)
    r2     = r_squared(sigmas, pred)
    resid  = np.std(sigmas - pred)
    print(f"\nDegree-{deg} fit  (R²={r2:.4f}, residual σ={resid:.4f}):")
    terms = [f"{c:+.6f}·m^{deg-i}" for i, c in enumerate(coeffs[:-1])]
    terms.append(f"{coeffs[-1]:+.6f}")
    print(f"  σ(m) = {' '.join(terms)}")
    print(f"  Coefficients (highest→lowest degree): {list(coeffs)}")

# Use degree 3 as final
coeffs3, ms, sigmas = fit_smile(train_records, degree=3)
pred3  = np.polyval(coeffs3, ms)
r2_3   = r_squared(sigmas, pred3)

print("\n" + "=" * 60)
print("FINAL: degree-3 smile polynomial")
print(f"  σ(m) = {coeffs3[0]:.6f}·m³ + {coeffs3[1]:.6f}·m² + {coeffs3[2]:.6f}·m + {coeffs3[3]:.6f}")
print(f"  where m = ln(K / spot)")

# ── FV table at reference spot=5250, per-day TTE ────────────────────────────

print("\n" + "=" * 60)
print("FV computed from smile vs. current static FV:")
print(f"{'Strike':<10} {'m':>8} {'σ_smile':>10} {'FV(TTE=7)':>12} {'FV(TTE=6)':>12} {'FV(TTE=5)':>12} {'Static FV':>12}")
STATIC = {5000:255.0, 5100:166.8, 5200:95.6, 5300:46.8, 5400:15.95, 5500:6.64}
S_ref  = 5250.0
for K in STRIKES:
    m      = math.log(K / S_ref)
    sigma  = float(np.polyval(coeffs3, m))
    sigma  = max(0.05, min(sigma, 2.0))
    fvs    = [bs_call(S_ref, K, tte/252, sigma) for tte in [7, 6, 5]]
    static = STATIC.get(K, "—")
    print(f"  VEV_{K:<5} {m:>8.4f} {sigma:>10.4f} "
          f"{fvs[0]:>12.2f} {fvs[1]:>12.2f} {fvs[2]:>12.2f} {static:>12}")

# ── Validate on day 0 ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Validation: computing smile FV error on day 0...")
val_records = collect_ivs([0])
print(f"  {len(val_records):,} observations")

val_ms     = np.array([r[0] for r in val_records])
val_sigmas = np.array([r[1] for r in val_records])
val_pred   = np.polyval(coeffs3, val_ms)
r2_val     = r_squared(val_sigmas, val_pred)
resid_val  = np.std(val_sigmas - val_pred)
print(f"  R²={r2_val:.4f}  residual σ={resid_val:.4f}")
print(f"  (Train R²={r2_3:.4f} for reference)")

print("\n" + "=" * 60)
print("COPY THIS INTO THE TRADER:")
c = coeffs3
print(f"    SMILE_COEFFS = ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}, {c[3]:.6f})")
print(f"    # σ(m) = C0·m³ + C1·m² + C2·m + C3   where m = ln(K/spot)")
print(f"    SMILE_SIGMA_MIN = 0.10")
print(f"    SMILE_SIGMA_MAX = 0.60")
