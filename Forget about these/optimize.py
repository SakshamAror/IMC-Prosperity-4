"""
Random search optimizer for round3_model_lever_mm.py.
Trains on days 1 and 2. Day 0 held out for overfitting check.

Run with:
    .venv/bin/python -u optimize.py
The -u flag disables output buffering so you see progress in real time.
"""
import sys, random, time
sys.stdout.reconfigure(line_buffering=True)   # flush after every line
sys.path.insert(0, '.venv/lib/python3.14/site-packages/prosperity4bt')

import importlib.util
from prosperity4bt.runner import run_backtest
from prosperity4bt.file_reader import FileSystemReader
from prosperity4bt.models import TradeMatchingMode
from pathlib import Path

# ── Load base trader ──────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('trader_mod', 'round3_model_lever_mm.py')
BASE_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE_MOD)
BaseTrader = BASE_MOD.Trader

FR  = FileSystemReader(Path('backtests'))
TMM = TradeMatchingMode.all

TRAIN_DAYS = [1, 2]
VAL_DAYS   = [0]

# ── PnL extraction ────────────────────────────────────────────────────────────
def get_pnl(result):
    final = {}
    for row in result.activity_logs:
        c = row.columns
        final[c[2]] = c[-1]
    return sum(final.values())

def eval_trader(trader_cls, days):
    total = 0
    for day in days:
        result = run_backtest(trader_cls(), FR, 3, day, False, TMM, False, False)
        total += get_pnl(result)
    return total

# ── Parameter search space ────────────────────────────────────────────────────
# (lo, hi, type)  type: 'float' | 'int'
SPACE = {
    # HYDROGEL
    'HYDROGEL_FAIR_VALUE':           (9984.0, 9996.0, 'float'),
    'HYDROGEL_MISPRICE_THRESHOLD':   (3,      10,     'int'),
    'HYDROGEL_TARGET_SCALE':         (4,      14,     'int'),
    'HYDROGEL_POSITION_SCALE':       (2,      9,      'int'),
    'HYDROGEL_DEVIATION_SCALE':      (0.2,    0.8,    'float'),
    'HYDROGEL_WARMUP_TICKS':         (0,      5000,   'int'),
    'HYDROGEL_AGGRESSIVE_THRESHOLD': (15,     50,     'int'),
    # VELVETFRUIT
    'VEV_FAIR_VALUE':                (5245.0, 5257.0, 'float'),
    'VEV_MISPRICE_THRESHOLD':        (2,      8,      'int'),
    'VEV_TARGET_SCALE':              (3,      14,     'int'),
    'VEV_POSITION_SCALE':            (1,      6,      'int'),
    'VEV_DEVIATION_SCALE':           (0.2,    0.8,    'float'),
    'VEV_WARMUP_TICKS':              (0,      3000,   'int'),
    # OPTIONS
    'OPT_TARGET_SCALE':              (5,      20,     'int'),
    'OPT_POSITION_SCALE':            (1,      5,      'int'),
    'OPT_DEVIATION_SCALE':           (0.2,    0.8,    'float'),
    'OPT_WARMUP_TICKS':              (0,      5000,   'int'),
}

def sample():
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        p[k] = random.randint(lo, hi) if t == 'int' else round(random.uniform(lo, hi), 2)
    return p

def make_trader(params):
    return type('T', (BaseTrader,), params)

# ── Baseline ──────────────────────────────────────────────────────────────────
print("Computing baseline (current params)...", flush=True)
print("  Running day 1...", flush=True)
baseline_train  = eval_trader(BaseTrader, [1])
print("  Running day 2...", flush=True)
baseline_train += eval_trader(BaseTrader, [2])
print("  Running day 0 (val)...", flush=True)
baseline_val    = eval_trader(BaseTrader, VAL_DAYS)
print(f"  Train (days 1+2): {baseline_train:,.0f}")
print(f"  Val   (day 0):    {baseline_val:,.0f}")
print()

# ── Random search ─────────────────────────────────────────────────────────────
N = 400
results = []  # [(train_pnl, params)]

print(f"Running {N} random trials on days 1+2  (one dot = 1 trial)...")
print(f"  Legend:  . = trial done  | = checkpoint every 10  * = new best")
print(f"  ", end="", flush=True)
t0 = time.time()
best_so_far = float('-inf')

for i in range(N):
    params = sample()
    try:
        pnl = eval_trader(make_trader(params), TRAIN_DAYS)
        results.append((pnl, params))
        is_best = pnl > best_so_far
        if is_best:
            best_so_far = pnl
        at_checkpoint = (i + 1) % 50 == 0
        marker = "*" if is_best else ("|" if (i + 1) % 10 == 0 else ".")
        print(marker, end="" if not at_checkpoint else "\n", flush=True)
    except Exception as e:
        print("E", end="", flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = elapsed / (i + 1)
        eta  = rate * (N - i - 1)
        print(f"  [{i+1:3d}/{N}]  best={best_so_far:,.0f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        print(f"  ", end="", flush=True)

print()
results.sort(key=lambda x: x[0], reverse=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s  ({elapsed/N:.2f}s per trial)")

# ── Report top results ────────────────────────────────────────────────────────
TOP_K = 10
print(f"\n{'='*70}")
print(f"TOP {TOP_K} RESULTS (train = days 1+2)")
print(f"{'='*70}")
for rank, (pnl, params) in enumerate(results[:TOP_K], 1):
    print(f"\n#{rank}  Train PnL: {pnl:,.0f}  (baseline: {baseline_train:,.0f})")
    for k, v in params.items():
        base_v = getattr(BaseTrader, k)
        marker = ' <--' if v != base_v else ''
        print(f"    {k:<40} = {v}{marker}")

# ── Validate best on day 0 ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("OVERFITTING CHECK: best params on held-out day 0")
print(f"{'='*70}")
best_pnl, best_params = results[0]
val_pnl = eval_trader(make_trader(best_params), VAL_DAYS)
print(f"  Best train PnL (days 1+2): {best_pnl:,.0f}")
print(f"  Val PnL        (day 0):    {val_pnl:,.0f}")
print(f"  Baseline val   (day 0):    {baseline_val:,.0f}")
delta = val_pnl - baseline_val
print(f"  Delta vs baseline val:     {delta:+,.0f}  ({'OK' if delta > -baseline_val*0.1 else 'OVERFIT WARNING'})")

# ── Print best params as ready-to-paste code ─────────────────────────────────
print(f"\n{'='*70}")
print("BEST PARAMS (copy into Trader class):")
print(f"{'='*70}")
_, best_params = results[0]
for k, v in best_params.items():
    base_v = getattr(BaseTrader, k)
    if v != base_v:
        print(f"    {k} = {v}")
