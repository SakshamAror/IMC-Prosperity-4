"""
Random search optimizer for round4_ema.py — EMA-blended FV hyperparameters.
Baseline = all EMA weights = 0 (anchored to static FVs).

Run with:
    .venv/bin/python -u optimize.py
"""
import sys, random, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.venv/lib/python3.14/site-packages')
sys.path.insert(0, '.venv/lib/python3.14/site-packages/prosperity4bt')

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from prosperity4bt.runner import run_backtest
from prosperity4bt.file_reader import FileReader
from prosperity4bt.models import TradeMatchingMode

# ── Custom file reader mapping round{N} -> ROUND_{N}/ ─────────────────────────
@contextmanager
def _cm(val):
    yield val

class RoundFileReader(FileReader):
    def __init__(self, base: Path) -> None:
        self._base = base

    def file(self, path_parts):
        round_folder = path_parts[0]
        n = round_folder[5:]
        fname = path_parts[-1]
        p = self._base / f"ROUND_{n}" / fname
        if not p.is_file():
            return _cm(None)
        return _cm(p)

# ── Load base trader ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('trader_mod', 'round4_ema.py')
BASE_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE_MOD)
BaseTrader = BASE_MOD.Trader

BASE_DIR = Path('.')
FR  = RoundFileReader(BASE_DIR)
TMM = TradeMatchingMode.all

TRAIN_DAYS = [(3, 1), (3, 2), (4, 3)]
VAL_DAYS   = [(3, 0)]

# ── PnL extraction ─────────────────────────────────────────────────────────────
def get_pnl(result):
    final = {}
    for row in result.activity_logs:
        c = row.columns
        final[c[2]] = c[-1]
    return sum(final.values())

def eval_trader(trader_cls, days):
    # Fresh instance per day: no inter-day instance-var bleed (theta TTE, etc.)
    total = 0
    for round_num, day_num in days:
        trader = trader_cls()
        result = run_backtest(trader, FR, round_num, day_num, False, TMM, False, False)
        total += get_pnl(result)
    return total

# ── Parameter search space ─────────────────────────────────────────────────────
# ALPHA: EMA decay rate.  alpha=0 → no update (effectively disabled).
#   alpha=0.0001 ≈ 10k-tick window;  0.0005 ≈ 2k-tick window;  0.00001 ≈ 100k-tick.
# WEIGHT: blend factor.  0 = anchor 100% to static FV;  0.5 = half-weight EMA.
SPACE = {
    # HYDROGEL EMA
    'HYD_EMA_ALPHA':   (0.0, 0.00005, 'float'),
    'HYD_EMA_WEIGHT':  (0.0, 0.50,   'float'),
    # VELVETFRUIT EMA
    'VEV_EMA_ALPHA':   (0.0, 0.00005, 'float'),
    'VEV_EMA_WEIGHT':  (0.0, 0.50,   'float'),
    # OPTIONS EMA
    'OPT_EMA_ALPHA':   (0.0, 0.00005, 'float'),
    'OPT_EMA_WEIGHT':  (0.0, 0.50,   'float'),
}

def sample():
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        if t == 'int':
            p[k] = random.randint(lo, hi)
        else:
            # extra precision for very small alphas
            p[k] = round(random.uniform(lo, hi), 6)
    return p

def make_trader(params):
    attrs = {k: params[k] for k in SPACE if k in params}
    return type('T', (BaseTrader,), attrs)

# Baseline: force all EMA weights to 0 (anchored to static FVs only)
BaselineTrader = type('Baseline', (BaseTrader,), {
    'HYD_EMA_WEIGHT': 0.0,
    'VEV_EMA_WEIGHT': 0.0,
    'OPT_EMA_WEIGHT': 0.0,
})

# ── Baseline ───────────────────────────────────────────────────────────────────
print("Computing baseline (current params)...", flush=True)
for round_num, day_num in TRAIN_DAYS:
    print(f"  Running round {round_num} day {day_num}...", flush=True)
baseline_train = eval_trader(BaselineTrader, TRAIN_DAYS)
print(f"  Running round 4 day 3 (val)...", flush=True)
baseline_val   = eval_trader(BaselineTrader, VAL_DAYS)
print(f"  Train (R3 days 0+1+2): {baseline_train:,.0f}")
print(f"  Val   (R4 day 3):      {baseline_val:,.0f}")
print()

# ── Random search ──────────────────────────────────────────────────────────────
N = 300
results = []

print(f"Running {N} random trials on R3 days 0+1+2  (one dot = 1 trial)...")
print(f"  Legend:  . = trial done  | = checkpoint every 10  * = new best")
print(f"  ", end="", flush=True)
t0 = time.time()
best_so_far = float('-inf')

for i in range(N):
    params = sample()
    try:
        trader_cls = make_trader(params)
        pnl = eval_trader(trader_cls, TRAIN_DAYS)
        results.append((pnl, params))
        is_best = pnl > best_so_far
        if is_best:
            best_so_far = pnl
        marker = "*" if is_best else ("|" if (i + 1) % 10 == 0 else ".")
        at_checkpoint = (i + 1) % 50 == 0
        print(marker, end="" if not at_checkpoint else "\n", flush=True)
    except Exception as e:
        print("E", end="", flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate    = elapsed / (i + 1)
        eta     = rate * (N - i - 1)
        print(f"  [{i+1:3d}/{N}]  best={best_so_far:,.0f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        print(f"  ", end="", flush=True)

print()
results.sort(key=lambda x: x[0], reverse=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s  ({elapsed/N:.2f}s per trial)")

# ── Report top results ─────────────────────────────────────────────────────────
TOP_K = 10
print(f"\n{'='*70}")
print(f"TOP {TOP_K} RESULTS (train = R3 days 0+1+2)")
print(f"{'='*70}")

# Safe baseline comparison (only for params that exist on BaseTrader)
base_params = {k: getattr(BaseTrader, k, None) for k in SPACE.keys()}

for rank, (pnl, params) in enumerate(results[:TOP_K], 1):
    print(f"\n#{rank}  Train PnL: {pnl:,.0f}  (baseline: {baseline_train:,.0f})")
    for k, v in params.items():
        bv = base_params.get(k)
        marker = ' <--' if v != bv else ''
        print(f"    {k:<35} = {v}{marker}")

# ── Validate best on R4 day 3 ──────────────────────────────────────────────────
print(f"\n{'='*70}")
print("OVERFITTING CHECK: best params on held-out R4 day 3")
print(f"{'='*70}")
best_pnl, best_params = results[0]
best_trader_cls = make_trader(best_params)
val_pnl = eval_trader(best_trader_cls, VAL_DAYS)
print(f"  Best train PnL (R3 days 0+1+2): {best_pnl:,.0f}")
print(f"  Val PnL        (R4 day 3):       {val_pnl:,.0f}")
print(f"  Baseline val   (R4 day 3):       {baseline_val:,.0f}")
delta = val_pnl - baseline_val
print(f"  Delta vs baseline val:           {delta:+,.0f}  "
      f"({'OK' if delta > -abs(baseline_val) * 0.1 else 'OVERFIT WARNING'})")

# ── Print best params as ready-to-paste code ──────────────────────────────────
print(f"\n{'='*70}")
print("BEST PARAMS (copy into Trader class):")
print(f"{'='*70}")
_, best_params = results[0]
for k in SPACE:
    print(f"    {k:<35} = {best_params[k]}")
