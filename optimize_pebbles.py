"""
Random search optimizer for round5_pebbles.py — PEBBLES_XL/XS z-score MR strategy.

Train: days 2 + 3  (in-sample)
Val:   day  4      (held-out — never seen during search)

Overfitting note:
  BG_POS is a directional bet size. An "optimal" value on 2 days of training
  data may not hold if the XL-XS trend weakens or reverses in live trading.
  Watch the val delta carefully for this param in particular.

Run with:
    .venv/bin/python -u optimize_pebbles.py
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


# ── File reader ───────────────────────────────────────────────────────────────
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

BASE_DIR = Path('.')
FR  = RoundFileReader(BASE_DIR)
TMM = TradeMatchingMode.all

TRAIN_DAYS = [(5, 2), (5, 3)]
VAL_DAYS   = [(5, 4)]


# ── Load strategy module ──────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('pebbles_mod', 'round5_pebbles.py')
pebbles_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pebbles_mod)


# ── Hyperparameter search space ───────────────────────────────────────────────
# (lo, hi, type)
# EXIT_Z is sampled relative to ENTRY_Z to enforce EXIT_Z < ENTRY_Z always.
SPACE = {
    'WINDOW':   (200,  1000, 'int'),
    'ENTRY_Z':  (1.5,  3.5,  'float'),
    'EXIT_Z':   (0.05, 0.9,  'float'),   # clamped to < ENTRY_Z after sampling
    'MAX_HOLD': (50,   600,  'int'),
    'BG_POS':   (0,    10,   'int'),
}

DEFAULTS = {
    'WINDOW':   pebbles_mod.WINDOW,
    'ENTRY_Z':  pebbles_mod.ENTRY_Z,
    'EXIT_Z':   pebbles_mod.EXIT_Z,
    'MAX_HOLD': pebbles_mod.MAX_HOLD,
    'BG_POS':   pebbles_mod.BG_POS,
}


def sample() -> dict:
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        p[k] = random.randint(int(lo), int(hi)) if t == 'int' else round(random.uniform(lo, hi), 3)
    # Enforce EXIT_Z < ENTRY_Z (exit must be tighter than entry)
    p['EXIT_Z'] = min(p['EXIT_Z'], p['ENTRY_Z'] - 0.1)
    return p


def apply_params(params: dict):
    for k, v in params.items():
        setattr(pebbles_mod, k, v)


# ── PnL extraction ────────────────────────────────────────────────────────────
def get_pnl(result) -> float:
    final = {}
    for row in result.activity_logs:
        c = row.columns
        final[c[2]] = c[-1]
    return sum(final.values())


def eval_params(params: dict, days: list) -> float:
    apply_params(params)
    total = 0.0
    for round_num, day_num in days:
        trader = pebbles_mod.Trader()
        result = run_backtest(trader, FR, round_num, day_num, False, TMM, False, False)
        total += get_pnl(result)
    return total


# ── Baseline (current params) ─────────────────────────────────────────────────
print("=" * 65)
print("PEBBLES optimizer — train=days2+3  val=day4")
print("=" * 65)
print()
print("Computing baseline (current params)...", flush=True)
for rn, dn in TRAIN_DAYS:
    print(f"  Running round {rn} day {dn}...", flush=True)
baseline_train = eval_params(DEFAULTS, TRAIN_DAYS)
print(f"  Running round 5 day 4 (val)...", flush=True)
baseline_val = eval_params(DEFAULTS, VAL_DAYS)
print(f"  Train (days 2+3): {baseline_train:,.0f}")
print(f"  Val   (day 4):    {baseline_val:,.0f}")
print()
print("Default params:")
for k, v in DEFAULTS.items():
    print(f"  {k:<12} = {v}")
print()


# ── Random search ─────────────────────────────────────────────────────────────
N = 400
results = []

print(f"Running {N} random trials on days 2+3  (one dot = 1 trial)")
print(f"  Legend:  . = trial  | = checkpoint every 10  * = new best  E = error")
print(f"  ", end="", flush=True)

t0 = time.time()
best_so_far = float('-inf')

for i in range(N):
    params = sample()
    try:
        pnl = eval_params(params, TRAIN_DAYS)
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
print(f"\nDone in {elapsed:.0f}s  ({elapsed/N:.2f}s per trial)\n")


# ── Top results ───────────────────────────────────────────────────────────────
TOP_K = 10
print("=" * 65)
print(f"TOP {TOP_K} RESULTS (train = days 2+3)")
print("=" * 65)

for rank, (pnl, params) in enumerate(results[:TOP_K], 1):
    print(f"\n#{rank}  Train PnL: {pnl:,.0f}  (baseline: {baseline_train:,.0f}  delta: {pnl-baseline_train:+,.0f})")
    for k, v in params.items():
        bv = DEFAULTS.get(k)
        marker = ' <--' if v != bv else ''
        print(f"    {k:<12} = {v}{marker}")


# ── Validate top 5 on held-out day 4 ─────────────────────────────────────────
print(f"\n{'='*65}")
print("OVERFITTING CHECK: top 5 train params on held-out day 4")
print(f"{'='*65}")
print(f"  Baseline val (day 4): {baseline_val:,.0f}\n")

for rank, (train_pnl, params) in enumerate(results[:5], 1):
    val_pnl = eval_params(params, VAL_DAYS)
    delta   = val_pnl - baseline_val
    flag    = "OK" if delta > -abs(baseline_val) * 0.15 else "OVERFIT WARNING"
    print(f"  #{rank}  train={train_pnl:,.0f}  val={val_pnl:,.0f}  "
          f"delta_vs_baseline={delta:+,.0f}  [{flag}]")


# ── Best params as copy-paste code ────────────────────────────────────────────
print(f"\n{'='*65}")
print("BEST PARAMS — copy into round5_pebbles.py:")
print(f"{'='*65}")
_, best_params = results[0]
for k, v in best_params.items():
    print(f"  {k:<12} = {v}")
