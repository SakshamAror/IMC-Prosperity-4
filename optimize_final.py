"""
Smart random-search optimizer for round4_final.py.
Train: ROUND_3 days 0, 1, 2.   Validation: ROUND_4 day 3 (held out).

80% of trials: full random sampling from search space.
20% of trials: local perturbation of the current best (exploitation).

Run with:
    .venv/bin/python -u optimize_final.py
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

# ── Custom file reader: round{N} → ROUND_{N}/ ─────────────────────────────────
@contextmanager
def _cm(val):
    yield val

class RoundFileReader(FileReader):
    def __init__(self, base: Path) -> None:
        self._base = base
    def file(self, path_parts):
        n = path_parts[0][5:]          # 'round3' → '3'
        p = self._base / f"ROUND_{n}" / path_parts[-1]
        return _cm(p if p.is_file() else None)

# ── Load base trader ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('trader_mod', 'round4_final.py')
BASE_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE_MOD)
BaseTrader = BASE_MOD.Trader

FR  = RoundFileReader(Path('.'))
TMM = TradeMatchingMode.all

TRAIN_DAYS = [(3, 0), (3, 1), (3, 2)]
VAL_DAYS   = [(4, 3)]

# ── PnL extraction ─────────────────────────────────────────────────────────────
def get_pnl(result):
    final = {}
    for row in result.activity_logs:
        c = row.columns
        final[c[2]] = c[-1]
    return sum(final.values())

def eval_trader(trader_cls, days):
    total = 0
    trader = trader_cls()
    for round_num, day_num in days:
        result = run_backtest(trader, FR, round_num, day_num, False, TMM, False, False)
        total += get_pnl(result)
    return total

# ── Search space ───────────────────────────────────────────────────────────────
# (lo, hi, type)  type: 'float' | 'int'
SPACE = {
    # HYDROGEL
    'HYDROGEL_FAIR_VALUE':           (9988.0, 9999.0, 'float'),
    'HYDROGEL_MISPRICE_THRESHOLD':   (1,      10,     'int'),
    'HYDROGEL_DEVIATION_SCALE':      (0.2,    1.5,    'float'),
    'HYDROGEL_POSITION_SCALE':       (1,      8,      'int'),
    'HYDROGEL_TARGET_SCALE':         (1,      12,     'int'),
    'HYDROGEL_AGGRESSIVE_THRESHOLD': (10,     80,     'int'),
    'HYDROGEL_AGGRESSIVE_SIZE':      (5,      60,     'int'),
    'HYDROGEL_WARMUP_TICKS':         (0,      8000,   'int'),
    # VELVETFRUIT
    'VEV_FAIR_VALUE':                (5240.0, 5255.0, 'float'),
    'VEV_MISPRICE_THRESHOLD':        (1,      10,     'int'),
    'VEV_DEVIATION_SCALE':           (0.2,    1.5,    'float'),
    'VEV_POSITION_SCALE':            (1,      6,      'int'),
    'VEV_TARGET_SCALE':              (3,      20,     'int'),
    'VEV_AGGRESSIVE_THRESHOLD':      (10,     60,     'int'),
    'VEV_AGGRESSIVE_SIZE':           (5,      60,     'int'),
    'VEV_WARMUP_TICKS':              (0,      5000,   'int'),
    # OPTIONS (shared across all strikes)
    'OPT_TARGET_SCALE':              (5,      25,     'int'),
    'OPT_DEVIATION_SCALE':           (5,      50,     'int'),
    'OPT_POSITION_SCALE':            (1,      8,      'int'),
    'OPT_AGGRESSIVE_SIZE':           (1,      15,     'int'),
    'OPT_WARMUP_TICKS':              (0,      5000,   'int'),
}

EXPLOIT_PROB = 0.20
PERTURB_KEYS = 3

def _rand(k):
    lo, hi, t = SPACE[k]
    return random.randint(lo, hi) if t == 'int' else round(random.uniform(lo, hi), 3)

def _perturb_val(k, v):
    lo, hi, t = SPACE[k]
    span = hi - lo
    if t == 'int':
        delta = random.randint(-max(1, span // 8), max(1, span // 8))
        return max(lo, min(hi, v + delta))
    else:
        delta = random.uniform(-span * 0.12, span * 0.12)
        return max(lo, min(hi, round(v + delta, 3)))

def sample(best_params=None):
    if best_params is not None and random.random() < EXPLOIT_PROB:
        p = dict(best_params)
        for k in random.sample(list(SPACE), k=PERTURB_KEYS):
            p[k] = _perturb_val(k, p[k])
        return p
    return {k: _rand(k) for k in SPACE}

def make_trader(params):
    return type('T', (BaseTrader,), dict(params))

# ── Baseline (single pass, timed) ─────────────────────────────────────────────
print("=" * 70)
print("round4_final.py optimizer  |  train=R3d0+R3d1+R3d2  |  val=R4d3")
print("=" * 70)
print("Computing baseline...", flush=True)
t_base = time.time()
baseline_train = eval_trader(BaseTrader, TRAIN_DAYS)
print(f"  Train done in {time.time()-t_base:.1f}s", flush=True)
t_base = time.time()
baseline_val = eval_trader(BaseTrader, VAL_DAYS)
print(f"  Val   done in {time.time()-t_base:.1f}s", flush=True)
print(f"  Baseline train (R3 d0+1+2): {baseline_train:>12,.0f}")
print(f"  Baseline val   (R4 d3):     {baseline_val:>12,.0f}")
print()

# ── Random search ──────────────────────────────────────────────────────────────
N = 500
results = []

print(f"Running {N} trials  (exploit={int(EXPLOIT_PROB*100)}%  random={int((1-EXPLOIT_PROB)*100)}%)")
print(f"  . trial done   | checkpoint every 10   * new best   E error")
print(f"  ", end="", flush=True)

t0 = time.time()
best_pnl    = float('-inf')
best_params = None

for i in range(N):
    params = sample(best_params)
    try:
        pnl = eval_trader(make_trader(params), TRAIN_DAYS)
        results.append((pnl, params))
        is_best = pnl > best_pnl
        if is_best:
            best_pnl    = pnl
            best_params = params
        at_check = (i + 1) % 50 == 0
        marker = '*' if is_best else ('|' if (i + 1) % 10 == 0 else '.')
        print(marker, end='' if not at_check else '\n', flush=True)
    except Exception:
        print('E', end='', flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate    = elapsed / (i + 1)
        eta     = rate * (N - i - 1)
        pct     = (i + 1) / N * 100
        n_ok    = len(results)
        print(f"  [{i+1:3d}/{N}  {pct:.0f}%]  ok={n_ok}  best={best_pnl:,.0f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        print(f"  ", end="", flush=True)

print()
results.sort(key=lambda x: x[0], reverse=True)
elapsed = time.time() - t0
n_ok = len(results)
n_err = N - n_ok
print(f"\nSearch done in {elapsed:.0f}s  "
      f"({n_ok} ok / {n_err} errors / {elapsed/N:.2f}s per trial)")

if not results:
    print("No successful trials. Exiting.")
    sys.exit(1)

# ── Top results ────────────────────────────────────────────────────────────────
TOP_K = 10
base_vals = {k: getattr(BaseTrader, k) for k in SPACE}

print(f"\n{'='*70}")
print(f"TOP {min(TOP_K, len(results))} RESULTS  (train = R3 d0+1+2)")
print(f"{'='*70}")
for rank, (pnl, params) in enumerate(results[:TOP_K], 1):
    delta = pnl - baseline_train
    print(f"\n#{rank}  Train PnL: {pnl:,.0f}  ({delta:+,.0f} vs baseline)")
    for k, v in params.items():
        bv = base_vals[k]
        marker = ' <--' if v != bv else ''
        print(f"    {k:<40} = {v}{marker}")

# ── Validate top 5 on R4 day 3 ────────────────────────────────────────────────
print(f"\n{'='*70}")
print("OVERFITTING CHECK  —  top 5 on held-out R4 day 3")
print(f"{'='*70}")
val_results = []
for rank, (tr_pnl, params) in enumerate(results[:5], 1):
    try:
        vp = eval_trader(make_trader(params), VAL_DAYS)
        val_results.append((vp, tr_pnl, params, rank))
        delta = vp - baseline_val
        flag  = 'OK' if delta > -abs(baseline_val) * 0.1 else 'OVERFIT?'
        print(f"  #{rank}  train={tr_pnl:>12,.0f}  val={vp:>12,.0f}  "
              f"({delta:+,.0f} vs baseline)  {flag}", flush=True)
    except Exception as e:
        print(f"  #{rank}  ERROR: {e}", flush=True)

if not val_results:
    print("All validation runs failed. Cannot produce robust recommendation.")
    sys.exit(1)

val_results.sort(key=lambda x: x[0], reverse=True)
best_val_pnl, best_val_tr, best_val_params, best_rank = val_results[0]
print(f"\n  Most robust: #{best_rank}  val={best_val_pnl:,.0f}  train={best_val_tr:,.0f}")

# ── Copy-paste output ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BEST BY TRAIN  (copy into Trader):")
print(f"{'='*70}")
for k, v in results[0][1].items():
    if v != base_vals[k]:
        print(f"    {k} = {v}")

print(f"\n{'='*70}")
print("MOST ROBUST  (best val score among top-5 train):")
print(f"{'='*70}")
for k, v in best_val_params.items():
    if v != base_vals[k]:
        print(f"    {k} = {v}")
