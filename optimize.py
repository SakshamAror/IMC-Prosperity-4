"""
Random search optimizer for round4_EMA.py.
Trains on ROUND_3 days 0, 1, 2.  Day 3 in ROUND_4 held out for validation.

Run with:
    .venv/bin/python -u optimize.py
The -u flag disables output buffering so you see progress in real time.
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
        # path_parts = ['round3', 'prices_round_3_day_0.csv']
        round_folder = path_parts[0]          # 'round3' or 'round4'
        n = round_folder[5:]                  # '3' or '4'
        fname = path_parts[-1]
        p = self._base / f"ROUND_{n}" / fname
        if not p.is_file():
            return _cm(None)
        return _cm(p)

# ── Load base trader ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('trader_mod', 'round4_EMA.py')
BASE_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE_MOD)
BaseTrader = BASE_MOD.Trader

BASE_DIR = Path('.')
FR  = RoundFileReader(BASE_DIR)
TMM = TradeMatchingMode.all

TRAIN_DAYS = [(3, 0), (3, 1), (3, 2)]   # (round, day)
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
    trader = trader_cls()   # single instance — EMA state persists across days via instance vars
    for round_num, day_num in days:
        result = run_backtest(trader, FR, round_num, day_num, False, TMM, False, False)
        total += get_pnl(result)
    return total

# ── Parameter search space ─────────────────────────────────────────────────────
# (lo, hi, type)  type: 'float' | 'int'
SPACE = {
    # HYDROGEL EMA
    'HYDROGEL_EMA_ALPHA':       (0.01, 0.40, 'float'),
    'HYDROGEL_DRIFT_SCALE':     (0.01, 0.30, 'float'),
    # VELVETFRUIT EMA
    'VEV_EMA_ALPHA':            (0.01, 0.20, 'float'),
    'VEV_DRIFT_SCALE':          (0.10, 1.50, 'float'),
    # Options: ema_ticks and drift_scale per strike
    'OPT_4000_EMA_TICKS':       (20,  500,  'int'),
    'OPT_4000_DRIFT':           (0.05, 0.80, 'float'),
    'OPT_4500_EMA_TICKS':       (20,  500,  'int'),
    'OPT_4500_DRIFT':           (0.05, 0.80, 'float'),
    'OPT_5000_EMA_TICKS':       (20,  500,  'int'),
    'OPT_5000_DRIFT':           (0.05, 0.80, 'float'),
    'OPT_5100_EMA_TICKS':       (20,  500,  'int'),
    'OPT_5100_DRIFT':           (0.05, 0.80, 'float'),
    'OPT_5200_EMA_TICKS':       (20,  300,  'int'),
    'OPT_5200_DRIFT':           (0.05, 1.00, 'float'),
    'OPT_5300_EMA_TICKS':       (20,  300,  'int'),
    'OPT_5300_DRIFT':           (0.05, 1.00, 'float'),
    'OPT_5400_EMA_TICKS':       (10,  200,  'int'),
    'OPT_5400_DRIFT':           (0.05, 1.20, 'float'),
    'OPT_5500_EMA_TICKS':       (10,  200,  'int'),
    'OPT_5500_DRIFT':           (0.05, 1.20, 'float'),
}

def sample():
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        p[k] = random.randint(lo, hi) if t == 'int' else round(random.uniform(lo, hi), 3)
    return p

def make_trader(params):
    # Build modified _OPT_PARAMS from base, replacing ema_ticks + drift_scale
    base_opt = dict(BaseTrader._OPT_PARAMS)
    new_opt = {}
    for K, (fv, thresh, clip, agg, _et, _ds) in base_opt.items():
        et_key = f'OPT_{K}_EMA_TICKS'
        ds_key = f'OPT_{K}_DRIFT'
        ema_ticks  = params.get(et_key, _et)
        drift_scale = params.get(ds_key, _ds)
        new_opt[K] = (fv, thresh, clip, agg, ema_ticks, drift_scale)

    attrs = {
        'HYDROGEL_EMA_ALPHA':  params['HYDROGEL_EMA_ALPHA'],
        'HYDROGEL_DRIFT_SCALE': params['HYDROGEL_DRIFT_SCALE'],
        'VEV_EMA_ALPHA':       params['VEV_EMA_ALPHA'],
        'VEV_DRIFT_SCALE':     params['VEV_DRIFT_SCALE'],
        '_OPT_PARAMS':         new_opt,
    }
    return type('T', (BaseTrader,), attrs)

# ── Baseline ───────────────────────────────────────────────────────────────────
print("Computing baseline (current params)...", flush=True)
for round_num, day_num in TRAIN_DAYS:
    print(f"  Running round {round_num} day {day_num}...", flush=True)
baseline_train = eval_trader(BaseTrader, TRAIN_DAYS)
print(f"  Running round 4 day 3 (val)...", flush=True)
baseline_val   = eval_trader(BaseTrader, VAL_DAYS)
print(f"  Train (R3 days 0+1+2): {baseline_train:,.0f}")
print(f"  Val   (R4 day 3):      {baseline_val:,.0f}")
print()

# ── Random search ──────────────────────────────────────────────────────────────
N = 300
results = []  # [(train_pnl, params)]

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

base_params = {
    'HYDROGEL_EMA_ALPHA':  BaseTrader.HYDROGEL_EMA_ALPHA,
    'HYDROGEL_DRIFT_SCALE': BaseTrader.HYDROGEL_DRIFT_SCALE,
    'VEV_EMA_ALPHA':       BaseTrader.VEV_EMA_ALPHA,
    'VEV_DRIFT_SCALE':     BaseTrader.VEV_DRIFT_SCALE,
}
for K, (_, _, _, _, et, ds) in BaseTrader._OPT_PARAMS.items():
    base_params[f'OPT_{K}_EMA_TICKS'] = et
    base_params[f'OPT_{K}_DRIFT']     = ds

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
print(f"    HYDROGEL_EMA_ALPHA  = {best_params['HYDROGEL_EMA_ALPHA']}")
print(f"    HYDROGEL_DRIFT_SCALE = {best_params['HYDROGEL_DRIFT_SCALE']}")
print(f"    VEV_EMA_ALPHA       = {best_params['VEV_EMA_ALPHA']}")
print(f"    VEV_DRIFT_SCALE     = {best_params['VEV_DRIFT_SCALE']}")
print(f"    _OPT_PARAMS = {{")
for K, (fv, thresh, clip, agg, _et, _ds) in BaseTrader._OPT_PARAMS.items():
    et = best_params.get(f'OPT_{K}_EMA_TICKS', _et)
    ds = best_params.get(f'OPT_{K}_DRIFT', _ds)
    print(f"        {K}: ({fv}, {thresh}, {clip}, {agg}, {et}, {ds}),")
print(f"    }}")
