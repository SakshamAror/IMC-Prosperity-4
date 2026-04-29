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
spec = importlib.util.spec_from_file_location('trader_mod', 'round5_MM.py')
BASE_MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE_MOD)
BaseTrader = BASE_MOD.Trader

BASE_DIR = Path('.')
FR  = RoundFileReader(BASE_DIR)
TMM = TradeMatchingMode.all

TRAIN_DAYS = [(5, 3), (5, 4)]
VAL_DAYS   = [(5, 2)]

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

SPACE = {
    # ── global regime control ONLY ──
    'VOL_THRESHOLD_SCALE': (0.7, 1.5, 'float'),
    'TREND_HOLD_SCALE': (0.7, 1.3, 'float'),
    'IMBALANCE_WEIGHT': (0.0, 0.6, 'float'),
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

# Baseline: current Round 5 strategy configuration (ENABLED + per-product params)
BaselineTrader = type('Baseline', (BaseTrader,), {

    # ── product enablement ─────────────────────────────────────
    'ENABLED': {
        # Galaxy Sounds
        'GALAXY_SOUNDS_BLACK_HOLES': True,
        'GALAXY_SOUNDS_DARK_MATTER': True,
        'GALAXY_SOUNDS_PLANETARY_RINGS': True,
        'GALAXY_SOUNDS_SOLAR_FLAMES': True,
        'GALAXY_SOUNDS_SOLAR_WINDS': True,

        # Microchip
        'MICROCHIP_CIRCLE': True,
        'MICROCHIP_OVAL': True,
        'MICROCHIP_RECTANGLE': True,
        'MICROCHIP_SQUARE': True,
        'MICROCHIP_TRIANGLE': True,

        # Oxygen Shake
        'OXYGEN_SHAKE_CHOCOLATE': True,
        'OXYGEN_SHAKE_EVENING_BREATH': True,
        'OXYGEN_SHAKE_GARLIC': True,
        'OXYGEN_SHAKE_MINT': True,
        'OXYGEN_SHAKE_MORNING_BREATH': True,

        # Panel
        'PANEL_1X2': True,
        'PANEL_1X4': True,
        'PANEL_2X2': True,
        'PANEL_2X4': True,
        'PANEL_4X4': True,

        # Pebbles
        'PEBBLES_XS': True,
        'PEBBLES_S': True,
        'PEBBLES_M': True,
        'PEBBLES_L': True,
        'PEBBLES_XL': True,

        # Robot
        'ROBOT_DISHES': True,
        'ROBOT_IRONING': True,
        'ROBOT_LAUNDRY': True,
        'ROBOT_MOPPING': True,
        'ROBOT_VACUUMING': True,

        # Sleep Pod
        'SLEEP_POD_COTTON': True,
        'SLEEP_POD_LAMB_WOOL': True,
        'SLEEP_POD_NYLON': True,
        'SLEEP_POD_POLYESTER': True,
        'SLEEP_POD_SUEDE': True,

        # Snackpack
        'SNACKPACK_CHOCOLATE': True,
        'SNACKPACK_PISTACHIO': True,
        'SNACKPACK_RASPBERRY': True,
        'SNACKPACK_STRAWBERRY': True,
        'SNACKPACK_VANILLA': True,

        # Translator
        'TRANSLATOR_ASTRO_BLACK': True,
        'TRANSLATOR_ECLIPSE_CHARCOAL': True,
        'TRANSLATOR_GRAPHITE_MIST': True,
        'TRANSLATOR_SPACE_GRAY': True,
        'TRANSLATOR_VOID_BLUE': True,

        # UV Visor
        'UV_VISOR_AMBER': True,
        'UV_VISOR_MAGENTA': True,
        'UV_VISOR_ORANGE': True,
        'UV_VISOR_RED': True,
        'UV_VISOR_YELLOW': True,
    },

    # ── per-product regime tuning (VOL_THRESHOLD) ───────────────
    'VOL_THRESHOLD': {
        'GALAXY_SOUNDS_BLACK_HOLES': 1.2,
        'GALAXY_SOUNDS_DARK_MATTER': 2.0,
        'GALAXY_SOUNDS_PLANETARY_RINGS': 1.2,
        'GALAXY_SOUNDS_SOLAR_FLAMES': 2.0,
        'GALAXY_SOUNDS_SOLAR_WINDS': 2.0,

        'MICROCHIP_CIRCLE': 2.0,
        'MICROCHIP_OVAL': 1.3,
        'MICROCHIP_RECTANGLE': 2.0,
        'MICROCHIP_SQUARE': 1.3,
        'MICROCHIP_TRIANGLE': 1.3,

        'OXYGEN_SHAKE_CHOCOLATE': 3.0,
        'OXYGEN_SHAKE_EVENING_BREATH': 3.5,
        'OXYGEN_SHAKE_GARLIC': 1.3,
        'OXYGEN_SHAKE_MINT': 2.0,
        'OXYGEN_SHAKE_MORNING_BREATH': 2.0,

        'PANEL_1X2': 2.0,
        'PANEL_1X4': 2.0,
        'PANEL_2X2': 2.0,
        'PANEL_2X4': 2.0,
        'PANEL_4X4': 2.0,

        'PEBBLES_L': 2.5,
        'PEBBLES_M': 2.5,
        'PEBBLES_S': 2.5,
        'PEBBLES_XL': 2.5,
        'PEBBLES_XS': 2.5,

        'ROBOT_DISHES': 4.0,
        'ROBOT_IRONING': 4.0,
        'ROBOT_LAUNDRY': 3.0,
        'ROBOT_MOPPING': 3.0,
        'ROBOT_VACUUMING': 3.0,

        'SLEEP_POD_COTTON': 1.4,
        'SLEEP_POD_LAMB_WOOL': 2.0,
        'SLEEP_POD_NYLON': 2.0,
        'SLEEP_POD_POLYESTER': 1.4,
        'SLEEP_POD_SUEDE': 1.4,

        'SNACKPACK_CHOCOLATE': 3.0,
        'SNACKPACK_PISTACHIO': 2.5,
        'SNACKPACK_RASPBERRY': 2.5,
        'SNACKPACK_STRAWBERRY': 2.5,
        'SNACKPACK_VANILLA': 2.5,

        'TRANSLATOR_ASTRO_BLACK': 2.0,
        'TRANSLATOR_ECLIPSE_CHARCOAL': 2.0,
        'TRANSLATOR_GRAPHITE_MIST': 2.0,
        'TRANSLATOR_SPACE_GRAY': 2.0,
        'TRANSLATOR_VOID_BLUE': 2.0,

        'UV_VISOR_AMBER': 1.2,
        'UV_VISOR_MAGENTA': 2.0,
        'UV_VISOR_ORANGE': 2.0,
        'UV_VISOR_RED': 2.0,
        'UV_VISOR_YELLOW': 2.0,
    },

    # ── per-product trend hold ──────────────────────────────────
    'TREND_HOLD_INIT': {
        'GALAXY_SOUNDS_BLACK_HOLES': 10,
        'GALAXY_SOUNDS_DARK_MATTER': 8,
        'GALAXY_SOUNDS_PLANETARY_RINGS': 10,
        'GALAXY_SOUNDS_SOLAR_FLAMES': 8,
        'GALAXY_SOUNDS_SOLAR_WINDS': 8,

        'MICROCHIP_CIRCLE': 8,
        'MICROCHIP_OVAL': 10,
        'MICROCHIP_RECTANGLE': 8,
        'MICROCHIP_SQUARE': 10,
        'MICROCHIP_TRIANGLE': 10,

        'OXYGEN_SHAKE_CHOCOLATE': 6,
        'OXYGEN_SHAKE_EVENING_BREATH': 6,
        'OXYGEN_SHAKE_GARLIC': 10,
        'OXYGEN_SHAKE_MINT': 8,
        'OXYGEN_SHAKE_MORNING_BREATH': 8,

        'ROBOT_DISHES': 5,
        'ROBOT_IRONING': 5,
        'ROBOT_LAUNDRY': 7,
        'ROBOT_MOPPING': 7,
        'ROBOT_VACUUMING': 7,

        'SLEEP_POD_COTTON': 10,
        'SLEEP_POD_LAMB_WOOL': 8,
        'SLEEP_POD_NYLON': 8,
        'SLEEP_POD_POLYESTER': 10,
        'SLEEP_POD_SUEDE': 10,

        'SNACKPACK_CHOCOLATE': 6,
        'SNACKPACK_PISTACHIO': 8,
        'SNACKPACK_RASPBERRY': 8,
        'SNACKPACK_STRAWBERRY': 8,
        'SNACKPACK_VANILLA': 8,

        'UV_VISOR_AMBER': 10,
        'UV_VISOR_MAGENTA': 8,
        'UV_VISOR_ORANGE': 8,
        'UV_VISOR_RED': 8,
        'UV_VISOR_YELLOW': 8,
    }

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
N = 400
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
