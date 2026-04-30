"""
optimize_modular.py — Random search over the 15 least-overfit-prone hyperparameters
in clean_modularS.py.

Train:      ROUND_5 days 3 + 4
Validation: ROUND_5 day 2  (shown live on every new best)

Run with:
    .venv/bin/python -u optimize_modular.py
"""
import sys, random, time, copy
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '.venv/lib/python3.14/site-packages')
sys.path.insert(0, '.venv/lib/python3.14/site-packages/prosperity4bt')

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from prosperity4bt.runner import run_backtest
from prosperity4bt.file_reader import FileReader
from prosperity4bt.models import TradeMatchingMode

# ── Custom file reader ─────────────────────────────────────────────────────────
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

# ── Load base module ───────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location('trader_mod', 'clean_modularS.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

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
    total = 0
    for round_num, day_num in days:
        trader = trader_cls()
        result = run_backtest(trader, FR, round_num, day_num, False, TMM, False, False)
        total += get_pnl(result)
    return total

# ── Search space (15 least-overfit-prone hyperparameters) ─────────────────────
#
#  Excluded top-7 (highest overfit risk):
#    POWER, EXIT_Z, ENTRY_Z, BOT_PNL_THRESHOLD, RISK_MULT, BASE_RISK, SECONDARY_SCALE
#
#  Rank  Parameter                         Current    Type    Range
#  8     TREND_HOLD_GS_PLANETARY_RINGS         7      int     3 – 10
#  8     TREND_HOLD_GARLIC                    10      int     5 – 10
#  10    BOT_STATS_DECAY                   0.995    float   0.970 – 0.9995
#  11    PEBBLES_MAX_HOLD                    200      int    80 – 400
#  11    ROBOT_MAX_HOLD                      150      int    50 – 300
#  12    PEBBLES_WINDOW                      500      int   200 – 800
#  12    ROBOT_WINDOW                        550      int   200 – 800
#  13    BOT_FWD_HORIZON                      20      int     5 – 60
#  14    BOT_RECENT_TICKS                    100      int    30 – 300
#  15    ORANGE_AVG_WINDOW                   200      int   100 – 400
#  16    BOT_MIN_TRADES                       20      int     5 – 60
#  17    BG_POS                               10      int     3 – 10
#  18    BOT_LOT                               2      int     1 – 5
#  19    DIR_BUDGET_ORANGE                     8      int     4 – 10
#  19    DIR_BUDGET_YELLOW                     8      int     4 – 10

DEFAULTS = {
    'TREND_HOLD_GS_PLANETARY_RINGS': 7,
    'TREND_HOLD_GARLIC':             10,
    'BOT_STATS_DECAY':               0.995,
    'PEBBLES_MAX_HOLD':              200,
    'ROBOT_MAX_HOLD':                150,
    'PEBBLES_WINDOW':                500,
    'ROBOT_WINDOW':                  550,
    'BOT_FWD_HORIZON':               20,
    'BOT_RECENT_TICKS':              100,
    'ORANGE_AVG_WINDOW':             200,
    'BOT_MIN_TRADES':                20,
    'BG_POS':                        10,
    'BOT_LOT':                       2,
    'DIR_BUDGET_ORANGE':             8,
    'DIR_BUDGET_YELLOW':             8,
}

SPACE = {
    'TREND_HOLD_GS_PLANETARY_RINGS': (3,   10,    'int'),
    'TREND_HOLD_GARLIC':             (5,   10,    'int'),
    'BOT_STATS_DECAY':               (0.970, 0.9995, 'float'),
    'PEBBLES_MAX_HOLD':              (80,  400,   'int'),
    'ROBOT_MAX_HOLD':                (50,  300,   'int'),
    'PEBBLES_WINDOW':                (200, 800,   'int'),
    'ROBOT_WINDOW':                  (200, 800,   'int'),
    'BOT_FWD_HORIZON':               (5,   60,    'int'),
    'BOT_RECENT_TICKS':              (30,  300,   'int'),
    'ORANGE_AVG_WINDOW':             (100, 400,   'int'),
    'BOT_MIN_TRADES':                (5,   60,    'int'),
    'BG_POS':                        (3,   10,    'int'),
    'BOT_LOT':                       (1,   5,     'int'),
    'DIR_BUDGET_ORANGE':             (4,   10,    'int'),
    'DIR_BUDGET_YELLOW':             (4,   10,    'int'),
}

def sample():
    p = {}
    for k, (lo, hi, t) in SPACE.items():
        p[k] = random.randint(lo, hi) if t == 'int' else round(random.uniform(lo, hi), 6)
    return p

# ── Trader factory ─────────────────────────────────────────────────────────────
def make_trader(params):
    PebblesCls = type('PebblesCls', (M.PebblesSector,), {
        'WINDOW':   params['PEBBLES_WINDOW'],
        'MAX_HOLD': params['PEBBLES_MAX_HOLD'],
        'BG_POS':   params['BG_POS'],
    })

    RobotCls = type('RobotCls', (M.RobotSector,), {
        'WINDOW':   params['ROBOT_WINDOW'],
        'MAX_HOLD': params['ROBOT_MAX_HOLD'],
    })

    DirCls = type('DirCls', (M.DirectionalSector,), {
        'BOT_STATS_DECAY':  params['BOT_STATS_DECAY'],
        'BOT_FWD_HORIZON':  params['BOT_FWD_HORIZON'],
        'BOT_RECENT_TICKS': params['BOT_RECENT_TICKS'],
        'BOT_MIN_TRADES':   params['BOT_MIN_TRADES'],
        'BOT_LOT':          params['BOT_LOT'],
        'ORANGE_AVG_WINDOW': params['ORANGE_AVG_WINDOW'],
        'MID_HIST_LEN':     params['ORANGE_AVG_WINDOW'] + 20,
        'DIR_BUDGET': {
            **M.DirectionalSector.DIR_BUDGET,
            'UV_VISOR_ORANGE': params['DIR_BUDGET_ORANGE'],
            'UV_VISOR_YELLOW': params['DIR_BUDGET_YELLOW'],
        },
    })

    GalaxyCls = type('GalaxyCls', (M.GalaxySoundsSector,), {
        'TREND_HOLD': {
            **M.GalaxySoundsSector.TREND_HOLD,
            'GALAXY_SOUNDS_PLANETARY_RINGS': params['TREND_HOLD_GS_PLANETARY_RINGS'],
        },
    })

    OxygenCls = type('OxygenCls', (M.OxygenShakeSector,), {
        'TREND_HOLD': {
            **M.OxygenShakeSector.TREND_HOLD,
            'OXYGEN_SHAKE_GARLIC': params['TREND_HOLD_GARLIC'],
        },
    })

    class CustomTrader(M.Trader):
        def __init__(self):
            self.sectors = [
                GalaxyCls(),
                M.MicrochipSector(),
                OxygenCls(),
                M.PanelSector(),
                M.SnackpackSector(),
                M.TranslatorSector(),
                DirCls(),
                PebblesCls(),
                RobotCls(),
                M.SleepPodSector(),
            ]

    return CustomTrader

# ── Baseline (default params) ──────────────────────────────────────────────────
BaselineTrader = make_trader(DEFAULTS)

print("=" * 72)
print("BASELINE (current clean_modularS.py params)")
print("=" * 72)
print("  Computing...", flush=True)
baseline_train = eval_trader(BaselineTrader, TRAIN_DAYS)
baseline_val   = eval_trader(BaselineTrader, VAL_DAYS)
print(f"  Train (days 3+4): {baseline_train:>12,.0f}")
print(f"  Val   (day  2  ): {baseline_val:>12,.0f}")
print()

# ── Random search ──────────────────────────────────────────────────────────────
N = 500
results = []

print(f"Random search: {N} trials   train=days 3+4   val=day 2")
print(f"  . trial done   | checkpoint/10   * new best (prints params + scores)")
print(f"  ", end="", flush=True)

t0             = time.time()
best_train     = float('-inf')
best_params    = None
best_val       = None

def print_best(rank_label, train_pnl, val_pnl, params):
    print(f"\n{'─'*72}")
    print(f"  NEW BEST  {rank_label}")
    print(f"  Train (days 3+4): {train_pnl:>12,.0f}   (baseline {baseline_train:,.0f}  delta {train_pnl-baseline_train:+,.0f})")
    print(f"  Val   (day  2  ): {val_pnl:>12,.0f}   (baseline {baseline_val:,.0f}  delta {val_pnl-baseline_val:+,.0f})")
    print(f"  {'Parameter':<35}  {'Value':>12}  {'Default':>12}  {'Delta':>8}")
    print(f"  {'─'*35}  {'─'*12}  {'─'*12}  {'─'*8}")
    for k, v in params.items():
        d = DEFAULTS[k]
        changed = ' <--' if v != d else ''
        if isinstance(v, float):
            print(f"  {k:<35}  {v:>12.6f}  {d:>12.6f}  {v-d:>+8.6f}{changed}")
        else:
            print(f"  {k:<35}  {v:>12}  {d:>12}  {v-d:>+8}{changed}")
    print(f"{'─'*72}")
    print(f"  ", end="", flush=True)

for i in range(N):
    params = sample()
    try:
        trader_cls = make_trader(params)
        train_pnl  = eval_trader(trader_cls, TRAIN_DAYS)
        results.append((train_pnl, params))

        is_best = train_pnl > best_train
        if is_best:
            best_train  = train_pnl
            best_params = params
            val_pnl     = eval_trader(trader_cls, VAL_DAYS)
            best_val    = val_pnl
            print_best(f"trial {i+1}", train_pnl, val_pnl, params)
        else:
            marker = '|' if (i + 1) % 10 == 0 else '.'
            at_ck  = (i + 1) % 50 == 0
            print(marker, end='' if not at_ck else '\n', flush=True)

    except Exception as e:
        print('E', end='', flush=True)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate    = elapsed / (i + 1)
        eta     = rate * (N - i - 1)
        print(f"  [{i+1:3d}/{N}]  best_train={best_train:,.0f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
        print(f"  ", end="", flush=True)

print()
results.sort(key=lambda x: x[0], reverse=True)
elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s  ({elapsed/N:.2f}s per trial)")

# ── Final report ───────────────────────────────────────────────────────────────
TOP_K = 10
print(f"\n{'='*72}")
print(f"TOP {TOP_K} BY TRAIN PnL")
print(f"{'='*72}")
for rank, (pnl, params) in enumerate(results[:TOP_K], 1):
    print(f"\n#{rank}  Train: {pnl:,.0f}  (baseline: {baseline_train:,.0f}  delta: {pnl-baseline_train:+,.0f})")
    for k, v in params.items():
        d      = DEFAULTS[k]
        marker = ' <--' if v != d else ''
        if isinstance(v, float):
            print(f"    {k:<35} = {v:.6f}  (default {d:.6f}){marker}")
        else:
            print(f"    {k:<35} = {v}  (default {d}){marker}")

print(f"\n{'='*72}")
print("OVERFITTING CHECK — best train params on held-out day 2")
print(f"{'='*72}")
print(f"  Best train PnL (days 3+4): {best_train:>12,.0f}")
print(f"  Val PnL        (day  2  ): {best_val:>12,.0f}")
print(f"  Baseline train (days 3+4): {baseline_train:>12,.0f}")
print(f"  Baseline val   (day  2  ): {baseline_val:>12,.0f}")
train_delta = best_train - baseline_train
val_delta   = best_val   - baseline_val
print(f"  Train delta vs baseline:   {train_delta:>+12,.0f}")
print(f"  Val   delta vs baseline:   {val_delta:>+12,.0f}")
ratio = val_delta / train_delta if train_delta != 0 else float('nan')
print(f"  Val/Train delta ratio:     {ratio:>12.2f}  "
      f"({'OK' if ratio > 0.4 else 'OVERFIT WARNING'})")

print(f"\n{'='*72}")
print("BEST PARAMS (copy into clean_modularS.py)")
print(f"{'='*72}")
for k, v in best_params.items():
    if isinstance(v, float):
        print(f"    {k:<35} = {v:.6f}")
    else:
        print(f"    {k:<35} = {v}")
