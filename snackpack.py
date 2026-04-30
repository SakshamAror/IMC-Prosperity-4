# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_mid(od):
    if od and od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None

def send_order(result, symbol, price, qty):
    if qty == 0: return
    result.setdefault(symbol, []).append(Order(symbol, int(price), int(qty)))

def update_ema(prev, val, span):
    a = 2.0 / (span + 1)
    return a * val + (1 - a) * prev

def rolling_mean_std(window):
    n = len(window)
    if n < 2: return 0.0, 1.0
    m = sum(window) / n
    s = sqrt(sum((x - m)**2 for x in window) / (n - 1))
    return m, max(s, 1.0)

def zscore(series):
    if len(series) < 10: return 0.0
    m, s = rolling_mean_std(series)
    return (series[-1] - m) / s

def bb(od): return max(od.buy_orders)
def ba(od): return min(od.sell_orders)

def hold_directional(symbol, od, pos, target, result):
    """Aggressively take liquidity to reach target position."""
    if not od.buy_orders or not od.sell_orders: return
    delta = target - pos
    if delta == 0: return
    if delta > 0:
        qty = min(delta, POS_LIMIT - pos)
        send_order(result, symbol, ba(od), qty)
    else:
        qty = max(delta, -(POS_LIMIT + pos))
        send_order(result, symbol, bb(od), qty)

def ema_mm(symbol, od, pos, ema, span, thresh, result):
    """EMA market making with MR skew. Returns updated EMA."""
    if not od.buy_orders or not od.sell_orders: return ema
    best_b, best_a = bb(od), ba(od)
    mid = (best_b + best_a) / 2
    ema = update_ema(ema, mid, span)
    dev = mid - ema

    our_bid = int(ema) - 1
    our_ask = int(ema) + 2

    if dev > thresh:
        skew = min(int(dev / thresh) * 2, 5)
        our_ask -= skew; our_bid -= skew
    elif dev < -thresh:
        skew = min(int(-dev / thresh) * 2, 5)
        our_bid += skew; our_ask += skew

    our_bid = min(our_bid, best_b)
    our_ask = max(our_ask, best_a)
    if our_bid >= our_ask: return ema

    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos
    if buy_cap  > 0: send_order(result, symbol, our_bid,  buy_cap)
    if sell_cap > 0: send_order(result, symbol, our_ask, -sell_cap)
    return ema


# ══════════════════════════════════════════════════════════════════════════════
#  SNACKPACK
#
#  Chart reading:
#    STRAWBERRY  → clean uptrend   +735  → hold +10, NO MM
#    CHOCOLATE   → clean downtrend -293  → hold -10, NO MM
#    PISTACHIO   → steepest down   -311  → hold -10, NO MM
#    RASPBERRY   → flat ±66              → aggressive EMA MM
#    VANILLA     → mild drift ±175       → aggressive EMA MM
#    CHOC-PIST spread (corr=0.82)        → tilt within shorts
#    RASP-VAN spread  (corr=0.61)        → overlay on MM
# ══════════════════════════════════════════════════════════════════════════════

SNACK_PRODUCTS = [
    "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA",
]

CP_WINDOW = 200; CP_ENTRY_Z = 1.4; CP_EXIT_Z = 0.25
RV_WINDOW = 150; RV_ENTRY_Z = 1.2; RV_EXIT_Z = 0.25

def snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result):
    mc, mp = get_mid(od_c), get_mid(od_p)
    if mc is None or mp is None: return
    cp_hist.append(mc - mp)
    if len(cp_hist) > CP_WINDOW: cp_hist[:] = cp_hist[-CP_WINDOW:]
    if len(cp_hist) < 20: return
    z = zscore(cp_hist)
    tgt_c = -7 if z > CP_ENTRY_Z else (-POS_LIMIT if z < -CP_ENTRY_Z else -POS_LIMIT)
    tgt_p = -POS_LIMIT if z > CP_ENTRY_Z else (-7 if z < -CP_ENTRY_Z else -POS_LIMIT)
    hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, tgt_c, result)
    hold_directional("SNACKPACK_PISTACHIO",  od_p, pos_p, tgt_p, result)

def snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result):
    mr, mv = get_mid(od_r), get_mid(od_v)
    if mr is None or mv is None: return
    rv_hist.append(mr - mv)
    if len(rv_hist) > RV_WINDOW: rv_hist[:] = rv_hist[-RV_WINDOW:]
    if len(rv_hist) < 20: return
    z = zscore(rv_hist)
    PAIR = 3
    if z > RV_ENTRY_Z:
        if pos_r > -PAIR: send_order(result, "SNACKPACK_RASPBERRY", bb(od_r), -(pos_r + PAIR))
        if pos_v < PAIR:  send_order(result, "SNACKPACK_VANILLA",   ba(od_v),  PAIR - pos_v)
    elif z < -RV_ENTRY_Z:
        if pos_v > -PAIR: send_order(result, "SNACKPACK_VANILLA",   bb(od_v), -(pos_v + PAIR))
        if pos_r < PAIR:  send_order(result, "SNACKPACK_RASPBERRY", ba(od_r),  PAIR - pos_r)
    elif abs(z) < RV_EXIT_Z:
        if pos_r != 0:
            send_order(result, "SNACKPACK_RASPBERRY", bb(od_r) if pos_r > 0 else ba(od_r), -pos_r)
        if pos_v != 0:
            send_order(result, "SNACKPACK_VANILLA",   bb(od_v) if pos_v > 0 else ba(od_v), -pos_v)

def run_snackpack(state: TradingState, mem: dict, result: dict):
    ods, positions = state.order_depths, state.position
    emas    = mem.get("snk_ema", {})
    cp_hist = mem.get("snk_cp",  [])
    rv_hist = mem.get("snk_rv",  [])

    od_c = ods.get("SNACKPACK_CHOCOLATE")
    od_p = ods.get("SNACKPACK_PISTACHIO")
    od_r = ods.get("SNACKPACK_RASPBERRY")
    od_s = ods.get("SNACKPACK_STRAWBERRY")
    od_v = ods.get("SNACKPACK_VANILLA")

    pos_c = positions.get("SNACKPACK_CHOCOLATE",  0)
    pos_p = positions.get("SNACKPACK_PISTACHIO",  0)
    pos_r = positions.get("SNACKPACK_RASPBERRY",  0)
    pos_s = positions.get("SNACKPACK_STRAWBERRY", 0)
    pos_v = positions.get("SNACKPACK_VANILLA",    0)

    # Strawberry: directional long
    if od_s: hold_directional("SNACKPACK_STRAWBERRY", od_s, pos_s, +POS_LIMIT, result)

    # Choc + Pist: directional short + spread overlay
    if od_c and od_p:
        snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result)
    else:
        if od_c: hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, -POS_LIMIT, result)
        if od_p: hold_directional("SNACKPACK_PISTACHIO",  od_p, pos_p, -POS_LIMIT, result)

    # Raspberry: aggressive EMA MM
    if od_r:
        mid_r = get_mid(od_r)
        if mid_r:
            emas.setdefault("SNACKPACK_RASPBERRY", mid_r)
            emas["SNACKPACK_RASPBERRY"] = ema_mm(
                "SNACKPACK_RASPBERRY", od_r, pos_r, emas["SNACKPACK_RASPBERRY"], 200, 30, result)

    # Vanilla: aggressive EMA MM
    if od_v:
        mid_v = get_mid(od_v)
        if mid_v:
            emas.setdefault("SNACKPACK_VANILLA", mid_v)
            emas["SNACKPACK_VANILLA"] = ema_mm(
                "SNACKPACK_VANILLA", od_v, pos_v, emas["SNACKPACK_VANILLA"], 200, 38, result)

    # Rasp-Van pair overlay
    if od_r and od_v:
        snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result)

    mem["snk_ema"] = emas
    mem["snk_cp"]  = cp_hist
    mem["snk_rv"]  = rv_hist


# ══════════════════════════════════════════════════════════════════════════════
#  ROBOTS
#
#  Chart reading:
#    MOPPING    → strong uptrend  to 12k+  → hold +10
#    DISHES     → uptrend D3-D4   to 11k+  → hold +10 (also strongest MR ac1=-0.23)
#    LAUNDRY    → mild uptrend            → hold +7
#    VACUUMING  → mild downtrend          → hold -6
#    IRONING    → strong downtrend to 7k  → hold -10
#    DISHES-IRONING spread (they diverge) → pair trade on top
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_PRODUCTS = [
    "ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
    "ROBOT_MOPPING", "ROBOT_VACUUMING",
]

# Directional targets from chart
ROBOT_TARGETS = {
    "ROBOT_MOPPING":    +10,   # strongest uptrend
    "ROBOT_DISHES":     +10,   # uptrend + strongest MR
    "ROBOT_LAUNDRY":    +7,    # mild uptrend
    "ROBOT_VACUUMING":  -6,    # mild downtrend
    "ROBOT_IRONING":    -10,   # strongest downtrend
}

# Dishes-Ironing pair spread — they diverge, trade the spread
DI_WINDOW = 300; DI_ENTRY_Z = 1.5; DI_EXIT_Z = 0.3

def run_robots(state: TradingState, mem: dict, result: dict):
    ods, positions = state.order_depths, state.position
    emas   = mem.get("rob_ema", {})
    di_hist = mem.get("rob_di",  [])

    # Dishes-Ironing spread overlay
    od_d = ods.get("ROBOT_DISHES")
    od_i = ods.get("ROBOT_IRONING")
    md = get_mid(od_d)
    mi = get_mid(od_i)

    if od_d and od_i and md and mi:
        di_hist.append(md - mi)
        if len(di_hist) > DI_WINDOW: di_hist[:] = di_hist[-DI_WINDOW:]

    # All robots: hold directional target
    for symbol, target in ROBOT_TARGETS.items():
        od  = ods.get(symbol)
        pos = positions.get(symbol, 0)
        if od is None: continue
        hold_directional(symbol, od, pos, target, result)

    # Dishes MR overlay on top of trend (ac1=-0.23 is strong)
    # When Dishes spikes far above its own EMA → lean sell slightly
    # This captures short-term reversions while still holding the trend
    if od_d and md:
        emas.setdefault("ROBOT_DISHES", md)
        emas["ROBOT_DISHES"] = update_ema(emas["ROBOT_DISHES"], md, 100)
        dev = md - emas["ROBOT_DISHES"]
        pos_d = positions.get("ROBOT_DISHES", 0)
        # Only sell a little when it spikes — never below trend target of +7
        if dev > 80 and pos_d > 7:
            send_order(result, "ROBOT_DISHES", bb(od_d), -(pos_d - 7))
        elif dev < -80 and pos_d < POS_LIMIT:
            send_order(result, "ROBOT_DISHES", ba(od_d), POS_LIMIT - pos_d)

    mem["rob_ema"] = emas
    mem["rob_di"]  = di_hist

# ══════════════════════════════════════════════════════════════════════════════
#  NORMAL TRADER ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

import json

class Trader:
    def run(self, state: TradingState):
        # persistent memory
        mem = json.loads(state.traderData) if state.traderData else {}

        result = {}

        # ── run strategy modules ──────────────────────────────────────────────
        run_snackpack(state, mem, result)
        run_robots(state, mem, result)

        # (optional hook for future UV/TRANSLATOR module if you wire it in)
        # compute_uv_translator_targets(...)

        trader_data = json.dumps(mem)
        return result, 0, trader_data