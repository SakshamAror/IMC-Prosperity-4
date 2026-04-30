
"""
round5_clustersGPT_860.py — Prosperity Trading Bot
====================================================
Architecture
------------
Each product sector is fully self-contained:

  ┌─────────────────────────────────────────────────────┐
  │  SECTOR BLOCK                                       │
  │  • ENABLED flags for its products                   │
  │  • Tuning constants                                 │
  │  • Strategy / helper functions                      │
  │  • A top-level  run_<sector>(state, mem, result)    │
  │    function called by Trader.run()                  │
  └─────────────────────────────────────────────────────┘

Global utilities (Logger, mid_price, market_make, …) live above all sectors
and are shared by any sector that needs them.

To add a new sector: copy any existing sector block, rename everything, wire
the new run_<sector>() call into Trader.run(), and add its state keys to the
mem-load / mem-save dictionaries. Nothing else needs to change.

Sectors
-------
  1. Galaxy Sounds    — market make / trend hold
  2. Microchip        — market make / trend hold
  3. Oxygen Shake     — market make / trend hold
  4. Panel            — market make / trend hold
  5. Pebbles          — XL/XS mean-reversion pair + MM for S/M/L
  6. Robot            — Dishes/Ironing MR pair + MM for the rest
  7. Sleep Pod        — CPS core cluster + secondary cluster overlay
  8. Snackpack        — market make
  9. Translator       — market make
 10. UV Visor         — market make / trend hold
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
from typing import Any
from math import sqrt
import json


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POS_LIMIT = 10   # hard position limit shared by every product


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Compact logger that stays within the Prosperity platform log-length cap."""

    MAX_LOG_LENGTH = 3750

    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(self._to_json([
            self._compress_state(state, ""),
            self._compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.MAX_LOG_LENGTH - base_length) // 3

        print(self._to_json([
            self._compress_state(state, self._truncate(state.traderData, max_item_length)),
            self._compress_orders(orders),
            conversions,
            self._truncate(trader_data, max_item_length),
            self._truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def _compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self._compress_listings(state.listings),
            self._compress_order_depths(state.order_depths),
            self._compress_trades(state.own_trades),
            self._compress_trades(state.market_trades),
            state.position,
            self._compress_observations(state.observations),
        ]

    def _compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def _compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def _compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values() for t in arr
        ]

    def _compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
            product: [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def _compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
        """Binary-search the longest prefix of *value* whose JSON length <= max_length."""
        lo, hi = 0, min(len(value), max_length)
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return result


logger = Logger()


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL UTILITIES  (shared by all sectors)
# ══════════════════════════════════════════════════════════════════════════════

def mid_price(od: OrderDepth) -> float | None:
    """Best-bid / best-ask mid-point, or None when either side is empty."""
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None

def rolling_mean_std(window: list[float]) -> tuple[float, float]:
    """Sample mean and std-dev of *window*.  Std-dev is floored at 1.0."""
    n = len(window)
    if n < 2:
        return 0.0, 1.0
    mean = sum(window) / n
    std  = sqrt(sum((x - mean) ** 2 for x in window) / (n - 1))
    return mean, max(std, 1.0)

def place_order_to_target(
    symbol: str,
    od: OrderDepth,
    current_pos: int,
    target_pos: int,
    orders_out: dict,
) -> None:
    """
    Emit one order to move *current_pos* toward *target_pos*.

    Passive (inside-spread) price when |delta| <= 3; crosses the spread
    aggressively for larger deltas.
    """
    if not od.buy_orders or not od.sell_orders:
        return
    delta = target_pos - current_pos
    if delta == 0:
        return
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    price = (best_bid + 1 if delta > 0 else best_ask - 1) if abs(delta) <= 3 \
            else (best_ask if delta > 0 else best_bid)
    orders_out[symbol] = [Order(symbol, price, delta)]

def market_make(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    """
    Post resting quotes one tick inside the best bid/ask.

    Quotes are sized to remaining capacity so position never exceeds POS_LIMIT.
    This is the global default strategy — any sector without a specialised
    function falls back to this.
    """
    if not od.buy_orders or not od.sell_orders:
        return []
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid
    if spread <= 0:
        return []
    our_bid = best_bid + 1 if spread >= 2 else best_bid
    our_ask = best_ask - 1 if spread >= 2 else best_ask
    if our_bid >= our_ask:
        return []
    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos
    orders: list[Order] = []
    if buy_cap  > 0: orders.append(Order(symbol, our_bid,   buy_cap))
    if sell_cap > 0: orders.append(Order(symbol, our_ask,  -sell_cap))
    return orders

def trade_trend(symbol: str, od: OrderDepth, pos: int, hold_size: int) -> list[Order]:
    """
    Hold a minimum long position of *hold_size*, then layer MM quotes above it.

    Used by any sector whose products have a persistent uptrend signal.
    """
    if not od.buy_orders or not od.sell_orders:
        return []
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid
    orders: list[Order] = []

    if pos < hold_size:
        orders.append(Order(symbol, best_ask, hold_size - pos))

    if spread > 0:
        our_bid     = best_bid + 1 if spread >= 2 else best_bid
        mm_buy_cap  = POS_LIMIT - max(pos, hold_size)
        mm_sell_cap = max(0, pos - hold_size)
        if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
        if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))

    return orders

def cluster_mean_price(products: list[str], order_depths: dict) -> float | None:
    """Equal-weighted average mid-price across *products*, or None."""
    mids = [
        m for p in products
        if (od := order_depths.get(p)) and (m := mid_price(od)) is not None
    ]
    return sum(mids) / len(mids) if mids else None

# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY HELPERS (Snackpack + Robot upgrade layer)
# ══════════════════════════════════════════════════════════════════════════════

def get_mid(od):
    if od and od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None


def send_order(result, symbol, price, qty):
    if qty == 0:
        return
    result.setdefault(symbol, []).append(Order(symbol, int(price), int(qty)))


def update_ema(prev, val, span):
    a = 2.0 / (span + 1)
    return a * val + (1 - a) * prev


def zscore(series):
    if len(series) < 10:
        return 0.0
    m, s = rolling_mean_std(series)
    return (series[-1] - m) / s


def bb(od): return max(od.buy_orders)
def ba(od): return min(od.sell_orders)


def hold_directional(symbol, od, pos, target, result):
    if not od.buy_orders or not od.sell_orders:
        return
    delta = target - pos
    if delta == 0:
        return
    if delta > 0:
        qty = min(delta, POS_LIMIT - pos)
        send_order(result, symbol, ba(od), qty)
    else:
        qty = max(delta, -(POS_LIMIT + pos))
        send_order(result, symbol, bb(od), qty)


def ema_mm(symbol, od, pos, ema, span, thresh, result):
    if not od.buy_orders or not od.sell_orders:
        return ema

    mid = (bb(od) + ba(od)) / 2
    ema = update_ema(ema, mid, span)
    dev = mid - ema

    bid = int(ema) - 1
    ask = int(ema) + 2

    if dev > thresh:
        skew = min(int(dev / thresh) * 2, 5)
        bid -= skew
        ask -= skew
    elif dev < -thresh:
        skew = min(int(-dev / thresh) * 2, 5)
        bid += skew
        ask += skew

    bid = min(bid, bb(od))
    ask = max(ask, ba(od))
    if bid >= ask:
        return ema

    if POS_LIMIT - pos > 0:
        send_order(result, symbol, bid, POS_LIMIT - pos)
    if POS_LIMIT + pos > 0:
        send_order(result, symbol, ask, -(POS_LIMIT + pos))

    return ema

def run_simple_sector(
    products: dict[str, bool],
    trend_hold: dict[str, int],
    state: TradingState,
    result: dict,
) -> None:
    """
    Generic sector runner: trend-hold for products listed in *trend_hold*,
    market-make everything else.  Called by sectors with no bespoke pair or
    cluster logic.
    """
    for symbol, enabled in products.items():
        if not enabled:
            continue
        od = state.order_depths.get(symbol)
        if not od:
            continue
        pos = state.position.get(symbol, 0)
        orders = (
            trade_trend(symbol, od, pos, trend_hold[symbol])
            if symbol in trend_hold
            else market_make(symbol, od, pos)
        )
        if orders:
            result[symbol] = orders


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 1 — GALAXY SOUNDS
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: trend-hold for products with a measured uptrend; market-make the rest.

GALAXY_SOUNDS_ENABLED: dict[str, bool] = {
    "GALAXY_SOUNDS_BLACK_HOLES":     True,
    "GALAXY_SOUNDS_DARK_MATTER":     True,
    "GALAXY_SOUNDS_PLANETARY_RINGS": True,
    "GALAXY_SOUNDS_SOLAR_FLAMES":    False,
    "GALAXY_SOUNDS_SOLAR_WINDS":     True,
}

# Minimum long position to maintain while the trend persists
GALAXY_SOUNDS_TREND_HOLD: dict[str, int] = {
    "GALAXY_SOUNDS_PLANETARY_RINGS": 7,
    "GALAXY_SOUNDS_BLACK_HOLES":     6,
    "GALAXY_SOUNDS_DARK_MATTER":     5,
}


def run_galaxy_sounds(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(GALAXY_SOUNDS_ENABLED, GALAXY_SOUNDS_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 2 — MICROCHIP
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: trend-hold for MICROCHIP_SQUARE; market-make the rest.

MICROCHIP_ENABLED: dict[str, bool] = {
    "MICROCHIP_CIRCLE":    True,
    "MICROCHIP_OVAL":      True,
    "MICROCHIP_RECTANGLE": True,
    "MICROCHIP_SQUARE":    True,
    "MICROCHIP_TRIANGLE":  True,
}

MICROCHIP_TREND_HOLD: dict[str, int] = {
    "MICROCHIP_SQUARE": 9,
}


def run_microchip(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(MICROCHIP_ENABLED, MICROCHIP_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 3 — OXYGEN SHAKE
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: trend-hold for GARLIC; market-make the rest.

OXYGEN_SHAKE_ENABLED: dict[str, bool] = {
    "OXYGEN_SHAKE_CHOCOLATE":      True,
    "OXYGEN_SHAKE_EVENING_BREATH": True,
    "OXYGEN_SHAKE_GARLIC":         True,
    "OXYGEN_SHAKE_MINT":           True,
    "OXYGEN_SHAKE_MORNING_BREATH": True,
}

OXYGEN_SHAKE_TREND_HOLD: dict[str, int] = {
    "OXYGEN_SHAKE_GARLIC": 6,
}


def run_oxygen_shake(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(OXYGEN_SHAKE_ENABLED, OXYGEN_SHAKE_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 4 — PANEL
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: trend-hold for PANEL_2X4; market-make the rest.

PANEL_ENABLED: dict[str, bool] = {
    "PANEL_1X2": False,
    "PANEL_1X4": True,
    "PANEL_2X2": True,
    "PANEL_2X4": True,
    "PANEL_4X4": False,
}

PANEL_TREND_HOLD: dict[str, int] = {
    "PANEL_2X4": 6,
}


def run_panel(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(PANEL_ENABLED, PANEL_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 5 — PEBBLES
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy:
#    XL / XS  -> mean-reversion pair on the XL-XS spread;
#                background long XL / short XS when no signal
#    S / M / L -> market-make only

PEBBLES_ENABLED: dict[str, bool] = {
    "PEBBLES_XS": True,
    "PEBBLES_S":  True,
    "PEBBLES_M":  False,
    "PEBBLES_L":  False,
    "PEBBLES_XL": True,
}

PEBBLES_MR_WINDOW   = 500   # rolling z-score window length
PEBBLES_MR_ENTRY_Z  = 2.5   # z-score to open a trade
PEBBLES_MR_EXIT_Z   = 0.3   # z-score to close a trade
PEBBLES_MR_MAX_HOLD = 200   # force-close after this many ticks
PEBBLES_BG_POS      = 10    # background long-XL / short-XS when flat

PEBBLES_MM_ONLY = {"PEBBLES_S", "PEBBLES_M", "PEBBLES_L"}


def _pebbles_xl_xs_mr(
    spread_history: list[int],
    hold_ticks: int,
    direction: int,
    od_xl: OrderDepth,
    od_xs: OrderDepth,
    pos_xl: int,
    pos_xs: int,
) -> tuple[dict, int, int]:
    """
    Mean-reversion on the XL - XS mid-price spread.

    direction:  +1 = long XL / short XS
                -1 = short XL / long XS
                 0 = flat (uses background position)
    Returns (orders, hold_ticks, direction).
    """
    orders: dict = {}

    if len(spread_history) >= 50:
        mean, std = rolling_mean_std(spread_history)
        z = (spread_history[-1] - mean) / std

        if direction != 0:
            hold_ticks += 1
            if abs(z) < PEBBLES_MR_EXIT_Z or hold_ticks >= PEBBLES_MR_MAX_HOLD:
                direction, hold_ticks = 0, 0

        if direction == 0:
            if   z >  PEBBLES_MR_ENTRY_Z + 0.5: direction, hold_ticks = -1, 0
            elif z < -PEBBLES_MR_ENTRY_Z - 0.5: direction, hold_ticks =  1, 0

        logger.print(f"pebbles_xl_xs  z={z:.2f}  dir={direction}  hold={hold_ticks}")

    target_xl =  POS_LIMIT * direction if direction != 0 else  PEBBLES_BG_POS
    target_xs = -POS_LIMIT * direction if direction != 0 else -PEBBLES_BG_POS

    place_order_to_target("PEBBLES_XL", od_xl, pos_xl, target_xl, orders)
    place_order_to_target("PEBBLES_XS", od_xs, pos_xs, target_xs, orders)

    return orders, hold_ticks, direction


def run_pebbles(state: TradingState, mem: dict, result: dict) -> dict:
    """Run all Pebbles strategies and return updated persistent state."""
    spread_history = mem.get("pebbles_history", [])
    hold_ticks     = mem.get("pebbles_hold", 0)
    direction      = mem.get("pebbles_dir", 0)

    od_xl = state.order_depths.get("PEBBLES_XL")
    od_xs = state.order_depths.get("PEBBLES_XS")

    # XL / XS — mean-reversion pair
    if PEBBLES_ENABLED.get("PEBBLES_XL") and od_xl and od_xs:
        mid_xl = mid_price(od_xl)
        mid_xs = mid_price(od_xs)
        if mid_xl is not None and mid_xs is not None:
            spread_history.append(int(mid_xl - mid_xs))
            if len(spread_history) > PEBBLES_MR_WINDOW:
                spread_history = spread_history[-PEBBLES_MR_WINDOW:]
            pos_xl = state.position.get("PEBBLES_XL", 0)
            pos_xs = state.position.get("PEBBLES_XS", 0)
            orders, hold_ticks, direction = _pebbles_xl_xs_mr(
                spread_history, hold_ticks, direction, od_xl, od_xs, pos_xl, pos_xs,
            )
            result.update(orders)

    # S / M / L — market-make only
    for symbol in PEBBLES_MM_ONLY:
        if not PEBBLES_ENABLED.get(symbol):
            continue
        od = state.order_depths.get(symbol)
        if not od:
            continue
        orders = market_make(symbol, od, state.position.get(symbol, 0))
        if orders:
            result[symbol] = orders

    return {
        "pebbles_history": spread_history,
        "pebbles_hold":    hold_ticks,
        "pebbles_dir":     direction,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 6 — ROBOT
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_PRODUCTS = [
    "ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
    "ROBOT_MOPPING", "ROBOT_VACUUMING",
]

ROBOT_TARGETS = {
    "ROBOT_MOPPING":   +10,
    "ROBOT_DISHES":    +10,
    "ROBOT_LAUNDRY":   +7,
    "ROBOT_VACUUMING": -6,
    "ROBOT_IRONING":   -10,
}

DI_WINDOW = 300
DI_ENTRY_Z = 1.5
DI_EXIT_Z = 0.3


def run_robot(state: TradingState, mem: dict, result: dict) -> dict:
    ods, positions = state.order_depths, state.position

    emas = mem.get("rob_ema", {})
    di_hist = mem.get("rob_di", [])

    od_d = ods.get("ROBOT_DISHES")
    od_i = ods.get("ROBOT_IRONING")

    md = get_mid(od_d)
    mi = get_mid(od_i)

    if od_d and od_i and md is not None and mi is not None:
        di_hist.append(md - mi)
        if len(di_hist) > DI_WINDOW:
            di_hist[:] = di_hist[-DI_WINDOW:]

    # Directional trend holds
    for symbol, target in ROBOT_TARGETS.items():
        od = ods.get(symbol)
        pos = positions.get(symbol, 0)
        if od is None:
            continue
        hold_directional(symbol, od, pos, target, result)

    # Dishes mean reversion overlay
    if od_d and md is not None:
        emas.setdefault("ROBOT_DISHES", md)
        emas["ROBOT_DISHES"] = update_ema(emas["ROBOT_DISHES"], md, 100)

        dev = md - emas["ROBOT_DISHES"]
        pos_d = positions.get("ROBOT_DISHES", 0)

        if dev > 80 and pos_d > 7:
            send_order(result, "ROBOT_DISHES", bb(od_d), -(pos_d - 7))
        elif dev < -80 and pos_d < POS_LIMIT:
            send_order(result, "ROBOT_DISHES", ba(od_d), POS_LIMIT - pos_d)

    mem["rob_ema"] = emas
    mem["rob_di"] = di_hist

    return {"rob_ema": emas, "rob_di": di_hist}

# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 7 — SLEEP POD
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy:
#    Core cluster  (Cotton / Polyester / Suede)   -> nonlinear deviation-weighted basket MR
#    Secondary cluster (Lamb Wool / Nylon)        -> linear mean-reversion overlay

SLEEP_POD_ENABLED: dict[str, bool] = {
    "SLEEP_POD_COTTON":    True,
    "SLEEP_POD_LAMB_WOOL": False,
    "SLEEP_POD_NYLON":     True,
    "SLEEP_POD_POLYESTER": True,
    "SLEEP_POD_SUEDE":     True,
}

# Core cluster (CPS) — nonlinear basket MR
SLEEP_POD_CORE_PRODUCTS   = ["SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"]
SLEEP_POD_CORE_BASE_RISK  = 28    # base notional risk
SLEEP_POD_CORE_RISK_MULT  = 2.8   # scaler to maintain active trading intensity
SLEEP_POD_CORE_TOTAL_RISK = SLEEP_POD_CORE_BASE_RISK * SLEEP_POD_CORE_RISK_MULT
SLEEP_POD_CORE_POWER      = 1.6   # exponent: amplifies stronger mispricings

# Secondary cluster — linear MR overlay
SLEEP_POD_SECONDARY_PRODUCTS = ["SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON"]
SLEEP_POD_SECONDARY_SCALE    = 10   # position sensitivity to deviation


def _sleep_pod_core_cluster(state: TradingState, result: dict) -> None:
    """
    Nonlinear deviation-weighted MR across the core Sleep Pod basket.

    Each leg's target is proportional to sign(dev) * |dev|^POWER,
    so larger mispricings attract disproportionately more capital.
    """
    factor = cluster_mean_price(SLEEP_POD_CORE_PRODUCTS, state.order_depths)
    if factor is None:
        return

    ods  = {p: state.order_depths.get(p) for p in SLEEP_POD_CORE_PRODUCTS}
    mids = {p: mid_price(od) for p, od in ods.items() if od and mid_price(od) is not None}
    devs = {p: mids[p] - factor for p in mids}

    if len(devs) < len(SLEEP_POD_CORE_PRODUCTS):
        return   # need all legs to trade the basket

    raw_weights = {
        p: (1 if dev >= 0 else -1) * (abs(dev) ** SLEEP_POD_CORE_POWER)
        for p, dev in devs.items()
    }
    weight_sum = max(sum(abs(w) for w in raw_weights.values()), 1e-9)

    for symbol, od_local in ods.items():
        if not od_local:
            continue
        pos         = state.position.get(symbol, 0)
        norm_weight = raw_weights[symbol] / weight_sum
        target      = int(-norm_weight * SLEEP_POD_CORE_TOTAL_RISK)
        best_bid    = max(od_local.buy_orders)
        best_ask    = min(od_local.sell_orders)
        price       = best_bid if target < 0 else best_ask
        qty         = max(-POS_LIMIT, min(POS_LIMIT, target - pos))
        if qty != 0:
            result[symbol] = [Order(symbol, price, qty)]


def _sleep_pod_secondary_cluster(state: TradingState, result: dict) -> None:
    """
    Linear MR overlay for the peripheral Sleep Pod products.

    Each product is traded toward the secondary cluster mean price;
    position is proportional to the deviation from that mean.
    """
    factor = cluster_mean_price(SLEEP_POD_SECONDARY_PRODUCTS, state.order_depths)
    if factor is None:
        return

    for symbol in SLEEP_POD_SECONDARY_PRODUCTS:
        if not SLEEP_POD_ENABLED.get(symbol):
            continue
        od = state.order_depths.get(symbol)
        if not od:
            continue
        price_mid = mid_price(od)
        if price_mid is None:
            continue
        deviation = price_mid - factor
        if abs(deviation) < 1:
            continue   # inside deadband — skip
        pos      = state.position.get(symbol, 0)
        target   = max(-POS_LIMIT, min(POS_LIMIT, -int(deviation * SLEEP_POD_SECONDARY_SCALE)))
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        price    = best_bid if target < 0 else best_ask
        qty      = max(-POS_LIMIT, min(POS_LIMIT, target - pos))
        if qty != 0:
            result[symbol] = [Order(symbol, price, qty)]


def run_sleep_pod(state: TradingState, mem: dict, result: dict) -> None:
    _sleep_pod_core_cluster(state, result)
    _sleep_pod_secondary_cluster(state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 8 — SNACKPACK
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: pure market-making across all variants.

# Pair-trade + spread parameters (from original alpha)
CP_WINDOW = 200
CP_ENTRY_Z = 1.4
CP_EXIT_Z = 0.25

RV_WINDOW = 150
RV_ENTRY_Z = 1.2
RV_EXIT_Z = 0.25

SNACKPACK_ENABLED: dict[str, bool] = {
    "SNACKPACK_CHOCOLATE":  True,
    "SNACKPACK_PISTACHIO":  True,
    "SNACKPACK_RASPBERRY":  True,
    "SNACKPACK_STRAWBERRY": True,
    "SNACKPACK_VANILLA":    True,
}

SNACKPACK_TREND_HOLD: dict[str, int] = {}   # no trend products in this sector


def snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result):
    mc, mp = get_mid(od_c), get_mid(od_p)
    if mc is None or mp is None:
        return

    cp_hist.append(mc - mp)
    if len(cp_hist) > CP_WINDOW:
        cp_hist[:] = cp_hist[-CP_WINDOW:]

    if len(cp_hist) < 20:
        return

    z = zscore(cp_hist)

    tgt_c = -7 if z > CP_ENTRY_Z else (-POS_LIMIT if z < -CP_ENTRY_Z else -POS_LIMIT)
    tgt_p = -POS_LIMIT if z > CP_ENTRY_Z else (-7 if z < -CP_ENTRY_Z else -POS_LIMIT)

    hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, tgt_c, result)
    hold_directional("SNACKPACK_PISTACHIO", od_p, pos_p, tgt_p, result)


def snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result):
    mr, mv = get_mid(od_r), get_mid(od_v)
    if mr is None or mv is None:
        return

    rv_hist.append(mr - mv)
    if len(rv_hist) > RV_WINDOW:
        rv_hist[:] = rv_hist[-RV_WINDOW:]

    if len(rv_hist) < 20:
        return

    z = zscore(rv_hist)
    PAIR = 3

    if z > RV_ENTRY_Z:
        if pos_r > -PAIR:
            send_order(result, "SNACKPACK_RASPBERRY", bb(od_r), -(pos_r + PAIR))
        if pos_v < PAIR:
            send_order(result, "SNACKPACK_VANILLA", ba(od_v), PAIR - pos_v)

    elif z < -RV_ENTRY_Z:
        if pos_v > -PAIR:
            send_order(result, "SNACKPACK_VANILLA", bb(od_v), -(pos_v + PAIR))
        if pos_r < PAIR:
            send_order(result, "SNACKPACK_RASPBERRY", ba(od_r), PAIR - pos_r)

    elif abs(z) < RV_EXIT_Z:
        if pos_r != 0:
            send_order(result, "SNACKPACK_RASPBERRY",
                       bb(od_r) if pos_r > 0 else ba(od_r), -pos_r)
        if pos_v != 0:
            send_order(result, "SNACKPACK_VANILLA",
                       bb(od_v) if pos_v > 0 else ba(od_v), -pos_v)


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

    pos_c = positions.get("SNACKPACK_CHOCOLATE", 0)
    pos_p = positions.get("SNACKPACK_PISTACHIO", 0)
    pos_r = positions.get("SNACKPACK_RASPBERRY", 0)
    pos_s = positions.get("SNACKPACK_STRAWBERRY", 0)
    pos_v = positions.get("SNACKPACK_VANILLA", 0)

    # Strawberry: strong long trend
    if od_s:
        hold_directional("SNACKPACK_STRAWBERRY", od_s, pos_s, +POS_LIMIT, result)

    # Chocolate + Pistachio: strong shorts
    if od_c:
        hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, -POS_LIMIT, result)
    if od_p:
        hold_directional("SNACKPACK_PISTACHIO", od_p, pos_p, -POS_LIMIT, result)

    # Raspberry: EMA MM
    if od_r:
        mid_r = get_mid(od_r)
        if mid_r is not None:
            emas.setdefault("SNACKPACK_RASPBERRY", mid_r)
            emas["SNACKPACK_RASPBERRY"] = ema_mm(
                "SNACKPACK_RASPBERRY", od_r, pos_r,
                emas["SNACKPACK_RASPBERRY"], 200, 30, result
            )

    # Vanilla: EMA MM
    if od_v:
        mid_v = get_mid(od_v)
        if mid_v is not None:
            emas.setdefault("SNACKPACK_VANILLA", mid_v)
            emas["SNACKPACK_VANILLA"] = ema_mm(
                "SNACKPACK_VANILLA", od_v, pos_v,
                emas["SNACKPACK_VANILLA"], 200, 38, result
            )

    # CP pair spread (Chocolate vs Pistachio)
    if od_c and od_p:
        snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result)

    # RV spread (Raspberry vs Vanilla)
    if od_r and od_v:
        snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result)

    mem["snk_ema"] = emas
    mem["snk_cp"] = cp_hist
    mem["snk_rv"] = rv_hist


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 9 — TRANSLATOR
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: pure market-making across enabled variants.

TRANSLATOR_ENABLED: dict[str, bool] = {
    "TRANSLATOR_ASTRO_BLACK":      True,
    "TRANSLATOR_ECLIPSE_CHARCOAL": True,
    "TRANSLATOR_GRAPHITE_MIST":    False,
    "TRANSLATOR_SPACE_GRAY":       False,
    "TRANSLATOR_VOID_BLUE":        True,
}

TRANSLATOR_TREND_HOLD: dict[str, int] = {}   # no trend products in this sector


def run_translator(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(TRANSLATOR_ENABLED, TRANSLATOR_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTOR 10 — UV VISOR
# ══════════════════════════════════════════════════════════════════════════════
#  Strategy: trend-hold for Magenta and Yellow; market-make the rest.

UV_VISOR_ENABLED: dict[str, bool] = {
    "UV_VISOR_AMBER":   True,
    "UV_VISOR_MAGENTA": True,
    "UV_VISOR_ORANGE":  True,
    "UV_VISOR_RED":     True,
    "UV_VISOR_YELLOW":  True,
}

UV_VISOR_TREND_HOLD: dict[str, int] = {
    "UV_VISOR_MAGENTA": 6,
    "UV_VISOR_YELLOW":  6,
}


def run_uv_visor(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(UV_VISOR_ENABLED, UV_VISOR_TREND_HOLD, state, result)


# ══════════════════════════════════════════════════════════════════════════════
#  TRADER — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Trader:
    """
    Orchestrates all sector runners each tick.

    To add a new sector:
      1. Write the sector block above (ENABLED dict, constants, run_<sector>()).
      2. Call run_<sector>(state, mem, result) in the "Run all sectors" block below.
      3. If the sector needs persistent state, merge its returned dict into new_mem.
    """

    def run(
        self, state: TradingState
    ) -> tuple[dict[Symbol, list[Order]], int, str]:

        result:      dict[Symbol, list[Order]] = {}
        conversions: int = 0

        # ── Load persisted state ───────────────────────────────────────────────
        mem = json.loads(state.traderData) if state.traderData else {}

        # ── Run all sectors ────────────────────────────────────────────────────
        # Stateless sectors (no persistent data needed)
        run_galaxy_sounds(state, mem, result)
        run_microchip(state, mem, result)
        run_oxygen_shake(state, mem, result)
        run_panel(state, mem, result)
        run_sleep_pod(state, mem, result)
        run_snackpack(state, mem, result)
        run_translator(state, mem, result)
        run_uv_visor(state, mem, result)

        # Stateful sectors (return updated state dicts to persist)
        pebbles_state = run_pebbles(state, mem, result)
        robot_state   = run_robot(state, mem, result)

        # ── Persist state ──────────────────────────────────────────────────────
        new_mem     = {**pebbles_state, **robot_state}
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
