
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
#  Strategy:
#    Dishes / Ironing  -> mean-reversion pair; falls back to MM when flat
#    Laundry / Mopping / Vacuuming -> market-make
#    (ROBOT_MOPPING excluded from trend hold: profitable only on day 3)

ROBOT_ENABLED: dict[str, bool] = {
    "ROBOT_DISHES":    True,
    "ROBOT_IRONING":   True,
    "ROBOT_LAUNDRY":   True,
    "ROBOT_MOPPING":   True,
    "ROBOT_VACUUMING": False,
}

# resid_ac1 ~= -0.20, half-life ~= 3 ticks. Short window tracks drifting spread mean.
ROBOT_DI_WINDOW   = 550
ROBOT_DI_ENTRY_Z  = 2.2   # ~5.6 % fire rate
ROBOT_DI_EXIT_Z   = 0.5
ROBOT_DI_MAX_HOLD = 150

ROBOT_MM_ONLY = {"ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING"}


def _robot_dishes_ironing_mr(
    spread_history: list[int],
    hold_ticks: int,
    direction: int,
    od_dishes: OrderDepth,
    od_ironing: OrderDepth,
    pos_dishes: int,
    pos_ironing: int,
) -> tuple[dict, int, int]:
    """
    Mean-reversion on the Dishes - Ironing mid-price spread.

    direction:  +1 = long Dishes / short Ironing
                -1 = short Dishes / long Ironing
                 0 = no signal (falls back to market-making both legs)
    Returns (orders, hold_ticks, direction).
    """
    orders: dict = {}

    if len(spread_history) >= 50:
        mean, std = rolling_mean_std(spread_history)
        z = (spread_history[-1] - mean) / std

        if direction != 0:
            hold_ticks += 1
            if abs(z) < ROBOT_DI_EXIT_Z or hold_ticks >= ROBOT_DI_MAX_HOLD:
                direction, hold_ticks = 0, 0

        if direction == 0:
            if   z >  ROBOT_DI_ENTRY_Z: direction, hold_ticks = -1, 0   # Dishes expensive
            elif z < -ROBOT_DI_ENTRY_Z: direction, hold_ticks =  1, 0   # Dishes cheap

    if direction != 0:
        place_order_to_target("ROBOT_DISHES",  od_dishes,  pos_dishes,   POS_LIMIT * direction, orders)
        place_order_to_target("ROBOT_IRONING", od_ironing, pos_ironing, -POS_LIMIT * direction, orders)
    else:
        dishes_mm  = market_make("ROBOT_DISHES",  od_dishes,  pos_dishes)
        ironing_mm = market_make("ROBOT_IRONING", od_ironing, pos_ironing)
        if dishes_mm:  orders["ROBOT_DISHES"]  = dishes_mm
        if ironing_mm: orders["ROBOT_IRONING"] = ironing_mm

    return orders, hold_ticks, direction


def run_robot(state: TradingState, mem: dict, result: dict) -> dict:
    """Run all Robot strategies and return updated persistent state."""
    spread_history = mem.get("di_history", [])
    hold_ticks     = mem.get("di_hold", 0)
    direction      = mem.get("di_dir", 0)

    od_dishes  = state.order_depths.get("ROBOT_DISHES")
    od_ironing = state.order_depths.get("ROBOT_IRONING")

    # Dishes / Ironing — MR pair
    if ROBOT_ENABLED.get("ROBOT_DISHES") and od_dishes and od_ironing:
        mid_d = mid_price(od_dishes)
        mid_i = mid_price(od_ironing)
        if mid_d is not None and mid_i is not None:
            spread_history.append(int(mid_d - mid_i))
            if len(spread_history) > ROBOT_DI_WINDOW:
                spread_history = spread_history[-ROBOT_DI_WINDOW:]
            pos_d = state.position.get("ROBOT_DISHES", 0)
            pos_i = state.position.get("ROBOT_IRONING", 0)
            orders, hold_ticks, direction = _robot_dishes_ironing_mr(
                spread_history, hold_ticks, direction,
                od_dishes, od_ironing, pos_d, pos_i,
            )
            result.update(orders)

    # Laundry / Mopping / Vacuuming — market-make only
    for symbol in ROBOT_MM_ONLY:
        if not ROBOT_ENABLED.get(symbol):
            continue
        od = state.order_depths.get(symbol)
        if not od:
            continue
        orders = market_make(symbol, od, state.position.get(symbol, 0))
        if orders:
            result[symbol] = orders

    return {
        "di_history": spread_history,
        "di_hold":    hold_ticks,
        "di_dir":     direction,
    }


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

SNACKPACK_ENABLED: dict[str, bool] = {
    "SNACKPACK_CHOCOLATE":  True,
    "SNACKPACK_PISTACHIO":  True,
    "SNACKPACK_RASPBERRY":  True,
    "SNACKPACK_STRAWBERRY": True,
    "SNACKPACK_VANILLA":    True,
}

SNACKPACK_TREND_HOLD: dict[str, int] = {}   # no trend products in this sector


def run_snackpack(state: TradingState, mem: dict, result: dict) -> None:
    run_simple_sector(SNACKPACK_ENABLED, SNACKPACK_TREND_HOLD, state, result)


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
