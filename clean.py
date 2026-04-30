"""
round5_clustersGPT_860.py — Prosperity Trading Bot
====================================================
Strategies:
  • Market Making        — default for most products
  • Trend Hold + MM      — products in TREND_HOLD
  • Pebbles XL/XS MR    — mean-reversion pair on spread
  • Dishes/Ironing MR   — mean-reversion pair on spread
  • CPS Cluster          — deviation-weighted basket MR (Sleep Pod core)
  • Secondary Cluster    — mean-reversion overlay (Sleep Pod periphery)
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
from typing import Any
from math import sqrt
import json


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Per-product on/off switch ──────────────────────────────────────────────────
ENABLED: dict[str, bool] = {
    # Galaxy Sounds
    "GALAXY_SOUNDS_BLACK_HOLES":      True,
    "GALAXY_SOUNDS_DARK_MATTER":      True,
    "GALAXY_SOUNDS_PLANETARY_RINGS":  True,
    "GALAXY_SOUNDS_SOLAR_FLAMES":     False,
    "GALAXY_SOUNDS_SOLAR_WINDS":      True,
    # Microchip
    "MICROCHIP_CIRCLE":    True,
    "MICROCHIP_OVAL":      True,
    "MICROCHIP_RECTANGLE": True,
    "MICROCHIP_SQUARE":    True,
    "MICROCHIP_TRIANGLE":  True,
    # Oxygen Shake
    "OXYGEN_SHAKE_CHOCOLATE":      True,
    "OXYGEN_SHAKE_EVENING_BREATH": True,
    "OXYGEN_SHAKE_GARLIC":         True,
    "OXYGEN_SHAKE_MINT":           True,
    "OXYGEN_SHAKE_MORNING_BREATH": True,
    # Panel
    "PANEL_1X2": False,
    "PANEL_1X4": True,
    "PANEL_2X2": True,
    "PANEL_2X4": True,
    "PANEL_4X4": False,
    # Pebbles  (XL/XS handled by MR strategy; L/M/S market-made)
    "PEBBLES_XS": True,
    "PEBBLES_S":  True,
    "PEBBLES_M":  False,
    "PEBBLES_L":  False,
    "PEBBLES_XL": True,
    # Robot
    "ROBOT_DISHES":    True,
    "ROBOT_IRONING":   True,
    "ROBOT_LAUNDRY":   True,
    "ROBOT_MOPPING":   True,
    "ROBOT_VACUUMING": False,
    # Sleep Pod
    "SLEEP_POD_COTTON":    True,
    "SLEEP_POD_LAMB_WOOL": False,
    "SLEEP_POD_NYLON":     True,
    "SLEEP_POD_POLYESTER": True,
    "SLEEP_POD_SUEDE":     True,
    # Snackpack
    "SNACKPACK_CHOCOLATE":  True,
    "SNACKPACK_PISTACHIO":  True,
    "SNACKPACK_RASPBERRY":  True,
    "SNACKPACK_STRAWBERRY": True,
    "SNACKPACK_VANILLA":    True,
    # Translator
    "TRANSLATOR_ASTRO_BLACK":      True,
    "TRANSLATOR_ECLIPSE_CHARCOAL": True,
    "TRANSLATOR_GRAPHITE_MIST":    False,
    "TRANSLATOR_SPACE_GRAY":       False,
    "TRANSLATOR_VOID_BLUE":        True,
    # UV Visor
    "UV_VISOR_AMBER":   True,
    "UV_VISOR_MAGENTA": True,
    "UV_VISOR_ORANGE":  True,
    "UV_VISOR_RED":     True,
    "UV_VISOR_YELLOW":  True,
}

# ── Global position limit (applies to every product) ──────────────────────────
POS_LIMIT = 10

# ── Pebbles XL/XS mean-reversion parameters ───────────────────────────────────
PEBBLES_WINDOW   = 500   # rolling window length for z-score
PEBBLES_ENTRY_Z  = 2.5   # z-score threshold to open a trade
PEBBLES_EXIT_Z   = 0.3   # z-score threshold to close a trade
PEBBLES_MAX_HOLD = 200   # force-close after this many ticks
PEBBLES_BG_POS   = 10    # background long-XL / short-XS when no MR signal

# ── Robot Dishes/Ironing pair mean-reversion parameters ───────────────────────
# resid_ac1 ≈ -0.20, half-life ≈ 3 ticks.  Short window tracks drifting spread.
DI_WINDOW   = 550
DI_ENTRY_Z  = 2.2   # ~5.6 % fire rate
DI_EXIT_Z   = 0.5
DI_MAX_HOLD = 150

# ── CPS (Sleep Pod core) cluster parameters ───────────────────────────────────
CPS_BASE_RISK        = 28     # base notional risk per cluster
CLUSTER_RISK_MULT    = 2.8    # scaler to maintain active trading intensity
CPS_TOTAL_RISK       = CPS_BASE_RISK * CLUSTER_RISK_MULT
CPS_POWER            = 1.6    # exponent for nonlinear deviation amplification

# ── Secondary Sleep Pod cluster parameters ────────────────────────────────────
SECONDARY_SCALE = 10   # sensitivity multiplier for secondary cluster trades

# ── Trend-hold product targets ────────────────────────────────────────────────
# Each value is the sqrt-scaled minimum long position to hold while trend lasts.
# (ROBOT_MOPPING excluded: only profitable day 3; loses on days 2 & 4.)
TREND_HOLD: dict[str, int] = {
    "MICROCHIP_SQUARE":              9,
    "SLEEP_POD_SUEDE":               8,
    "SLEEP_POD_POLYESTER":           8,
    "GALAXY_SOUNDS_PLANETARY_RINGS": 7,
    "UV_VISOR_MAGENTA":              6,
    "OXYGEN_SHAKE_GARLIC":           6,
    "SLEEP_POD_COTTON":              6,
    "UV_VISOR_YELLOW":               6,
    "PANEL_2X4":                     6,
    "GALAXY_SOUNDS_BLACK_HOLES":     6,
    "GALAXY_SOUNDS_DARK_MATTER":     5,
}

# ── Cluster membership ────────────────────────────────────────────────────────
CPS_CLUSTER_PRODUCTS = ["SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"]
SECONDARY_CLUSTER_PRODUCTS = ["SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON"]

# Products handled by dedicated strategies (excluded from the generic loop)
SPECIALIST_PRODUCTS = (
    {"PEBBLES_XL", "PEBBLES_XS", "ROBOT_DISHES", "ROBOT_IRONING"}
    | set(CPS_CLUSTER_PRODUCTS)
    | set(SECONDARY_CLUSTER_PRODUCTS)
)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    """Compact logger that fits within the Prosperity platform log-length limit."""

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
        base_length = len(
            self._to_json([
                self._compress_state(state, ""),
                self._compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.MAX_LOG_LENGTH - base_length) // 3

        print(self._to_json([
            self._compress_state(state, self._truncate(state.traderData, max_item_length)),
            self._compress_orders(orders),
            conversions,
            self._truncate(trader_data, max_item_length),
            self._truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    # ── Private compression helpers ───────────────────────────────────────────

    def _compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
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
            for arr in trades.values()
            for t in arr
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
        """Binary-search the longest prefix of *value* whose JSON length ≤ max_length."""
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
# LOW-LEVEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mid_price(od: OrderDepth) -> float | None:
    """Return the mid-point between best bid and best ask, or None if unavailable."""
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None


def rolling_mean_std(window: list[float]) -> tuple[float, float]:
    """Sample mean and std-dev of *window* (std-dev floored at 1.0)."""
    n = len(window)
    if n < 2:
        return 0.0, 1.0
    mean = sum(window) / n
    std = sqrt(sum((x - mean) ** 2 for x in window) / (n - 1))
    return mean, max(std, 1.0)


def place_order_to_target(
    symbol: str,
    od: OrderDepth,
    current_pos: int,
    target_pos: int,
    orders_out: dict,
) -> None:
    """
    Emit a single order to move *current_pos* toward *target_pos*.

    Uses a passive (inside-spread) price when delta ≤ 3, otherwise
    crosses the spread aggressively.
    """
    if not od.buy_orders or not od.sell_orders:
        return
    delta = target_pos - current_pos
    if delta == 0:
        return

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)

    if abs(delta) <= 3:
        price = best_bid + 1 if delta > 0 else best_ask - 1
    else:
        price = best_ask if delta > 0 else best_bid

    orders_out[symbol] = [Order(symbol, price, delta)]


# ══════════════════════════════════════════════════════════════════════════════
# MARKET MAKING
# ══════════════════════════════════════════════════════════════════════════════

def market_make(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    """
    Post resting quotes one tick inside the best bid/ask.

    Quotes are sized to the remaining capacity on each side so that
    position never exceeds POS_LIMIT.
    """
    if not od.buy_orders or not od.sell_orders:
        return []

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread = best_ask - best_bid

    if spread <= 0:
        return []

    our_bid = best_bid + 1 if spread >= 2 else best_bid
    our_ask = best_ask - 1 if spread >= 2 else best_ask

    if our_bid >= our_ask:
        return []

    buy_capacity  = POS_LIMIT - pos
    sell_capacity = POS_LIMIT + pos
    orders: list[Order] = []

    if buy_capacity  > 0: orders.append(Order(symbol, our_bid,   buy_capacity))
    if sell_capacity > 0: orders.append(Order(symbol, our_ask,  -sell_capacity))

    return orders


# ══════════════════════════════════════════════════════════════════════════════
# TREND HOLD + MARKET MAKING
# ══════════════════════════════════════════════════════════════════════════════

def trade_trend(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    """
    Maintain a minimum long position equal to TREND_HOLD[symbol], then
    layer market-making orders above that floor.
    """
    if not od.buy_orders or not od.sell_orders:
        return []

    hold_size = TREND_HOLD[symbol]
    best_bid  = max(od.buy_orders)
    best_ask  = min(od.sell_orders)
    spread    = best_ask - best_bid
    orders: list[Order] = []

    # Build up to the trend-hold floor
    if pos < hold_size:
        orders.append(Order(symbol, best_ask, hold_size - pos))

    # Overlay market making above the floor
    if spread > 0:
        our_bid      = best_bid + 1 if spread >= 2 else best_bid
        mm_buy_cap   = POS_LIMIT - max(pos, hold_size)
        mm_sell_cap  = max(0, pos - hold_size)

        if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
        if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))

    return orders


# ══════════════════════════════════════════════════════════════════════════════
# PEBBLES XL / XS — MEAN REVERSION PAIR
# ══════════════════════════════════════════════════════════════════════════════

def strategy_pebbles_xl_xs(
    spread_history: list[int],
    hold_ticks: int,
    direction: int,
    od_xl: OrderDepth,
    od_xs: OrderDepth,
    pos_xl: int,
    pos_xs: int,
) -> tuple[dict, int, int]:
    """
    Mean-reversion on the XL − XS mid-price spread.

    Returns (orders, updated_hold_ticks, updated_direction).
    Direction:  +1 = long XL / short XS,  -1 = short XL / long XS,  0 = flat (background).
    """
    orders: dict = {}

    if len(spread_history) >= 50:
        mean, std = rolling_mean_std(spread_history)
        z = (spread_history[-1] - mean) / std

        # Manage open position
        if direction != 0:
            hold_ticks += 1
            if abs(z) < PEBBLES_EXIT_Z or hold_ticks >= PEBBLES_MAX_HOLD:
                direction, hold_ticks = 0, 0

        # Check for new entry signal
        if direction == 0:
            if z > PEBBLES_ENTRY_Z + 0.5:
                direction, hold_ticks = -1, 0
            elif z < -PEBBLES_ENTRY_Z - 0.5:
                direction, hold_ticks = 1, 0

        logger.print(f"pebbles_xl_xs  z={z:.2f}  dir={direction}  hold={hold_ticks}")

    target_xl = POS_LIMIT * direction if direction != 0 else  PEBBLES_BG_POS
    target_xs = -POS_LIMIT * direction if direction != 0 else -PEBBLES_BG_POS

    place_order_to_target("PEBBLES_XL", od_xl, pos_xl, target_xl, orders)
    place_order_to_target("PEBBLES_XS", od_xs, pos_xs, target_xs, orders)

    return orders, hold_ticks, direction


# ══════════════════════════════════════════════════════════════════════════════
# ROBOT DISHES / IRONING — MEAN REVERSION PAIR
# ══════════════════════════════════════════════════════════════════════════════

def strategy_dishes_ironing(
    spread_history: list[int],
    hold_ticks: int,
    direction: int,
    od_dishes: OrderDepth,
    od_ironing: OrderDepth,
    pos_dishes: int,
    pos_ironing: int,
) -> tuple[dict, int, int]:
    """
    Mean-reversion on the Dishes − Ironing mid-price spread.

    Falls back to market making on both legs when no MR signal is active.
    Returns (orders, updated_hold_ticks, updated_direction).
    Direction:  +1 = long Dishes / short Ironing,  -1 = the reverse,  0 = no signal.
    """
    orders: dict = {}

    if len(spread_history) >= 50:
        mean, std = rolling_mean_std(spread_history)
        z = (spread_history[-1] - mean) / std

        # Manage open position
        if direction != 0:
            hold_ticks += 1
            if abs(z) < DI_EXIT_Z or hold_ticks >= DI_MAX_HOLD:
                direction, hold_ticks = 0, 0

        # Check for new entry signal
        if direction == 0:
            if z > DI_ENTRY_Z:       # Dishes expensive vs Ironing → short Dishes
                direction, hold_ticks = -1, 0
            elif z < -DI_ENTRY_Z:    # Dishes cheap vs Ironing → long Dishes
                direction, hold_ticks = 1, 0

    if direction != 0:
        place_order_to_target("ROBOT_DISHES",  od_dishes,  pos_dishes,   POS_LIMIT * direction, orders)
        place_order_to_target("ROBOT_IRONING", od_ironing, pos_ironing, -POS_LIMIT * direction, orders)
    else:
        dishes_orders  = market_make("ROBOT_DISHES",  od_dishes,  pos_dishes)
        ironing_orders = market_make("ROBOT_IRONING", od_ironing, pos_ironing)
        if dishes_orders:  orders["ROBOT_DISHES"]  = dishes_orders
        if ironing_orders: orders["ROBOT_IRONING"] = ironing_orders

    return orders, hold_ticks, direction


# ══════════════════════════════════════════════════════════════════════════════
# CPS CLUSTER — SLEEP POD CORE (nonlinear deviation-weighted basket MR)
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_mean_price(products: list[str], order_depths: dict) -> float | None:
    """Return the equal-weighted average mid-price of *products*, or None."""
    mids = [
        mid_price(od)
        for p in products
        if (od := order_depths.get(p)) is not None
        and mid_price(od) is not None
    ]
    return sum(mids) / len(mids) if mids else None


def trade_cps_cluster(state: TradingState, orders_out: dict) -> None:
    """
    Nonlinear deviation-weighted mean-reversion across the CPS_CLUSTER_PRODUCTS basket.

    Each product's allocation is proportional to (deviation ^ CPS_POWER), so
    outliers receive a disproportionately large position.
    """
    factor = _cluster_mean_price(CPS_CLUSTER_PRODUCTS, state.order_depths)
    if factor is None:
        return

    ods  = {p: state.order_depths.get(p) for p in CPS_CLUSTER_PRODUCTS}
    mids = {p: mid_price(od) for p, od in ods.items() if od and mid_price(od) is not None}
    devs = {p: mids[p] - factor for p in mids}

    if len(devs) < len(CPS_CLUSTER_PRODUCTS):
        return   # need all legs to trade the basket

    # Nonlinear weights: sign(dev) * |dev|^CPS_POWER
    raw_weights = {
        p: (1 if dev >= 0 else -1) * (abs(dev) ** CPS_POWER)
        for p, dev in devs.items()
    }
    weight_sum = max(sum(abs(w) for w in raw_weights.values()), 1e-9)

    for symbol, od_local in ods.items():
        if not od_local:
            continue

        pos = state.position.get(symbol, 0)
        norm_weight = raw_weights[symbol] / weight_sum
        target = int(-norm_weight * CPS_TOTAL_RISK)

        best_bid = max(od_local.buy_orders)
        best_ask = min(od_local.sell_orders)
        price = best_bid if target < 0 else best_ask
        qty = max(-POS_LIMIT, min(POS_LIMIT, target - pos))

        if qty != 0:
            orders_out[symbol] = [Order(symbol, price, qty)]


# ══════════════════════════════════════════════════════════════════════════════
# SECONDARY SLEEP POD CLUSTER (mean-reversion overlay)
# ══════════════════════════════════════════════════════════════════════════════

def trade_secondary_cluster(state: TradingState, orders_out: dict) -> None:
    """
    Mean-reversion overlay for SECONDARY_CLUSTER_PRODUCTS.

    Each product is traded toward the cluster mean; positions are proportional
    to the deviation from that mean.
    """
    factor = _cluster_mean_price(SECONDARY_CLUSTER_PRODUCTS, state.order_depths)
    if factor is None:
        return

    for symbol in SECONDARY_CLUSTER_PRODUCTS:
        od = state.order_depths.get(symbol)
        if not od:
            continue

        price_mid = mid_price(od)
        if price_mid is None:
            continue

        deviation = price_mid - factor
        if abs(deviation) < 1:
            continue   # inside deadband — no action

        pos    = state.position.get(symbol, 0)
        target = -int(deviation * SECONDARY_SCALE)
        target = max(-POS_LIMIT, min(POS_LIMIT, target))

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        price = best_bid if target < 0 else best_ask
        qty   = max(-POS_LIMIT, min(POS_LIMIT, target - pos))

        if qty != 0:
            orders_out[symbol] = [Order(symbol, price, qty)]


# ══════════════════════════════════════════════════════════════════════════════
# TRADER — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

class Trader:

    def run(
        self, state: TradingState
    ) -> tuple[dict[Symbol, list[Order]], int, str]:

        result:      dict[Symbol, list[Order]] = {}
        conversions: int = 0

        # ── Load persisted state ───────────────────────────────────────────────
        mem = json.loads(state.traderData) if state.traderData else {}

        pebbles_history   = mem.get("pebbles_history", [])
        pebbles_hold      = mem.get("pebbles_hold", 0)
        pebbles_direction = mem.get("pebbles_dir", 0)

        di_history   = mem.get("di_history", [])
        di_hold      = mem.get("di_hold", 0)
        di_direction = mem.get("di_dir", 0)

        # ── Pebbles XL / XS mean-reversion strategy ───────────────────────────
        od_xl = state.order_depths.get("PEBBLES_XL")
        od_xs = state.order_depths.get("PEBBLES_XS")

        if ENABLED.get("PEBBLES_XL") and od_xl and od_xs:
            mid_xl = mid_price(od_xl)
            mid_xs = mid_price(od_xs)

            if mid_xl is not None and mid_xs is not None:
                pebbles_history.append(int(mid_xl - mid_xs))
                if len(pebbles_history) > PEBBLES_WINDOW:
                    pebbles_history = pebbles_history[-PEBBLES_WINDOW:]

                pos_xl = state.position.get("PEBBLES_XL", 0)
                pos_xs = state.position.get("PEBBLES_XS", 0)

                xl_xs_orders, pebbles_hold, pebbles_direction = strategy_pebbles_xl_xs(
                    pebbles_history, pebbles_hold, pebbles_direction,
                    od_xl, od_xs, pos_xl, pos_xs,
                )
                result.update(xl_xs_orders)

        # ── Robot Dishes / Ironing mean-reversion strategy ────────────────────
        od_dishes  = state.order_depths.get("ROBOT_DISHES")
        od_ironing = state.order_depths.get("ROBOT_IRONING")

        if ENABLED.get("ROBOT_DISHES") and od_dishes and od_ironing:
            mid_dishes  = mid_price(od_dishes)
            mid_ironing = mid_price(od_ironing)

            if mid_dishes is not None and mid_ironing is not None:
                di_history.append(int(mid_dishes - mid_ironing))
                if len(di_history) > DI_WINDOW:
                    di_history = di_history[-DI_WINDOW:]

                pos_dishes  = state.position.get("ROBOT_DISHES", 0)
                pos_ironing = state.position.get("ROBOT_IRONING", 0)

                di_orders, di_hold, di_direction = strategy_dishes_ironing(
                    di_history, di_hold, di_direction,
                    od_dishes, od_ironing, pos_dishes, pos_ironing,
                )
                result.update(di_orders)

        # ── Sleep Pod cluster strategies ──────────────────────────────────────
        trade_cps_cluster(state, result)
        trade_secondary_cluster(state, result)

        # ── All remaining products (trend-hold or market-make) ────────────────
        for symbol, od in state.order_depths.items():
            if symbol in SPECIALIST_PRODUCTS:
                continue
            if not ENABLED.get(symbol, False):
                continue

            pos = state.position.get(symbol, 0)
            orders = (
                trade_trend(symbol, od, pos)
                if symbol in TREND_HOLD
                else market_make(symbol, od, pos)
            )
            if orders:
                result[symbol] = orders

        # ── Persist state ─────────────────────────────────────────────────────
        new_mem = {
            "pebbles_history": pebbles_history,
            "pebbles_hold":    pebbles_hold,
            "pebbles_dir":     pebbles_direction,
            "di_history":      di_history,
            "di_hold":         di_hold,
            "di_dir":          di_direction,
        }
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data