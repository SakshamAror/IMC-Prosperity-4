from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict

from math import sqrt
import json

# ─────────────────────────────────────────────────────────────
# STRATEGY OVERVIEW (READ FIRST)
# ─────────────────────────────────────────────────────────────
# This trading system is a hybrid multi-regime market maker.
#
# 1. Pebbles (PEBBLES_XL/XS)
#    - Pure statistical mean reversion strategy
#    - Trades spread between XL and XS legs
#
# 2. All other products
#    - Default: Market Making (MM)
#    - Two additional adaptive overlays:
#
#    (a) REGIME SWITCHING (per-product)
#        - Uses volatility (rel_vol) to switch between:
#          • MM mode: passive liquidity provision
#          • TREND mode: directional taker execution
#
#    (b) MARKET STRUCTURE ADJUSTMENT (MM enhancement)
#        - Order book imbalance adjusts quoting skew
#        - Inventory skew reduces directional exposure
#
# 3. Key idea
#    - Calm markets → provide liquidity (earn spread)
#    - Volatile markets → follow momentum (take liquidity)
#    - Inventory + imbalance prevent adverse selection
#
# NOTE:
# - All enhancements are modular and can be removed:
#   • VOL_THRESHOLD → controls regime switching
#   • TREND_HOLD_INIT → controls trend duration
#   • imbalance/skew block → controls MM quoting bias
# ─────────────────────────────────────────────────────────────


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json([
                self.compress_state(state, ""),
                self.compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                           obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()

# ── Product toggles ───────────────────────────────────────────────────────────
ENABLED = {
    # Galaxy Sounds
    'GALAXY_SOUNDS_BLACK_HOLES':      True,
    'GALAXY_SOUNDS_DARK_MATTER':      True,
    'GALAXY_SOUNDS_PLANETARY_RINGS':  True,
    'GALAXY_SOUNDS_SOLAR_FLAMES':     False,
    'GALAXY_SOUNDS_SOLAR_WINDS':      True,
    # Microchip
    'MICROCHIP_CIRCLE':      False,
    'MICROCHIP_OVAL':        True,
    'MICROCHIP_RECTANGLE':   True,
    'MICROCHIP_SQUARE':      False,
    'MICROCHIP_TRIANGLE':    True,
    # Oxygen Shake
    'OXYGEN_SHAKE_CHOCOLATE':       True,
    'OXYGEN_SHAKE_EVENING_BREATH':  True,
    'OXYGEN_SHAKE_GARLIC':          False,
    'OXYGEN_SHAKE_MINT':            False,
    'OXYGEN_SHAKE_MORNING_BREATH':  True,
    # Panel
    'PANEL_1X2':  True,
    'PANEL_1X4':  True,
    'PANEL_2X2':  True,
    'PANEL_2X4':  True,
    'PANEL_4X4':  True,
    # Pebbles — XL/XS handled by MR strategy; L/M/S market-made
    'PEBBLES_XS':  True,
    'PEBBLES_S':   True,
    'PEBBLES_M':   True,
    'PEBBLES_L':   True,   
    'PEBBLES_XL':  True,
    # Robot
    'ROBOT_DISHES':    True,
    'ROBOT_IRONING':   True,
    'ROBOT_LAUNDRY':   True,
    'ROBOT_MOPPING':   False,
    'ROBOT_VACUUMING': False,
    # Sleep Pod
    'SLEEP_POD_COTTON':     True,
    'SLEEP_POD_LAMB_WOOL':  False,
    'SLEEP_POD_NYLON':      True,
    'SLEEP_POD_POLYESTER':  True,
    'SLEEP_POD_SUEDE':      True,
    # Snackpack
    'SNACKPACK_CHOCOLATE':   True,
    'SNACKPACK_PISTACHIO':   True,
    'SNACKPACK_RASPBERRY':   True,
    'SNACKPACK_STRAWBERRY':  True,
    'SNACKPACK_VANILLA':     True,
    # Translator
    'TRANSLATOR_ASTRO_BLACK':      True,
    'TRANSLATOR_ECLIPSE_CHARCOAL': True,
    'TRANSLATOR_GRAPHITE_MIST':    False,
    'TRANSLATOR_SPACE_GRAY':       False,
    'TRANSLATOR_VOID_BLUE':        True,
    # UV Visor
    'UV_VISOR_AMBER':    True,
    'UV_VISOR_MAGENTA':  False,
    'UV_VISOR_ORANGE':   True,
    'UV_VISOR_RED':      True,
    'UV_VISOR_YELLOW':   True,
}
# ─────────────────────────────────────────────────────────────────────────────

# ── Pebbles ───────────────────────────────────────────────────────────────────
WINDOW   = 500   # rolling window for z-score
ENTRY_Z  = 2.5   # z-score threshold to enter MR trade
EXIT_Z   = 0.3   # z-score threshold to exit MR trade
MAX_HOLD = 200   # force-exit MR trade after this many ticks
BG_POS   = 10    # background long XL / short XS when no MR signal
# ─────────────────────────────────────────────────────────────────────────────

POS_LIMIT = 10   # hard position limit for every product
USE_MAX_SIZE = False  # if False → adaptive sizing

# ── Per-product regime tuning ───────────────────────────────
VOL_THRESHOLD = {
    # Galaxy Sounds (mostly trending)
    'GALAXY_SOUNDS_BLACK_HOLES': 1.2,
    'GALAXY_SOUNDS_DARK_MATTER': 2.0,
    'GALAXY_SOUNDS_PLANETARY_RINGS': 1.2,
    'GALAXY_SOUNDS_SOLAR_FLAMES': 2.0,
    'GALAXY_SOUNDS_SOLAR_WINDS': 2.0,

    # Microchip
    'MICROCHIP_CIRCLE': 2.0,
    'MICROCHIP_OVAL': 1.3,
    'MICROCHIP_RECTANGLE': 2.0,
    'MICROCHIP_SQUARE': 1.3,
    'MICROCHIP_TRIANGLE': 1.3,

    # Oxygen Shake
    'OXYGEN_SHAKE_CHOCOLATE': 3.0,
    'OXYGEN_SHAKE_EVENING_BREATH': 3.5,
    'OXYGEN_SHAKE_GARLIC': 1.3,
    'OXYGEN_SHAKE_MINT': 2.0,
    'OXYGEN_SHAKE_MORNING_BREATH': 2.0,

    # Panel
    'PANEL_1X2': 2.0,
    'PANEL_1X4': 2.0,
    'PANEL_2X2': 2.0,
    'PANEL_2X4': 2.0,
    'PANEL_4X4': 2.0,

    # Pebbles (MM products here)
    'PEBBLES_L': 2.5,
    'PEBBLES_M': 2.5,
    'PEBBLES_S': 2.5,
    'PEBBLES_XL': 2.5,
    'PEBBLES_XS': 2.5,

    # Robot
    'ROBOT_DISHES': 4.0,
    'ROBOT_IRONING': 4.0,
    'ROBOT_LAUNDRY': 3.0,
    'ROBOT_MOPPING': 3.0,
    'ROBOT_VACUUMING': 3.0,

    # Sleep Pod
    'SLEEP_POD_COTTON': 1.4,
    'SLEEP_POD_LAMB_WOOL': 2.0,
    'SLEEP_POD_NYLON': 2.0,
    'SLEEP_POD_POLYESTER': 1.4,
    'SLEEP_POD_SUEDE': 1.4,

    # Snackpack
    'SNACKPACK_CHOCOLATE': 3.0,
    'SNACKPACK_PISTACHIO': 2.5,
    'SNACKPACK_RASPBERRY': 2.5,
    'SNACKPACK_STRAWBERRY': 2.5,
    'SNACKPACK_VANILLA': 2.5,

    # Translator
    'TRANSLATOR_ASTRO_BLACK': 2.0,
    'TRANSLATOR_ECLIPSE_CHARCOAL': 2.0,
    'TRANSLATOR_GRAPHITE_MIST': 2.0,
    'TRANSLATOR_SPACE_GRAY': 2.0,
    'TRANSLATOR_VOID_BLUE': 2.0,

    # UV Visor
    'UV_VISOR_AMBER': 1.2,
    'UV_VISOR_MAGENTA': 2.0,
    'UV_VISOR_ORANGE': 2.0,
    'UV_VISOR_RED': 2.0,
    'UV_VISOR_YELLOW': 2.0,
}

TREND_HOLD_INIT = {
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


# ─────────────────────────────────────────────────────────────────────────────



# ── Helpers ───────────────────────────────────────────────────────────────────
def mid_price(od: OrderDepth):
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None


def rolling_mean_std(window: list) -> tuple:
    n = len(window)
    if n < 2:
        return 0.0, 1.0
    m = sum(window) / n
    s = sqrt(sum((x - m) ** 2 for x in window) / (n - 1))
    return m, max(s, 1.0)


def place(symbol: str, od: OrderDepth, cur: int, tgt: int, result: dict):
    delta = tgt - cur
    if not od.buy_orders or not od.sell_orders:
        return
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    if abs(delta) <= 3:
        price = best_bid + 1 if delta > 0 else best_ask - 1
    else:
        price = best_ask if delta > 0 else best_bid
    if delta != 0:
        result[symbol] = [Order(symbol, price, delta)]


# ── Pebbles XL-XS MR strategy ────────────────────────────────────────────────
def strategy_xl_xs(history: list, hold: int, direction: int,
                   od_xl: OrderDepth, od_xs: OrderDepth,
                   pos_xl: int, pos_xs: int) -> tuple[dict, int, int]:
    orders = {}

    if len(history) >= 50:
        mean, std = rolling_mean_std(history)
        z = (history[-1] - mean) / std

        if direction != 0:
            hold += 1
            if abs(z) < EXIT_Z or hold >= MAX_HOLD:
                direction = 0
                hold = 0

        if direction == 0:
            if z > ENTRY_Z + 0.5:
                direction = -1
                hold = 0
            elif z < -ENTRY_Z - 0.5:
                direction = 1
                hold = 0

        logger.print(f"xl_xs z={z:.2f} dir={direction} hold={hold}")

    target_xl = POS_LIMIT * direction if direction != 0 else BG_POS
    target_xs = -POS_LIMIT * direction if direction != 0 else -BG_POS

    place("PEBBLES_XL", od_xl, pos_xl, target_xl, orders)
    place("PEBBLES_XS", od_xs, pos_xs, target_xs, orders)
    return orders, hold, direction


# ── Market making ─────────────────────────────────────────────────────────────
def market_make(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    orders: list[Order] = []
    if not od.buy_orders or not od.sell_orders:
        return orders

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid

    mid = (best_bid + best_ask) / 2

    if not hasattr(market_make, "last_mid"):
        market_make.last_mid = {}

    prev_mid = market_make.last_mid.get(symbol)
    market_make.last_mid[symbol] = mid

    # ── volatility + regime state ───────────────────────────────
    if not hasattr(market_make, "vol_ema"):
        market_make.vol_ema = {}
    if not hasattr(market_make, "mode"):
        market_make.mode = {}
    if not hasattr(market_make, "trend_hold"):
        market_make.trend_hold = {}
    if not hasattr(market_make, "trend_dir"):
        market_make.trend_dir = {}

    prev_mid = market_make.last_mid.get(symbol)

    vol = 0
    if prev_mid is not None:
        vol = abs(mid - prev_mid)

    prev_vol = market_make.vol_ema.get(symbol, vol)
    market_make.vol_ema[symbol] = 0.2 * vol + 0.8 * prev_vol

    rel_vol = market_make.vol_ema[symbol] / max(1, spread)

    # ── regime decision ─────────────────────────────────────────
    threshold = VOL_THRESHOLD.get(symbol, 2.0)
    mode = "MM"
    if rel_vol > threshold:
        mode = "TREND"
    market_make.mode[symbol] = mode

    trend = 0
    if prev_mid is not None:
        trend = mid - prev_mid

    # ── order book imbalance ────────────────────────────────────
    # >>> IMBALANCE MODULE START (SAFE TO REMOVE)
    bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
    ask_vol = sum(od.sell_orders.values()) if od.sell_orders else 0
    imbalance = 0
    denom = bid_vol + ask_vol
    if denom > 0:
        imbalance = (bid_vol - ask_vol) / denom
    # <<< IMBALANCE MODULE END

    # >>> SKEW MODULE START (SAFE TO REMOVE)
    skew = pos / POS_LIMIT
    skew = skew - 0.3 * imbalance
    # <<< SKEW MODULE END

    # ── TREND MODE OVERRIDE ─────────────────────────────────────
    if market_make.mode.get(symbol, "MM") == "TREND":

        hold = market_make.trend_hold.get(symbol, 0)
        direction = market_make.trend_dir.get(symbol, 0)

        # initialize direction if new trend
        if hold == 0:
            direction = 1 if trend > 0 else -1
            market_make.trend_dir[symbol] = direction
            hold = TREND_HOLD_INIT.get(symbol, 8)
        else:
            hold -= 1

        market_make.trend_hold[symbol] = hold

        orders: list[Order] = []

        size = max(1, POS_LIMIT // 3)

        if direction > 0:
            # buy aggressively
            orders.append(Order(symbol, best_ask, size))
        else:
            # sell aggressively
            orders.append(Order(symbol, best_bid, -size))

        return orders

    our_bid = best_bid + 1 - int(skew > 0)
    our_ask = best_ask - 1 + int(skew < 0)


    if our_bid >= our_ask:
        return orders

    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos

    if USE_MAX_SIZE:
        buy_size  = buy_cap
        sell_size = sell_cap
    else:
        base = max(1, spread // 2)
        buy_size  = min(buy_cap, base)
        sell_size = min(sell_cap, base)

    if buy_size > 0:
        orders.append(Order(symbol, our_bid, buy_size))
    if sell_size > 0:
        orders.append(Order(symbol, our_ask, -sell_size))

    return orders




class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── Load state ────────────────────────────────────────────────
        mem       = json.loads(state.traderData) if state.traderData else {}
        history   = mem.get("h", [])
        hold      = mem.get("hold", 0)
        direction = mem.get("dir", 0)

        od_xl = state.order_depths.get("PEBBLES_XL")
        od_xs = state.order_depths.get("PEBBLES_XS")

        # ── Pebbles XL-XS MR strategy ────────────────────────────────
        if ENABLED.get('PEBBLES_XL', False) and od_xl and od_xs:
            mid_xl = mid_price(od_xl)
            mid_xs = mid_price(od_xs)

            if mid_xl is not None and mid_xs is not None:
                history.append(int(mid_xl - mid_xs))
                if len(history) > WINDOW:
                    history = history[-WINDOW:]

                pos_xl = state.position.get("PEBBLES_XL", 0)
                pos_xs = state.position.get("PEBBLES_XS", 0)

                orders_1, hold, direction = strategy_xl_xs(
                    history, hold, direction, od_xl, od_xs, pos_xl, pos_xs)
                result.update(orders_1)

        # ── All other products ────────────────────────────────────────
        SKIP = {"PEBBLES_XL", "PEBBLES_XS"}
        for symbol, od in state.order_depths.items():
            if symbol in SKIP:
                continue
            if not ENABLED.get(symbol, False):
                continue

            pos = state.position.get(symbol, 0)
            orders = market_make(symbol, od, pos)

            if orders:
                result[symbol] = orders

        # ── Save state ────────────────────────────────────────────────
        new_mem     = {"h": history, "hold": hold, "dir": direction}
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
