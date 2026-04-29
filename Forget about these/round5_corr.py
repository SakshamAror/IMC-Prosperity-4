from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
from math import sqrt
import json


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
    'MICROCHIP_CIRCLE':      True,
    'MICROCHIP_OVAL':        True,
    'MICROCHIP_RECTANGLE':   True,
    'MICROCHIP_SQUARE':      True,
    'MICROCHIP_TRIANGLE':    True,
    # Oxygen Shake
    'OXYGEN_SHAKE_CHOCOLATE':       True,
    'OXYGEN_SHAKE_EVENING_BREATH':  True,
    'OXYGEN_SHAKE_GARLIC':          True,
    'OXYGEN_SHAKE_MINT':            True,
    'OXYGEN_SHAKE_MORNING_BREATH':  True,
    # Panel
    'PANEL_1X2':  False,
    'PANEL_1X4':  True,
    'PANEL_2X2':  True,
    'PANEL_2X4':  True,
    'PANEL_4X4':  False,
    # Pebbles — XL/XS handled by MR strategy; L/M/S market-made
    'PEBBLES_XS':  True,
    'PEBBLES_S':   True,
    'PEBBLES_M':   False,
    'PEBBLES_L':   True,
    'PEBBLES_XL':  True,
    # Robot
    'ROBOT_DISHES':    True,
    'ROBOT_IRONING':   True,
    'ROBOT_LAUNDRY':   True,
    'ROBOT_MOPPING':   True,
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
    'UV_VISOR_MAGENTA':  True,
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

# Genuine uptrends — hold size sqrt-scaled to trend strength
TREND_HOLD: dict[str, int] = {
    "MICROCHIP_SQUARE":              9,
    "SLEEP_POD_SUEDE":               7,
    "SLEEP_POD_POLYESTER":           7,
    "GALAXY_SOUNDS_PLANETARY_RINGS": 7,
    "UV_VISOR_MAGENTA":              6,
    "OXYGEN_SHAKE_GARLIC":           6,
    "SLEEP_POD_COTTON":              6,
    "UV_VISOR_YELLOW":               6,
    "ROBOT_MOPPING":                 6,
    "PANEL_2X4":                     6,
    "GALAXY_SOUNDS_BLACK_HOLES":     6,
    "GALAXY_SOUNDS_DARK_MATTER":     5,
}

# ── EMA exit for all trend-hold products ─────────────────────────────────────
# Correlation analysis shows a SYSTEMATIC day-4 reversal across GALAXY, UV_VISOR,
# MICROCHIP, SLEEP_POD, and TRANSLATOR — multiple categories flip on the same day.
# A single EMA30/EMA200 crossover applied uniformly to all TREND_HOLD products
# catches this reversal regardless of which category leads or lags.
# Cross-category tick correlations are all <0.15 — not worth trading on directly.
EMA_FAST_ALPHA = 2.0 / (30  + 1)   # span = 30
EMA_SLOW_ALPHA = 2.0 / (200 + 1)   # span = 200
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

    if spread <= 0:
        return orders

    our_bid = best_bid + 1 if spread >= 2 else best_bid
    our_ask = best_ask - 1 if spread >= 2 else best_ask

    if our_bid >= our_ask:
        return orders

    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos

    if buy_cap  > 0: orders.append(Order(symbol, our_bid,  buy_cap))
    if sell_cap > 0: orders.append(Order(symbol, our_ask, -sell_cap))

    return orders


# ── Trend hold + EMA exit (all trend products) ───────────────────────────────
def trade_trend(symbol: str, od: OrderDepth, pos: int,
                ema: dict) -> tuple[list[Order], dict]:
    orders: list[Order] = []
    if not od.buy_orders or not od.sell_orders:
        return orders, ema

    hold     = TREND_HOLD[symbol]
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid
    mid      = (best_bid + best_ask) / 2.0

    # Update online EMAs (initialise to mid on first tick for this symbol)
    ef = symbol + "_f"
    es = symbol + "_s"
    ema_fast = EMA_FAST_ALPHA * mid + (1 - EMA_FAST_ALPHA) * ema.get(ef, mid)
    ema_slow = EMA_SLOW_ALPHA * mid + (1 - EMA_SLOW_ALPHA) * ema.get(es, mid)
    ema[ef] = ema_fast
    ema[es] = ema_slow

    trending = ema_fast >= ema_slow

    # ── EMA says trend reversed: exit hold, fall back to symmetric MM ──
    if not trending:
        if pos > 0:
            orders.append(Order(symbol, best_bid, -pos))
            return orders, ema
        if spread > 0:
            our_bid = best_bid + 1 if spread >= 2 else best_bid
            our_ask = best_ask - 1 if spread >= 2 else best_ask
            if our_bid < our_ask:
                buy_cap  = POS_LIMIT - pos
                sell_cap = POS_LIMIT + pos
                if buy_cap  > 0: orders.append(Order(symbol, our_bid,  buy_cap))
                if sell_cap > 0: orders.append(Order(symbol, our_ask, -sell_cap))
        return orders, ema

    # ── Trend intact: hold + MM on remaining capacity ────────────────
    if pos < hold:
        orders.append(Order(symbol, best_ask, hold - pos))

    if spread > 0:
        our_bid     = best_bid + 1 if spread >= 2 else best_bid
        mm_buy_cap  = POS_LIMIT - max(pos, hold)
        mm_sell_cap = max(0, pos - hold)
        if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
        if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))

    return orders, ema


class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── Load state ────────────────────────────────────────────────
        mem       = json.loads(state.traderData) if state.traderData else {}
        history   = mem.get("h", [])
        hold      = mem.get("hold", 0)
        direction = mem.get("dir", 0)
        ema       = mem.get("ema", {})

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

        # ── All other products (incl. PEBBLES_L/M/S) ─────────────────
        for symbol, od in state.order_depths.items():
            if symbol in ("PEBBLES_XL", "PEBBLES_XS"):
                continue
            if not ENABLED.get(symbol, False):
                continue

            pos = state.position.get(symbol, 0)

            if symbol in TREND_HOLD:
                orders, ema = trade_trend(symbol, od, pos, ema)
            else:
                orders = market_make(symbol, od, pos)

            if orders:
                result[symbol] = orders

        # ── Save state ────────────────────────────────────────────────
        new_mem     = {"h": history, "hold": hold, "dir": direction, "ema": ema}
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
