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
    'PEBBLES_L':   False,   
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
# ROBOT_MOPPING removed: only profitable day 3, loses days 2+4 in trend hold → pure MM
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

# ─────────────────────────────────────────────────────────────────────────────

# ── ROBOT DISHES-IRONING pair MR ─────────────────────────────────────────────
# resid_ac1 = -0.20, HL ≈ 3 ticks. Spread mean drifts day-to-day so use
# short rolling window to track local level.
DI_WINDOW   = 550   # short: adapts to drifting spread mean
DI_ENTRY_Z  = 2.2   # ~5.6% fire rate
DI_EXIT_Z   = 0.5
DI_MAX_HOLD = 150
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

    # OBI-weighted fair value: heavy bid side → fv closer to ask, vice versa
    bid_vol = od.buy_orders[best_bid]
    ask_vol = -od.sell_orders[best_ask]
    total   = bid_vol + ask_vol
    fv = (bid_vol * best_ask + ask_vol * best_bid) / total if total > 0 else (best_bid + best_ask) / 2.0

    if spread >= 2:
        our_bid = max(best_bid, min(int(fv),     best_ask - 1))
        our_ask = max(our_bid + 1, min(int(fv) + 1, best_ask))
    else:
        our_bid = best_bid
        our_ask = best_ask

    if our_bid >= our_ask:
        return orders

    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos

    if buy_cap  > 0: orders.append(Order(symbol, our_bid,  buy_cap))
    if sell_cap > 0: orders.append(Order(symbol, our_ask, -sell_cap))

    return orders


# ── ROBOT DISHES-IRONING pair strategy ───────────────────────────────────────
def strategy_dishes_ironing(di_hist: list, di_hold: int, di_dir: int,
                             od_d: OrderDepth, od_i: OrderDepth,
                             pos_d: int, pos_i: int) -> tuple[dict, int, int]:
    orders: dict = {}

    if len(di_hist) >= 50:
        mean, std = rolling_mean_std(di_hist)
        z = (di_hist[-1] - mean) / std

        if di_dir != 0:
            di_hold += 1
            if abs(z) < DI_EXIT_Z or di_hold >= DI_MAX_HOLD:
                di_dir = 0
                di_hold = 0

        if di_dir == 0:
            if z > DI_ENTRY_Z:      # DISHES expensive vs IRONING
                di_dir = -1
                di_hold = 0
            elif z < -DI_ENTRY_Z:   # DISHES cheap vs IRONING
                di_dir = 1
                di_hold = 0

    if di_dir != 0:
        place("ROBOT_DISHES",  od_d, pos_d,  POS_LIMIT * di_dir,  orders)
        place("ROBOT_IRONING", od_i, pos_i, -POS_LIMIT * di_dir,  orders)
    else:
        d_ords = market_make("ROBOT_DISHES",  od_d, pos_d)
        i_ords = market_make("ROBOT_IRONING", od_i, pos_i)
        if d_ords: orders["ROBOT_DISHES"]  = d_ords
        if i_ords: orders["ROBOT_IRONING"] = i_ords

    return orders, di_hold, di_dir



# ── Trend hold + MM (all TREND_HOLD products) ────────────────────────────────
def trade_trend(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    orders: list[Order] = []
    if not od.buy_orders or not od.sell_orders:
        return orders

    hold     = TREND_HOLD[symbol]
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid

    if pos < hold:
        orders.append(Order(symbol, best_ask, hold - pos))

    if spread > 0:
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]
        total   = bid_vol + ask_vol
        fv = (bid_vol * best_ask + ask_vol * best_bid) / total if total > 0 else (best_bid + best_ask) / 2.0
        if spread >= 2:
            our_bid = max(best_bid, min(int(fv),     best_ask - 1))
            our_ask = max(our_bid + 1, min(int(fv) + 1, best_ask))
        else:
            our_bid = best_bid
            our_ask = best_ask
        mm_buy_cap  = POS_LIMIT - max(pos, hold)
        mm_sell_cap = max(0, pos - hold)
        if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
        if mm_sell_cap > 0: orders.append(Order(symbol, our_ask, -mm_sell_cap))

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
        di_hist   = mem.get("dh", [])
        di_hold   = mem.get("dhld", 0)
        di_dir    = mem.get("ddir", 0)

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

        # ── ROBOT DISHES-IRONING pair strategy ───────────────────────
        od_d = state.order_depths.get("ROBOT_DISHES")
        od_i = state.order_depths.get("ROBOT_IRONING")
        if ENABLED.get('ROBOT_DISHES', False) and od_d and od_i:
            mid_d = mid_price(od_d)
            mid_i = mid_price(od_i)
            if mid_d is not None and mid_i is not None:
                di_hist.append(int(mid_d - mid_i))
                if len(di_hist) > DI_WINDOW:
                    di_hist = di_hist[-DI_WINDOW:]
                pos_d = state.position.get("ROBOT_DISHES", 0)
                pos_i = state.position.get("ROBOT_IRONING", 0)
                orders_di, di_hold, di_dir = strategy_dishes_ironing(
                    di_hist, di_hold, di_dir, od_d, od_i, pos_d, pos_i)
                result.update(orders_di)

        # ── All other products ────────────────────────────────────────
        SKIP = {"PEBBLES_XL", "PEBBLES_XS", "ROBOT_DISHES", "ROBOT_IRONING"}
        for symbol, od in state.order_depths.items():
            if symbol in SKIP:
                continue
            if not ENABLED.get(symbol, False):
                continue

            pos = state.position.get(symbol, 0)

            if symbol in TREND_HOLD:
                orders = trade_trend(symbol, od, pos)
            else:
                orders = market_make(symbol, od, pos)

            if orders:
                result[symbol] = orders

        # ── Save state ────────────────────────────────────────────────
        new_mem     = {"h": history, "hold": hold, "dir": direction,
                       "dh": di_hist, "dhld": di_hold, "ddir": di_dir}
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
