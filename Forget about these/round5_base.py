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

# ── Hyperparameters ───────────────────────────────────────────────────────────

# ── Pebbles ───────────────────────────────────────────────────────────────────
WINDOW    = 500   # rolling window for z-score
ENTRY_Z   = 2.5   # z-score threshold to enter MR trade
EXIT_Z    = 0.3   # z-score threshold to exit MR trade
MAX_HOLD  = 200   # force-exit MR trade after this many ticks
BG_POS = 10        # Pos in the background to buy and hold XL-XS
# ──────────────────────────────────────────────────────────────────────────────

POS_LIMIT   = 10   # hard position limit for every product

# Genuine uptrends: positive slope on BOTH day 2 and day 3, R² > 0.3
# Hold size sqrt-scaled to trend strength (avg=6.42, range 5-9)
TREND_HOLD: dict[str, int] = {
    "MICROCHIP_SQUARE":              9,  # +62.7%  R²=0.937
    "SLEEP_POD_SUEDE":               7,  # +29.8%  R²=0.851
    "SLEEP_POD_POLYESTER":           7,  # +26.7%  R²=0.878
    "GALAXY_SOUNDS_PLANETARY_RINGS": 7,  # +26.5%  R²=0.740
    "UV_VISOR_MAGENTA":              6,  # +18.6%  R²=0.806
    "OXYGEN_SHAKE_GARLIC":           6,  # +16.6%  R²=0.583
    "SLEEP_POD_COTTON":              6,  # +15.7%  R²=0.704
    "UV_VISOR_YELLOW":               6,  # +15.5%  R²=0.557
    "ROBOT_MOPPING":                 6,  # +15.4%  R²=0.374
    "PANEL_2X4":                     6,  # +12.7%  R²=0.611
    "GALAXY_SOUNDS_BLACK_HOLES":     6,  # +11.1%  R²=0.405
    "GALAXY_SOUNDS_DARK_MATTER":     5,  # +8.4%   R²=0.550
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Helper functions for Pebbles strategies ──────────────────────────────────
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
    if delta > 0 and od.sell_orders:
        result[symbol] = [Order(symbol, min(od.sell_orders), delta)]
    elif delta < 0 and od.buy_orders:
        result[symbol] = [Order(symbol, max(od.buy_orders), delta)]


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
            if z > ENTRY_Z:
                direction = -1
                hold = 0
            elif z < -ENTRY_Z:
                direction = 1
                hold = 0

        logger.print(f"xl_xs z={z:.2f} dir={direction} hold={hold}")

    target_xl = POS_LIMIT * direction if direction != 0 else BG_POS
    target_xs = -POS_LIMIT * direction if direction != 0 else -BG_POS

    place("PEBBLES_XL", od_xl, pos_xl, target_xl, orders)
    place("PEBBLES_XS", od_xs, pos_xs, target_xs, orders)
    return orders, hold, direction


def strategy_lsm_hedge(direction: int,
                       od_l: OrderDepth, od_s: OrderDepth, od_m: OrderDepth,
                       pos_l: int, pos_s: int, pos_m: int) -> dict:
    orders = {}
    tgt = -direction * POS_LIMIT
    if od_l: place("PEBBLES_L", od_l, pos_l, tgt, orders)
    if od_s: place("PEBBLES_S", od_s, pos_s, tgt, orders)
    if od_m: place("PEBBLES_M", od_m, pos_m, tgt, orders)
    return orders


def market_make(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    """Symmetric passive MM — step one tick inside spread when possible.
    Position skew is implicit via buy_cap / sell_cap."""
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


def trade_trend(symbol: str, od: OrderDepth, pos: int) -> list[Order]:
    """Buy-and-hold TREND_HOLD[symbol] units, MM the remaining capacity.

    Directional phase (pos < hold):
      - Taker buy at best_ask to reach hold quickly.

    MM phase (pos >= hold):
      - Buy cap:  POS_LIMIT - pos        (above hold, up to limit)
      - Sell cap: pos - hold             (only sell what's above the floor)
      - Skewed: bid one tick inside spread, ask at best_ask (reluctant to sell).
    """
    orders: list[Order] = []

    if not od.buy_orders or not od.sell_orders:
        return orders

    hold     = TREND_HOLD[symbol]
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid

    # ── Directional: reach hold ───────────────────────────────────────────────
    if pos < hold:
        orders.append(Order(symbol, best_ask, hold - pos))

    # ── MM with remaining capacity, skewed upward ─────────────────────────────
    if spread <= 0:
        return orders

    our_bid     = best_bid + 1 if spread >= 2 else best_bid
    mm_buy_cap  = POS_LIMIT - max(pos, hold)
    mm_sell_cap = max(0, pos - hold)

    if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
    if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))

    return orders



class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── Pebbles state ─────────────────────────────────────────────
        mem       = json.loads(state.traderData) if state.traderData else {}
        history   = mem.get("h", [])
        hold      = mem.get("hold", 0)
        direction = mem.get("dir", 0)

        od_xl = state.order_depths.get("PEBBLES_XL")
        od_xs = state.order_depths.get("PEBBLES_XS")
        od_l  = state.order_depths.get("PEBBLES_L")
        od_s  = state.order_depths.get("PEBBLES_S")
        od_m  = state.order_depths.get("PEBBLES_M")

        # ── Run Pebbles strategy (override base logic) ────────────────
        if od_xl and od_xs:
            mid_xl = mid_price(od_xl)
            mid_xs = mid_price(od_xs)

            if mid_xl is not None and mid_xs is not None:
                history.append(int(mid_xl - mid_xs))
                if len(history) > WINDOW:
                    history = history[-WINDOW:]

                pos_xl = state.position.get("PEBBLES_XL", 0)
                pos_xs = state.position.get("PEBBLES_XS", 0)
                pos_l  = state.position.get("PEBBLES_L",  0)
                pos_s  = state.position.get("PEBBLES_S",  0)
                pos_m  = state.position.get("PEBBLES_M",  0)

                orders_1, hold, direction = strategy_xl_xs(
                    history, hold, direction, od_xl, od_xs, pos_xl, pos_xs)

                orders_2 = strategy_lsm_hedge(
                    direction, od_l, od_s, od_m, pos_l, pos_s, pos_m)

                result.update(orders_1)
                result.update(orders_2)

        # ── Non-pebbles products use base strategy ────────────────────
        for symbol, od in state.order_depths.items():
            if symbol.startswith("PEBBLES"):
                continue

            pos = state.position.get(symbol, 0)

            if symbol in TREND_HOLD:
                orders = trade_trend(symbol, od, pos)
            else:
                orders = market_make(symbol, od, pos)

            if orders:
                result[symbol] = orders

        # ── Save Pebbles state ────────────────────────────────────────
        new_mem     = {"h": history, "hold": hold, "dir": direction}
        trader_data = json.dumps(new_mem, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
