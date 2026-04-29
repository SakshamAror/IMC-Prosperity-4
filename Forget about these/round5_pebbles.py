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
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

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
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

# ── Hyperparameters ───────────────────────────────────────────────────────────
WINDOW    = 500   # rolling window for z-score
ENTRY_Z   = 2.5   # z-score threshold to enter MR trade
EXIT_Z    = 0.3   # z-score threshold to exit MR trade
MAX_HOLD  = 200   # force-exit MR trade after this many ticks
BG_POS = 10        # Pos in the background to buy and hold XL-XS

POS_LIMIT = 10    # position limit per product
# ─────────────────────────────────────────────────────────────────────────────


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
    """Z-score MR on XL-XS spread + background long XL / short XS."""
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
    """Hedge XL-XS MR via L+S+M (basket complement: L+S+M = 50000 - XS - XL).
    direction=-1 (short XL) → long L+S+M; direction=+1 (long XL) → short L+S+M."""
    orders = {}
    tgt = -direction * POS_LIMIT  # opposite to XL direction; 0 when flat
    if od_l: place("PEBBLES_L", od_l, pos_l, tgt, orders)
    if od_s: place("PEBBLES_S", od_s, pos_s, tgt, orders)
    if od_m: place("PEBBLES_M", od_m, pos_m, tgt, orders)
    return orders


class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        mem       = json.loads(state.traderData) if state.traderData else {}
        history   = mem.get("h", [])
        hold      = mem.get("hold", 0)
        direction = mem.get("dir", 0)

        od_xl = state.order_depths.get("PEBBLES_XL")
        od_xs = state.order_depths.get("PEBBLES_XS")
        od_l  = state.order_depths.get("PEBBLES_L")
        od_s  = state.order_depths.get("PEBBLES_S")
        od_m  = state.order_depths.get("PEBBLES_M")

        if not od_xl or not od_xs:
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData

        mid_xl = mid_price(od_xl)
        mid_xs = mid_price(od_xs)
        if mid_xl is None or mid_xs is None:
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData

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

        new_mem     = {"h": history, "hold": hold, "dir": direction}
        trader_data = json.dumps(new_mem, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data