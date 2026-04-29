from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
from math import sqrt, exp
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
                compressed.append([trade.symbol, trade.price, trade.quantity,
                                    trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice,
                observation.transportFees, observation.exportTariff,
                observation.importTariff, observation.sugarPrice,
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
WINDOW      = 200    # rolling window for z-score
ENTRY_Z     = 1.5    # enter MR trade when |z| > ENTRY_Z
EXIT_Z      = 0.3    # exit MR trade when |z| < EXIT_Z
MAX_HOLD    = 200    # force-exit after this many ticks
SIGMOID_K   = 1.5    # steepness of sigmoid sizing (higher = more aggressive)
POS_LIMIT   = 10     # hard position limit per product
MM_SPREAD   = 0      # market-make this many ticks inside best bid/ask

# Butter leg signs: XS=+1, S=-1, M=+1, L=-1, XL=+1
PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
SIGNS    = [+2, -1, +1, -1, +2]   # butter = XS - S + M - L + XL
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


def sigmoid_size(excess: float) -> int:
    """Maps excess z above ENTRY_Z to position size via sigmoid.
    excess=0 -> 0, excess=large -> POS_LIMIT."""
    raw = POS_LIMIT * 2 * (1 / (1 + exp(-SIGMOID_K * excess)) - 0.5)
    return min(POS_LIMIT, max(0, round(raw)))


def strategy_butter(history: list, hold: int, direction: int,
                    ods: dict) -> tuple[dict, int, int]:
    """Z-score MR on butter = XS - S + M - L + XL.
    Sigmoid sizing based on deviation above ENTRY_Z.
    Returns: {symbol: target_position}, hold, direction."""
    targets = {p: 0 for p in PRODUCTS}

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

        if direction != 0:
            excess = max(0.0, abs(z) - ENTRY_Z)
            size   = sigmoid_size(excess)
        else:
            size = 0

        logger.print(f"z={z:.2f} dir={direction} size={size} hold={hold}")

        for prod, sign in zip(PRODUCTS, SIGNS):
            targets[prod] = size * direction * sign

    return targets, hold, direction


def market_make(symbol: str, od: OrderDepth, cur_pos: int,
                used: int, orders: list):
    """Post passive bid/ask using remaining capacity after directional orders.
    used = signed quantity already committed this tick."""
    if not od.buy_orders or not od.sell_orders:
        return

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    if best_ask - best_bid < 2:
        return   # spread too tight to safely post inside

    mm_bid = best_bid + MM_SPREAD
    mm_ask = best_ask - MM_SPREAD
    if mm_bid >= mm_ask:
        return

    # Remaining buy/sell capacity
    net = cur_pos + used
    buy_cap  = POS_LIMIT - net
    sell_cap = POS_LIMIT + net

    if buy_cap > 0:
        orders.append(Order(symbol, mm_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(symbol, mm_ask, -sell_cap))


class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        mem       = json.loads(state.traderData) if state.traderData else {}
        history   = mem.get("h",    [])
        hold      = mem.get("hold", 0)
        direction = mem.get("dir",  0)

        ods = {p: state.order_depths.get(p) for p in PRODUCTS}

        # Compute butter value this tick
        mids = {p: mid_price(ods[p]) for p in PRODUCTS if ods[p]}
        if all(p in mids for p in PRODUCTS):
            butter_val = sum(mids[p] * s for p, s in zip(PRODUCTS, SIGNS))
            history.append(int(butter_val))
            if len(history) > WINDOW:
                history = history[-WINDOW:]

        # Directional butter targets
        targets, hold, direction = strategy_butter(history, hold, direction, ods)

        # Build orders: directional leg first, then market-make with remainder
        for prod, sign in zip(PRODUCTS, SIGNS):
            od  = ods.get(prod)
            if od is None:
                continue

            cur     = state.position.get(prod, 0)
            tgt     = targets[prod]
            delta   = tgt - cur
            prod_orders: list[Order] = []

            # Directional order
            if delta > 0 and od.sell_orders:
                prod_orders.append(Order(prod, min(od.sell_orders), delta))
            elif delta < 0 and od.buy_orders:
                prod_orders.append(Order(prod, max(od.buy_orders), delta))

            # Market-make with remaining capacity
            market_make(prod, od, cur, delta, prod_orders)

            if prod_orders:
                result[prod] = prod_orders

        new_mem     = {"h": history, "hold": hold, "dir": direction}
        trader_data = json.dumps(new_mem, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
