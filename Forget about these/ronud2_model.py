from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
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

class Trader:
    """
    THE BUG THAT CAUSED 700 PnL:
    Timestamps restart at 0 each day (0..999900 per day).
    Our old formula used state.timestamp directly as a global counter,
    giving fair_value = 11000 + 0.001*0 = 11000 on day 2 when true price is ~14000.
    Bot thought the market was massively overpriced and refused to buy.

    THE FIX:
    - Track day transitions in traderData (when timestamp resets = new day)
    - Day 2 (Round 2) base price = 14000
    - fair_value_IPR = day_base + 0.001 * timestamp
      e.g. day 2 ts=0 → 14000, ts=999900 → 14999.9

    IPR trend: +1000 XIRECs per day, slope 0.001/timestamp unit, residual noise std=2.4
    ACO: mean-reverts tightly to 10000 (std=5, max_dev=21, spread~16)
    """

    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 50,
        "ASH_COATED_OSMIUM": 50,
    }

    # IPR per-day base prices (day index → starting fair value)
    # day -1 → 11000, day 0 → 12000, day 1 → 13000, day 2 → 14000, day 3 → 15000
    IPR_DAY_BASE = {-1: 11000, 0: 12000, 1: 13000, 2: 14000, 3: 15000, 4: 16000}
    IPR_SLOPE = 0.001  # per timestamp unit

    def run(self, state: TradingState):
        # ── Load persistent state ──────────────────────────────────────────────
        trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}

        # ── Day detection: track when timestamp resets (new day) ───────────────
        last_ts = trader_data.get("last_ts", -1)
        day_index = trader_data.get("day_index", 0)

        # When timestamp goes backwards (or jumps to near 0 from near 1M) = new day
        if last_ts > 0 and state.timestamp < last_ts - 500000:
            day_index += 1

        trader_data["last_ts"] = state.timestamp
        trader_data["day_index"] = day_index

        # ── IPR fair value for this tick ───────────────────────────────────────
        # day_index=0 corresponds to the first day of THIS round (day 2 globally)
        # Base starts at 14000 and increases by 1000 each day
        ipr_base = 14000 + day_index * 1000
        ipr_fair = ipr_base + self.IPR_SLOPE * state.timestamp

        # ── Trade each product ─────────────────────────────────────────────────
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            od: OrderDepth = state.order_depths[product]
            pos = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._trade_ipr(od, pos, limit, ipr_fair)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self._trade_aco(od, pos, limit)
            else:
                result[product] = []

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # INTARIAN PEPPER ROOT — ride the +1000/day linear trend
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_ipr(self, od: OrderDepth, pos: int, limit: int, fair: float) -> List[Order]:
        orders: List[Order] = []
        buy_capacity = limit - pos
        sell_capacity = limit + pos

        # 1. Aggressively hit all ask levels if priced below fair + 5
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price > fair + 5:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, qty))
                    buy_capacity -= qty
                if buy_capacity <= 0:
                    break

        # 2. Post resting bids just below fair to catch any dip
        if buy_capacity > 0:
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else int(fair + 8)
            bid_price = min(int(fair - 1), best_ask - 1)
            # Two-level bid ladder
            tier1 = max(buy_capacity // 2, 1)
            orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price, min(tier1, buy_capacity)))
            rem = buy_capacity - tier1
            if rem > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bid_price - 3, rem))

        # 3. Safety valve only — don't sell unless price massively above fair
        if pos > 0 and sell_capacity > 0 and od.buy_orders:
            best_bid = max(od.buy_orders.keys())
            if best_bid > fair + 15:
                qty = min(od.buy_orders[best_bid], sell_capacity, 5)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # ASH-COATED OSMIUM — market making around fair value 10000
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_aco(self, od: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        FAIR = 10000

        buy_capacity = limit - pos
        sell_capacity = limit + pos

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else FAIR - 8
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else FAIR + 8
        mid = (best_bid + best_ask) / 2
        deviation = mid - FAIR

        # 1. Take clear misprices immediately
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= FAIR - 3:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_price, qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= FAIR + 3:
                    break
                qty = min(od.buy_orders[bid_price], sell_capacity)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_price, -qty))
                    sell_capacity -= qty

        # 2. Market making — post inside spread, skewed by position and deviation
        pos_skew = pos / limit if limit != 0 else 0
        skew = int(round(deviation * 0.4 + pos_skew * 3))

        our_bid = int(best_bid + 1 - skew)
        our_ask = int(best_ask - 1 - skew)

        # Prevent crossing
        if our_bid >= our_ask:
            our_bid = FAIR - 2
            our_ask = FAIR + 2

        # Clip to sane range around fair
        our_bid = max(our_bid, FAIR - 8)
        our_ask = min(our_ask, FAIR + 8)

        base_size = max(3, min(8, int(2 + abs(deviation))))

        if buy_capacity > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_bid, min(base_size, buy_capacity)))
        if sell_capacity > 0:
            orders.append(Order("ASH_COATED_OSMIUM", our_ask, -min(base_size, sell_capacity)))

        # 3. Aggressive fade when price is very far from fair
        if deviation < -10 and buy_capacity > 0:
            extra = min(buy_capacity, 10)
            orders.append(Order("ASH_COATED_OSMIUM", best_ask, extra))
        elif deviation > 10 and sell_capacity > 0:
            extra = min(sell_capacity, 10)
            orders.append(Order("ASH_COATED_OSMIUM", best_bid, -extra))

        return orders
