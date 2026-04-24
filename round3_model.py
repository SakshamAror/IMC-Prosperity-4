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
    ROUND 3 STRATEGY OVERVIEW
    --------------------------
    HYDROGEL_PACK:
        Same mean-reversion strategy as ACO from Round 2. Mean-reverts to 10000
        (std ~32, spread ~16). Hyperparameters unchanged from ACO.

    VELVETFRUIT_EXTRACT / VEV_XXXX options:
        TODO — implement IV mean-reversion strategy.
        Stubs return empty orders for now.

    CHANGELOG
    ---------
    [R3 v1] Replaced IPR (linear trend) and ACO with HYDROGEL_PACK using the
            same mean-reversion logic. Added empty stubs for VEV options.
    """

    # TODO: verify all limits from the round 3 wiki before submitting
    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":        50,
        "VELVETFRUIT_EXTRACT":  50,
        "VEV_4000":             50,
        "VEV_4500":             50,
        "VEV_5000":             50,
        "VEV_5100":             50,
        "VEV_5200":             50,
        "VEV_5300":             50,
        "VEV_5400":             50,
        "VEV_5500":             50,
        "VEV_6000":             50,
        "VEV_6500":             50,
    }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        trader_data: dict = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "HYDROGEL_PACK":
                result[product] = self._trade_hydrogel(od, pos, limit)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit, trader_data)
            elif product.startswith("VEV_"):
                result[product] = self._trade_vev_option(product, od, pos, limit, trader_data)
            else:
                result[product] = []

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK — market making around fair value 10000
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, od: OrderDepth, pos: int, limit: int) -> List[Order]:
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
                    orders.append(Order("HYDROGEL_PACK", ask_price, qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= FAIR + 3:
                    break
                qty = min(od.buy_orders[bid_price], sell_capacity)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", bid_price, -qty))
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
            orders.append(Order("HYDROGEL_PACK", our_bid, min(base_size, buy_capacity)))
        if sell_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", our_ask, -min(base_size, sell_capacity)))

        # 3. Aggressive fade when price is very far from fair
        if deviation < -10 and buy_capacity > 0:
            extra = min(buy_capacity, 10)
            orders.append(Order("HYDROGEL_PACK", best_ask, extra))
        elif deviation > 10 and sell_capacity > 0:
            extra = min(sell_capacity, 10)
            orders.append(Order("HYDROGEL_PACK", best_bid, -extra))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT — stub (underlying for VEV options)
    # TODO: implement once options strategy is finalised in the notebook.
    # Do NOT trade the underlying directionally alongside options — opaque PnL
    # attribution (jmerle P2 lesson). Only use for delta hedging if needed.
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int,
                           trader_data: dict) -> List[Order]:
        # TODO: implement delta hedge or market making on the underlying
        return []

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX OPTIONS — stub (IV mean-reversion strategy)
    # TODO: implement once IV z-score analysis is complete in the notebook.
    # Primary signal: buy option when per-strike IV z-score < -2,
    #                 sell option when IV z-score > +2, exit at ±0.5.
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_vev_option(self, symbol: str, od: OrderDepth, pos: int,
                          limit: int, trader_data: dict) -> List[Order]:
        # TODO: implement IV z-score strategy
        return []

