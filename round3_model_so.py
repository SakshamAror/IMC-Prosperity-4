from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
from math import log, sqrt, exp, erf
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
    [R3 v2] Reverted VEV IV signal back to original (buy cheap IV, sell expensive IV).
            Reversed signal lost 177k; original signal expected to gain ~177k.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # ITEM TOGGLES
    # ──────────────────────────────────────────────────────────────────────────
    HYDROGEL = True
    VELVETFRUIT = True
    OPTIONS = False

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    HYDROGEL_FAIR_VALUE = 9990.8
    HYDROGEL_MISPRICE_THRESHOLD = 32
    HYDROGEL_CLIP_RANGE = 100
    HYDROGEL_SPREAD_OFFSET = 1
    HYDROGEL_DEVIATION_SCALE = 0.4
    HYDROGEL_POSITION_SCALE = 12
    HYDROGEL_TARGET_SCALE = 3.5   # target_pos = -deviation * 5; deviation=10 → target=±50
    HYDROGEL_AGGRESSIVE_THRESHOLD = 40
    HYDROGEL_AGGRESSIVE_SIZE = 40
    HYDROGEL_WARMUP_TICKS = 10_000        # ramp target_scale 0→full over this window

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    VEV_FAIR_VALUE = 5250
    VEV_MISPRICE_THRESHOLD = 16
    VEV_CLIP_RANGE = 100
    VEV_SPREAD_OFFSET = 1
    VEV_DEVIATION_SCALE = 0.4
    VEV_POSITION_SCALE = 12
    VEV_TARGET_SCALE = 4
    VEV_AGGRESSIVE_THRESHOLD = 30
    VEV_AGGRESSIVE_SIZE = 40

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options - IV Z-Score Mean Reversion
    # ──────────────────────────────────────────────────────────────────────────
    VEV_OPT_T              = 5.0               # fixed; simulation has no time decay, T=5.0 gives most stable IV (std=0.00080)
    VEV_OPT_EMA_ALPHA      = 0.005             # mean(1 - lag-1 ACF) across 5300/5400/5500 with T=5.0: 0.0066/0.0031/0.0057
    VEV_OPT_Z_ENTRY        = 2.0               # ~5% of ticks exceed ±2σ; validate with 11c z-score histogram
    VEV_OPT_Z_EXIT         = 0.5
    VEV_OPT_SIZE           = 3                 # sized to ~50% of typical L1 depth
    VEV_OPT_SEED_IV        = {5300: 0.0146, 5400: 0.0137, 5500: 0.0148}   # T=5.0 mean IV per strike
    VEV_OPT_SEED_VAR       = {5300: 6.4e-7,  5400: 9.2e-7, 5500: 8.0e-7}  # T=5.0 var(IV) per strike; old 0.001 was ~1250x too large
    VEV_OPT_LIQUID_STRIKES = frozenset({5300, 5400, 5500})  # 5000/5100: spread>4 + 1 trade; 5200: only 15 trades
    VEV_OPTION_LEVERAGE = 0.5

    # TODO: verify all limits from the round 3 wiki before submitting
    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":        200,
        "VELVETFRUIT_EXTRACT":  200,
        "VEV_4000":             300,
        "VEV_4500":             300,
        "VEV_5000":             300,
        "VEV_5100":             300,
        "VEV_5200":             300,
        "VEV_5300":             300,
        "VEV_5400":             300,
        "VEV_5500":             300,
        "VEV_6000":             300,
        "VEV_6500":             300,
    }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        trader_data: dict = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}

        result: Dict[str, List[Order]] = {}

        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if vev_od and vev_od.buy_orders and vev_od.sell_orders:
            spot = (max(vev_od.buy_orders) + min(vev_od.sell_orders)) / 2
        else:
            spot = self.VEV_FAIR_VALUE

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "HYDROGEL_PACK":
                result[product] = self._trade_hydrogel(od, pos, limit, state.timestamp)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit)
            elif product.startswith("VEV_"):
                result[product] = self._trade_vev_option(product, od, pos, limit, spot, trader_data)
            else:
                result[product] = []

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK — market making around fair value 9987.5
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0) -> List[Order]:
        if (not self.HYDROGEL): return []

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else self.HYDROGEL_FAIR_VALUE - self.HYDROGEL_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else self.HYDROGEL_FAIR_VALUE + self.HYDROGEL_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - self.HYDROGEL_FAIR_VALUE

        # Ramp target_scale from 0 → full over warmup window to avoid binary flipping at start
        ramp = min(1.0, timestamp / self.HYDROGEL_WARMUP_TICKS) if self.HYDROGEL_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.HYDROGEL_TARGET_SCALE * ramp

        # Target position scales with edge: short when above fair, long when below
        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        # 1. Take clear misprices immediately
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= self.HYDROGEL_FAIR_VALUE - self.HYDROGEL_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= self.HYDROGEL_FAIR_VALUE + self.HYDROGEL_MISPRICE_THRESHOLD:
                    break
                qty = min(od.buy_orders[bid_price], sell_capacity)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", int(bid_price), -qty))
                    sell_capacity -= qty

        # 2. Passive quotes toward target
        pos_skew = pos / limit if limit != 0 else 0
        skew = int(round(deviation * self.HYDROGEL_DEVIATION_SCALE + pos_skew * self.HYDROGEL_POSITION_SCALE))

        our_bid = int(best_bid + self.HYDROGEL_SPREAD_OFFSET - skew)
        our_ask = int(best_ask - self.HYDROGEL_SPREAD_OFFSET - skew)

        if our_bid >= our_ask:
            our_bid = int(self.HYDROGEL_FAIR_VALUE - 2)
            our_ask = int(self.HYDROGEL_FAIR_VALUE + 2)

        our_bid = max(our_bid, int(self.HYDROGEL_FAIR_VALUE - self.HYDROGEL_CLIP_RANGE))
        our_ask = min(our_ask, int(self.HYDROGEL_FAIR_VALUE + self.HYDROGEL_CLIP_RANGE))

        if buy_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", our_ask, -sell_capacity))

        # 3. Aggressive hit when deviation is very large
        if deviation < -self.HYDROGEL_AGGRESSIVE_THRESHOLD and buy_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", int(best_ask), min(buy_capacity, self.HYDROGEL_AGGRESSIVE_SIZE)))
        elif deviation > self.HYDROGEL_AGGRESSIVE_THRESHOLD and sell_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", int(best_bid), -min(sell_capacity, self.HYDROGEL_AGGRESSIVE_SIZE)))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT — mean reversion around fair value 5250
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int) -> List[Order]:
        if (not self.VELVETFRUIT): return []

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else self.VEV_FAIR_VALUE - self.VEV_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else self.VEV_FAIR_VALUE + self.VEV_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - self.VEV_FAIR_VALUE

        target = int(max(-limit, min(limit, -deviation * self.VEV_TARGET_SCALE)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        # 1. Take clear misprices immediately
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= self.VEV_FAIR_VALUE - self.VEV_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= self.VEV_FAIR_VALUE + self.VEV_MISPRICE_THRESHOLD:
                    break
                qty = min(od.buy_orders[bid_price], sell_capacity)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", int(bid_price), -qty))
                    sell_capacity -= qty

        # 2. Passive quotes toward target
        pos_skew = pos / limit if limit != 0 else 0
        skew = int(round(deviation * self.VEV_DEVIATION_SCALE + pos_skew * self.VEV_POSITION_SCALE))

        our_bid = int(best_bid + self.VEV_SPREAD_OFFSET - skew)
        our_ask = int(best_ask - self.VEV_SPREAD_OFFSET - skew)

        if our_bid >= our_ask:
            our_bid = int(self.VEV_FAIR_VALUE - 2)
            our_ask = int(self.VEV_FAIR_VALUE + 2)

        our_bid = max(our_bid, int(self.VEV_FAIR_VALUE - self.VEV_CLIP_RANGE))
        our_ask = min(our_ask, int(self.VEV_FAIR_VALUE + self.VEV_CLIP_RANGE))

        if buy_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", our_ask, -sell_capacity))

        # 3. Aggressive hit when deviation is very large
        if deviation < -self.VEV_AGGRESSIVE_THRESHOLD and buy_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), min(buy_capacity, self.VEV_AGGRESSIVE_SIZE)))
        elif deviation > self.VEV_AGGRESSIVE_THRESHOLD and sell_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -min(sell_capacity, self.VEV_AGGRESSIVE_SIZE)))

        return orders


    def _trade_vev_option(self, symbol: str, od: OrderDepth, pos: int,
                          limit: int, spot: float, trader_data: dict) -> List[Order]:
        if (not self.OPTIONS): return []

        """
        Proportional leverage strategy:
        - Scale option position to a fraction of velvetfruit target position
        - Keeps exposure consistent instead of event-based buying
        """
        orders: List[Order] = []
        
        # Compute velvetfruit signal
        vev_deviation = spot - self.VEV_FAIR_VALUE
        vev_target = int(max(-limit, min(limit, -vev_deviation * self.VEV_TARGET_SCALE)))

        # Desired option position (proportional leverage)
        desired_option_pos = int(vev_target * self.VEV_OPTION_LEVERAGE)

        # How much we need to trade to reach target
        delta = desired_option_pos - pos

        buy_capacity = max(0, limit - pos)
        sell_capacity = max(0, pos + limit)

        # Bullish → use ITM call (VEV_5200)
        if symbol == "VEV_5200" and delta > 0:
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                qty = min(delta, -od.sell_orders[best_ask], buy_capacity)
                if qty > 0:
                    orders.append(Order(symbol, int(best_ask), qty))

        # Bearish → use ITM put (VEV_5300)
        elif symbol == "VEV_5300" and delta > 0:
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                qty = min(delta, -od.sell_orders[best_ask], buy_capacity)
                if qty > 0:
                    orders.append(Order(symbol, int(best_ask), qty))

        return orders

