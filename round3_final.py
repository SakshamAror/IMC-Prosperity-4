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

    # ──────────────────────────────────────────────────────────────────────────
    # ITEM TOGGLES
    # ──────────────────────────────────────────────────────────────────────────
    HYDROGEL = True
    VELVETFRUIT = True
    OPTIONS = True

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    HYDROGEL_FAIR_VALUE = 9995.13
    HYDROGEL_MISPRICE_THRESHOLD = 4       # halved: fire on smaller misprices
    HYDROGEL_CLIP_RANGE = 100
    HYDROGEL_SPREAD_OFFSET = 1
    HYDROGEL_DEVIATION_SCALE = 0.69
    HYDROGEL_POSITION_SCALE = 4           # reduced: less skew → quotes stay tight → more fills
    HYDROGEL_TARGET_SCALE = 4             # doubled: larger targets → more position cycling
    HYDROGEL_AGGRESSIVE_THRESHOLD = 40
    HYDROGEL_AGGRESSIVE_SIZE = 40
    HYDROGEL_WARMUP_TICKS = 4490        # faster ramp: start trading at full scale sooner

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    VEV_FAIR_VALUE = 5248.39
    VEV_MISPRICE_THRESHOLD = 5            # halved
    VEV_CLIP_RANGE = 60
    VEV_SPREAD_OFFSET = 1
    VEV_DEVIATION_SCALE = 0.61
    VEV_POSITION_SCALE = 1                # reduced: less inventory skew → more fills
    VEV_TARGET_SCALE = 11                  # doubled
    VEV_AGGRESSIVE_THRESHOLD = 30
    VEV_AGGRESSIVE_SIZE = 40
    VEV_WARMUP_TICKS = 772             # warmup matching HYDROGEL

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options - Static FV market-making
    # Historical FVs calibrated at TTE≈7d (sigma=0.20, S=5250).
    # Live FVs recalibrated at TTE=5d (sigma=0.20, S=5250) via Black-Scholes.
    # Deep ITM (4000, 4500) unchanged — dominated by intrinsic value.
    # SET LIVE_SUBMISSION=True BEFORE submitting to live round.
    # Per-strike: (fair_value, misprice_threshold, clip_range, agg_dev_threshold)
    # ──────────────────────────────────────────────────────────────────────────
    LIVE_SUBMISSION = False   # ← flip to True for live round (TTE starts at 5d)

    _OPT_HIST = {             # backtest: TTE≈7d
        4000: (1250.1,  5, 60, 25),
        4500: ( 750.1,  4, 60, 20),
        5000: ( 255.0,  1, 40, 15),
        5100: ( 166.8,  1, 30, 10),
        5200: (  95.6,  1, 20,  8),
        5300: (  46.8,  1, 12,  6),
        5400: (  15.95, 1, 15,  5),
        5500: (   6.64, 1,  6,  3),
    }
    _OPT_LIVE = {             # live: ~50% theta correction toward TTE=5d (conservative)
        4000: (1250.1,  5, 60, 25),   # deep ITM — unchanged
        4500: ( 750.1,  4, 60, 20),   # deep ITM — unchanged
        5000: ( 254.0,  1, 40, 15),   # -1.0  (~0.5 day theta)
        5100: ( 164.5,  1, 30, 10),   # -2.3  (~0.5 day theta)
        5200: (  91.0,  1, 20,  8),   # -4.6  (~0.5 day theta)
        5300: (  42.0,  1, 12,  6),   # -4.8  (~0.5 day theta)
        5400: (  14.2,  1, 15,  5),   # -1.75 (~0.5 day theta)
        5500: (   5.0,  1,  6,  3),   # -1.64 (~0.5 day theta)
    }
    OPT_MR_STRIKES = _OPT_LIVE if LIVE_SUBMISSION else _OPT_HIST
    OPT_TARGET_SCALE    = 11              # increased: larger targets → more capacity
    OPT_SPREAD_OFFSET   = 1
    OPT_DEVIATION_SCALE = 0.24
    OPT_POSITION_SCALE  = 2              # reduced: less skew → more passive fills
    OPT_AGGRESSIVE_SIZE = 5              # enabled: fade large moves aggressively
    OPT_WARMUP_TICKS    = 2510          # faster ramp matching HYDROGEL_WARMUP_TICKS

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

        # ── Day tracking (timestamp resets to 0 each new day) ─────────────────
        last_ts   = trader_data.get("last_ts", -1)
        day_index = trader_data.get("day_index", 0)
        if last_ts > 0 and state.timestamp < last_ts - 500_000:
            day_index += 1
        trader_data["last_ts"]   = state.timestamp
        trader_data["day_index"] = day_index

        # ── Spot price ─────────────────────────────────────────────────────────
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if vev_od and vev_od.buy_orders and vev_od.sell_orders:
            spot = (max(vev_od.buy_orders) + min(vev_od.sell_orders)) / 2
        else:
            spot = self.VEV_FAIR_VALUE

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "HYDROGEL_PACK":
                result[product] = self._trade_hydrogel(od, pos, limit, state.timestamp)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit, state.timestamp)
            elif product.startswith("VEV_"):
                result[product] = self._trade_vev_option(product, od, pos, limit, spot, state.timestamp)
            else:
                result[product] = []

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0) -> List[Order]:
        if not self.HYDROGEL: return []

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else self.HYDROGEL_FAIR_VALUE - self.HYDROGEL_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else self.HYDROGEL_FAIR_VALUE + self.HYDROGEL_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - self.HYDROGEL_FAIR_VALUE

        ramp = min(1.0, timestamp / self.HYDROGEL_WARMUP_TICKS) if self.HYDROGEL_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.HYDROGEL_TARGET_SCALE * ramp

        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

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

        if deviation < -self.HYDROGEL_AGGRESSIVE_THRESHOLD and buy_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", int(best_ask), min(buy_capacity, self.HYDROGEL_AGGRESSIVE_SIZE)))
        elif deviation > self.HYDROGEL_AGGRESSIVE_THRESHOLD and sell_capacity > 0:
            orders.append(Order("HYDROGEL_PACK", int(best_bid), -min(sell_capacity, self.HYDROGEL_AGGRESSIVE_SIZE)))

        # ── Penny-jump + join remaining capacity ───────────────────────────────
        if od.buy_orders and od.sell_orders:
            buys_placed  = sum(o.quantity for o in orders if o.quantity > 0)
            sells_placed = sum(-o.quantity for o in orders if o.quantity < 0)
            mm_buy  = max(0, limit - pos - buys_placed)
            mm_sell = max(0, limit + pos - sells_placed)
            pj_bid  = int(best_bid + 1)
            pj_ask  = int(best_ask - 1)
            if pj_bid < pj_ask:
                pj_buy,  join_buy  = mm_buy  // 2, mm_buy  - mm_buy  // 2
                pj_sell, join_sell = mm_sell // 2, mm_sell - mm_sell // 2
                if pj_buy   > 0: orders.append(Order("HYDROGEL_PACK", pj_bid,          pj_buy))
                if pj_sell  > 0: orders.append(Order("HYDROGEL_PACK", pj_ask,         -pj_sell))
                if join_buy  > 0: orders.append(Order("HYDROGEL_PACK", int(best_bid),  join_buy))
                if join_sell > 0: orders.append(Order("HYDROGEL_PACK", int(best_ask), -join_sell))
            else:
                if mm_buy  > 0: orders.append(Order("HYDROGEL_PACK", int(best_bid),  mm_buy))
                if mm_sell > 0: orders.append(Order("HYDROGEL_PACK", int(best_ask), -mm_sell))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0) -> List[Order]:
        if not self.VELVETFRUIT: return []

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else self.VEV_FAIR_VALUE - self.VEV_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else self.VEV_FAIR_VALUE + self.VEV_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - self.VEV_FAIR_VALUE

        ramp = min(1.0, timestamp / self.VEV_WARMUP_TICKS) if self.VEV_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.VEV_TARGET_SCALE * ramp
        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

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

        if deviation < -self.VEV_AGGRESSIVE_THRESHOLD and buy_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), min(buy_capacity, self.VEV_AGGRESSIVE_SIZE)))
        elif deviation > self.VEV_AGGRESSIVE_THRESHOLD and sell_capacity > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -min(sell_capacity, self.VEV_AGGRESSIVE_SIZE)))

        # ── Penny-jump + join remaining capacity ───────────────────────────────
        if od.buy_orders and od.sell_orders:
            buys_placed  = sum(o.quantity for o in orders if o.quantity > 0)
            sells_placed = sum(-o.quantity for o in orders if o.quantity < 0)
            mm_buy  = max(0, limit - pos - buys_placed)
            mm_sell = max(0, limit + pos - sells_placed)
            pj_bid  = int(best_bid + 1)
            pj_ask  = int(best_ask - 1)
            if pj_bid < pj_ask:
                pj_buy,  join_buy  = mm_buy  // 2, mm_buy  - mm_buy  // 2
                pj_sell, join_sell = mm_sell // 2, mm_sell - mm_sell // 2
                if pj_buy   > 0: orders.append(Order("VELVETFRUIT_EXTRACT", pj_bid,          pj_buy))
                if pj_sell  > 0: orders.append(Order("VELVETFRUIT_EXTRACT", pj_ask,         -pj_sell))
                if join_buy  > 0: orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid),  join_buy))
                if join_sell > 0: orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), -join_sell))
            else:
                if mm_buy  > 0: orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid),  mm_buy))
                if mm_sell > 0: orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), -mm_sell))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_vev_option(self, symbol: str, od: OrderDepth, pos: int,
                          limit: int, spot: float, timestamp: int = 0) -> List[Order]:
        if not self.OPTIONS:
            return []

        K = int(symbol.split("_")[1])
        if K not in self.OPT_MR_STRIKES:
            return []

        fv, threshold, clip, agg_thresh = self.OPT_MR_STRIKES[K]
        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else fv - clip
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fv + clip
        mid = (best_bid + best_ask) / 2
        deviation = mid - fv

        ramp = min(1.0, timestamp / self.OPT_WARMUP_TICKS) if self.OPT_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.OPT_TARGET_SCALE * ramp
        target        = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= fv - threshold:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order(symbol, int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= fv + threshold:
                    break
                qty = min(od.buy_orders[bid_price], sell_capacity)
                if qty > 0:
                    orders.append(Order(symbol, int(bid_price), -qty))
                    sell_capacity -= qty

        pos_skew = pos / limit if limit != 0 else 0
        skew = int(round(deviation * self.OPT_DEVIATION_SCALE + pos_skew * self.OPT_POSITION_SCALE))

        our_bid = int(best_bid + self.OPT_SPREAD_OFFSET - skew)
        our_ask = int(best_ask - self.OPT_SPREAD_OFFSET - skew)

        if our_bid >= our_ask:
            our_bid = int(fv - 2)
            our_ask = int(fv + 2)

        our_bid = max(our_bid, max(1, int(fv - clip)))
        our_ask = min(our_ask, int(fv + clip))

        if buy_capacity > 0:
            orders.append(Order(symbol, our_bid, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(symbol, our_ask, -sell_capacity))

        if deviation < -agg_thresh and buy_capacity > 0:
            orders.append(Order(symbol, int(best_ask), min(buy_capacity, self.OPT_AGGRESSIVE_SIZE)))
        elif deviation > agg_thresh and sell_capacity > 0:
            orders.append(Order(symbol, int(best_bid), -min(sell_capacity, self.OPT_AGGRESSIVE_SIZE)))

        # ── Penny-jump + join remaining capacity ───────────────────────────────
        if od.buy_orders and od.sell_orders:
            buys_placed  = sum(o.quantity for o in orders if o.quantity > 0)
            sells_placed = sum(-o.quantity for o in orders if o.quantity < 0)
            mm_buy  = max(0, limit - pos - buys_placed)
            mm_sell = max(0, limit + pos - sells_placed)
            pj_bid  = max(1, int(best_bid + 1))
            pj_ask  = int(best_ask - 1)
            if pj_bid < pj_ask:
                pj_buy,  join_buy  = mm_buy  // 2, mm_buy  - mm_buy  // 2
                pj_sell, join_sell = mm_sell // 2, mm_sell - mm_sell // 2
                if pj_buy   > 0: orders.append(Order(symbol, pj_bid,          pj_buy))
                if pj_sell  > 0: orders.append(Order(symbol, pj_ask,         -pj_sell))
                if join_buy  > 0: orders.append(Order(symbol, int(best_bid),  join_buy))
                if join_sell > 0: orders.append(Order(symbol, int(best_ask), -join_sell))
            else:
                if mm_buy  > 0: orders.append(Order(symbol, max(1, int(best_bid)),  mm_buy))
                if mm_sell > 0: orders.append(Order(symbol, int(best_ask),         -mm_sell))

        return orders

