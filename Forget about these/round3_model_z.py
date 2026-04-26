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
    HYDROGEL = False
    VELVETFRUIT = True
    OPTIONS = False

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    HYDROGEL_FAIR_VALUE = 9990.8   # seed only — overridden by EMA each tick
    HYDROGEL_EMA_ALPHA  = 0.002    # slow-adapting (~346-tick half-life)
    HYDROGEL_MISPRICE_THRESHOLD = 6       # halved: fire on smaller misprices
    HYDROGEL_CLIP_RANGE = 100
    HYDROGEL_SPREAD_OFFSET = 1
    HYDROGEL_DEVIATION_SCALE = 0.4
    HYDROGEL_POSITION_SCALE = 5           # reduced: less skew → quotes stay tight → more fills
    HYDROGEL_TARGET_SCALE = 7             # doubled: larger targets → more position cycling
    HYDROGEL_AGGRESSIVE_THRESHOLD = 30
    HYDROGEL_AGGRESSIVE_SIZE = 40
    HYDROGEL_WARMUP_TICKS = 2_500         # faster ramp: start trading at full scale sooner

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT - Adaptive Mean Reversion (New Model)
    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive mean reversion (new model)
    VEV_EMA_ALPHA = 0.15
    VEV_VOL_ALPHA = 0.05

    # signal thresholds
    # tighter, more selective trading
    VEV_ENTRY_SIGMA = 1.5
    VEV_AGG_SIGMA = 2.5

    # spread protection threshold (NEW)
    VEV_MIN_EDGE = 1.2

    # inventory dampener (NEW)
    VEV_INV_BETA = 0.3

    # microstructure alpha
    VEV_IMB_ALPHA = 0.8

    # trend filter strength
    VEV_TREND_BETA = 0.8

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options - BS fair value with time decay
    # TTE_DAY0 = 8 for historical backtest; change to 5 for live submission.
    # Per-strike: (IV, misprice_threshold, clip_range, agg_dev_threshold)
    # ──────────────────────────────────────────────────────────────────────────
    OPT_TTE_DAY0 = 8                     # TTE (days) at start of historical day 0
    OPT_MR_STRIKES = {
        # K:  (IV,   threshold, clip, agg_thresh)
        4000: (0.25,  5, 60, 25),
        4500: (0.22,  4, 60, 20),
        5000: (0.20,  1, 40, 15),
        5100: (0.20,  1, 30, 10),
        5200: (0.20,  1, 20,  8),
        5300: (0.20,  1, 12,  6),
        5400: (0.21,  1, 15,  5),
        5500: (0.22,  1,  6,  3),
    }
    OPT_TARGET_SCALE    = 10
    OPT_SPREAD_OFFSET   = 1
    OPT_DEVIATION_SCALE = 0.4
    OPT_POSITION_SCALE  = 2
    OPT_AGGRESSIVE_SIZE = 5
    OPT_WARMUP_TICKS    = 2_500

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

        # ── Adaptive HYDROGEL fair value (slow EMA, seed = HYDROGEL_FAIR_VALUE) ─
        h_od = state.order_depths.get("HYDROGEL_PACK")
        if h_od and h_od.buy_orders and h_od.sell_orders:
            h_mid = (max(h_od.buy_orders) + min(h_od.sell_orders)) / 2
            h_fv  = (self.HYDROGEL_EMA_ALPHA * h_mid
                     + (1 - self.HYDROGEL_EMA_ALPHA) * trader_data.get("h_fv", self.HYDROGEL_FAIR_VALUE))
        else:
            h_fv = trader_data.get("h_fv", self.HYDROGEL_FAIR_VALUE)
        trader_data["h_fv"] = h_fv

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
                result[product] = self._trade_hydrogel(od, pos, limit, state.timestamp, h_fv)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit, state.timestamp, trader_data)
            elif product.startswith("VEV_"):
                result[product] = self._trade_vev_option(product, od, pos, limit, spot, state.timestamp, day_index)
            else:
                result[product] = []

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0, fair_value: float = None) -> List[Order]:
        if not self.HYDROGEL: return []

        fv = fair_value if fair_value is not None else self.HYDROGEL_FAIR_VALUE
        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else fv - self.HYDROGEL_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fv + self.HYDROGEL_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - fv

        ramp = min(1.0, timestamp / self.HYDROGEL_WARMUP_TICKS) if self.HYDROGEL_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.HYDROGEL_TARGET_SCALE * ramp

        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= fv - self.HYDROGEL_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= fv + self.HYDROGEL_MISPRICE_THRESHOLD:
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
            our_bid = int(fv - 2)
            our_ask = int(fv + 2)

        our_bid = max(our_bid, int(fv - self.HYDROGEL_CLIP_RANGE))
        our_ask = min(our_ask, int(fv + self.HYDROGEL_CLIP_RANGE))

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
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0, trader_data: dict = None) -> List[Order]:
        if not self.VELVETFRUIT:
            return []

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else 0

        if best_bid == 0 and best_ask == 0:
            return []

        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else (best_bid or best_ask)

        if trader_data is None:
            trader_data = {}

        # ─────────────────────────────
        # EMA mean + volatility
        # ─────────────────────────────
        prev_mean = trader_data.get("vev_ema_mean", mid)
        mean = self.VEV_EMA_ALPHA * mid + (1 - self.VEV_EMA_ALPHA) * prev_mean
        trader_data["vev_ema_mean"] = mean

        prev_var = trader_data.get("vev_ema_var", 1.0)
        diff = mid - mean
        var = self.VEV_VOL_ALPHA * (diff * diff) + (1 - self.VEV_VOL_ALPHA) * prev_var
        trader_data["vev_ema_var"] = var
        std = max(1.0, var ** 0.5)

        # ─────────────────────────────
        # signal (alpha core)
        # ─────────────────────────────
        bid_prices = sorted(od.buy_orders.keys(), reverse=True)[:3]
        ask_prices = sorted(od.sell_orders.keys())[:3]

        bid_vol = sum(od.buy_orders[p] for p in bid_prices) if bid_prices else 0
        ask_vol = -sum(od.sell_orders[p] for p in ask_prices) if ask_prices else 0

        denom = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / denom if denom > 0 else 0

        prev_mean_trend = trader_data.get("vev_prev_mean", mean)
        trend = mean - prev_mean_trend
        trader_data["vev_prev_mean"] = mean

        z_like = (mid - mean) / std
        signal = z_like + self.VEV_IMB_ALPHA * imbalance - self.VEV_TREND_BETA * trend

        inv_penalty = (pos / limit) * self.VEV_INV_BETA
        signal -= inv_penalty

        # ─────────────────────────────
        # Layer 1: Market Making (passive)
        # ─────────────────────────────
        skew = (pos / limit) * 0.5 if limit else 0

        mm_bid = int(mean - 1 - skew)
        mm_ask = int(mean + 1 - skew)

        mm_size = max(1, limit // 4)

        if od.sell_orders and pos < limit:
            orders.append(Order("VELVETFRUIT_EXTRACT", mm_bid, mm_size))
        if od.buy_orders and pos > -limit:
            orders.append(Order("VELVETFRUIT_EXTRACT", mm_ask, -mm_size))

        # ─────────────────────────────
        # Layer 2: Alpha overlay (aggressive fills on strong signal)
        # Note: edge_buy = mean - best_ask is always ~-2.5 (half spread),
        # so we gate only on signal strength, not edge.
        # ─────────────────────────────
        if signal < -self.VEV_AGG_SIGMA:
            qty = min(limit - pos, 40)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))
        elif signal < -self.VEV_ENTRY_SIGMA:
            qty = min(limit - pos, 20)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))

        if signal > self.VEV_AGG_SIGMA:
            qty = min(limit + pos, 40)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))
        elif signal > self.VEV_ENTRY_SIGMA:
            qty = min(limit + pos, 20)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))

        return orders

    def _bs_call(self, S: float, K: float, T_days: float, sigma: float) -> float:
        """Black-Scholes call price. T_days in calendar days, r=0."""
        T = T_days / 252
        if T <= 1e-6 or S <= 0:
            return max(0.0, S - K)
        d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        return S * N(d1) - K * N(d2)

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_vev_option(self, symbol: str, od: OrderDepth, pos: int,
                          limit: int, spot: float, timestamp: int = 0,
                          day_index: int = 0) -> List[Order]:
        if not self.OPTIONS:
            return []

        K = int(symbol.split("_")[1])
        if K not in self.OPT_MR_STRIKES:
            return []

        iv, threshold, clip, agg_thresh = self.OPT_MR_STRIKES[K]
        # TTE decreases by 1 per day and linearly within a day
        T_days = (self.OPT_TTE_DAY0 - day_index) - timestamp / 1_000_000
        fv = self._bs_call(spot, K, max(T_days, 0.01), iv)
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

