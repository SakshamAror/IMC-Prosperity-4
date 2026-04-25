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
    HYDROGEL_FAIR_VALUE = 9990.8          # seed only; EMA adapts within session
    HYDROGEL_EMA_ALPHA = 0.002            # slow: wall-mid half-life ≈ 350 ticks
    HYDROGEL_MISPRICE_THRESHOLD = 12
    HYDROGEL_CLIP_RANGE = 100
    HYDROGEL_SPREAD_OFFSET = 1
    HYDROGEL_DEVIATION_SCALE = 0.4
    HYDROGEL_POSITION_SCALE = 12
    HYDROGEL_TARGET_SCALE = 2.0           # ACF(1)=0.998 → slow reversion; was 3.5, overbets
    HYDROGEL_AGGRESSIVE_THRESHOLD = 40
    HYDROGEL_AGGRESSIVE_SIZE = 40
    HYDROGEL_WARMUP_TICKS = 500           # was 10k = entire day; 500 ticks is enough

    # ──────────────────────────────────────────────────────────────────────────
    # VELVETFRUIT_EXTRACT - Mean Reversion Strategy
    # ──────────────────────────────────────────────────────────────────────────
    VEV_FAIR_VALUE = 5248.0               # seed only; EMA adapts within session
    VEV_EMA_ALPHA = 0.005                 # slightly faster than hydrogel; spread is tighter
    VEV_MISPRICE_THRESHOLD = 4            # market spread=5; take when ask ≥4 below fair
    VEV_CLIP_RANGE = 60
    VEV_SPREAD_OFFSET = 1
    VEV_DEVIATION_SCALE = 0.4
    VEV_POSITION_SCALE = 4
    VEV_TARGET_SCALE = 2.5                # ACF(1)=0.997 → slow reversion; was 4
    VEV_AGGRESSIVE_THRESHOLD = 30
    VEV_AGGRESSIVE_SIZE = 40

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options - Fair Value Model + Market Making
    # ──────────────────────────────────────────────────────────────────────────
    # Coefficients: (intercept, moneyness, days_remaining, moneyness^2) — fit offline, paste here
    OPT_FV_COEFFS: Dict[int, tuple] = {
        5100: (30.7077, 0.6464, 4.1338, 0.000627),
        5200: (38.8591, 0.5599, 4.4991, 0.000597),
        5300: (43.2072, 0.4246, 3.9257, 0.000434),
        5400: (29.8800, 0.2727, 3.1140, 0.000364),
        5500: (12.2952, 0.0519, 1.7156, -0.000048),
    }
    OPT_MAX_DAYS      = 7
    OPT_MM_SPREAD     = 8     # wide enough to cover ~5-unit model bias on 5100-5300
    OPT_MM_SIZE       = 1     # small to limit one-sided accumulation
    OPT_SKEW_COEF     = 2
    OPT_ITM_ARB_THRESH = 2    # plan spec: only buy deep ITM if ask < intrinsic - 2
    OPT_ITM_MAX_SIZE  = 10    # cap arb buy per tick to limit delta buildup
    OPT_DELTA_LIMIT   = 50
    OPT_SPIKE_THRESH  = 30
    OPT_SPIKE_RESUME  = 20
    OPT_LEVERAGE_FRAC = 0.25  # option pos = 25% of VEV MR target (ITM calls only)

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

        # ── Day tracking (timestamp resets to 0 each new day) ─────────────────
        last_ts   = trader_data.get("last_ts", -1)
        day_index = trader_data.get("day_index", 0)
        if last_ts > 0 and state.timestamp < last_ts - 500_000:
            day_index += 1
        trader_data["last_ts"]   = state.timestamp
        trader_data["day_index"] = day_index
        days_remaining = max(0, self.OPT_MAX_DAYS - day_index)

        # ── Adaptive fair values via wall-mid EMA ─────────────────────────────
        # Wall mid uses the max-volume level on each side (L2 here), which is
        # more stable than raw mid and tracks the market maker's anchored price.
        def wall_mid(od, seed):
            if not od or not od.buy_orders or not od.sell_orders:
                return seed
            wb = max(od.buy_orders, key=lambda p: od.buy_orders[p])
            wa = min(od.sell_orders, key=lambda p: od.sell_orders[p])
            return (wb + wa) / 2

        hyd_od = state.order_depths.get("HYDROGEL_PACK")
        hyd_fv  = trader_data.get("hyd_fv_ema", self.HYDROGEL_FAIR_VALUE)
        hyd_fv  = self.HYDROGEL_EMA_ALPHA * wall_mid(hyd_od, hyd_fv) + (1 - self.HYDROGEL_EMA_ALPHA) * hyd_fv
        trader_data["hyd_fv_ema"] = hyd_fv

        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_fv  = trader_data.get("vev_fv_ema", self.VEV_FAIR_VALUE)
        vev_fv  = self.VEV_EMA_ALPHA * wall_mid(vev_od, vev_fv) + (1 - self.VEV_EMA_ALPHA) * vev_fv
        trader_data["vev_fv_ema"] = vev_fv

        # ── Spot price (for options) ───────────────────────────────────────────
        if vev_od and vev_od.buy_orders and vev_od.sell_orders:
            spot = (max(vev_od.buy_orders) + min(vev_od.sell_orders)) / 2
        else:
            spot = vev_fv

        # ── Spike detection ────────────────────────────────────────────────────
        prev_spot    = trader_data.get("prev_spot", spot)
        spike_active = trader_data.get("spike_active", False)
        if abs(spot - prev_spot) > self.OPT_SPIKE_THRESH:
            spike_active = True
        elif spike_active and abs(spot - prev_spot) < self.OPT_SPIKE_RESUME:
            spike_active = False
        trader_data["prev_spot"]    = spot
        trader_data["spike_active"] = spike_active

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "HYDROGEL_PACK":
                result[product] = self._trade_hydrogel(od, pos, limit, state.timestamp, hyd_fv)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit, vev_fv)
            elif product.startswith("VEV_"):
                result[product] = self._trade_vev_option(product, od, pos, limit, spot, vev_fv)
            else:
                result[product] = []

        # ── Delta hedge via VELVETFRUIT_EXTRACT ───────────────────────────────
        if self.OPTIONS and vev_od:
            net_delta = sum(
                state.position.get(f"VEV_{K}", 0) * self._opt_delta(K, spot, days_remaining)
                for K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
            )
            if abs(net_delta) > self.OPT_DELTA_LIMIT:
                target_delta = self.OPT_DELTA_LIMIT if net_delta > 0 else -self.OPT_DELTA_LIMIT
                hedge = int(-(net_delta - target_delta))
                vev_pos   = state.position.get("VELVETFRUIT_EXTRACT", 0)
                vev_limit = self.POSITION_LIMITS.get("VELVETFRUIT_EXTRACT", 200)
                hedge = max(-vev_limit - vev_pos, min(vev_limit - vev_pos, hedge))
                if hedge > 0 and vev_od.sell_orders:
                    ask = min(vev_od.sell_orders)
                    qty = min(hedge, -vev_od.sell_orders[ask])
                    if qty > 0:
                        result.setdefault("VELVETFRUIT_EXTRACT", []).append(Order("VELVETFRUIT_EXTRACT", int(ask), qty))
                elif hedge < 0 and vev_od.buy_orders:
                    bid = max(vev_od.buy_orders)
                    qty = max(hedge, -vev_od.buy_orders[bid])
                    if qty < 0:
                        result.setdefault("VELVETFRUIT_EXTRACT", []).append(Order("VELVETFRUIT_EXTRACT", int(bid), qty))

        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

    # ──────────────────────────────────────────────────────────────────────────
    # HYDROGEL_PACK — market making around fair value 9987.5
    # ──────────────────────────────────────────────────────────────────────────
    def _trade_hydrogel(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0, fair_value: float = None) -> List[Order]:
        if (not self.HYDROGEL): return []
        if fair_value is None: fair_value = self.HYDROGEL_FAIR_VALUE

        orders: List[Order] = []

        FV = fair_value
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else FV - self.HYDROGEL_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else FV + self.HYDROGEL_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - FV

        ramp = min(1.0, timestamp / self.HYDROGEL_WARMUP_TICKS) if self.HYDROGEL_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.HYDROGEL_TARGET_SCALE * ramp

        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        # 1. Take clear misprices immediately
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= FV - self.HYDROGEL_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("HYDROGEL_PACK", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= FV + self.HYDROGEL_MISPRICE_THRESHOLD:
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
            our_bid = int(FV - 2)
            our_ask = int(FV + 2)

        our_bid = max(our_bid, int(FV - self.HYDROGEL_CLIP_RANGE))
        our_ask = min(our_ask, int(FV + self.HYDROGEL_CLIP_RANGE))

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
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int, fair_value: float = None) -> List[Order]:
        if (not self.VELVETFRUIT): return []
        if fair_value is None: fair_value = self.VEV_FAIR_VALUE

        orders: List[Order] = []
        FV = fair_value

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else FV - self.VEV_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else FV + self.VEV_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - FV

        target = int(max(-limit, min(limit, -deviation * self.VEV_TARGET_SCALE)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        # 1. Take clear misprices immediately
        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= FV - self.VEV_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= FV + self.VEV_MISPRICE_THRESHOLD:
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
            our_bid = int(FV - 2)
            our_ask = int(FV + 2)

        our_bid = max(our_bid, int(FV - self.VEV_CLIP_RANGE))
        our_ask = min(our_ask, int(FV + self.VEV_CLIP_RANGE))

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


    def _opt_fv(self, K: int, S: float, days: int) -> float:
        if K <= 5000:
            return max(S - K, 0.0)
        b, m_coef, d_coef, m2_coef = self.OPT_FV_COEFFS[K]
        m = S - K
        return b + m_coef * m + d_coef * days + m2_coef * m * m

    def _opt_delta(self, K: int, S: float, days: int) -> float:
        if K <= 5000:
            return 1.0 if S > K else 0.0
        return (self._opt_fv(K, S + 1, days) - self._opt_fv(K, S - 1, days)) / 2

    def _trade_vev_option(self, symbol: str, od: OrderDepth, pos: int,
                      limit: int, spot: float, vev_fv: float = None) -> List[Order]:
        if not self.OPTIONS:
            return []

        K = int(symbol.split("_")[1])
        if K in (6000, 6500):
            return []

        orders: List[Order] = []
        buy_cap = limit - pos
        sell_cap = limit + pos
        intrinsic = max(spot - K, 0.0)

        # Intrinsic arb: buy deep ITM if ask < intrinsic - threshold
        best_ask = min(od.sell_orders) if od.sell_orders else None
        if best_ask is not None and best_ask < intrinsic - self.OPT_ITM_ARB_THRESH and buy_cap > 0:
            qty = min(-od.sell_orders[best_ask], buy_cap, self.OPT_ITM_MAX_SIZE)
            if qty > 0:
                orders.append(Order(symbol, int(best_ask), qty))
                buy_cap -= qty

        # Leverage: mirror VEV MR signal via ITM calls (5200, 5300 only)
        if K in (5200, 5300) and vev_fv is not None:
            vev_lim = self.POSITION_LIMITS.get("VELVETFRUIT_EXTRACT", 200)
            vev_dev = spot - vev_fv
            vev_target = int(max(-vev_lim, min(vev_lim, -vev_dev * self.VEV_TARGET_SCALE)))
            desired = int(vev_target * self.OPT_LEVERAGE_FRAC)
            diff = desired - pos
            if diff > 0 and buy_cap > 0 and od.sell_orders:
                ask = min(od.sell_orders)
                qty = min(diff, buy_cap, -od.sell_orders[ask])
                if qty > 0:
                    orders.append(Order(symbol, int(ask), qty))
            elif diff < 0 and sell_cap > 0 and od.buy_orders:
                bid = max(od.buy_orders)
                qty = min(-diff, sell_cap, od.buy_orders[bid])
                if qty > 0:
                    orders.append(Order(symbol, int(bid), -qty))

        return orders

