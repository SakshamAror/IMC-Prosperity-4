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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0: return max(S - K, 0.0)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


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
    HYDROGEL_FAIR_VALUE = 9992.06
    HYDROGEL_MISPRICE_THRESHOLD = 4       # halved: fire on smaller misprices
    HYDROGEL_CLIP_RANGE = 100
    HYDROGEL_SPREAD_OFFSET = 1
    HYDROGEL_DEVIATION_SCALE = 0.69
    HYDROGEL_POSITION_SCALE = 4           # reduced: less skew → quotes stay tight → more fills
    HYDROGEL_TARGET_SCALE = 4             # doubled: larger targets → more position cycling
    HYDROGEL_AGGRESSIVE_THRESHOLD = 40
    HYDROGEL_AGGRESSIVE_SIZE = 40
    HYDROGEL_WARMUP_TICKS = 4490        # faster ramp: start trading at full scale sooner
    # EMA drift: dynamic_fv = static_fv + ema_deviation * DRIFT_SCALE
    # EMA_ALPHA controls responsiveness (τ = 1/alpha ticks); DRIFT_SCALE caps how far FV can shift.
    HYDROGEL_EMA_ALPHA  = 0.059        
    HYDROGEL_DRIFT_SCALE = 0.439           # absorb x% of persistent deviation into FV

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
    VEV_EMA_ALPHA   = 0.059             
    VEV_DRIFT_SCALE = 0.439

    # ──────────────────────────────────────────────────────────────────────────
    # VEV_XXXX Options - EMA-drifted FV from LIVE calibration point
    # LIVE: day3 mean for deep ITM; 50% theta correction for OTM (TTE=4d).
    # FV = live_fv + ema_of_deviation * OPT_DRIFT_SCALE
    # EMA tracks persistent mispricings; drift scale caps how far FV can shift.
    # VEV_5400 and VEV_5500 excluded — too illiquid near expiry.
    # Per-strike: (live_fv, threshold, clip, agg_thresh, ema_ticks, drift_scale)
    # ──────────────────────────────────────────────────────────────────────────
    _OPT_PARAMS = {        # (fair_value, threshold, clip, agg_thresh, ema_ticks, drift_scale)
        4000: (1250.1,  5, 60, 25, 30, 0.466),
        4500: ( 750.1,  4, 60, 20, 445, 0.091),
        5000: ( 254.0,  1, 40, 15, 28, 0.297),
        5100: ( 164.5,  1, 30, 10, 66, 0.08),
        5200: (  42.0,  1, 20,  8, 189, 0.599),
        5300: (  41.18,  1, 12,  6, 140, 0.311),
        5400: (  14.2,  1, 15,  5, 44, 0.22),  
        5500: (   5.0,  1,  6,  3, 49, 0.791),
    }

    OPT_TARGET_SCALE    = 11              # increased: larger targets → more capacity
    OPT_SPREAD_OFFSET   = 1
    OPT_DEVIATION_SCALE = 23    # relative: applied to deviation/fv so skew is proportional across strikes
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

        # ── EMA-drifted fair values for HYDROGEL and VELVETFRUIT ───────────────
        # dynamic_fv = static_fv + ema_of_deviation * DRIFT_SCALE
        # EMA tracks slow persistent shifts; DRIFT_SCALE caps how far FV can move.
        hyd_od = state.order_depths.get("HYDROGEL_PACK")
        if hyd_od and hyd_od.buy_orders and hyd_od.sell_orders:
            hyd_mid = (max(hyd_od.buy_orders) + min(hyd_od.sell_orders)) / 2
            hyd_ema = trader_data.get("hyd_ema", getattr(self, '_hyd_ema', 0.0))
            hyd_ema += self.HYDROGEL_EMA_ALPHA * (hyd_mid - self.HYDROGEL_FAIR_VALUE - hyd_ema)
            trader_data["hyd_ema"] = hyd_ema
            self._hyd_ema = hyd_ema
            hyd_fv = self.HYDROGEL_FAIR_VALUE + hyd_ema * self.HYDROGEL_DRIFT_SCALE
        else:
            hyd_fv = self.HYDROGEL_FAIR_VALUE

        vev_ema = trader_data.get("vev_ema", getattr(self, '_vev_ema', 0.0))
        vev_ema += self.VEV_EMA_ALPHA * (spot - self.VEV_FAIR_VALUE - vev_ema)
        trader_data["vev_ema"] = vev_ema
        self._vev_ema = vev_ema
        vev_fv = self.VEV_FAIR_VALUE + vev_ema * self.VEV_DRIFT_SCALE

        # ── Per-option EMA-drifted fair values ────────────────────────────────
        opt_fv: Dict[int, float] = {}
        for K, (live_fv, _, _, _, ema_ticks, drift_scale) in self._OPT_PARAMS.items():
            od_k = state.order_depths.get(f"VEV_{K}")
            if od_k and od_k.buy_orders and od_k.sell_orders:
                mid_k = (max(od_k.buy_orders) + min(od_k.sell_orders)) / 2
                ema   = trader_data.get(f"opt_ema_{K}", getattr(self, f'_opt_ema_{K}', 0.0))
                ema  += (1 / ema_ticks) * (mid_k - live_fv - ema)
                trader_data[f"opt_ema_{K}"] = ema
                setattr(self, f'_opt_ema_{K}', ema)
                opt_fv[K] = live_fv + ema * drift_scale
            else:
                opt_fv[K] = live_fv

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos   = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 50)

            if product == "HYDROGEL_PACK":
                result[product] = self._trade_hydrogel(od, pos, limit, state.timestamp, hyd_fv)
            elif product == "VELVETFRUIT_EXTRACT":
                result[product] = self._trade_velvetfruit(od, pos, limit, state.timestamp, vev_fv)
            elif product.startswith("VEV_"):
                K = int(product.split("_")[1])
                result[product] = self._trade_vev_option(product, od, pos, limit, opt_fv.get(K), state.timestamp)
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
    def _trade_velvetfruit(self, od: OrderDepth, pos: int, limit: int, timestamp: int = 0, fair_value: float = None) -> List[Order]:
        if not self.VELVETFRUIT: return []
        fv = fair_value if fair_value is not None else self.VEV_FAIR_VALUE

        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else fv - self.VEV_CLIP_RANGE
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else fv + self.VEV_CLIP_RANGE
        mid = (best_bid + best_ask) / 2
        deviation = mid - fv

        ramp = min(1.0, timestamp / self.VEV_WARMUP_TICKS) if self.VEV_WARMUP_TICKS > 0 else 1.0
        effective_target_scale = self.VEV_TARGET_SCALE * ramp
        target = int(max(-limit, min(limit, -deviation * effective_target_scale)))
        buy_capacity  = min(max(0, target - pos), limit - pos)
        sell_capacity = min(max(0, pos - target), limit + pos)

        if od.sell_orders and buy_capacity > 0:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price >= fv - self.VEV_MISPRICE_THRESHOLD:
                    break
                qty = min(-od.sell_orders[ask_price], buy_capacity)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", int(ask_price), qty))
                    buy_capacity -= qty

        if od.buy_orders and sell_capacity > 0:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price <= fv + self.VEV_MISPRICE_THRESHOLD:
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
            our_bid = int(fv - 2)
            our_ask = int(fv + 2)

        our_bid = max(our_bid, int(fv - self.VEV_CLIP_RANGE))
        our_ask = min(our_ask, int(fv + self.VEV_CLIP_RANGE))

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
                          limit: int, fair_value: float, timestamp: int = 0) -> List[Order]:
        if not self.OPTIONS:
            return []

        K = int(symbol.split("_")[1])
        if K not in self._OPT_PARAMS or fair_value is None:
            return []

        _, threshold, clip, agg_thresh, _, _ = self._OPT_PARAMS[K]
        fv = fair_value
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
        skew = int(round((deviation / fv) * self.OPT_DEVIATION_SCALE + pos_skew * self.OPT_POSITION_SCALE))

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