"""
Round 5 — revised strategy after backtest diagnosis.

ROOT CAUSES OF -3.26M:
  1. EMA-MM with alpha=0.001 lags 200–450 ticks on trending products
     → quotes are way off-market → constant adverse selection
  2. Full-limit quote sizes (50) → one trending day wipes a product
  3. PEBBLES basket arb never fires: max correction ±3.6 < half-spread (4.9–8.3)

FIXES:
  A. SNACKPACK   — keep EMA-MM (stable FV, it was the only profitable group)
  B. MR products — fast EMA (α=0.01), aggressive fill + passive penny-jump fallback
  C. PEBBLES     — passive penny-jump with basket-adjusted FV for quoting
  D. All others  — penny-jump MM (market-relative, no EMA lag) + inventory cap

Position limit: 50 on every product.
"""

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import json


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations: Observation) -> list[Any]:
        co = {}
        for product, o in observations.conversionObservations.items():
            co[product] = [o.bidPrice, o.askPrice, o.transportFees,
                           o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
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
                out = candidate; lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()
LIMIT = 10


class Trader:

    # ── Product toggles (set False to disable a product) ─────────────────────
    ENABLED = {
        # Galaxy Sounds
        'GALAXY_SOUNDS_BLACK_HOLES':      False,
        'GALAXY_SOUNDS_DARK_MATTER':      False,
        'GALAXY_SOUNDS_PLANETARY_RINGS':  False,
        'GALAXY_SOUNDS_SOLAR_FLAMES':     False,
        'GALAXY_SOUNDS_SOLAR_WINDS':      False,
        # Microchip
        'MICROCHIP_CIRCLE':      False,
        'MICROCHIP_OVAL':        False,
        'MICROCHIP_RECTANGLE':   False,
        'MICROCHIP_SQUARE':      False,
        'MICROCHIP_TRIANGLE':    False,
        # Oxygen Shake
        'OXYGEN_SHAKE_CHOCOLATE':       False,
        'OXYGEN_SHAKE_EVENING_BREATH':  False,
        'OXYGEN_SHAKE_GARLIC':          False,
        'OXYGEN_SHAKE_MINT':            False,
        'OXYGEN_SHAKE_MORNING_BREATH':  False,
        # Panel
        'PANEL_1X2':  False,
        'PANEL_1X4':  False,
        'PANEL_2X2':  False,
        'PANEL_2X4':  False,
        'PANEL_4X4':  False,
        # Pebbles
        'PEBBLES_XS':  True,
        'PEBBLES_S':   True,
        'PEBBLES_M':   True,
        'PEBBLES_L':   True,
        'PEBBLES_XL':  True,
        # Robot
        'ROBOT_DISHES':    False,
        'ROBOT_IRONING':   False,
        'ROBOT_LAUNDRY':   False,
        'ROBOT_MOPPING':   False,
        'ROBOT_VACUUMING': False,
        # Sleep Pod
        'SLEEP_POD_COTTON':     False,
        'SLEEP_POD_LAMB_WOOL':  False,
        'SLEEP_POD_NYLON':      False,
        'SLEEP_POD_POLYESTER':  False,
        'SLEEP_POD_SUEDE':      False,
        # Snackpack
        'SNACKPACK_CHOCOLATE':   False,
        'SNACKPACK_PISTACHIO':   False,
        'SNACKPACK_RASPBERRY':   False,
        'SNACKPACK_STRAWBERRY':  False,
        'SNACKPACK_VANILLA':     False,
        # Translator
        'TRANSLATOR_ASTRO_BLACK':      False,
        'TRANSLATOR_ECLIPSE_CHARCOAL': False,
        'TRANSLATOR_GRAPHITE_MIST':    False,
        'TRANSLATOR_SPACE_GRAY':       False,
        'TRANSLATOR_VOID_BLUE':        False,
        # UV Visor
        'UV_VISOR_AMBER':    False,
        'UV_VISOR_MAGENTA':  False,
        'UV_VISOR_ORANGE':   False,
        'UV_VISOR_RED':      False,
        'UV_VISOR_YELLOW':   False,
    }

    # ── PEBBLES ───────────────────────────────────────────────────────────────
    PEBBLES    = ['PEBBLES_XS', 'PEBBLES_S', 'PEBBLES_M', 'PEBBLES_L', 'PEBBLES_XL']
    PEBBLE_SUM = 50_000

    # ── MR products: (ema_alpha, threshold) ──────────────────────────────────
    # Fast EMA (α=0.01, half-life ~70 ticks) to track the current price level.
    # Threshold > half-spread ensures each aggressive fill is net-positive.
    MR_PARAMS = {
        'ROBOT_DISHES':                (0.01, 8),    # spread~7,  ac1=-0.23
        'ROBOT_IRONING':               (0.01, 7),    # spread~6,  ac1=-0.13
        'OXYGEN_SHAKE_EVENING_BREATH': (0.01, 8),    # spread~12, ac1=-0.12
        'OXYGEN_SHAKE_CHOCOLATE':      (0.01, 8),    # spread~12, ac1=-0.09
    }

    # ── SNACKPACK: EMA-MM (stable intra-day FV) ──────────────────────────────
    SNACKPACK_PARAMS = {
        'SNACKPACK_RASPBERRY':  (0.0002, 8),
        'SNACKPACK_VANILLA':    (0.0002, 8),
        'SNACKPACK_CHOCOLATE':  (0.0002, 8),
    }

    # ── Trending products: linear fair value = day_base + slope * timestamp ──
    # (day0_base, intraday_slope, buy_thresh, sell_thresh)
    # day0_base  : fair value at timestamp=0 on the first day of the round
    # slope      : price change per timestamp unit (e.g. -0.000489 for -489/day)
    # buy_thresh : buy aggressively when ask <= fair + buy_thresh
    # sell_thresh: safety sell only when bid >= fair + sell_thresh
    # Set day0_base=0 until calibrated from backtest data.
    TREND_PARAMS = {
        'SNACKPACK_PISTACHIO':  (0, -0.000489, 5, 15),   # ~-489/day drift
        'SNACKPACK_STRAWBERRY': (0,  0.000436, 5, 15),   # ~+436/day drift
    }

    # ── Penny-jump MM products (all others) ──────────────────────────────────
    # Market-relative quoting: bid at best_bid+1, ask at best_ask-1.
    # No EMA needed — eliminates lag-based adverse selection entirely.
    PJ_PRODUCTS = {
        # Robots (mild MR, handled separately above for dishes/ironing)
        'ROBOT_LAUNDRY', 'ROBOT_MOPPING', 'ROBOT_VACUUMING',
        # Oxygen Shake (remaining)
        'OXYGEN_SHAKE_MINT', 'OXYGEN_SHAKE_MORNING_BREATH', 'OXYGEN_SHAKE_GARLIC',
        # Translator
        'TRANSLATOR_ASTRO_BLACK', 'TRANSLATOR_ECLIPSE_CHARCOAL',
        'TRANSLATOR_GRAPHITE_MIST', 'TRANSLATOR_SPACE_GRAY', 'TRANSLATOR_VOID_BLUE',
        # Panel
        'PANEL_1X2', 'PANEL_1X4', 'PANEL_2X2', 'PANEL_2X4', 'PANEL_4X4',
        # Sleep Pod
        'SLEEP_POD_COTTON', 'SLEEP_POD_LAMB_WOOL', 'SLEEP_POD_NYLON',
        'SLEEP_POD_POLYESTER', 'SLEEP_POD_SUEDE',
        # UV Visor
        'UV_VISOR_AMBER', 'UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE',
        'UV_VISOR_RED', 'UV_VISOR_YELLOW',
        # Galaxy Sounds
        'GALAXY_SOUNDS_BLACK_HOLES', 'GALAXY_SOUNDS_DARK_MATTER',
        'GALAXY_SOUNDS_PLANETARY_RINGS', 'GALAXY_SOUNDS_SOLAR_FLAMES',
        'GALAXY_SOUNDS_SOLAR_WINDS',
        # Microchip
        'MICROCHIP_CIRCLE', 'MICROCHIP_OVAL', 'MICROCHIP_RECTANGLE',
        'MICROCHIP_SQUARE', 'MICROCHIP_TRIANGLE',
    }

    # Inventory cap for penny-jump: stop quoting one side beyond this.
    PJ_INVENTORY_CAP = 30    # 60% of LIMIT
    PJ_QUOTE_SIZE    = 5

    # ── helpers ───────────────────────────────────────────────────────────────

    def _mid(self, od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2
        return None

    def _ema(self, old, val, alpha):
        return val if old is None else (1 - alpha) * old + alpha * val

    # ── strategy E: linear-trend fair value ──────────────────────────────────

    def _trade_trend(self, product: str, od: OrderDepth, pos: int,
                     fair: float, buy_thresh: float, sell_thresh: float,
                     result: dict):
        orders = []
        buy_cap  = LIMIT - pos
        sell_cap = LIMIT + pos

        # 1. Hit all ask levels priced at or below fair + buy_thresh
        if od.sell_orders and buy_cap > 0:
            for ask_px in sorted(od.sell_orders):
                if ask_px > fair + buy_thresh:
                    break
                qty = min(-od.sell_orders[ask_px], buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask_px, qty))
                    buy_cap -= qty
                if buy_cap <= 0:
                    break

        # 2. Rest bids just below fair to catch dips (two-level ladder)
        if buy_cap > 0 and od.sell_orders:
            best_ask = min(od.sell_orders)
            bid_px   = min(int(fair - 1), best_ask - 1)
            tier1    = max(buy_cap // 2, 1)
            orders.append(Order(product, bid_px, min(tier1, buy_cap)))
            rem = buy_cap - tier1
            if rem > 0:
                orders.append(Order(product, bid_px - 3, rem))

        # 3. Safety valve: sell only when price massively above fair
        if pos > 0 and sell_cap > 0 and od.buy_orders:
            best_bid = max(od.buy_orders)
            if best_bid > fair + sell_thresh:
                qty = min(od.buy_orders[best_bid], sell_cap, 5)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

        if orders:
            result[product] = orders

    # ── strategy A: SNACKPACK EMA-MM ─────────────────────────────────────────

    def _trade_snackpack(self, product: str, od: OrderDepth, pos: int,
                         ema: float, half_spread: int, result: dict):
        if ema is None:
            return
        pos_pct = pos / LIMIT
        skew    = round(pos_pct * half_spread)
        bid_px  = round(ema - half_spread - skew)
        ask_px  = round(ema + half_spread - skew)
        if bid_px >= ask_px:
            bid_px, ask_px = round(ema) - 1, round(ema) + 1

        orders = []
        if LIMIT - pos > 0:
            orders.append(Order(product, bid_px,  LIMIT - pos))
        if LIMIT + pos > 0:
            orders.append(Order(product, ask_px, -(LIMIT + pos)))
        if orders:
            result[product] = orders

    # ── strategy B: MR + passive penny-jump fallback ─────────────────────────

    def _trade_mr(self, product: str, od: OrderDepth, pos: int,
                  ema: float, threshold: float, result: dict):
        mid = self._mid(od)
        if mid is None or ema is None:
            self._trade_pj(product, od, pos, result)
            return

        dev    = mid - ema
        orders = []

        if dev > threshold and pos > -LIMIT:
            # Overbought → sell aggressively at best bid
            if od.buy_orders:
                best_bid = max(od.buy_orders)
                qty = min(LIMIT + pos, 2, abs(od.buy_orders[best_bid]))
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

        elif dev < -threshold and pos < LIMIT:
            # Oversold → buy aggressively at best ask
            if od.sell_orders:
                best_ask = min(od.sell_orders)
                qty = min(LIMIT - pos, 2, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))

        if orders:
            result[product] = orders
        else:
            # No strong MR signal: passive penny-jump to capture spread
            self._trade_pj(product, od, pos, result)

    # ── strategy C: PEBBLES ───────────────────────────────────────────────────

    def _trade_pebbles(self, state: TradingState, emas: dict, result: dict):
        ods = {p: state.order_depths.get(p) for p in self.PEBBLES}

        # trade only XL as proxy instrument
        p = "PEBBLES_XL"
        od = ods.get(p)
        if od is None:
            return

        # compute full basket signal
        mids = {}
        for k, o in ods.items():
            if o is None:
                return
            m = self._mid(o)
            if m is None:
                return
            mids[k] = m

        basket_sum = sum(mids.values())
        dev = basket_sum - self.PEBBLE_SUM

        pos = state.position.get(p, 0)

        # ── FLATTEN ZONE ──
        if abs(dev) <= 5:
            if pos > 0 and od.buy_orders:
                best_bid = max(od.buy_orders)
                qty = min(pos, abs(od.buy_orders[best_bid]))
                if qty > 0:
                    result.setdefault(p, []).append(Order(p, best_bid, -qty))

            elif pos < 0 and od.sell_orders:
                best_ask = min(od.sell_orders)
                qty = min(-pos, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    result.setdefault(p, []).append(Order(p, best_ask, qty))

            return

        # ── TRADE ZONE ──
        if abs(dev) < 12:
            return

        # direction: basket too high → short XL, too low → long XL
        if dev > 0:
            # short XL
            if od.buy_orders and pos > -LIMIT:
                best_bid = max(od.buy_orders)
                qty = min(LIMIT + pos, abs(od.buy_orders[best_bid]))
                if qty > 0:
                    result.setdefault(p, []).append(Order(p, best_bid, -qty))

        else:
            # long XL
            if od.sell_orders and pos < LIMIT:
                best_ask = min(od.sell_orders)
                qty = min(LIMIT - pos, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    result.setdefault(p, []).append(Order(p, best_ask, qty))

    # ── strategy D: penny-jump market making ─────────────────────────────────

    def _trade_pj(self, product: str, od: OrderDepth, pos: int, result: dict):
        if not od.buy_orders or not od.sell_orders:
            return
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        spread   = best_ask - best_bid

        # Need room to penny-jump; fall back to joining when spread = 2
        if spread >= 3:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
        elif spread == 2:
            our_bid = best_bid
            our_ask = best_ask
        else:
            return   # spread = 1, no profit possible

        orders   = []
        buy_qty  = min(LIMIT - pos, self.PJ_QUOTE_SIZE) if pos <  self.PJ_INVENTORY_CAP else 0
        sell_qty = min(LIMIT + pos, self.PJ_QUOTE_SIZE) if pos > -self.PJ_INVENTORY_CAP else 0

        if buy_qty  > 0: orders.append(Order(product, our_bid,  buy_qty))
        if sell_qty > 0: orders.append(Order(product, our_ask, -sell_qty))
        if orders:
            result[product] = orders

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        td   = json.loads(state.traderData) if state.traderData else {}
        emas = td.get('emas', {})
        result: dict[str, list[Order]] = {}

        # ── Day detection (timestamp resets to ~0 each new day) ───────────────
        last_ts   = td.get('last_ts', -1)
        day_index = td.get('day_index', 0)
        if last_ts > 0 and state.timestamp < last_ts - 500_000:
            day_index += 1
        td['last_ts']   = state.timestamp
        td['day_index'] = day_index

        # ── E — trending products ─────────────────────────────────────────────
        for p, (day0_base, slope, buy_thresh, sell_thresh) in self.TREND_PARAMS.items():
            if not self.ENABLED.get(p, True) or day0_base == 0:
                continue
            od = state.order_depths.get(p)
            if od:
                fair = day0_base + day_index * slope * 1_000_000 + slope * state.timestamp
                self._trade_trend(p, od, state.position.get(p, 0),
                                  fair, buy_thresh, sell_thresh, result)

        # Update EMAs for SNACKPACK and MR products
        for p, (alpha, _) in {**self.SNACKPACK_PARAMS, **self.MR_PARAMS}.items():
            od = state.order_depths.get(p)
            if od:
                mid = self._mid(od)
                if mid is not None:
                    emas[p] = self._ema(emas.get(p), mid, alpha)

        # A — SNACKPACK
        for p, (_, half_spread) in self.SNACKPACK_PARAMS.items():
            if not self.ENABLED.get(p, True):
                continue
            od = state.order_depths.get(p)
            if od:
                self._trade_snackpack(p, od, state.position.get(p, 0),
                                      emas.get(p), half_spread, result)

        # B — MR products
        for p, (_, threshold) in self.MR_PARAMS.items():
            if not self.ENABLED.get(p, True):
                continue
            od = state.order_depths.get(p)
            if od:
                self._trade_mr(p, od, state.position.get(p, 0),
                               emas.get(p), threshold, result)

        # C — PEBBLES (skip entire basket if any pebble is disabled)
        if all(self.ENABLED.get(p, True) for p in self.PEBBLES):
            self._trade_pebbles(state, emas, result)

        # D — penny-jump products
        for p in self.PJ_PRODUCTS:
            if not self.ENABLED.get(p, True):
                continue
            od = state.order_depths.get(p)
            if od:
                self._trade_pj(p, od, state.position.get(p, 0), result)

        td['emas'] = emas
        trader_data_str = json.dumps(td)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str
