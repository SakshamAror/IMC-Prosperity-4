"""
clean_modular.py — Prosperity Trading Bot (modular sector architecture)
========================================================================
Each product family is an independent Sector class with its own config,
state, and run() method. Trader just aggregates them.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)
from typing import Any
from math import sqrt, log
import json



# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

class Logger:
    MAX_LOG_LENGTH = 3750

    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self._to_json([
                self._compress_state(state, ""),
                self._compress_orders(orders),
                conversions, "", "",
            ])
        )
        max_item_length = (self.MAX_LOG_LENGTH - base_length) // 3

        print(self._to_json([
            self._compress_state(state, self._truncate(state.traderData, max_item_length)),
            self._compress_orders(orders),
            conversions,
            self._truncate(trader_data, max_item_length),
            self._truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def _compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self._compress_listings(state.listings),
            self._compress_order_depths(state.order_depths),
            self._compress_trades(state.own_trades),
            self._compress_trades(state.market_trades),
            state.position,
            self._compress_observations(state.observations),
        ]

    def _compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def _compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def _compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values()
            for t in arr
        ]

    def _compress_observations(self, observations: Observation) -> list[Any]:
        conversion_obs = {
            product: [
                obs.bidPrice, obs.askPrice, obs.transportFees,
                obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex,
            ]
            for product, obs in observations.conversionObservations.items()
        }
        return [observations.plainValueObservations, conversion_obs]

    def _compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def _to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def _truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        result = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                result = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return result


logger = Logger()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def mid_price(od: OrderDepth) -> float | None:
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None


def rolling_mean_std(window: list[float]) -> tuple[float, float]:
    n = len(window)
    if n < 2:
        return 0.0, 1.0
    mean = sum(window) / n
    std = sqrt(sum((x - mean) ** 2 for x in window) / (n - 1))
    return mean, max(std, 1.0)


def place_order_to_target(
    symbol: str,
    od: OrderDepth,
    current_pos: int,
    target_pos: int,
    orders_out: dict,
    pos_limit: int = 10,
) -> None:
    if not od.buy_orders or not od.sell_orders:
        return
    delta = target_pos - current_pos
    if delta == 0:
        return
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    price = (best_bid + 1 if delta > 0 else best_ask - 1) if abs(delta) <= 3 else (best_ask if delta > 0 else best_bid)
    orders_out[symbol] = [Order(symbol, price, delta)]

def market_make(symbol: str, od: OrderDepth, pos: int, pos_limit: int = 10) -> list[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread = best_ask - best_bid
    if spread <= 0:
        return []
    our_bid = best_bid + 1 if spread >= 2 else best_bid
    our_ask = best_ask - 1 if spread >= 2 else best_ask
    if our_bid >= our_ask:
        return []
    buy_capacity  = pos_limit - pos
    sell_capacity = pos_limit + pos
    orders: list[Order] = []
    if buy_capacity  > 0: orders.append(Order(symbol, our_bid,   buy_capacity))
    if sell_capacity > 0: orders.append(Order(symbol, our_ask,  -sell_capacity))
    return orders

def trade_trend(
    symbol: str,
    od: OrderDepth,
    pos: int,
    trend_hold_map: dict[str, int],
    pos_limit: int = 10,
) -> list[Order]:
    if not od.buy_orders or not od.sell_orders:
        return []
    hold_size = trend_hold_map[symbol]
    best_bid  = max(od.buy_orders)
    best_ask  = min(od.sell_orders)
    spread    = best_ask - best_bid
    orders: list[Order] = []
    if hold_size >= 0:
        if pos < hold_size:
            orders.append(Order(symbol, best_ask, hold_size - pos))
        if spread > 0:
            our_bid     = best_bid + 1 if spread >= 2 else best_bid
            mm_buy_cap  = pos_limit - max(pos, hold_size)
            mm_sell_cap = max(0, pos - hold_size)
            if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
            if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))
    else:  # short hold
        if pos > hold_size:
            orders.append(Order(symbol, best_bid, hold_size - pos))
        if spread > 0:
            our_bid     = best_bid + 1 if spread >= 2 else best_bid
            our_ask     = best_ask - 1 if spread >= 2 else best_ask
            mm_sell_cap = pos_limit + min(pos, hold_size)
            mm_buy_cap  = max(0, hold_size - pos)
            if mm_sell_cap > 0: orders.append(Order(symbol, our_ask,  -mm_sell_cap))
            if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
    return orders


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR BASE
# ══════════════════════════════════════════════════════════════════════════════

class Sector:
    ACTIVE: bool = True  # set False on a subclass to disable the whole sector

    def load_state(self, mem: dict) -> None:
        pass

    def save_state(self, mem: dict) -> None:
        pass

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE SECTOR BASE  (market-make or trend-hold, no rolling state)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleProductSector(Sector):
    """
    Handles a flat list of products with either trend-hold + MM or pure MM.
    Subclasses declare PRODUCTS, ENABLED, TREND_HOLD, and optionally POS_LIMIT.
    """
    PRODUCTS:    list[str]       = []
    ENABLED:     dict[str, bool] = {}
    TREND_HOLD:  dict[str, int]  = {}
    POS_LIMIT:   int             = 10

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        orders: dict[Symbol, list[Order]] = {}
        for symbol in self.PRODUCTS:
            if not self.ENABLED.get(symbol, False):
                continue
            od = state.order_depths.get(symbol)
            if not od:
                continue
            pos = state.position.get(symbol, 0)
            if symbol in self.TREND_HOLD:
                result = trade_trend(symbol, od, pos, self.TREND_HOLD, self.POS_LIMIT)
            else:
                result = market_make(symbol, od, pos, self.POS_LIMIT)
            if result:
                orders[symbol] = result
        return orders


# ══════════════════════════════════════════════════════════════════════════════
# GALAXY SOUNDS SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class GalaxySoundsSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ]
    ENABLED = {
        "GALAXY_SOUNDS_BLACK_HOLES":     True,
        "GALAXY_SOUNDS_DARK_MATTER":     True,
        "GALAXY_SOUNDS_PLANETARY_RINGS": False,
        "GALAXY_SOUNDS_SOLAR_FLAMES":    True,
        "GALAXY_SOUNDS_SOLAR_WINDS":     True,
    }
    TREND_HOLD = {
        "GALAXY_SOUNDS_BLACK_HOLES":     8,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MICROCHIP SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class MicrochipSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ]
    ENABLED = {
        "MICROCHIP_CIRCLE":    True,
        "MICROCHIP_OVAL":      True,
        "MICROCHIP_RECTANGLE": True,
        "MICROCHIP_SQUARE":    True,
        "MICROCHIP_TRIANGLE":  True,
    }
    TREND_HOLD = {
        "MICROCHIP_OVAL":      -10,
        "MICROCHIP_SQUARE":    9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OXYGEN SHAKE SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class OxygenShakeSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ]
    ENABLED = {
        "OXYGEN_SHAKE_CHOCOLATE":      True,  # handled by OxygenMRSector
        "OXYGEN_SHAKE_EVENING_BREATH": True,  # handled by OxygenMRSector
        "OXYGEN_SHAKE_GARLIC":         True,
        "OXYGEN_SHAKE_MINT":           True,
        "OXYGEN_SHAKE_MORNING_BREATH": True,
    }
    TREND_HOLD = {
        "OXYGEN_SHAKE_GARLIC": 10,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PANEL SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class PanelSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "PANEL_1X2",
        "PANEL_1X4",
        "PANEL_2X2",
        "PANEL_2X4",
        "PANEL_4X4",
    ]
    ENABLED = {
        "PANEL_1X2": False,
        "PANEL_1X4": True,
        "PANEL_2X2": True,
        "PANEL_2X4": True,
        "PANEL_4X4": True,
    }
    TREND_HOLD = {
        "PANEL_2X4": 6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SNACKPACK SECTOR
#
#  STRAWBERRY  → hold +10  (clean uptrend)
#  CHOCOLATE   → hold -10  (clean downtrend)
#  PISTACHIO   → hold -10  (steepest down)
#  CHOC-PIST spread (corr=0.82) → tilt within shorts
#  RASPBERRY   → aggressive EMA MM  +  RASP-VAN spread overlay
#  VANILLA     → aggressive EMA MM  +  RASP-VAN spread overlay
# ══════════════════════════════════════════════════════════════════════════════

class SnackpackSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ]
    ENABLED = {
        "SNACKPACK_CHOCOLATE":  True,
        "SNACKPACK_PISTACHIO":  True,
        "SNACKPACK_RASPBERRY":  True,
        "SNACKPACK_STRAWBERRY": True,
        "SNACKPACK_VANILLA":    True,
    }
    TREND_HOLD = {}


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATOR SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TranslatorSector(SimpleProductSector):
    ACTIVE = True
    PRODUCTS = [
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ]
    ENABLED = {
        "TRANSLATOR_ASTRO_BLACK":      True,
        "TRANSLATOR_ECLIPSE_CHARCOAL": False,
        "TRANSLATOR_GRAPHITE_MIST":    True,
        "TRANSLATOR_SPACE_GRAY":       True,
        "TRANSLATOR_VOID_BLUE":        True,
    }
    TREND_HOLD = {}


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTIONAL SECTOR  (UV Visor + Translator + selective OxygenShake)
#
#  Three sub-engines share a LIMIT=10 budget per product:
#    (a) Bot-mirror: score counterparties on RED+MAGENTA forward PnL,
#        lean BOT_LOT in the direction of winning bots.
#    (b) Structural always-on bets (STRUCTURAL_BIAS): validated day-over-day
#        drifts from days 2-4; held at sign*DIR_BUDGET unless a winning bot
#        just traded against us within BOT_RECENT_TICKS.
#    (c) ORANGE laggard fade: reactive short when mid < 200-tick rolling avg;
#        cover on bot buy or 1% recovery above avg.
#  Snap-back and pair engines disabled (ACF too weak vs spread cost).
# ══════════════════════════════════════════════════════════════════════════════

class DirectionalSector(Sector):
    ACTIVE = True
    LIMIT  = 10
    TICK   = 100

    VISORS = (
        'UV_VISOR_AMBER', 'UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE',
        'UV_VISOR_RED',   'UV_VISOR_YELLOW',
    )
    DIRECTIONAL_PRODUCTS = VISORS + (
        'TRANSLATOR_VOID_BLUE',
        'TRANSLATOR_SPACE_GRAY',
        'TRANSLATOR_ASTRO_BLACK',
    )

    STRUCTURAL_BIAS = {
        'UV_VISOR_AMBER':       -1,   # -28.7% cumulative drift
        'UV_VISOR_MAGENTA':     +1,   # +15.3%
        'UV_VISOR_RED':         +1,   # +17.2%
        'TRANSLATOR_VOID_BLUE': +1,   # +15.6%
        'TRANSLATOR_SPACE_GRAY':-1,   # -15.7%
        'TRANSLATOR_ASTRO_BLACK':-1,  # -10.4%
    }

    BOT_LEARN_PRODUCTS = ('UV_VISOR_RED', 'UV_VISOR_MAGENTA')
    BOT_FWD_HORIZON    = 20
    BOT_MIN_TRADES     = 20
    BOT_PNL_THRESHOLD  = 1.5
    BOT_STATS_DECAY    = 0.995
    BOT_LOT            = 2
    BOT_RECENT_TICKS   = 100

    DIR_BUDGET = {
        'UV_VISOR_AMBER':        10,
        'UV_VISOR_MAGENTA':      10,
        'UV_VISOR_RED':          10,
        'UV_VISOR_ORANGE':        8,
        'UV_VISOR_YELLOW':        8,
        'TRANSLATOR_VOID_BLUE':  10,
        'TRANSLATOR_SPACE_GRAY': 10,
        'TRANSLATOR_ASTRO_BLACK':10,
    }

    ORANGE_AVG_WINDOW = 200
    MID_HIST_LEN      = 220

    def load_state(self, mem: dict) -> None:
        self._mem = mem

    def save_state(self, mem: dict) -> None:
        pass

    @staticmethod
    def _mid(od: OrderDepth):
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _ema(old, val, alpha):
        return val if old is None else (1.0 - alpha) * old + alpha * val

    @staticmethod
    def _is_real_cp(name):
        return isinstance(name, str) and name and name != 'SUBMISSION'

    def _winning_bots(self, bot_stats):
        winners = set()
        for cp, st in bot_stats.items():
            n_b, pnl_b = st.get('n_buy', 0.0), st.get('pnl_buy', 0.0)
            n_s, pnl_s = st.get('n_sell', 0.0), st.get('pnl_sell', 0.0)
            if n_b >= self.BOT_MIN_TRADES and pnl_b / n_b > self.BOT_PNL_THRESHOLD:
                winners.add((cp, 'buy'))
            if n_s >= self.BOT_MIN_TRADES and pnl_s / n_s > self.BOT_PNL_THRESHOLD:
                winners.add((cp, 'sell'))
        return winners

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        td = self._mem
        result: dict[Symbol, list[Order]] = {}

        # day tracking
        last_ts     = td.get('last_ts', -1)
        day_index   = td.get('day_index', 0)
        day_changed = last_ts > 0 and state.timestamp < last_ts - 500_000
        if day_changed:
            day_index += 1
        gts = day_index * 1_000_000 + state.timestamp
        td['day_index'] = day_index
        td['last_ts']   = state.timestamp

        last_mid        = td.get('last_mid',        {})
        bot_stats       = td.get('bot_stats',       {})
        bot_pending     = td.get('bot_pending',     [])
        mid_hist        = td.get('mid_hist',        {})
        bot_recent_buy  = td.get('bot_recent_buy',  {})
        bot_recent_sell = td.get('bot_recent_sell', {})
        dir_pos         = td.get('dir_pos',         {})

        if day_changed:
            dir_pos = {}

        # 1. Snapshot mids
        mids: dict[str, float] = {}
        for p in self.DIRECTIONAL_PRODUCTS:
            m = self._mid(state.order_depths.get(p))
            if m is not None:
                mids[p] = m

        # capacity tracking
        rem_buy  = {p: self.LIMIT - state.position.get(p, 0) for p in self.DIRECTIONAL_PRODUCTS}
        rem_sell = {p: self.LIMIT + state.position.get(p, 0) for p in self.DIRECTIONAL_PRODUCTS}

        def add(prod, price, qty):
            if qty == 0:
                return
            result.setdefault(prod, []).append(Order(prod, int(price), int(qty)))
            if qty > 0:
                rem_buy[prod]  = max(0, rem_buy[prod]  - qty)
            else:
                rem_sell[prod] = max(0, rem_sell[prod] + qty)

        # 2. Update mid history
        for p in self.DIRECTIONAL_PRODUCTS:
            if p in mids:
                hist = mid_hist.setdefault(p, [])
                hist.append(round(mids[p], 1))
                if len(hist) > self.MID_HIST_LEN:
                    del hist[: len(hist) - self.MID_HIST_LEN]

        # 3. Decay bot stats
        for cp, st in bot_stats.items():
            for k in ('n_buy', 'pnl_buy', 'n_sell', 'pnl_sell'):
                st[k] = st.get(k, 0.0) * self.BOT_STATS_DECAY

        # 4. Resolve pending bot trades (forward-mark PnL on LEARN_PRODUCTS)
        still_pending = []
        for entry in bot_pending:
            cp, side, prod, price, ts0 = entry
            if gts - ts0 < self.BOT_FWD_HORIZON * self.TICK:
                still_pending.append(entry)
                continue
            if prod not in mids or prod not in self.BOT_LEARN_PRODUCTS:
                continue
            fwd = mids[prod] - price if side == 'buy' else price - mids[prod]
            st  = bot_stats.setdefault(cp, {'n_buy': 0.0, 'pnl_buy': 0.0,
                                            'n_sell': 0.0, 'pnl_sell': 0.0})
            if side == 'buy':
                st['n_buy']   += 1.0
                st['pnl_buy'] += fwd
            else:
                st['n_sell']   += 1.0
                st['pnl_sell'] += fwd
        bot_pending = still_pending

        # 5. Ingest market trades; update bot recency
        winners: set = self._winning_bots(bot_stats)
        bot_buy_now:  dict[str, int] = {}
        bot_sell_now: dict[str, int] = {}
        for p in self.DIRECTIONAL_PRODUCTS:
            for t in state.market_trades.get(p, []):
                buyer  = t.buyer  if self._is_real_cp(t.buyer)  else None
                seller = t.seller if self._is_real_cp(t.seller) else None
                if p in self.BOT_LEARN_PRODUCTS:
                    if buyer  is not None: bot_pending.append([buyer,  'buy',  p, t.price, gts])
                    if seller is not None: bot_pending.append([seller, 'sell', p, t.price, gts])
                if buyer  is not None and (buyer,  'buy')  in winners:
                    bot_buy_now[p]    = bot_buy_now.get(p, 0)  + t.quantity
                    bot_recent_buy[p] = gts
                if seller is not None and (seller, 'sell') in winners:
                    bot_sell_now[p]    = bot_sell_now.get(p, 0) + t.quantity
                    bot_recent_sell[p] = gts

        if len(bot_pending) > 4000:
            bot_pending = bot_pending[-4000:]

        # 6. Compute directional targets
        target_dir: dict[str, int] = {p: 0 for p in self.DIRECTIONAL_PRODUCTS}

        # (i) bot mirror
        for p in self.DIRECTIONAL_PRODUCTS:
            budget = self.DIR_BUDGET.get(p, 0)
            if not budget:
                continue
            net = bot_buy_now.get(p, 0) - bot_sell_now.get(p, 0)
            if net > 0:
                target_dir[p] = min( budget,  self.BOT_LOT)
            elif net < 0:
                target_dir[p] = max(-budget, -self.BOT_LOT)

        # (ii) structural always-on bets
        for p, bias in self.STRUCTURAL_BIAS.items():
            if bias > 0:
                contrarian = gts - bot_recent_sell.get(p, -(10**9)) <= self.BOT_RECENT_TICKS * self.TICK
            else:
                contrarian = gts - bot_recent_buy.get(p,  -(10**9)) <= self.BOT_RECENT_TICKS * self.TICK
            if not contrarian:
                target_dir[p] = bias * self.DIR_BUDGET[p]

        # (iii) ORANGE laggard fade
        orange      = 'UV_VISOR_ORANGE'
        orange_hist = mid_hist.get(orange, [])
        if len(orange_hist) >= self.ORANGE_AVG_WINDOW:
            window        = orange_hist[-self.ORANGE_AVG_WINDOW:]
            avg           = sum(window) / len(window)
            cur           = window[-1]
            recent_buy    = gts - bot_recent_buy.get(orange, -(10**9)) <= self.BOT_RECENT_TICKS * self.TICK
            already_short = dir_pos.get(orange, 0) < 0
            if recent_buy or cur > avg * 1.01:
                target_dir[orange] = 0
            elif cur < avg or already_short:
                target_dir[orange] = -self.DIR_BUDGET[orange]

        # 7. Emit orders toward target
        for p in self.DIRECTIONAL_PRODUCTS:
            target  = target_dir[p]
            our_pos = dir_pos.get(p, 0)
            delta   = target - our_pos
            if delta == 0:
                continue
            od = state.order_depths.get(p)
            if od is None:
                continue
            if delta > 0 and od.sell_orders:
                best_ask = min(od.sell_orders)
                avail    = -od.sell_orders[best_ask]
                qty      = min(delta, rem_buy[p], avail)
                if qty > 0:
                    add(p, best_ask, qty)
                    dir_pos[p] = our_pos + qty
            elif delta < 0 and od.buy_orders:
                best_bid = max(od.buy_orders)
                avail    = od.buy_orders[best_bid]
                qty      = min(-delta, rem_sell[p], avail)
                if qty > 0:
                    add(p, best_bid, -qty)
                    dir_pos[p] = our_pos - qty

        # 8. Persist state
        for p, m in mids.items():
            last_mid[p] = m

        td['last_mid']        = last_mid
        td['bot_stats']       = bot_stats
        td['bot_pending']     = bot_pending
        td['mid_hist']        = mid_hist
        td['bot_recent_buy']  = bot_recent_buy
        td['bot_recent_sell'] = bot_recent_sell
        td['dir_pos']         = dir_pos

        return result


# ══════════════════════════════════════════════════════════════════════════════
# PEBBLES SECTOR  (XL/XS pair MR  +  S/M/L standalone MM)
# ══════════════════════════════════════════════════════════════════════════════

class PebblesSector(Sector):
    ACTIVE = True
    # Pair mean-reversion config
    WINDOW   = 500
    ENTRY_Z  = 2.5
    EXIT_Z   = 0.3
    MAX_HOLD = 200
    BG_POS   = 10   # background long-XL / short-XS when flat
    POS_LIMIT = 10

    # Standalone products (not part of the pair strategy)
    STANDALONE: dict[str, bool] = {
        "PEBBLES_S": True,
        "PEBBLES_M": False,
        "PEBBLES_L": True,
    }

    def __init__(self) -> None:
        self.history:   list[int] = []
        self.hold:      int       = 0
        self.direction: int       = 0   # +1=long XL/short XS, -1=reverse, 0=flat

    def load_state(self, mem: dict) -> None:
        self.history   = mem.get("pebbles_history", [])
        self.hold      = mem.get("pebbles_hold",    0)
        self.direction = mem.get("pebbles_dir",     0)

    def save_state(self, mem: dict) -> None:
        mem["pebbles_history"] = self.history
        mem["pebbles_hold"]    = self.hold
        mem["pebbles_dir"]     = self.direction

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        orders: dict[Symbol, list[Order]] = {}

        od_xl = state.order_depths.get("PEBBLES_XL")
        od_xs = state.order_depths.get("PEBBLES_XS")

        if od_xl and od_xs:
            mid_xl = mid_price(od_xl)
            mid_xs = mid_price(od_xs)
            if mid_xl is not None and mid_xs is not None:
                self.history.append(int(mid_xl - mid_xs))
                if len(self.history) > self.WINDOW:
                    self.history = self.history[-self.WINDOW:]

                self._update_signal()

                pos_xl = state.position.get("PEBBLES_XL", 0)
                pos_xs = state.position.get("PEBBLES_XS", 0)

                target_xl = self.POS_LIMIT * self.direction if self.direction != 0 else  self.BG_POS
                target_xs = -self.POS_LIMIT * self.direction if self.direction != 0 else -self.BG_POS

                place_order_to_target("PEBBLES_XL", od_xl, pos_xl, target_xl, orders, self.POS_LIMIT)
                place_order_to_target("PEBBLES_XS", od_xs, pos_xs, target_xs, orders, self.POS_LIMIT)

        for symbol, enabled in self.STANDALONE.items():
            if not enabled:
                continue
            od = state.order_depths.get(symbol)
            if not od:
                continue
            pos = state.position.get(symbol, 0)
            result = market_make(symbol, od, pos, self.POS_LIMIT)
            if result:
                orders[symbol] = result

        return orders

    def _update_signal(self) -> None:
        if len(self.history) < 50:
            return
        mean, std = rolling_mean_std(self.history)
        z = (self.history[-1] - mean) / std

        if self.direction != 0:
            self.hold += 1
            if abs(z) < self.EXIT_Z or self.hold >= self.MAX_HOLD:
                self.direction, self.hold = 0, 0

        if self.direction == 0:
            if z > self.ENTRY_Z + 0.5:
                self.direction, self.hold = -1, 0
            elif z < -self.ENTRY_Z - 0.5:
                self.direction, self.hold = 1, 0

        logger.print(f"pebbles z={z:.2f} dir={self.direction} hold={self.hold}")


# ══════════════════════════════════════════════════════════════════════════════
# ROBOT SECTOR  (Dishes/Ironing pair MR  +  Laundry/Mopping/Vacuuming MM)
#
#  Entry into MR position gated on cs_disp >= p90 (market-wide cross-sectional
#  return std) — only trade when MR edge is historically strongest.
# ══════════════════════════════════════════════════════════════════════════════

class RobotSector(Sector):
    ACTIVE = True
    # Pair mean-reversion config
    WINDOW        = 550
    ENTRY_Z       = 2.2
    EXIT_Z        = 0.5
    MAX_HOLD      = 150
    POS_LIMIT     = 10
    CS_DISP_P90   = 0.001371  # gate: only enter MR when market dispersion is high

    # Standalone products (not part of the pair strategy)
    STANDALONE: dict[str, bool] = {
    }

    def __init__(self) -> None:
        self.history:   list[int] = []
        self.hold:      int       = 0
        self.direction: int       = 0   # +1=long Dishes/short Ironing, -1=reverse, 0=flat

    def load_state(self, mem: dict) -> None:
        self.history   = mem.get("di_history", [])
        self.hold      = mem.get("di_hold",    0)
        self.direction = mem.get("di_dir",     0)
        self._cs_disp  = mem.get("cs_disp",    0.0)

    def save_state(self, mem: dict) -> None:
        mem["di_history"] = self.history
        mem["di_hold"]    = self.hold
        mem["di_dir"]     = self.direction

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        orders: dict[Symbol, list[Order]] = {}

        od_dishes  = state.order_depths.get("ROBOT_DISHES")
        od_ironing = state.order_depths.get("ROBOT_IRONING")

        if od_dishes and od_ironing:
            mid_d = mid_price(od_dishes)
            mid_i = mid_price(od_ironing)
            if mid_d is not None and mid_i is not None:
                self.history.append(int(mid_d - mid_i))
                if len(self.history) > self.WINDOW:
                    self.history = self.history[-self.WINDOW:]

                self._update_signal(self._cs_disp)

                pos_dishes  = state.position.get("ROBOT_DISHES",  0)
                pos_ironing = state.position.get("ROBOT_IRONING", 0)

                if self.direction != 0:
                    place_order_to_target("ROBOT_DISHES",  od_dishes,  pos_dishes,   self.POS_LIMIT * self.direction,  orders, self.POS_LIMIT)
                    place_order_to_target("ROBOT_IRONING", od_ironing, pos_ironing, -self.POS_LIMIT * self.direction, orders, self.POS_LIMIT)
                else:
                    d_orders = market_make("ROBOT_DISHES",  od_dishes,  pos_dishes,  self.POS_LIMIT)
                    i_orders = market_make("ROBOT_IRONING", od_ironing, pos_ironing, self.POS_LIMIT)
                    if d_orders: orders["ROBOT_DISHES"]  = d_orders
                    if i_orders: orders["ROBOT_IRONING"] = i_orders

        for symbol, enabled in self.STANDALONE.items():
            if not enabled:
                continue
            od = state.order_depths.get(symbol)
            if not od:
                continue
            pos = state.position.get(symbol, 0)
            result = market_make(symbol, od, pos, self.POS_LIMIT)
            if result:
                orders[symbol] = result

        return orders

    def _update_signal(self, cs_disp: float) -> None:
        if len(self.history) < 50:
            return
        mean, std = rolling_mean_std(self.history)
        z = (self.history[-1] - mean) / std

        if self.direction != 0:
            self.hold += 1
            if abs(z) < self.EXIT_Z or self.hold >= self.MAX_HOLD:
                self.direction, self.hold = 0, 0

        if self.direction == 0 and cs_disp >= self.CS_DISP_P90:
            if z > self.ENTRY_Z:
                self.direction, self.hold = -1, 0
            elif z < -self.ENTRY_Z:
                self.direction, self.hold = 1, 0


# ══════════════════════════════════════════════════════════════════════════════
# SLEEP POD SECTOR  (CPS core cluster  +  secondary cluster)
# ══════════════════════════════════════════════════════════════════════════════

class SleepPodSector(Sector):
    ACTIVE = True
    # Core cluster (nonlinear deviation-weighted MR)
    CORE_PRODUCTS = ["SLEEP_POD_COTTON", "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"]
    CORE_ENABLED: dict[str, bool] = {
        "SLEEP_POD_COTTON":    True,
        "SLEEP_POD_POLYESTER": True,
        "SLEEP_POD_SUEDE":     True,
    }
    BASE_RISK  = 28
    RISK_MULT  = 2.8
    POWER      = 1.6

    # Secondary cluster (linear deviation MR)
    SECONDARY_PRODUCTS = ["SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON"]
    SECONDARY_ENABLED: dict[str, bool] = {
        "SLEEP_POD_LAMB_WOOL": False,
        "SLEEP_POD_NYLON":     True,
    }
    SECONDARY_SCALE = 10

    POS_LIMIT = 10

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        orders: dict[Symbol, list[Order]] = {}
        self._run_core_cluster(state, orders)
        self._run_secondary_cluster(state, orders)
        return orders

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_mean(products: list[str], order_depths: dict) -> float | None:
        mids = [
            mid_price(od)
            for p in products
            if (od := order_depths.get(p)) is not None
            and mid_price(od) is not None
        ]
        return sum(mids) / len(mids) if mids else None

    def _run_core_cluster(self, state: TradingState, orders_out: dict) -> None:
        factor = self._cluster_mean(self.CORE_PRODUCTS, state.order_depths)
        if factor is None:
            return

        ods  = {p: state.order_depths.get(p) for p in self.CORE_PRODUCTS}
        mids = {p: mid_price(od) for p, od in ods.items() if od and mid_price(od) is not None}
        devs = {p: mids[p] - factor for p in mids}

        if len(devs) < len(self.CORE_PRODUCTS):
            return

        raw_weights = {
            p: (1 if dev >= 0 else -1) * (abs(dev) ** self.POWER)
            for p, dev in devs.items()
        }
        weight_sum = max(sum(abs(w) for w in raw_weights.values()), 1e-9)
        total_risk = self.BASE_RISK * self.RISK_MULT

        for symbol, od_local in ods.items():
            if not od_local or not self.CORE_ENABLED.get(symbol, False):
                continue
            pos = state.position.get(symbol, 0)
            norm_weight = raw_weights[symbol] / weight_sum
            target = int(-norm_weight * total_risk)
            best_bid = max(od_local.buy_orders)
            best_ask = min(od_local.sell_orders)
            price = best_bid if target < 0 else best_ask
            qty = max(-self.POS_LIMIT, min(self.POS_LIMIT, target - pos))
            if qty != 0:
                orders_out[symbol] = [Order(symbol, price, qty)]

    def _run_secondary_cluster(self, state: TradingState, orders_out: dict) -> None:
        factor = self._cluster_mean(self.SECONDARY_PRODUCTS, state.order_depths)
        if factor is None:
            return

        for symbol in self.SECONDARY_PRODUCTS:
            if not self.SECONDARY_ENABLED.get(symbol, False):
                continue
            od = state.order_depths.get(symbol)
            if not od:
                continue
            price_mid = mid_price(od)
            if price_mid is None:
                continue
            deviation = price_mid - factor
            if abs(deviation) < 1:
                continue
            pos    = state.position.get(symbol, 0)
            target = max(-self.POS_LIMIT, min(self.POS_LIMIT, -int(deviation * self.SECONDARY_SCALE)))
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            price = best_bid if target < 0 else best_ask
            qty   = max(-self.POS_LIMIT, min(self.POS_LIMIT, target - pos))
            if qty != 0:
                orders_out[symbol] = [Order(symbol, price, qty)]


# ══════════════════════════════════════════════════════════════════════════════
# OXYGEN MR SECTOR  (CHOCOLATE + EVENING_BREATH individual z-score MR)
#
#  Both products have AC1 = -0.09 to -0.12 (genuine MR).
#  Entry gated on cs_disp >= p90 (market-wide cross-sectional return std)
#  to only trade when MR edge is historically strongest.
# ══════════════════════════════════════════════════════════════════════════════

class OxygenMRSector(Sector):
    ACTIVE      = True
    CS_DISP_P90 = 0.001371  # p90 of cs_disp from 3-day history
    WINDOW      = 400
    ENTRY_Z     = 2.2
    EXIT_Z      = 0.5
    MAX_HOLD    = 150
    POS_LIMIT   = 10

    MR_PRODUCTS = ["OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH"]
    # Short mem keys to keep traderData JSON compact
    _HKEYS = {"OXYGEN_SHAKE_CHOCOLATE": "omrh0", "OXYGEN_SHAKE_EVENING_BREATH": "omrh1"}
    _KKEYS = {"OXYGEN_SHAKE_CHOCOLATE": "omrk0", "OXYGEN_SHAKE_EVENING_BREATH": "omrk1"}
    _DKEYS = {"OXYGEN_SHAKE_CHOCOLATE": "omrd0", "OXYGEN_SHAKE_EVENING_BREATH": "omrd1"}

    def load_state(self, mem: dict) -> None:
        self._cs_disp = mem.get("cs_disp", 0.0)
        self._history = {p: mem.get(self._HKEYS[p], []) for p in self.MR_PRODUCTS}
        self._hold    = {p: mem.get(self._KKEYS[p],  0) for p in self.MR_PRODUCTS}
        self._dir     = {p: mem.get(self._DKEYS[p],  0) for p in self.MR_PRODUCTS}

    def save_state(self, mem: dict) -> None:
        for p in self.MR_PRODUCTS:
            mem[self._HKEYS[p]] = self._history[p]
            mem[self._KKEYS[p]] = self._hold[p]
            mem[self._DKEYS[p]] = self._dir[p]

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        result: dict[Symbol, list[Order]] = {}
        cs_disp = self._cs_disp

        for p in self.MR_PRODUCTS:
            od = state.order_depths.get(p)
            if not od:
                continue
            m = mid_price(od)
            if m is None:
                continue

            pos  = state.position.get(p, 0)
            hist = self._history[p]
            hist.append(int(m))
            if len(hist) > self.WINDOW:
                hist[:] = hist[-self.WINDOW:]

            direction = self._dir[p]
            hold      = self._hold[p]

            if len(hist) >= 50:
                mean, std = rolling_mean_std(hist)
                z = (hist[-1] - mean) / std

                if direction != 0:
                    hold += 1
                    if abs(z) < self.EXIT_Z or hold >= self.MAX_HOLD:
                        direction = 0
                        hold = 0

                if direction == 0 and cs_disp >= self.CS_DISP_P90:
                    if z > self.ENTRY_Z:
                        direction = -1
                        hold = 0
                    elif z < -self.ENTRY_Z:
                        direction = 1
                        hold = 0

            self._dir[p]  = direction
            self._hold[p] = hold

            if direction != 0:
                target = self.POS_LIMIT * direction
                place_order_to_target(p, od, pos, target, result, self.POS_LIMIT)
            else:
                orders = market_make(p, od, pos, self.POS_LIMIT)
                if orders:
                    result[p] = orders

        return result


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CLUSTER SECTOR
#
#  Four products from different families share a common uptrend.
#  Strategy: hold +5 static long on each (trend component) + use remaining
#  ±5 capacity to MR against the instantaneous cluster mean (4-product avg).
#  If any product's spread vs cluster exceeds THRESHOLD (1.5x historical max),
#  fall back to pure MM for that product until spread normalises.
# ══════════════════════════════════════════════════════════════════════════════

class CrossClusterSector(Sector):
    ACTIVE = True
    PRODUCTS = [
        "SNACKPACK_RASPBERRY",
        "UV_VISOR_RED",
        "ROBOT_DISHES",
        "SNACKPACK_STRAWBERRY",
    ]
    HOLD      = 5      # static long on each product
    MR_BUDGET = 5      # position reserved for cluster MR
    POS_LIMIT = 10
    THRESHOLD = 2000   # 1.5x max spread seen across 3 days (max was 1326)
    ENTRY     = 400    # raw spread to enter MR trade (~1.2 std)
    EXIT      = 100    # raw spread to exit MR trade

    def load_state(self, mem: dict) -> None:
        self._mem = mem
        self._mr: dict[str, int] = {p: mem.get(f"cc_{p}", 0) for p in self.PRODUCTS}

    def save_state(self, mem: dict) -> None:
        for p in self.PRODUCTS:
            mem[f"cc_{p}"] = self._mr[p]

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        result: dict[Symbol, list[Order]] = {}

        mids: dict[str, float] = {}
        for p in self.PRODUCTS:
            od = state.order_depths.get(p)
            if od:
                m = mid_price(od)
                if m is not None:
                    mids[p] = m
        if len(mids) < 4:
            return result

        cluster = sum(mids.values()) / len(mids)

        for p in self.PRODUCTS:
            if p not in mids:
                continue
            od = state.order_depths.get(p)
            if not od:
                continue
            pos   = state.position.get(p, 0)
            spread = mids[p] - cluster

            if abs(spread) >= self.THRESHOLD:
                orders = market_make(p, od, pos, self.POS_LIMIT)
                if orders:
                    result[p] = orders
                self._mr[p] = 0
                continue

            mr = self._mr[p]
            if spread > self.ENTRY:
                mr = -self.MR_BUDGET
            elif spread < -self.ENTRY:
                mr = +self.MR_BUDGET
            elif abs(spread) < self.EXIT:
                mr = 0
            self._mr[p] = mr

            target = max(-self.POS_LIMIT, min(self.POS_LIMIT, self.HOLD + mr))
            place_order_to_target(p, od, pos, target, result, self.POS_LIMIT)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# TRADER — AGGREGATES ALL SECTORS
# ══════════════════════════════════════════════════════════════════════════════

class Trader:
    """
    Instantiates one Sector per product family.
    Each tick: load state → run → save state → merge orders.
    Adding a new product family = add a new Sector subclass to self.sectors.
    """

    def __init__(self) -> None:
        self.sectors: list[Sector] = [
            GalaxySoundsSector(),
            MicrochipSector(),
            OxygenShakeSector(),
            OxygenMRSector(),
            PanelSector(),
            SnackpackSector(),
            TranslatorSector(),
            DirectionalSector(),
            PebblesSector(),
            RobotSector(),
            SleepPodSector(),
            CrossClusterSector(),
        ]

    @staticmethod
    def _compute_cs_disp(prev_mids: dict, curr_mids: dict) -> float:
        log_rets = [
            log(curr_mids[s] / prev_mids[s])
            for s in curr_mids
            if s in prev_mids and prev_mids[s] > 0
        ]
        if len(log_rets) < 10:
            return 0.0
        n = len(log_rets)
        mean_r = sum(log_rets) / n
        var_r  = sum((r - mean_r) ** 2 for r in log_rets) / n
        return sqrt(var_r)

    def run(
        self, state: TradingState
    ) -> tuple[dict[Symbol, list[Order]], int, str]:

        result:      dict[Symbol, list[Order]] = {}
        conversions: int = 0

        mem = json.loads(state.traderData) if state.traderData else {}

        # Compute cross-sectional dispersion (std of log-returns across all products).
        # Stored in mem["cs_disp"] so MR sectors can gate new entries on it.
        curr_mids      = {s: mid_price(od) for s, od in state.order_depths.items()
                          if mid_price(od) is not None}
        mem["cs_disp"] = self._compute_cs_disp(mem.get("apm", {}), curr_mids)
        mem["apm"]     = curr_mids

        for sector in self.sectors:
            if not sector.ACTIVE:
                continue
            sector.load_state(mem)
            orders = sector.run(state)
            result.update(orders)
            sector.save_state(mem)

        trader_data = json.dumps(mem, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
