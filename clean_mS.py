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
from math import sqrt
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
    if pos < hold_size:
        orders.append(Order(symbol, best_ask, hold_size - pos))
    if spread > 0:
        our_bid     = best_bid + 1 if spread >= 2 else best_bid
        mm_buy_cap  = pos_limit - max(pos, hold_size)
        mm_sell_cap = max(0, pos - hold_size)
        if mm_buy_cap  > 0: orders.append(Order(symbol, our_bid,   mm_buy_cap))
        if mm_sell_cap > 0: orders.append(Order(symbol, best_ask, -mm_sell_cap))
    return orders


POS_LIMIT = 10


def get_mid(od):
    if od and od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2
    return None


def send_order(result, symbol, price, qty):
    if qty == 0:
        return
    result.setdefault(symbol, []).append(Order(symbol, int(price), int(qty)))


def update_ema(prev, val, span):
    a = 2.0 / (span + 1)
    return a * val + (1 - a) * prev


def zscore(series):
    if len(series) < 10:
        return 0.0
    m, s = rolling_mean_std(series)
    return (series[-1] - m) / s


def bb(od): return max(od.buy_orders)
def ba(od): return min(od.sell_orders)


def hold_directional(symbol, od, pos, target, result):
    if not od.buy_orders or not od.sell_orders:
        return
    delta = target - pos
    if delta == 0:
        return
    if delta > 0:
        qty = min(delta, POS_LIMIT - pos)
        send_order(result, symbol, ba(od), qty)
    else:
        qty = max(delta, -(POS_LIMIT + pos))
        send_order(result, symbol, bb(od), qty)


def ema_mm(symbol, od, pos, ema, span, thresh, result):
    if not od.buy_orders or not od.sell_orders:
        return ema
    best_b, best_a = bb(od), ba(od)
    mid = (best_b + best_a) / 2
    ema = update_ema(ema, mid, span)
    dev = mid - ema

    our_bid = int(ema) - 1
    our_ask = int(ema) + 2

    if dev > thresh:
        skew = min(int(dev / thresh) * 2, 5)
        our_ask -= skew; our_bid -= skew
    elif dev < -thresh:
        skew = min(int(-dev / thresh) * 2, 5)
        our_bid += skew; our_ask += skew

    our_bid = min(our_bid, best_b)
    our_ask = max(our_ask, best_a)
    if our_bid >= our_ask:
        return ema

    buy_cap  = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos
    if buy_cap  > 0: send_order(result, symbol, our_bid,  buy_cap)
    if sell_cap > 0: send_order(result, symbol, our_ask, -sell_cap)
    return ema


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
    ACTIVE = False
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
        "GALAXY_SOUNDS_PLANETARY_RINGS": True,
        "GALAXY_SOUNDS_SOLAR_FLAMES":    False,
        "GALAXY_SOUNDS_SOLAR_WINDS":     True,
    }
    TREND_HOLD = {
        "GALAXY_SOUNDS_PLANETARY_RINGS": 7,
        "GALAXY_SOUNDS_BLACK_HOLES":     6,
        "GALAXY_SOUNDS_DARK_MATTER":     5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MICROCHIP SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class MicrochipSector(SimpleProductSector):
    ACTIVE = False
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
        "MICROCHIP_SQUARE": 9,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OXYGEN SHAKE SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class OxygenShakeSector(SimpleProductSector):
    ACTIVE = False
    PRODUCTS = [
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ]
    ENABLED = {
        "OXYGEN_SHAKE_CHOCOLATE":      True,
        "OXYGEN_SHAKE_EVENING_BREATH": True,
        "OXYGEN_SHAKE_GARLIC":         True,
        "OXYGEN_SHAKE_MINT":           True,
        "OXYGEN_SHAKE_MORNING_BREATH": True,
    }
    TREND_HOLD = {
        "OXYGEN_SHAKE_GARLIC": 6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PANEL SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class PanelSector(SimpleProductSector):
    ACTIVE = False
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
        "PANEL_4X4": False,
    }
    TREND_HOLD = {
        "PANEL_2X4": 6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SNACKPACK SECTOR
#
#  STRAWBERRY  → hold +10  (clean uptrend)
#  CHOCOLATE   → hold -10  (clean downtrend)
#  PISTACHIO   → steepest down  → hold -10
#  CHOC-PIST spread (corr=0.82) → tilt within shorts
#  RASPBERRY   → aggressive EMA MM  +  RASP-VAN spread overlay
#  VANILLA     → aggressive EMA MM  +  RASP-VAN spread overlay
# ══════════════════════════════════════════════════════════════════════════════

CP_WINDOW = 200; CP_ENTRY_Z = 1.4; CP_EXIT_Z = 0.25
RV_WINDOW = 150; RV_ENTRY_Z = 1.2; RV_EXIT_Z = 0.25


def snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result):
    mc, mp = get_mid(od_c), get_mid(od_p)
    if mc is None or mp is None:
        return
    cp_hist.append(mc - mp)
    if len(cp_hist) > CP_WINDOW: cp_hist[:] = cp_hist[-CP_WINDOW:]
    if len(cp_hist) < 20:
        return
    z = zscore(cp_hist)
    tgt_c = -7 if z > CP_ENTRY_Z else (-POS_LIMIT if z < -CP_ENTRY_Z else -POS_LIMIT)
    tgt_p = -POS_LIMIT if z > CP_ENTRY_Z else (-7 if z < -CP_ENTRY_Z else -POS_LIMIT)
    hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, tgt_c, result)
    hold_directional("SNACKPACK_PISTACHIO",  od_p, pos_p, tgt_p, result)


def snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result):
    mr, mv = get_mid(od_r), get_mid(od_v)
    if mr is None or mv is None:
        return
    rv_hist.append(mr - mv)
    if len(rv_hist) > RV_WINDOW: rv_hist[:] = rv_hist[-RV_WINDOW:]
    if len(rv_hist) < 20:
        return
    z = zscore(rv_hist)
    PAIR = 3
    if z > RV_ENTRY_Z:
        if pos_r > -PAIR: send_order(result, "SNACKPACK_RASPBERRY", bb(od_r), -(pos_r + PAIR))
        if pos_v < PAIR:  send_order(result, "SNACKPACK_VANILLA",   ba(od_v),  PAIR - pos_v)
    elif z < -RV_ENTRY_Z:
        if pos_v > -PAIR: send_order(result, "SNACKPACK_VANILLA",   bb(od_v), -(pos_v + PAIR))
        if pos_r < PAIR:  send_order(result, "SNACKPACK_RASPBERRY", ba(od_r),  PAIR - pos_r)
    elif abs(z) < RV_EXIT_Z:
        if pos_r != 0:
            send_order(result, "SNACKPACK_RASPBERRY", bb(od_r) if pos_r > 0 else ba(od_r), -pos_r)
        if pos_v != 0:
            send_order(result, "SNACKPACK_VANILLA",   bb(od_v) if pos_v > 0 else ba(od_v), -pos_v)


def run_snackpack(state: TradingState, mem: dict, result: dict):
    ods, positions = state.order_depths, state.position
    emas    = mem.get("snk_ema", {})
    cp_hist = mem.get("snk_cp",  [])
    rv_hist = mem.get("snk_rv",  [])

    od_c = ods.get("SNACKPACK_CHOCOLATE")
    od_p = ods.get("SNACKPACK_PISTACHIO")
    od_r = ods.get("SNACKPACK_RASPBERRY")
    od_s = ods.get("SNACKPACK_STRAWBERRY")
    od_v = ods.get("SNACKPACK_VANILLA")

    pos_c = positions.get("SNACKPACK_CHOCOLATE",  0)
    pos_p = positions.get("SNACKPACK_PISTACHIO",  0)
    pos_r = positions.get("SNACKPACK_RASPBERRY",  0)
    pos_s = positions.get("SNACKPACK_STRAWBERRY", 0)
    pos_v = positions.get("SNACKPACK_VANILLA",    0)

    if od_s: hold_directional("SNACKPACK_STRAWBERRY", od_s, pos_s, +POS_LIMIT, result)

    if od_c and od_p:
        snack_cp_spread(od_c, od_p, pos_c, pos_p, cp_hist, result)
    else:
        if od_c: hold_directional("SNACKPACK_CHOCOLATE", od_c, pos_c, -POS_LIMIT, result)
        if od_p: hold_directional("SNACKPACK_PISTACHIO",  od_p, pos_p, -POS_LIMIT, result)

    if od_r:
        mid_r = get_mid(od_r)
        if mid_r:
            emas.setdefault("SNACKPACK_RASPBERRY", mid_r)
            emas["SNACKPACK_RASPBERRY"] = ema_mm(
                "SNACKPACK_RASPBERRY", od_r, pos_r, emas["SNACKPACK_RASPBERRY"], 200, 30, result)

    if od_v:
        mid_v = get_mid(od_v)
        if mid_v:
            emas.setdefault("SNACKPACK_VANILLA", mid_v)
            emas["SNACKPACK_VANILLA"] = ema_mm(
                "SNACKPACK_VANILLA", od_v, pos_v, emas["SNACKPACK_VANILLA"], 200, 38, result)

    if od_r and od_v:
        snack_rv_spread(od_r, od_v, pos_r, pos_v, rv_hist, result)

    mem["snk_ema"] = emas
    mem["snk_cp"]  = cp_hist
    mem["snk_rv"]  = rv_hist


class SnackpackSector(Sector):
    ACTIVE = True

    def load_state(self, mem: dict) -> None:
        self._mem = mem

    def save_state(self, mem: dict) -> None:
        pass

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        result: dict[Symbol, list[Order]] = {}
        run_snackpack(state, self._mem, result)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# TRANSLATOR SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TranslatorSector(SimpleProductSector):
    ACTIVE = False
    PRODUCTS = [
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ]
    ENABLED = {
        "TRANSLATOR_ASTRO_BLACK":      True,
        "TRANSLATOR_ECLIPSE_CHARCOAL": True,
        "TRANSLATOR_GRAPHITE_MIST":    False,
        "TRANSLATOR_SPACE_GRAY":       False,
        "TRANSLATOR_VOID_BLUE":        True,
    }
    TREND_HOLD = {}


# ══════════════════════════════════════════════════════════════════════════════
# UV VISOR SECTOR
# ══════════════════════════════════════════════════════════════════════════════

class UVVisorSector(SimpleProductSector):
    ACTIVE = False
    PRODUCTS = [
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_YELLOW",
    ]
    ENABLED = {
        "UV_VISOR_AMBER":   True,
        "UV_VISOR_MAGENTA": True,
        "UV_VISOR_ORANGE":  True,
        "UV_VISOR_RED":     True,
        "UV_VISOR_YELLOW":  True,
    }
    TREND_HOLD = {
        "UV_VISOR_MAGENTA": 6,
        "UV_VISOR_YELLOW":  6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PEBBLES SECTOR  (XL/XS pair MR  +  S/M/L standalone MM)
# ══════════════════════════════════════════════════════════════════════════════

class PebblesSector(Sector):
    ACTIVE = False
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
        "PEBBLES_L": False,
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
# ROBOT SECTOR
#
#  MOPPING    → strong uptrend  → hold +10
#  DISHES     → uptrend + strongest MR (ac1=-0.23)  → hold +10
#  LAUNDRY    → mild uptrend  → hold +7
#  VACUUMING  → mild downtrend  → hold -6
#  IRONING    → strong downtrend  → hold -10
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_TARGETS = {
    "ROBOT_MOPPING":   +10,
    "ROBOT_DISHES":    +10,
    "ROBOT_LAUNDRY":   +7,
    "ROBOT_VACUUMING": -6,
    "ROBOT_IRONING":   -10,
}

DI_WINDOW = 300; DI_ENTRY_Z = 1.5; DI_EXIT_Z = 0.3


def run_robots(state: TradingState, mem: dict, result: dict):
    ods, positions = state.order_depths, state.position
    emas    = mem.get("rob_ema", {})
    di_hist = mem.get("rob_di",  [])

    od_d = ods.get("ROBOT_DISHES")
    od_i = ods.get("ROBOT_IRONING")
    md = get_mid(od_d)
    mi = get_mid(od_i)

    if od_d and od_i and md and mi:
        di_hist.append(md - mi)
        if len(di_hist) > DI_WINDOW: di_hist[:] = di_hist[-DI_WINDOW:]

    for symbol, target in ROBOT_TARGETS.items():
        od  = ods.get(symbol)
        pos = positions.get(symbol, 0)
        if od is None: continue
        hold_directional(symbol, od, pos, target, result)

    if od_d and md:
        emas.setdefault("ROBOT_DISHES", md)
        emas["ROBOT_DISHES"] = update_ema(emas["ROBOT_DISHES"], md, 100)
        dev = md - emas["ROBOT_DISHES"]
        pos_d = positions.get("ROBOT_DISHES", 0)
        if dev > 80 and pos_d > 7:
            send_order(result, "ROBOT_DISHES", bb(od_d), -(pos_d - 7))
        elif dev < -80 and pos_d < POS_LIMIT:
            send_order(result, "ROBOT_DISHES", ba(od_d), POS_LIMIT - pos_d)

    mem["rob_ema"] = emas
    mem["rob_di"]  = di_hist


class RobotSector(Sector):
    ACTIVE = True

    def load_state(self, mem: dict) -> None:
        self._mem = mem

    def save_state(self, mem: dict) -> None:
        pass

    def run(self, state: TradingState) -> dict[Symbol, list[Order]]:
        result: dict[Symbol, list[Order]] = {}
        run_robots(state, self._mem, result)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SLEEP POD SECTOR  (CPS core cluster  +  secondary cluster)
# ══════════════════════════════════════════════════════════════════════════════

class SleepPodSector(Sector):
    ACTIVE = False
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
            PanelSector(),
            SnackpackSector(),
            TranslatorSector(),
            UVVisorSector(),
            PebblesSector(),
            RobotSector(),
            SleepPodSector(),
        ]

    def run(
        self, state: TradingState
    ) -> tuple[dict[Symbol, list[Order]], int, str]:

        result:      dict[Symbol, list[Order]] = {}
        conversions: int = 0

        mem = json.loads(state.traderData) if state.traderData else {}

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
