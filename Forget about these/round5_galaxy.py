from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""), self.compress_orders(orders),
            conversions, "", "",
        ]))
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
                state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations):
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                           obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
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

# ── Products ──────────────────────────────────────────────────────────────────
BH      = "GALAXY_SOUNDS_BLACK_HOLES"
OTHERS  = [
    "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "GALAXY_SOUNDS_SOLAR_WINDS",
]
PRODUCTS = [BH] + OTHERS

# ── Hyperparameters ───────────────────────────────────────────────────────────
POS_LIMIT    = 10

# ── Black Holes: 70% buy-and-hold, 30% MM ────────────────────────────────────
BH_ALLOC  = 7     # max units for directional buy-and-hold
MM_ALLOC  = 3     # units reserved for MM on top / below the BH floor

# Graduated entry: scale in as price confirms the uptrend
BH_TIER1       = 3      # initial position on trend confirmation
BH_TIER2       = 5      # scale-up when price up BH_TIER1_GAIN from entry
BH_TIER3       = 7      # full position when price up BH_TIER2_GAIN from entry
BH_TIER1_GAIN  = 0.005  # +0.5% from entry to add tier 2
BH_TIER2_GAIN  = 0.010  # +1.0% from entry to add tier 3

# Entry condition: OLS trend slope
BH_TREND_WINDOW = 500
BH_SLOPE_MIN    = 0.002

# Stop-loss
BH_TRAIL_PCT  = 0.020   # trailing stop: exit if price < HWM * (1 - TRAIL_PCT)
BH_STOP1_PCT  = 0.015   # hard stop 1: price < entry * (1 - STOP1) → reduce to tier 1
BH_STOP2_PCT  = 0.030   # hard stop 2: price < entry * (1 - STOP2) → close all

# ── Market making (all products) ──────────────────────────────────────────────
EMA_FAST         = 30     # ticks for fair-value EMA
INVENTORY_K      = 0.2    # base inventory skew per excess unit
INVENTORY_K_AMPL = 1.5    # amplifier when fair-value confirms reducing position

# ── Loss-streak cooldown (per product) ───────────────────────────────────────
LOSS_WINDOW    = 100       # ticks to measure MTM streak
LOSS_THRESHOLD = -20      # cumulative MTM loss to trigger cooldown
COOLDOWN_TICKS = 20       # ticks in defensive mode
# ─────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def ols_slope(prices: list, window: int) -> float:
    w = prices[-window:]
    n = len(w)
    if n < 10:
        return 0.0
    sx    = n * (n - 1) / 2
    sx2   = n * (n - 1) * (2 * n - 1) / 6
    sy    = sum(w)
    sxy   = sum(i * w[i] for i in range(n))
    denom = n * sx2 - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def microprice(od: OrderDepth) -> float:
    bid     = max(od.buy_orders)
    ask     = min(od.sell_orders)
    bid_vol = od.buy_orders[bid]
    ask_vol = abs(od.sell_orders[ask])
    total   = bid_vol + ask_vol
    if total == 0:
        return (bid + ask) / 2
    return (bid * ask_vol + ask * bid_vol) / total


# ── Black Holes buy-and-hold state machine ────────────────────────────────────

def bh_update(mid: float, prices: list,
              bh_tgt: int, entry: float, hwm: float
              ) -> tuple[int, float, float]:
    """
    Update BH state machine each tick.

    Returns: (new_target, new_entry, new_hwm)
      new_target = 0..BH_ALLOC

    Stop conditions (checked first):
      Trailing stop  : price < hwm  * (1 - BH_TRAIL_PCT) → close all
      Hard stop 2    : price < entry * (1 - BH_STOP2_PCT) → close all
      Hard stop 1    : price < entry * (1 - BH_STOP1_PCT) → reduce to tier 1

    Scale-in (only while in position without a fresh stop):
      Tier 2 unlock  : price > entry * (1 + BH_TIER1_GAIN)
      Tier 3 unlock  : price > entry * (1 + BH_TIER2_GAIN)

    Entry condition (when flat):
      OLS slope over BH_TREND_WINDOW > BH_SLOPE_MIN
    """

    # ── In a position: update HWM then check stops ────────────────────────────
    if bh_tgt > 0:
        hwm = max(hwm, mid)

        trail_stop = hwm  * (1 - BH_TRAIL_PCT)
        hard_stop2 = entry * (1 - BH_STOP2_PCT)
        hard_stop1 = entry * (1 - BH_STOP1_PCT)

        if mid < trail_stop or mid < hard_stop2:
            logger.print(f"BH full stop  mid={mid:.0f} entry={entry:.0f} hwm={hwm:.0f}")
            return 0, 0.0, 0.0

        if mid < hard_stop1 and bh_tgt > BH_TIER1:
            logger.print(f"BH reduce T1  mid={mid:.0f}")
            return BH_TIER1, entry, hwm

        # Scale in
        if bh_tgt < BH_TIER3 and mid > entry * (1 + BH_TIER2_GAIN):
            return BH_TIER3, entry, hwm
        if bh_tgt < BH_TIER2 and mid > entry * (1 + BH_TIER1_GAIN):
            return BH_TIER2, entry, hwm

        return bh_tgt, entry, hwm

    # ── Flat: check for entry ─────────────────────────────────────────────────
    slope = ols_slope(prices, BH_TREND_WINDOW)
    if slope > BH_SLOPE_MIN:
        logger.print(f"BH enter T1  mid={mid:.0f} slope={slope:.4f}")
        return BH_TIER1, mid, mid   # entry and hwm = current price

    return 0, 0.0, 0.0


def place_taker(symbol: str, od: OrderDepth, pos: int, target: int) -> tuple[list[Order], int]:
    """Taker order toward target. Returns (orders, delta_placed)."""
    orders = []
    delta  = target - pos
    filled = 0
    if delta > 0 and od.sell_orders:
        best_ask = min(od.sell_orders)
        qty      = min(delta, abs(od.sell_orders[best_ask]))
        if qty > 0:
            orders.append(Order(symbol, best_ask, qty))
            filled = qty
    elif delta < 0 and od.buy_orders:
        best_bid = max(od.buy_orders)
        qty      = min(-delta, od.buy_orders[best_bid])
        if qty > 0:
            orders.append(Order(symbol, best_bid, -qty))
            filled = -qty
    return orders, filled


# ── Market making ─────────────────────────────────────────────────────────────

def market_make(symbol: str, od: OrderDepth, pos: int,
                ema: float, cooldown: int,
                sell_floor: int = 0) -> list[Order]:
    """
    Reservation-price MM with directional penny-jumping and inventory skew.

    sell_floor: minimum position we're allowed to sell down to (protects BH core).
    cooldown  : if > 0, quote passively (no penny-jump) with halved size cap.
    """
    orders: list[Order] = []

    if not od.buy_orders or not od.sell_orders:
        return orders

    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    spread   = best_ask - best_bid

    if spread < 2:
        return orders

    mid = (best_bid + best_ask) / 2

    # ── Lean: EMA fair-value (adverse-selection filter) ───────────────────────
    if   mid < ema: lean =  1   # price below fair value → bullish lean
    elif mid > ema: lean = -1   # price above fair value → bearish lean
    else:           lean =  0

    # ── Inventory skew ────────────────────────────────────────────────────────
    excess      = pos - sell_floor   # excess above the BH floor
    k_effective = INVENTORY_K
    if excess > 0 and mid > ema:     # long excess + price elevated → amplify sell
        k_effective *= INVENTORY_K_AMPL
    elif excess < 0 and mid < ema:   # short vs floor + price depressed → amplify buy
        k_effective *= INVENTORY_K_AMPL
    inv_shift = int(round(k_effective * (pos - 0)))  # skew toward neutral (0)

    # ── Quotes ────────────────────────────────────────────────────────────────
    if cooldown > 0:
        bid_price = best_bid   # no improvement during cooldown
        ask_price = best_ask
    else:
        bid_price = (best_bid + 1) if lean >= 0 else best_bid
        ask_price = (best_ask - 1) if lean <= 0 else best_ask

    bid_price -= inv_shift
    ask_price -= inv_shift

    bid_price = max(best_bid, min(bid_price, best_ask - 1))
    ask_price = max(best_bid + 1, min(ask_price, best_ask))

    if bid_price >= ask_price:
        return orders

    size_cap = (POS_LIMIT // 2) if cooldown > 0 else POS_LIMIT
    buy_cap  = max(0, min(size_cap - pos,  POS_LIMIT - pos))
    sell_cap = max(0, min(size_cap + pos,  pos - sell_floor))   # respect BH floor

    if buy_cap  > 0: orders.append(Order(symbol, bid_price,  buy_cap))
    if sell_cap > 0: orders.append(Order(symbol, ask_price, -sell_cap))

    return orders


# ── Trader ────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        mem   = json.loads(state.traderData) if state.traderData else {}
        alpha = 2.0 / (EMA_FAST + 1)

        # ── Black Holes ───────────────────────────────────────────────────────
        od_bh = state.order_depths.get(BH)
        if od_bh and od_bh.buy_orders and od_bh.sell_orders:
            mid_bh = (max(od_bh.buy_orders) + min(od_bh.sell_orders)) / 2
            pos_bh = state.position.get(BH, 0)

            # EMA and streak tracking
            ema_bh   = mem.get("bhe", mid_bh)
            prev_bh  = mem.get("bhm", mid_bh)
            pnl_bh   = mem.get("bhl", [])
            cd_bh    = mem.get("bhc", 0)
            ema_bh   = alpha * mid_bh + (1 - alpha) * ema_bh

            mtm_bh = pos_bh * (mid_bh - prev_bh)
            pnl_bh.append(mtm_bh)
            if len(pnl_bh) > LOSS_WINDOW: pnl_bh = pnl_bh[-LOSS_WINDOW:]
            if cd_bh > 0:
                cd_bh -= 1
            elif len(pnl_bh) == LOSS_WINDOW and sum(pnl_bh) < LOSS_THRESHOLD:
                cd_bh = COOLDOWN_TICKS
                logger.print(f"BH cooldown (streak={sum(pnl_bh):.1f})")

            # OLS history
            bh_prices = mem.get("bhp", [])
            bh_prices.append(int(mid_bh))
            if len(bh_prices) > BH_TREND_WINDOW + 10:
                bh_prices = bh_prices[-(BH_TREND_WINDOW + 10):]

            # BH state machine
            bh_tgt = mem.get("bht", 0)
            entry  = mem.get("bhn", 0.0)   # entry price
            hwm    = mem.get("bhw", 0.0)   # high-water mark

            bh_tgt, entry, hwm = bh_update(mid_bh, bh_prices, bh_tgt, entry, hwm)

            # Orders: taker for BH target, then MM with remaining capacity
            bh_orders, filled = place_taker(BH, od_bh, pos_bh, bh_tgt)
            proj_pos    = pos_bh + filled
            # BH floor for MM: allow MM to sell up to MM_ALLOC units below BH target
            bh_floor    = max(0, bh_tgt - MM_ALLOC)
            mm_orders   = market_make(BH, od_bh, proj_pos, ema_bh, cd_bh,
                                      sell_floor=bh_floor)

            all_bh = bh_orders + mm_orders
            if all_bh:
                result[BH] = all_bh

            mem["bhe"] = ema_bh
            mem["bhm"] = mid_bh
            mem["bhl"] = pnl_bh
            mem["bhc"] = cd_bh
            mem["bhp"] = bh_prices
            mem["bht"] = bh_tgt
            mem["bhn"] = entry
            mem["bhw"] = hwm

        # ── Other Galaxy Sounds: pure MM ──────────────────────────────────────
        for symbol in OTHERS:
            od = state.order_depths.get(symbol)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
            pos = state.position.get(symbol, 0)

            k        = symbol[-3:]
            ema      = mem.get(k + "e", mid)
            prev_mid = mem.get(k + "m", mid)
            pnl_hist = mem.get(k + "l", [])
            cooldown = mem.get(k + "c", 0)

            ema = alpha * mid + (1 - alpha) * ema

            mtm = pos * (mid - prev_mid)
            pnl_hist.append(mtm)
            if len(pnl_hist) > LOSS_WINDOW: pnl_hist = pnl_hist[-LOSS_WINDOW:]
            if cooldown > 0:
                cooldown -= 1
            elif len(pnl_hist) == LOSS_WINDOW and sum(pnl_hist) < LOSS_THRESHOLD:
                cooldown = COOLDOWN_TICKS
                logger.print(f"{k} cooldown (streak={sum(pnl_hist):.1f})")

            orders = market_make(symbol, od, pos, ema, cooldown, sell_floor=0)
            if orders:
                result[symbol] = orders

            mem[k + "e"] = ema
            mem[k + "m"] = mid
            mem[k + "l"] = pnl_hist
            mem[k + "c"] = cooldown

        trader_data = json.dumps(mem, separators=(",", ":"))
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
