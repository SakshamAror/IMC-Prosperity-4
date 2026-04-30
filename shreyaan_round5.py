"""
Round 5 — pure directional / bot-mirror trader.
UV_VISOR + OXYGEN_SHAKE + TRANSLATOR. Snap-back and pair engines disabled.

Edge source: structural daily drifts (validated on
prices_round_5_day_{2,3,4}.csv). Every family is fundamentally trending
day-over-day; the lag-1 ACF signal is too weak to beat spread cost so
mean-reversion / pair / market-making layers were stripped out.

  1. Round-5 directional layer (the ONLY active layer).
     Round 5 exposes counterparty IDs in state.market_trades.{buyer,seller}.
     Three sub-engines share the LIMIT=10 budget per product.

       (a) Online bot-mirror.
           Score every counterparty by forward mark-to-market PnL at
           +20 ticks on RED + MAGENTA trades only (cleanest training
           signal — both have positive drift across days 2-4). A bot
           is "winning" if avg fwd-PnL > BOT_PNL_THRESHOLD on at least
           BOT_MIN_TRADES samples. When a winning bot trades any of
           our directional products we lean ±BOT_LOT in their
           direction, capped by DIR_BUDGET[p].
           NOTE: the historical CSVs in round5/days2-4 have empty
           buyer/seller fields, so this engine produces no signal in
           local backtest. It activates only on live submission data.

       (b) Structural always-on bets (STRUCTURAL_BIAS).
           High-conviction (full DIR_BUDGET=10):
             AMBER          short: -28.7% cumulative drift days 2-4
             MAGENTA        long:  +15.3%
             RED            long:  +17.2%
             GARLIC         long:  +38.9%
             VOID_BLUE      long:  +15.6% (10000 → 11564)
             SPACE_GRAY     short: -15.7% (10000 → 8429)
             ASTRO_BLACK    short: -10.4% (10000 → 8964)
           Lower-conviction "best direction" OXY bets (DIR_BUDGET=5):
             CHOC  long:  +7.1% cumulative; 1 big up, 2 small down
             EB    short: -5.8%; 1 big down, 2 mixed
             MB    short: -4.5%; 1 big down, 2 mixed
           Held at sign(bias) * DIR_BUDGET unconditionally; only override
           is a winning bot trading against us in the last
           BOT_RECENT_TICKS (the market's earliest reversal warning).
           No rolling-avg / momentum safety filter (an earlier version
           churned ~340 round-trips per day on normal price noise).

       (c) ORANGE laggard fade.
           ORANGE has no consistent direction day-over-day so we trade
           it reactively. If current mid is below the 200-tick rolling
           average, short up to DIR_BUDGET[ORANGE]. Once short, hold
           (no active cover) until either a winning bot buys ORANGE or
           mid pushes >1% above the avg (genuine recovery, not noise).

     Coordination: only one layer active, but dir_pos[p] is still kept
     as an INTENTION ledger in traderData so the engine sizes orders
     by its own delta (not by raw state.position) — a guard against
     re-introducing snap-back/pair later.

What we DON'T do here:

  * No naive market-making. Spreads are 4-7 ticks but mid moves >5 every tick
    is common, so quoting at fixed mid +/- h is a constant-bleed strategy
    (sim showed -150k to -360k per product).
  * No snap-back on ANY product. The 1-tick reverter that originally
    targeted TRANSLATORs (ac1 ≈ 0.0) lost ~150k on the 3-day backtest;
    the same engine on VISORs / OXY families lost ~169k. ACF too weak
    relative to spread.
  * No pair-spread reversion. The TRANSLATOR pairs (ASTRO/GRAPHITE,
    ECLIPSE/VOID_BLUE) had statistically reasonable spread MR but each
    leg's directional drift dragged the pair into directional bias and
    fought the new structural bets.
  * No structural bet on YELLOW (no consistent daily direction:
    +15.7%, +4.0%, -16.5%), on TRANSLATOR_ECLIPSE_CHARCOAL (-2.8%
    cumulative — < 1 spread of edge), on TRANSLATOR_GRAPHITE_MIST
    (-2.1% — same), or on the SNACKPACK / PEBBLES / ROBOT families.

Position math (LIMIT = 10 per product, snap-back & pair empty):

  product                         directional   total
  -----------------------------   -----------   -----
  UV_VISOR_AMBER                      10  (-)     10
  UV_VISOR_MAGENTA                    10  (+)     10
  UV_VISOR_RED                        10  (+)     10
  UV_VISOR_ORANGE                      8  (-/0)    8
  UV_VISOR_YELLOW                      8  (mirr)   8
  OXYGEN_SHAKE_GARLIC                 10  (+)     10
  OXYGEN_SHAKE_CHOCOLATE               5  (+)      5
  OXYGEN_SHAKE_EVENING_BREATH          5  (-)      5
  OXYGEN_SHAKE_MORNING_BREATH          5  (-)      5
  TRANSLATOR_VOID_BLUE                10  (+)     10
  TRANSLATOR_SPACE_GRAY               10  (-)     10
  TRANSLATOR_ASTRO_BLACK              10  (-)     10

  Direction column: (+) long, (-) short, (-/0) reactive short, (mirr)
  bot-mirror only. All within LIMIT=10.
"""

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import json


# ── Logger (unchanged from round5_claudia.py) ────────────────────────────────

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([self.compress_state(state, ""), self.compress_orders(orders), conversions, "", ""]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations):
        co = {}
        for product, o in observations.conversionObservations.items():
            co[product] = [o.bidPrice, o.askPrice, o.transportFees,
                           o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
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


# ── Strategy ──────────────────────────────────────────────────────────────────

LIMIT = 10
TICK = 100  # IMC timestamp granularity


class Trader:

    # How fast the |dm| sigma estimate reacts to a new tick (~50-tick half-life)
    SIGMA_ALPHA = 0.02
    # How fast the pair-spread mean drifts (~700-tick half-life ≈ 1/14 of a day)
    SPREAD_ALPHA = 0.001

    # ── Single-tick mean-reversion ("snap-back") ─────────────────────────────
    # COMPLETELY EMPTY now. All product families show structural directional
    # drifts that snap-back fights, paying spread on every fade. The earlier
    # iterations migrated VISORs and OXY products to the directional engine,
    # cutting -180k+ of losses. TRANSLATORs follow the same pattern:
    # ALL 5 have lag-1 ACF ≈ 0 (random walk in returns) but strong daily
    # drifts of opposite signs per product. Snap-back was losing ~150k
    # across the 3-day backtest. Migrated to the directional engine
    # (VOID_BLUE long, SPACE_GRAY short, ASTRO_BLACK short).
    # (k_in_sigma, hold_ticks, max_pos, lot, sigma_init)
    REVERT_CFG = {}

    # ── Pair-spread reversion (intra-family) ─────────────────────────────────
    # COMPLETELY EMPTY. The TRANSLATOR pairs (ASTRO/GRAPHITE,
    # ECLIPSE/VOID_BLUE) appeared promising on intra-day spread mean-
    # reversion stats, but the per-product drifts are so strong and
    # opposite-signed that the pair quickly accumulates a directional
    # bias which then fights the new TRANSLATOR structural bets.
    # Cleaner to drop pairs entirely and rely on directional bets.
    # (a, b, k_in, k_out, lot, max_hold_ticks, sigma_init)
    PAIR_CFG = []

    # ── Round-5 directional layer ────────────────────────────────────────────
    VISORS = ('UV_VISOR_AMBER', 'UV_VISOR_MAGENTA', 'UV_VISOR_ORANGE',
              'UV_VISOR_RED',   'UV_VISOR_YELLOW')

    # All products handled by the directional engine. VISORs + 4 OXY shakes
    # + 3 TRANSLATORs. Used by every sub-loop in the directional layer (mid
    # history, market-trade ingestion, target initialisation, order emission).
    DIRECTIONAL_PRODUCTS = VISORS + (
        'OXYGEN_SHAKE_GARLIC',
        'OXYGEN_SHAKE_CHOCOLATE',
        'OXYGEN_SHAKE_EVENING_BREATH',
        'OXYGEN_SHAKE_MORNING_BREATH',
        'TRANSLATOR_VOID_BLUE',
        'TRANSLATOR_SPACE_GRAY',
        'TRANSLATOR_ASTRO_BLACK',
    )

    # Persistent directional bias (signed unit; multiplied by DIR_BUDGET).
    # +1 = always-long unless winning bot SELLS the product recently.
    # -1 = always-short unless winning bot BUYS the product recently.
    # ORANGE is *not* in here — it's reactive (fade vs rolling avg), see (iii).
    # YELLOW is also out — no consistent direction days 2-4 (+15.7, +4.0, -16.5).
    # MINT is skipped (cumulative +156 over 3 days is rounding noise).
    #
    # The 3 "noisy" OXY structural bets (CHOC/EB/MB) win 1-2 of 3 days each
    # but their best-direction net PnL across all 3 days is positive. They
    # carry more variance than GARLIC/VISOR bets (one big losing day each)
    # so they get HALF budget — see DIR_BUDGET below.
    STRUCTURAL_BIAS = {
        'UV_VISOR_AMBER':                 -1,   # -28.7% cumulative drift
        'UV_VISOR_MAGENTA':               +1,   # +15.3% cumulative drift
        'UV_VISOR_RED':                   +1,   # +17.2% cumulative drift
        'OXYGEN_SHAKE_GARLIC':            +1,   # +38.9% cumulative drift
        'OXYGEN_SHAKE_CHOCOLATE':         +1,   # +7.1% (1 big up day, 2 small down)
        'OXYGEN_SHAKE_EVENING_BREATH':    -1,   # -5.8% (1 big down day, 2 mixed)
        'OXYGEN_SHAKE_MORNING_BREATH':    -1,   # -4.5% (1 big down day, 2 mixed)
        # ── TRANSLATORs ──────────────────────────────────────────────────
        # ECLIPSE_CHARCOAL (-2.8%) and GRAPHITE_MIST (-2.1%) are EXCLUDED:
        # their 3-day cumulative drift is < 1 spread, so we'd be paying
        # more in spread than collecting in drift. Only the 3 strong ones
        # are bet; the 2 weak ones are simply not traded by any layer.
        'TRANSLATOR_VOID_BLUE':           +1,   # +15.6% (10000→11564)
        'TRANSLATOR_SPACE_GRAY':          -1,   # -15.7% (10000→8429)
        'TRANSLATOR_ASTRO_BLACK':         -1,   # -10.4% (10000→8964)
    }

    # Products we use to *score* counterparties. RED + MAGENTA both trended
    # up across days 2-4, so any bot persistently buying them is on the
    # winning side; persistent sellers are losing. Cleanest training signal.
    BOT_LEARN_PRODUCTS = ('UV_VISOR_RED', 'UV_VISOR_MAGENTA')

    # Forward horizon (ticks) over which we mark a counterparty's trade to
    # market. 20 ticks ≈ same scale as our snap-back hold; long enough that
    # noise washes out, short enough to stay relevant within a single day.
    BOT_FWD_HORIZON = 20

    # Min trades on the LEARN_PRODUCTS before we trust a counterparty's
    # win-rate. Below this, we ignore them — guards against random-walk
    # bots that look profitable from a handful of lucky fills.
    BOT_MIN_TRADES = 20

    # Avg fwd-PnL threshold (in price-ticks per trade) above which a
    # counterparty counts as "winning". RED/MAGENTA daily moves are 50-200
    # ticks, so demanding +1.5 over a 20-tick horizon screens out noise
    # without being so strict we never see anyone qualify.
    BOT_PNL_THRESHOLD = 1.5

    # Cap on lookback for bot stats. We exponentially decay older samples
    # via a fixed alpha rather than a hard window — newer trades dominate
    # so a previously-good bot that turns bad gets dropped quickly.
    BOT_STATS_DECAY = 0.995  # half-life ≈ 138 trades

    # Per-product directional budget: maximum |position| the directional
    # layer may hold. Full LIMIT=10 for high-conviction always-on bets;
    # half (5) for the noisier OXY shakes that lose on at least one day
    # of the 3-day backtest sample (CHOC/EB/MB).
    DIR_BUDGET = {
        'UV_VISOR_AMBER':                 10,
        'UV_VISOR_MAGENTA':               10,
        'UV_VISOR_RED':                   10,
        'UV_VISOR_ORANGE':                 8,
        'UV_VISOR_YELLOW':                 8,
        'OXYGEN_SHAKE_GARLIC':            10,   # consistent +drift, full size
        'OXYGEN_SHAKE_CHOCOLATE':          5,   # noisier, half size
        'OXYGEN_SHAKE_EVENING_BREATH':     5,   # noisier, half size
        'OXYGEN_SHAKE_MORNING_BREATH':     5,   # noisier, half size
        # TRANSLATORs: 2/3 winning days each at +10u, with daily moves of
        # 500-1700 ticks. Net ≈ +10-15k per product, full size justified.
        'TRANSLATOR_VOID_BLUE':           10,   # +15.6% drift, full size
        'TRANSLATOR_SPACE_GRAY':          10,   # -15.7% drift, full size
        'TRANSLATOR_ASTRO_BLACK':         10,   # -10.4% drift, full size
    }

    # Bot-mirror lot per fired signal (cap at this regardless of bot size).
    BOT_LOT = 2

    # ORANGE laggard fade params.
    ORANGE_AVG_WINDOW    = 200    # rolling-mean window in ticks (=200 mids)
    ORANGE_Z_THRESHOLD   = 1.0    # z-score below which we open the short

    # Mid-history retention. We need 200 for ORANGE, plus a bit of cushion.
    MID_HIST_LEN = 220

    # Bot-mirror "recency" — a winning bot's buy-side conviction on AMBER
    # is treated as relevant for this many ticks (overrides the structural
    # short during that window).
    BOT_RECENT_TICKS = 100

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mid(od: OrderDepth):
        if od and od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def _ema(old, val, alpha):
        return val if old is None else (1.0 - alpha) * old + alpha * val

    def _all_products(self):
        s = set(self.REVERT_CFG.keys())
        for cfg in self.PAIR_CFG:
            s.add(cfg[0]); s.add(cfg[1])
        # Every directional product (VISORs + GARLIC) needs its mid tracked
        # in step 1, even those with no other engine.
        s.update(self.DIRECTIONAL_PRODUCTS)
        return s

    # ── Round-5 directional helpers ───────────────────────────────────────────

    @staticmethod
    def _is_real_cp(name):
        # Counterparty must be a non-empty string that is not our own marker.
        # Backtester emits "SUBMISSION" for self-trades and "" for anonymous
        # market trades on data without IDs (round 5 days 2-4 sample CSV).
        return isinstance(name, str) and name and name != 'SUBMISSION'

    def _winning_bots(self, bot_stats):
        # A counterparty wins if its decayed avg fwd-PnL on RED+MAGENTA
        # is above threshold and they have enough samples. Returns a set
        # of (cp, side) where side ∈ {'buy', 'sell'} indicating which
        # direction is profitable for that bot — we only mirror the
        # profitable direction, not every trade they do.
        winners = set()
        for cp, st in bot_stats.items():
            n_b   = st.get('n_buy',  0.0)
            pnl_b = st.get('pnl_buy', 0.0)
            n_s   = st.get('n_sell', 0.0)
            pnl_s = st.get('pnl_sell', 0.0)
            if n_b >= self.BOT_MIN_TRADES and pnl_b / n_b > self.BOT_PNL_THRESHOLD:
                winners.add((cp, 'buy'))
            if n_s >= self.BOT_MIN_TRADES and pnl_s / n_s > self.BOT_PNL_THRESHOLD:
                winners.add((cp, 'sell'))
        return winners

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        td = json.loads(state.traderData) if state.traderData else {}
        result: dict[Symbol, list[Order]] = {}

        # ── day index + global timestamp (continuous across days) ────────────
        last_ts   = td.get('last_ts', -1)
        day_index = td.get('day_index', 0)
        day_changed = (last_ts > 0 and state.timestamp < last_ts - 500_000)
        if day_changed:
            day_index += 1
        gts = day_index * 1_000_000 + state.timestamp
        td['day_index'] = day_index
        td['last_ts']   = state.timestamp

        last_mid     = td.get('last_mid', {})
        sig_ema      = td.get('sig_ema', {})
        rev_open_ts  = td.get('rev_open_ts', {})       # product -> open gts
        spread_ema   = td.get('spread_ema', {})        # str(idx) -> ema
        spread_sig   = td.get('spread_sig', {})        # str(idx) -> sigma
        pair_state   = td.get('pair_state', {})        # str(idx) -> {pos, open_ts}

        # ── Round-5 directional state ────────────────────────────────────────
        # bot_stats: counterparty -> {n_buy, pnl_buy, n_sell, pnl_sell}, all
        #   decayed-EMA style (sums get multiplied by BOT_STATS_DECAY each tick).
        # bot_pending: list of [cp, side, product, price, gts] awaiting
        #   forward-PnL evaluation at gts + BOT_FWD_HORIZON*TICK.
        # mid_hist: VISOR product -> rolling list of mids (last MID_HIST_LEN).
        # bot_recent_buy / bot_recent_sell: VISOR -> last gts a winning bot
        #   bought / sold that VISOR (used to override structural shorts).
        bot_stats        = td.get('bot_stats', {})
        bot_pending      = td.get('bot_pending', [])
        mid_hist         = td.get('mid_hist', {})
        bot_recent_buy   = td.get('bot_recent_buy', {})
        bot_recent_sell  = td.get('bot_recent_sell', {})
        # dir_pos: how much of state.position[VISOR] is "ours" (the
        # directional layer's). Tracked separately so we don't accidentally
        # try to flatten the snap-back / pair engines' positions when our
        # own target happens to be 0. Day-resets like other position state.
        dir_pos          = td.get('dir_pos', {})

        # If a new day started, our previous-day position trackers are
        # invalidated (state.position resets). Drop them.
        # Bot stats / mid_hist are intentionally KEPT across days — bot
        # behaviour is expected to persist, and price history adds context.
        if day_changed:
            rev_open_ts = {}
            pair_state  = {}
            dir_pos     = {}

        all_prods = self._all_products()

        # ── 1. Snapshot mids and update |dm| sigma ────────────────────────────
        mids: dict[str, float] = {}
        for p in all_prods:
            od = state.order_depths.get(p)
            m = self._mid(od)
            if m is not None:
                mids[p] = m
        for p, m in mids.items():
            prev = last_mid.get(p)
            if prev is not None:
                sig_init = self.REVERT_CFG.get(p, (None, None, None, None, 10.0))[4] or 10.0
                sig_ema[p] = self._ema(sig_ema.get(p), abs(m - prev), self.SIGMA_ALPHA) \
                             if sig_ema.get(p) is not None else sig_init

        # ── capacity tracking shared between engines ──────────────────────────
        rem_buy  = {p: LIMIT - state.position.get(p, 0) for p in all_prods}
        rem_sell = {p: LIMIT + state.position.get(p, 0) for p in all_prods}

        def add(prod, price, qty):
            if qty == 0:
                return
            result.setdefault(prod, []).append(Order(prod, int(price), int(qty)))
            if qty > 0:
                rem_buy[prod] = max(0, rem_buy[prod] - qty)
            else:
                rem_sell[prod] = max(0, rem_sell[prod] + qty)

        # ── 2. Single-tick snap-back per product ──────────────────────────────
        for p, (k_in, hold, max_pos, lot, sig_init) in self.REVERT_CFG.items():
            od = state.order_depths.get(p)
            if od is None or p not in mids:
                continue
            if not od.buy_orders or not od.sell_orders:
                continue
            prev = last_mid.get(p)
            if prev is None:
                continue
            sigma = sig_ema.get(p, sig_init) or sig_init
            if sigma <= 0:
                continue
            pos = state.position.get(p, 0)
            dm  = mids[p] - prev
            z   = dm / sigma

            # ── Hard exit: 'hold' ticks have elapsed since opening ────────────
            opened = rev_open_ts.get(p)
            if opened is not None and pos != 0 and gts - opened >= hold * TICK:
                if pos > 0 and rem_sell[p] > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders)
                    avail    = od.buy_orders[best_bid]
                    qty = min(pos, rem_sell[p], avail)
                    if qty > 0:
                        add(p, best_bid, -qty)
                elif pos < 0 and rem_buy[p] > 0 and od.sell_orders:
                    best_ask = min(od.sell_orders)
                    avail    = -od.sell_orders[best_ask]
                    qty = min(-pos, rem_buy[p], avail)
                    if qty > 0:
                        add(p, best_ask, qty)
                rev_open_ts.pop(p, None)
                # Don't simultaneously open a fresh leg — wait one more tick.
                continue

            # ── New entry on a big jump (in opposite direction) ───────────────
            if z > k_in and pos > -max_pos and od.buy_orders:
                best_bid = max(od.buy_orders)
                avail    = od.buy_orders[best_bid]
                qty = min(lot, max_pos + pos, rem_sell[p], avail)
                if qty > 0:
                    add(p, best_bid, -qty)
                    rev_open_ts[p] = gts
            elif z < -k_in and pos < max_pos and od.sell_orders:
                best_ask = min(od.sell_orders)
                avail    = -od.sell_orders[best_ask]
                qty = min(lot, max_pos - pos, rem_buy[p], avail)
                if qty > 0:
                    add(p, best_ask, qty)
                    rev_open_ts[p] = gts

        # ── 3. Pair-spread reversion ──────────────────────────────────────────
        for idx, (a, b, k_in, k_out, lot, max_hold, sigma_init) in enumerate(self.PAIR_CFG):
            key = str(idx)
            if a not in mids or b not in mids:
                continue
            od_a = state.order_depths.get(a); od_b = state.order_depths.get(b)
            if not (od_a and od_b and od_a.buy_orders and od_a.sell_orders
                    and od_b.buy_orders and od_b.sell_orders):
                continue

            sp = mids[a] - mids[b]
            e  = spread_ema.get(key)
            if e is None:
                # bootstrap: one tick of warmup before trading this pair
                spread_ema[key] = sp
                spread_sig[key] = sigma_init
                continue
            sigma = max(spread_sig.get(key, sigma_init), 1.0)
            z = (sp - e) / sigma
            spread_ema[key] = (1.0 - self.SPREAD_ALPHA) * e + self.SPREAD_ALPHA * sp
            spread_sig[key] = (1.0 - self.SIGMA_ALPHA)  * sigma + self.SIGMA_ALPHA  * abs(sp - e)

            ps = pair_state.get(key, {'pos': 0, 'open_ts': 0})

            # ── Close existing leg ───────────────────────────────────────────
            if ps['pos'] != 0:
                expired = gts - ps['open_ts'] >= max_hold * TICK
                if abs(z) < k_out or expired:
                    pos_a = state.position.get(a, 0)
                    pos_b = state.position.get(b, 0)
                    if ps['pos'] > 0:
                        # we're long a / short b → sell a, buy b
                        qa = min(lot, pos_a, rem_sell[a],
                                 od_a.buy_orders[max(od_a.buy_orders)] if od_a.buy_orders else 0)
                        qb = min(lot, -pos_b, rem_buy[b],
                                 -od_b.sell_orders[min(od_b.sell_orders)] if od_b.sell_orders else 0)
                        qty = min(qa, qb)
                        if qty > 0:
                            add(a, max(od_a.buy_orders),  -qty)
                            add(b, min(od_b.sell_orders),  qty)
                    else:
                        qa = min(lot, -pos_a, rem_buy[a],
                                 -od_a.sell_orders[min(od_a.sell_orders)] if od_a.sell_orders else 0)
                        qb = min(lot, pos_b, rem_sell[b],
                                 od_b.buy_orders[max(od_b.buy_orders)] if od_b.buy_orders else 0)
                        qty = min(qa, qb)
                        if qty > 0:
                            add(a, min(od_a.sell_orders),  qty)
                            add(b, max(od_b.buy_orders), -qty)
                    pair_state[key] = {'pos': 0, 'open_ts': 0}

            # ── Open fresh leg (only if currently flat) ──────────────────────
            if pair_state.get(key, {'pos': 0})['pos'] == 0:
                if z > k_in:
                    # spread too high → short a, long b
                    avail_a = od_a.buy_orders[max(od_a.buy_orders)]
                    avail_b = -od_b.sell_orders[min(od_b.sell_orders)]
                    qty = min(lot, rem_sell[a], rem_buy[b], avail_a, avail_b)
                    if qty > 0:
                        add(a, max(od_a.buy_orders),  -qty)
                        add(b, min(od_b.sell_orders),  qty)
                        pair_state[key] = {'pos': -1, 'open_ts': gts}
                elif z < -k_in:
                    avail_a = -od_a.sell_orders[min(od_a.sell_orders)]
                    avail_b = od_b.buy_orders[max(od_b.buy_orders)]
                    qty = min(lot, rem_buy[a], rem_sell[b], avail_a, avail_b)
                    if qty > 0:
                        add(a, min(od_a.sell_orders),  qty)
                        add(b, max(od_b.buy_orders), -qty)
                        pair_state[key] = {'pos': 1, 'open_ts': gts}

        # ── 4. Round-5 directional layer (VISORs + GARLIC) ────────────────────

        # 4a. Update mid history per directional product (used by ORANGE
        #     laggard fade and to give bot-mirror context).
        for p in self.DIRECTIONAL_PRODUCTS:
            if p in mids:
                hist = mid_hist.setdefault(p, [])
                hist.append(round(mids[p], 1))
                if len(hist) > self.MID_HIST_LEN:
                    del hist[: len(hist) - self.MID_HIST_LEN]

        # 4b. Decay existing bot stats one tick (so a once-good bot fades
        #     out if it stops trading or starts losing).
        for cp, st in bot_stats.items():
            for k in ('n_buy', 'pnl_buy', 'n_sell', 'pnl_sell'):
                st[k] = st.get(k, 0.0) * self.BOT_STATS_DECAY

        # 4c. Resolve pending bot trades that are old enough to mark.
        #     For each, compute forward-mark PnL using current mid (which
        #     is gts ticks later than the trade), and add to that
        #     counterparty's score on LEARN_PRODUCTS only.
        still_pending = []
        for entry in bot_pending:
            cp, side, prod, price, ts0 = entry
            if gts - ts0 < self.BOT_FWD_HORIZON * TICK:
                still_pending.append(entry)
                continue
            if prod not in mids:
                # Price feed gap — drop the entry rather than carry forever.
                continue
            if prod not in self.BOT_LEARN_PRODUCTS:
                # Not used for scoring; nothing to do (we only learn from
                # RED/MAGENTA per design).
                continue
            fwd = mids[prod] - price if side == 'buy' else price - mids[prod]
            st = bot_stats.setdefault(cp, {'n_buy': 0.0, 'pnl_buy': 0.0,
                                            'n_sell': 0.0, 'pnl_sell': 0.0})
            if side == 'buy':
                st['n_buy']   = st.get('n_buy', 0.0)   + 1.0
                st['pnl_buy'] = st.get('pnl_buy', 0.0) + fwd
            else:
                st['n_sell']   = st.get('n_sell', 0.0)   + 1.0
                st['pnl_sell'] = st.get('pnl_sell', 0.0) + fwd
        bot_pending = still_pending

        # 4d. Ingest current-tick market trades on every directional product.
        #     For LEARN_PRODUCTS we queue them for later forward-mark; on
        #     every directional product we record bot recency so winners
        #     can be mirrored on any of our positions (incl. GARLIC).
        winners = self._winning_bots(bot_stats)
        # Flag products where a winning bot just bought / sold this tick.
        bot_buy_now  : dict[str, int] = {}  # product -> qty bought by winners
        bot_sell_now : dict[str, int] = {}
        for p in self.DIRECTIONAL_PRODUCTS:
            for t in state.market_trades.get(p, []):
                # The trade itself happened at t.timestamp; we read it now.
                buyer  = t.buyer if self._is_real_cp(t.buyer)  else None
                seller = t.seller if self._is_real_cp(t.seller) else None
                # Queue for forward-mark scoring (LEARN_PRODUCTS only).
                if p in self.BOT_LEARN_PRODUCTS:
                    if buyer is not None:
                        bot_pending.append([buyer,  'buy',  p, t.price, gts])
                    if seller is not None:
                        bot_pending.append([seller, 'sell', p, t.price, gts])
                # Mirror logic: if this trade involves a known winner, lean.
                if buyer is not None and (buyer, 'buy') in winners:
                    bot_buy_now[p]  = bot_buy_now.get(p, 0)  + t.quantity
                    bot_recent_buy[p] = gts
                if seller is not None and (seller, 'sell') in winners:
                    bot_sell_now[p] = bot_sell_now.get(p, 0) + t.quantity
                    bot_recent_sell[p] = gts

        # Cap pending list size — defensive against runaway growth on
        # high-trade-volume days. 4000 entries ≈ ~150 KB JSON, well
        # within traderData budget if we ever approach it.
        if len(bot_pending) > 4000:
            bot_pending = bot_pending[-4000:]

        # 4e. Compute target directional position per product.
        target_dir: dict[str, int] = {p: 0 for p in self.DIRECTIONAL_PRODUCTS}

        # (i) Bot mirror — if a winning bot just traded one of our products
        #     this tick, lean BOT_LOT in their direction (capped by budget).
        for p in self.DIRECTIONAL_PRODUCTS:
            budget = self.DIR_BUDGET[p]
            net = bot_buy_now.get(p, 0) - bot_sell_now.get(p, 0)
            if net > 0:
                target_dir[p] = min(budget, self.BOT_LOT)
            elif net < 0:
                target_dir[p] = max(-budget, -self.BOT_LOT)

        # (ii) Structural always-on bets per STRUCTURAL_BIAS.
        #      Each product is held at sign(bias) * DIR_BUDGET as long as no
        #      winning bot has traded *against* us recently:
        #        - long bias (+1):  cover if winning bot SOLD recently
        #        - short bias (-1): cover if winning bot BOUGHT recently
        #      No rolling-avg / momentum filter — those caused ~340 target
        #      flips/day on noise during earlier iterations.
        for p, bias in self.STRUCTURAL_BIAS.items():
            if bias > 0:
                contrarian = (gts - bot_recent_sell.get(p, -10**9)
                              <= self.BOT_RECENT_TICKS * TICK)
            else:
                contrarian = (gts - bot_recent_buy.get(p, -10**9)
                              <= self.BOT_RECENT_TICKS * TICK)
            if not contrarian:
                target_dir[p] = bias * self.DIR_BUDGET[p]

        # (iii) ORANGE laggard fade.
        #      ORANGE has no consistent direction day-over-day (day 4 it
        #      collapsed -9.2%, days 2-3 slightly up). Trade it reactively:
        #      short when mid drops below the 200-tick rolling avg. Cover
        #      when mid recovers >1% above avg, or a winning bot buys.
        orange = 'UV_VISOR_ORANGE'
        orange_hist = mid_hist.get(orange, [])
        if len(orange_hist) >= self.ORANGE_AVG_WINDOW:
            window = orange_hist[-self.ORANGE_AVG_WINDOW:]
            avg = sum(window) / len(window)
            cur = window[-1]
            recent_orange_buy = (gts - bot_recent_buy.get(orange, -10**9)
                                  <= self.BOT_RECENT_TICKS * TICK)
            already_short = dir_pos.get(orange, 0) < 0
            big_recovery = cur > avg * 1.01
            if recent_orange_buy or big_recovery:
                target_dir[orange] = 0
            elif cur < avg or already_short:
                target_dir[orange] = -self.DIR_BUDGET[orange]

        # 4f. Emit orders to drive *our* contribution toward target_dir.
        #     We track dir_pos[p] as an INTENTION ledger for the directional
        #     layer only; snap-back / pair maintain their own positions in
        #     state.position. Delta is computed against our own ledger so we
        #     never try to flatten another engine's leg.
        for p in self.DIRECTIONAL_PRODUCTS:
            target = target_dir[p]
            our_pos = dir_pos.get(p, 0)
            delta   = target - our_pos
            if delta == 0:
                continue
            od = state.order_depths.get(p)
            if od is None:
                continue
            if delta > 0 and od.sell_orders:
                # Need to BUY |delta| more — lift best ask, capped by
                # remaining buy headroom and available volume at that level.
                best_ask  = min(od.sell_orders)
                avail     = -od.sell_orders[best_ask]
                qty       = min(delta, rem_buy[p], avail)
                if qty > 0:
                    add(p, best_ask, qty)
                    dir_pos[p] = our_pos + qty
            elif delta < 0 and od.buy_orders:
                # Need to SELL |delta| more — hit best bid.
                best_bid  = max(od.buy_orders)
                avail     = od.buy_orders[best_bid]
                qty       = min(-delta, rem_sell[p], avail)
                if qty > 0:
                    add(p, best_bid, -qty)
                    dir_pos[p] = our_pos - qty

        # ── 5. Persist state ──────────────────────────────────────────────────
        for p, m in mids.items():
            last_mid[p] = m

        td['last_mid']        = last_mid
        td['sig_ema']         = sig_ema
        td['rev_open_ts']     = rev_open_ts
        td['spread_ema']      = spread_ema
        td['spread_sig']      = spread_sig
        td['pair_state']      = pair_state
        td['bot_stats']       = bot_stats
        td['bot_pending']     = bot_pending
        td['mid_hist']        = mid_hist
        td['bot_recent_buy']  = bot_recent_buy
        td['bot_recent_sell'] = bot_recent_sell
        td['dir_pos']         = dir_pos

        trader_data_str = json.dumps(td)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str
