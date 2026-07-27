"""
Backtest for the liquidity-sweep / CISD / IFVG model.

Mirrors the TradingView indicator's logic exactly, but charges spread on every
trade and can run over hundreds of days — the two things the on-chart results
table cannot do.

Run:  python ict.py XAUUSD 180 0.30 15      symbol, days, spread, timeframe(min)
      python ict.py XAUUSD 180 0.30 15 --tp=pool      target the next pool
      python ict.py XAUUSD 180 0.30 15 --tp=range --adr=1.3
      python ict.py XAUUSD 365 0.30 15 --sweep        grid search, 60/40 split
      python ict.py XAUUSD 180 0.30 15 --manual       ignore the timeframe preset

Settings default to the same per-timeframe ladder the Pine indicator applies on
chart, so a backtest measures what you are actually looking at. --manual opts
out and uses the bare IctParams defaults.

The model:
  SWEEP        price takes out a prior swing; each pool is used once
  CISD         close back through the open of the first candle of the last
               opposing run — delivery has flipped
  REVERSAL     swept a high, then turned down -> short
  CONTINUATION swept a high and HELD above it -> long
  IFVG         optional: require price inside an inverted fair value gap

Targets (`tp_mode`) — the stop is always structural, from the sweep itself:
  atr    entry +/- risk * rr_mult. Blind to the chart: it demands the same
         multiple whether or not there is anywhere for price to go.
  pool   the next untouched opposing liquidity pool. If the chart offers less
         than `min_rr` of room to it, the trade is declined rather than taken
         with a target invented out of thin air.
  range  the high/low of the last 24h — the right target in a rotation.

`max_adr_used` is the impact filter: how far the last 24h has already travelled
against a normal day. Above ~1.3 the session has spent its range and a fresh
2R demand is unlikely to be paid.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace

import numpy as np

from config import AppConfig, load_config
from mt5_connector import MT5Connector

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

TF_MAP = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4"}


@dataclass
class IctParams:
    pivot_len: int = 8
    max_pools: int = 6
    require_close_back: bool = False
    arm_bars: int = 12
    trade_reversal: bool = True
    trade_continuation: bool = True
    min_run_len: int = 2
    use_ifvg: bool = False
    min_fvg_atr: float = 0.25
    sl_buf_atr: float = 0.25
    max_risk_atr: float = 1.5
    # max_risk_atr only ever tightens the stop. Without a floor, a continuation
    # entry taken right at the break sits a hair from its anchor and gets a stop
    # inside one candle of noise — right direction, stopped out anyway.
    min_risk_atr: float = 0.5
    rr_mult: float = 2.0
    cooldown_bars: int = 5

    # Where the target comes from:
    #   atr   — entry +/- risk * rr_mult. Blind to structure; asks for the same
    #           multiple whether or not the chart has room to deliver it.
    #   pool  — the next untouched opposing liquidity pool. Price is drawn to
    #           resting stops, so that level is where the move is actually going.
    #   range — the high/low of the last day. Correct target in a rotation.
    tp_mode: str = "atr"
    tp_buf_atr: float = 0.10      # take profit just BEFORE the level, not at it
    min_rr: float = 1.0           # structural target closer than this -> no trade
    # ...and a ceiling. Risk is capped at max_risk_atr, so leaving reward
    # unbounded lets the nearest unswept pool sit a whole day's range away while
    # the stop sits inside one bar of noise. Those resolve on different clocks
    # and the stop always wins. Clamp the target to something reachable.
    max_struct_rr: float = 5.0

    # "How big is the impact": how stretched the last 24h already is against a
    # normal day. 0 disables. 1.3 means skip when the day has already run 30%
    # further than usual, because the fuel for another leg is mostly spent.
    adr_days: int = 14
    max_adr_used: float = 0.0

    def label(self) -> str:
        mode = ("rev+cont" if self.trade_reversal and self.trade_continuation
                else "rev" if self.trade_reversal else "cont")
        tp = f"tp:{self.tp_mode}" + (f"x{self.rr_mult:g}" if self.tp_mode == "atr" else "")
        adr = f" adr<{self.max_adr_used:g}" if self.max_adr_used > 0 else ""
        return (f"piv{self.pivot_len} arm{self.arm_bars} run{self.min_run_len} "
                f"{tp} {mode}{adr}{' +ifvg' if self.use_ifvg else ''}")


@dataclass
class Trade:
    bar: int                 # index of the signal bar; fill is bar + 1
    is_buy: bool
    entry: float
    sl: float
    tp: float
    profit_usd: float
    r: float
    bars_held: int
    reason: str
    kind: str


@dataclass
class Market:
    symbol: str
    tf_minutes: int
    spread: float
    usd_per_unit: float
    lot: float
    time: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray


def _atr(h, l, c, n=14):
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.full(len(tr), np.nan)
    if len(tr) > n:
        out[n] = tr[1:n + 1].mean()
        for i in range(n + 1, len(tr)):          # Wilder smoothing, as ta.atr
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def _window(x, w, fn):
    """Rolling fn over w bars, aligned so index i covers bars i-w+1 .. i."""
    out = np.full(len(x), np.nan)
    if len(x) >= w >= 1:
        out[w - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, w), axis=1)
    return out


def _day_stretch(m: Market, adr_days: int):
    """
    How far the last 24h has travelled versus a normal 24h.

    1.0 is an average day. Above ~1.3 the session has already delivered more
    than its usual range, which is the state the chart was in when a 2xATR
    target sat above the entire day's rotation.
    """
    bpd = max(4, int(round(24 * 60 / m.tf_minutes)))
    n = len(m.c)
    if n < bpd * 2:
        return np.full(n, np.nan), bpd
    rng = _window(m.h, bpd, np.max) - _window(m.l, bpd, np.min)
    w = min(bpd * adr_days, n - bpd)
    typical = _window(np.nan_to_num(rng), w, np.mean)
    typical[:bpd + w] = np.nan          # not enough history to call it typical
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(typical > 0, rng / typical, np.nan), bpd


def load(symbol: str, days: int, spread_override: float | None,
         tf_minutes: int, cfg: AppConfig, quiet: bool = False) -> Market:
    tf_const = getattr(mt5, f"TIMEFRAME_{TF_MAP[tf_minutes]}")
    bars = min(int(days * 24 * 60 / tf_minutes) + 300, 100_000)

    conn = MT5Connector(cfg.mt5)
    conn.connect()
    if not quiet:
        print(f"Fetching {symbol}: {bars} bars of {tf_minutes}m")
    df = conn.get_ohlcv(symbol, tf_const, bars)
    info = conn.get_symbol_info(symbol)
    conn.disconnect()

    spread = spread_override if spread_override is not None else (info.ask - info.bid)
    lot = cfg.fixed_lots.get(symbol) or info.volume_min
    upu = lot * info.trade_contract_size

    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    if not quiet:
        print(f"Lot {lot} | spread {spread:.2f} (costs ${spread*upu:.2f}/trade) "
              f"| ${upu:.2f} per 1.00 move")
    return Market(symbol, tf_minutes, spread, upu, lot,
                  df["time"].to_numpy(), df["open"].to_numpy(float), h, l, c,
                  _atr(h, l, c))


def compute_signals(m: Market, p: IctParams):
    """
    Returns (side, sl, kind, pool_target, range_target) arrays.
    side: +1 buy, -1 sell, 0 none.

    Everything here is decided on the CLOSE of bar i using only bars <= i.
    The final target is deliberately NOT resolved here: it depends on the real
    fill price, which is the next bar's open, so `simulate` derives it. These
    arrays carry the two structural LEVELS instead, which keeps them
    independent of both rr_mult and tp_mode — the sweep caches on that.
    """
    n = len(m.c)
    side = np.zeros(n, dtype=np.int8)
    sl_a = np.full(n, np.nan)
    kind_a = np.zeros(n, dtype=np.int8)          # 1 reversal, 2 continuation
    pool_a = np.full(n, np.nan)                  # next opposing liquidity pool
    rng_a = np.full(n, np.nan)                   # high/low of the last day
    stretch, bpd = _day_stretch(m, p.adr_days)
    day_hi = _window(m.h, bpd, np.max)
    day_lo = _window(m.l, bpd, np.min)
    o, h, l, c, atr = m.o, m.h, m.l, m.c, m.atr
    pv = p.pivot_len

    pool_hi: list[float] = []
    pool_lo: list[float] = []
    swept_hi_lvl = swept_lo_lvl = np.nan
    sweep_hi_px = sweep_lo_px = np.nan
    armed_bull = armed_bear = -10 ** 9
    cur_dn_open = cur_up_open = np.nan
    cur_dn_len = cur_up_len = 0
    cisd_bull = cisd_bear = np.nan
    fvgs: list[dict] = []
    last_sig = -10 ** 9

    for i in range(pv + 2, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        # ── new liquidity pools (a pivot confirms pv bars after it forms) ────
        pi = i - pv
        if pi - pv >= 0:
            # h[pi+1:i+1] is the pv bars AFTER the pivot, all of them <= i, so
            # confirming the pivot here uses no future information. It doubles
            # as the "price has not already traded through it" test.
            hp = h[pi]
            if hp > h[pi - pv:pi].max() and hp > h[pi + 1:i + 1].max():
                pool_hi.append(hp)
                del pool_hi[:-p.max_pools]
            lp = l[pi]
            if lp < l[pi - pv:pi].min() and lp < l[pi + 1:i + 1].min():
                pool_lo.append(lp)
                del pool_lo[:-p.max_pools]

        # ── sweeps: each pool can only be taken once ─────────────────────────
        swept_hi = swept_lo = False
        for lvl in [x for x in pool_hi if h[i] > x and (not p.require_close_back or c[i] < x)]:
            swept_hi = True
            swept_hi_lvl = lvl
            sweep_hi_px = h[i]
        pool_hi = [x for x in pool_hi if not (h[i] > x and (not p.require_close_back or c[i] < x))]

        for lvl in [x for x in pool_lo if l[i] < x and (not p.require_close_back or c[i] > x)]:
            swept_lo = True
            swept_lo_lvl = lvl
            sweep_lo_px = l[i]
        pool_lo = [x for x in pool_lo if not (l[i] < x and (not p.require_close_back or c[i] > x))]

        if swept_hi:
            armed_bear = i
        if swept_lo:
            armed_bull = i

        # ── CISD ─────────────────────────────────────────────────────────────
        is_dn = c[i] < o[i]
        is_up = c[i] > o[i]
        pdn = c[i - 1] < o[i - 1]
        pup = c[i - 1] > o[i - 1]
        if is_dn:
            if not pdn:
                cur_dn_open, cur_dn_len = o[i], 1
            else:
                cur_dn_len += 1
        if is_up:
            if not pup:
                cur_up_open, cur_up_len = o[i], 1
            else:
                cur_up_len += 1
        if is_up and pdn and cur_dn_len >= p.min_run_len and not np.isnan(cur_dn_open):
            cisd_bull = cur_dn_open
        if is_dn and pup and cur_up_len >= p.min_run_len and not np.isnan(cur_up_open):
            cisd_bear = cur_up_open

        fired_bull = not np.isnan(cisd_bull) and c[i] > cisd_bull and c[i - 1] <= cisd_bull
        fired_bear = not np.isnan(cisd_bear) and c[i] < cisd_bear and c[i - 1] >= cisd_bear

        # ── fair value gaps and their inversion ──────────────────────────────
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= atr[i] * p.min_fvg_atr:
            fvgs.append({"top": l[i], "bot": h[i - 2], "dir": 1, "inv": False})
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) >= atr[i] * p.min_fvg_atr:
            fvgs.append({"top": l[i - 2], "bot": h[i], "dir": -1, "inv": False})
        del fvgs[:-30]

        in_bull_ifvg = in_bear_ifvg = False
        for g in fvgs:
            if not g["inv"]:
                if (g["dir"] == 1 and c[i] < g["bot"]) or (g["dir"] == -1 and c[i] > g["top"]):
                    g["inv"] = True
            else:
                if g["dir"] == -1 and l[i] <= g["top"] and h[i] >= g["bot"]:
                    in_bull_ifvg = True
                if g["dir"] == 1 and h[i] >= g["top"] and l[i] <= g["bot"]:
                    in_bear_ifvg = True

        # ── entry ────────────────────────────────────────────────────────────
        if i - last_sig < p.cooldown_bars:
            continue
        bull_armed = i - armed_bull <= p.arm_bars
        bear_armed = i - armed_bear <= p.arm_bars

        rev_buy = p.trade_reversal and bull_armed and fired_bull and not np.isnan(sweep_lo_px)
        rev_sell = p.trade_reversal and bear_armed and fired_bear and not np.isnan(sweep_hi_px)
        cont_buy = (p.trade_continuation and bear_armed and fired_bull
                    and not np.isnan(swept_hi_lvl) and c[i] > swept_hi_lvl)
        cont_sell = (p.trade_continuation and bull_armed and fired_bear
                     and not np.isnan(swept_lo_lvl) and c[i] < swept_lo_lvl)

        want_buy = (rev_buy or cont_buy) and (not p.use_ifvg or in_bull_ifvg)
        want_sell = (rev_sell or cont_sell) and (not p.use_ifvg or in_bear_ifvg)
        if want_buy == want_sell:                 # neither, or contradictory
            continue

        # the day has already run further than normal — skip the late entry
        if p.max_adr_used > 0 and not np.isnan(stretch[i]) and stretch[i] > p.max_adr_used:
            continue

        if want_buy:
            anchor = sweep_lo_px if rev_buy else swept_hi_lvl
            sl = max(anchor - atr[i] * p.sl_buf_atr, c[i] - atr[i] * p.max_risk_atr)
            if p.min_risk_atr > 0:
                sl = min(sl, c[i] - atr[i] * min(p.min_risk_atr, p.max_risk_atr))
            if c[i] - sl <= 0:
                continue
            side[i], sl_a[i] = 1, sl
            kind_a[i] = 1 if rev_buy else 2
            above = [x for x in pool_hi if x > c[i]]
            if above:
                pool_a[i] = min(above) - atr[i] * p.tp_buf_atr
            if not np.isnan(day_hi[i]) and day_hi[i] > c[i]:
                rng_a[i] = day_hi[i] - atr[i] * p.tp_buf_atr
        else:
            anchor = sweep_hi_px if rev_sell else swept_lo_lvl
            sl = min(anchor + atr[i] * p.sl_buf_atr, c[i] + atr[i] * p.max_risk_atr)
            if p.min_risk_atr > 0:
                sl = max(sl, c[i] + atr[i] * min(p.min_risk_atr, p.max_risk_atr))
            if sl - c[i] <= 0:
                continue
            side[i], sl_a[i] = -1, sl
            kind_a[i] = 1 if rev_sell else 2
            below = [x for x in pool_lo if x < c[i]]
            if below:
                pool_a[i] = max(below) + atr[i] * p.tp_buf_atr
            if not np.isnan(day_lo[i]) and day_lo[i] < c[i]:
                rng_a[i] = day_lo[i] + atr[i] * p.tp_buf_atr
        last_sig = i

    return side, sl_a, kind_a, pool_a, rng_a


def simulate(m: Market, p: IctParams, sig=None,
             start: int = 0, end: int | None = None) -> list[Trade]:
    """
    Replay the signals. `sig` may be a precomputed compute_signals() result,
    which lets the sweep reuse one pass across every rr_mult.

    The signal is decided on the close of bar i, so the earliest realistic fill
    is the OPEN of bar i+1 — never c[i], which you cannot transact at. Exits are
    then checked from that same bar i+1 onward, stop before target.
    """
    side, sl_a, kind_a, pool_a, rng_a = compute_signals(m, p) if sig is None else sig
    end = len(m.c) if end is None else end
    cost = m.spread * m.usd_per_unit
    trades: list[Trade] = []
    skipped = no_room = 0
    i = start
    while i < end - 1:
        if side[i] == 0:
            i += 1
            continue
        is_buy = side[i] > 0
        sl = sl_a[i]
        entry = m.o[i + 1]                    # fill at the next bar's open
        risk = (entry - sl) if is_buy else (sl - entry)
        if risk <= 0:                         # gapped through the stop
            skipped += 1
            i += 1
            continue

        if p.tp_mode == "atr":
            tp = entry + risk * p.rr_mult if is_buy else entry - risk * p.rr_mult
        else:
            tp = pool_a[i] if p.tp_mode == "pool" else rng_a[i]
            if not np.isnan(tp) and p.max_struct_rr > 0:
                cap = p.max_struct_rr * risk
                tp = (min(tp, entry + cap) if is_buy else max(tp, entry - cap))
            # No level to aim at, or the chart is not offering enough room to
            # justify the risk. Refusing the trade IS the decision.
            reward = (tp - entry) if is_buy else (entry - tp)
            if np.isnan(tp) or reward / risk < p.min_rr:
                no_room += 1
                i += 1
                continue

        j = i + 1
        while j < end:
            # stop checked before target: assume the adverse move came first
            hit_sl = m.l[j] <= sl if is_buy else m.h[j] >= sl
            hit_tp = m.h[j] >= tp if is_buy else m.l[j] <= tp
            if hit_sl or hit_tp:
                px = sl if hit_sl else tp
                move = (px - entry) if is_buy else (entry - px)
                trades.append(Trade(i, is_buy, entry, sl, tp,
                                    move * m.usd_per_unit - cost,
                                    (move / risk) - cost / (risk * m.usd_per_unit),
                                    j - i, "SL" if hit_sl else "TP",
                                    "rev" if kind_a[i] == 1 else "cont"))
                break
            j += 1
        else:
            break                             # still open at the end of data
        i = j + 1
    simulate.skipped = skipped
    simulate.no_room = no_room
    return trades


def stats(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "net": 0.0, "r": 0.0, "avg_r": 0.0, "per": 0.0,
                "wr": 0.0, "pf": 0.0, "dd": 0.0}
    wins = [t for t in trades if t.profit_usd > 0]
    losses = [t for t in trades if t.profit_usd <= 0]
    gw = sum(t.profit_usd for t in wins)
    gl = abs(sum(t.profit_usd for t in losses))
    eq = peak = dd = 0.0
    for t in trades:
        eq += t.profit_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    tot_r = sum(t.r for t in trades)
    return {"n": len(trades), "net": eq, "r": tot_r, "avg_r": tot_r / len(trades),
            # Lots are FIXED, so every trade risks a different number of dollars.
            # That makes average R a poor description of what the account does;
            # dollars per trade is the number that pays for the spread.
            "per": eq / len(trades),
            "wr": len(wins) / len(trades) * 100,
            "pf": (gw / gl) if gl > 0 else float("inf"), "dd": dd}


def report(m: Market, p: IctParams, trades: list[Trade]) -> None:
    s = stats(trades)
    print("-" * 62)
    if not s["n"]:
        print("No trades. Loosen pivot_len / arm_bars, or enable both premises.")
        return
    days = (m.time[-1] - m.time[0]) / np.timedelta64(1, "D")
    print(f"Period          : {days:.0f} days on {m.tf_minutes}m")
    print(f"Trades          : {s['n']}  ({s['n']/max(days,1):.2f}/day)")
    print(f"Win rate        : {s['wr']:.1f}%")
    print(f"Profit factor   : {s['pf']:.2f}" if s["pf"] != float("inf") else "Profit factor   : inf")
    print(f"Max drawdown    : -${s['dd']:.2f}")
    print(f"Avg hold        : {sum(t.bars_held for t in trades)/s['n']*m.tf_minutes:.0f} min")
    print("-" * 62)
    print(f"NET             : ${s['net']:+.2f}   ({s['r']:+.1f}R, avg {s['avg_r']:+.2f}R)")
    print("-" * 62)
    for kind in ("rev", "cont"):
        sub = [t for t in trades if t.kind == kind]
        if sub:
            k = stats(sub)
            print(f"  {kind:<5} {k['n']:>4} trades  win {k['wr']:>5.1f}%  "
                  f"avg {k['avg_r']:+.2f}R  net ${k['net']:+.2f}")
    if getattr(simulate, "no_room", 0):
        print(f"\n{simulate.no_room} signals declined — chart offered less than "
              f"{p.min_rr:g}R of room to the target level")
    if getattr(simulate, "skipped", 0):
        print(f"{simulate.skipped} signals skipped — next open gapped through the stop")
    print("\nEntries fill at the next bar's open, never at the signal bar's close.")
    print("Spread is charged per trade. Slippage and variable spread are not,")
    print("so treat anything under about +0.1R average as break-even.")


# The same ladder the Pine indicator applies on chart. A faster chart is
# noisier and pays a far larger share of its risk in spread (~18% on M1 against
# ~2.5% on M15), so every threshold tightens as the bars get quicker. Kept here
# so a backtest measures what the chart actually runs.
TF_PRESET = {
    1:  dict(pivot_len=20, arm_bars=10, min_run_len=4, cooldown_bars=20,
             min_risk_atr=1.00, max_struct_rr=3.0, min_rr=1.8,
             max_adr_used=1.15, require_close_back=True, tp_mode="pool"),
    5:  dict(pivot_len=14, arm_bars=12, min_run_len=3, cooldown_bars=12,
             min_risk_atr=0.80, max_struct_rr=4.0, min_rr=1.5,
             max_adr_used=1.25, require_close_back=True, tp_mode="pool"),
    15: dict(pivot_len=10, arm_bars=12, min_run_len=2, cooldown_bars=6,
             min_risk_atr=0.50, max_struct_rr=5.0, min_rr=1.2,
             max_adr_used=1.40, require_close_back=False, tp_mode="pool"),
    30: dict(pivot_len=8, arm_bars=12, min_run_len=2, cooldown_bars=4,
             min_risk_atr=0.50, max_struct_rr=6.0, min_rr=1.0,
             max_adr_used=0.0, require_close_back=False, tp_mode="pool"),
}


def preset_for(tf_minutes: int) -> dict:
    for k in (1, 5, 15):
        if tf_minutes <= k:
            return TF_PRESET[k]
    return TF_PRESET[30]


GRID = {
    "pivot_len": [5, 8, 12],
    "arm_bars": [6, 12, 20],
    "min_run_len": [1, 2],
    "rr_mult": [1.5, 2.0, 3.0],
    "mode": ["rev", "cont", "both"],
    "tp_mode": ["atr", "pool", "range"],
    "max_adr_used": [0.0, 1.3],
}


def run_sweep(m: Market, base: IctParams) -> None:
    n = len(m.c)
    split = int(n * 0.60)
    combos = list(itertools.product(*GRID.values()))
    print(f"\nTrain bars 0–{split} | Test bars {split}–{n}")
    print(f"Testing {len(combos)} combinations…\n")

    # rr_mult only means anything for the atr target, so drop the duplicates
    combos = [c for c in combos if c[5] == "atr" or c[3] == GRID["rr_mult"][0]]

    rows = []
    cache: dict = {}
    for idx, (pl, ab, mr, rr, mode, tpm, adr) in enumerate(combos, 1):
        if idx % 40 == 0:
            print(f"  …{idx}/{len(combos)}")
        p = replace(base, pivot_len=pl, arm_bars=ab, min_run_len=mr, rr_mult=rr,
                    tp_mode=tpm, max_adr_used=adr,
                    trade_reversal=mode in ("rev", "both"),
                    trade_continuation=mode in ("cont", "both"))
        # signals carry both structural levels and are independent of rr_mult
        # and tp_mode, so one pass serves every target choice
        key = (pl, ab, mr, mode, adr)
        if key not in cache:
            cache[key] = compute_signals(m, p)
        sig = cache[key]
        tr = stats(simulate(m, p, sig, 0, split))
        te = stats(simulate(m, p, sig, split, n))
        if tr["n"] < 20 or te["n"] < 20:
            continue
        rows.append((p, tr, te))

    if not rows:
        print("No combination produced enough trades in both halves.")
        return
    rows.sort(key=lambda r: r[1]["avg_r"], reverse=True)

    print("\n" + "=" * 82)
    print("TOP 12 BY TRAIN — the TEST columns are the ones that matter")
    print("=" * 82)
    print(f"{'settings':<40}{'train R':>8}{'train $':>9}{'test R':>8}"
          f"{'test $':>9}{'test n':>8}")
    print("-" * 82)
    for p, tr, te in rows[:12]:
        print(f"{p.label():<40}{tr['avg_r']:>+8.2f}{tr['per']:>+9.2f}"
              f"{te['avg_r']:>+8.2f}{te['per']:>+9.2f}{te['n']:>8}")

    # Must clear the bar in BOTH currencies: risk-normalised and in dollars.
    robust = [r for r in rows
              if r[2]["avg_r"] >= 0.10 and r[1]["avg_r"] > 0
              and r[2]["per"] > 0 and r[1]["per"] > 0]
    robust.sort(key=lambda r: r[2]["per"], reverse=True)
    print("\n" + "=" * 82)
    print("HELD UP OUT-OF-SAMPLE (test >= +0.10R, and made money on both halves)")
    print("=" * 82)
    if not robust:
        print("None.")
        print("\nThat is a real answer. No setting in this grid shows an edge that")
        print("survives on unseen data, so this model is not worth automating as is.")
        return
    for p, tr, te in robust[:8]:
        print(f"{p.label():<40}{tr['avg_r']:>+8.2f}{tr['per']:>+9.2f}"
              f"{te['avg_r']:>+8.2f}{te['per']:>+9.2f}{te['n']:>8}")


def main() -> int:
    if mt5 is None:
        print("MetaTrader5 package not available (Windows only).")
        return 1
    cfg = load_config()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    symbol = args[0] if args else cfg.symbols[0]
    days = int(args[1]) if len(args) > 1 else 90
    spread = float(args[2]) if len(args) > 2 else None
    tf = int(args[3]) if len(args) > 3 else 15
    if tf not in TF_MAP:
        print(f"Timeframe must be one of {sorted(TF_MAP)}")
        return 1

    base = IctParams()
    if "--manual" not in sys.argv:
        base = replace(base, **preset_for(tf))
        print(f"Preset for {TF_MAP[tf]}: {base.label()}")
    for a in sys.argv[1:]:
        if a.startswith("--tp="):
            base = replace(base, tp_mode=a.split("=", 1)[1])
        elif a.startswith("--adr="):
            base = replace(base, max_adr_used=float(a.split("=", 1)[1]))
        elif a.startswith("--minrr="):
            base = replace(base, min_rr=float(a.split("=", 1)[1]))
        elif a.startswith("--maxrr="):
            base = replace(base, max_struct_rr=float(a.split("=", 1)[1]))
    if base.tp_mode not in ("atr", "pool", "range"):
        print("--tp must be atr, pool or range")
        return 1

    m = load(symbol, days, spread, tf, cfg)
    if "--sweep" in sys.argv:
        run_sweep(m, base)
    else:
        print(f"Params: {base.label()}")
        report(m, base, simulate(m, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
