"""
Backtest engine — replays settings over historical MT5 data.

Run:  python backtest.py                      (SYMBOLS[0] from .env, 14 days)
      python backtest.py XAUUSD 30            (symbol, days)
      python backtest.py XAUUSD 30 0.30       (…and an explicit spread)

Also used as a library by sweep.py: load_market() fetches and precomputes
everything that does not depend on the tunable settings, then simulate()
can be run many times cheaply with different Params.

Honesty notes baked into the model:
  * spread is charged on every trade
  * signals are read off COMPLETED bars, filled at the NEXT bar's open
  * the trailing floor uses the peak as of the previous bar's close, so a
    trade can never open and trail-exit inside one bar
  * stops are checked before targets within a bar
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from analyzer import TechnicalAnalyzer
from config import AppConfig, load_config
from mt5_connector import MT5Connector

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

WARMUP = 260


@dataclass
class Params:
    """Every setting the sweep is allowed to vary."""
    signal_threshold: int
    min_adx: float
    require_momentum: bool
    sl_atr_mult: float
    rr_ratio: float
    trail_activate: float = 0.0
    trail_distance: float = 0.0
    quick_profit: float = 0.0
    max_loss: float = 0.0
    max_bars: int = 0

    def label(self) -> str:
        s = f"thr{self.signal_threshold} adx{self.min_adx:.0f} sl{self.sl_atr_mult:g} rr{self.rr_ratio:g}"
        if self.trail_activate > 0:
            s += f" trail{self.trail_activate:g}/{self.trail_distance:g}"
        return s

    @classmethod
    def from_config(cls, cfg: AppConfig, tf_minutes: int) -> "Params":
        max_bars = 0
        if cfg.max_trade_age_seconds > 0:
            max_bars = max(1, cfg.max_trade_age_seconds // (tf_minutes * 60))
        elif cfg.max_trade_age_minutes > 0:
            max_bars = max(1, cfg.max_trade_age_minutes // tf_minutes)
        return cls(
            signal_threshold=cfg.risk.signal_threshold,
            min_adx=cfg.risk.min_adx,
            require_momentum=cfg.risk.require_momentum,
            sl_atr_mult=cfg.risk.sl_atr_multiplier,
            rr_ratio=cfg.risk.rr_ratio,
            trail_activate=cfg.trail_activate_usd,
            trail_distance=cfg.trail_distance_usd,
            quick_profit=cfg.quick_profit_usd,
            max_loss=cfg.max_loss_usd,
            max_bars=max_bars,
        )


@dataclass
class SimTrade:
    direction: str
    entry_price: float
    exit_price: float
    profit_usd: float
    bars_held: int
    reason: str


@dataclass
class MarketData:
    """Everything precomputed that does not depend on tunable settings."""
    symbol: str
    tf_minutes: int
    spread: float
    usd_per_unit: float
    lot: float
    times: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    buy_score: np.ndarray   # -1 where no valid signal could be formed
    sell_score: np.ndarray
    adx: np.ndarray
    macd_hist: np.ndarray
    median_atr: float
    median_range: float
    first_valid: int          # first bar with usable higher-timeframe context


def _tf_constants(cfg: AppConfig) -> tuple[int, int, int, int, int, int]:
    """(macro, mid, trigger, macro_min, mid_min, trigger_min) for the active mode."""
    if cfg.balanced_mode:
        return mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5, 60, 15, 5
    if cfg.scalp_mode or cfg.turbo_mode:
        return mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M1, 15, 5, 1
    return mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_M15, 1440, 240, 15


def load_market(symbol: str, days: int, spread_override: float | None,
                cfg: AppConfig, quiet: bool = False) -> MarketData:
    analyzer = TechnicalAnalyzer(cfg.indicators, cfg.risk)
    connector = MT5Connector(cfg.mt5)
    connector.connect()

    tf_macro, tf_mid, tf_trigger, macro_min, mid_min, tf_minutes = _tf_constants(cfg)

    # Every timeframe must span the requested period, plus WARMUP bars of its
    # own history. Under-fetching the mid/macro series silently truncates the
    # tradeable window without changing the reported period.
    minutes = days * 24 * 60
    trig_bars = min(int(minutes / tf_minutes) + 300, 50_000)
    mid_bars = min(int(minutes / mid_min) + WARMUP + 50, 50_000)
    macro_bars = min(int(minutes / macro_min) + WARMUP + 50, 50_000)

    if not quiet:
        print(f"Fetching {symbol}: {trig_bars} x {tf_minutes}m, "
              f"{mid_bars} x {mid_min}m, {macro_bars} x {macro_min}m")
    df_trigger = connector.get_ohlcv(symbol, tf_trigger, trig_bars)
    df_mid = connector.get_ohlcv(symbol, tf_mid, mid_bars)
    df_macro = connector.get_ohlcv(symbol, tf_macro, macro_bars)

    info = connector.get_symbol_info(symbol)
    market_open = connector.is_market_open(symbol)
    connector.disconnect()

    live_spread = info.ask - info.bid
    spread = spread_override if spread_override is not None else live_spread
    lot = cfg.fixed_lots.get(symbol)
    if lot is None:
        lot = info.volume_min
        if not quiet:
            print(f"!  No {symbol.upper()}_LOT in .env — using broker minimum {lot}")
    usd_per_unit = lot * info.trade_contract_size

    if not quiet:
        if spread_override is not None:
            print(f"!  Using spread override ${spread:.2f} (live tick shows ${live_spread:.2f})")
        elif not market_open:
            print(f"!  Market appears CLOSED — ${live_spread:.2f} is a weekend spread and is")
            print(f"   far wider than normal. Pass a real one, e.g. "
                  f"python backtest.py {symbol} {days} 0.30")
        print(f"Lot {lot} | spread ${spread:.2f} (costs ${spread*usd_per_unit:.2f}/trade) "
              f"| ${usd_per_unit:.2f} per $1 move")
        print("Computing indicators…")

    df_trigger = analyzer._compute_indicators(df_trigger)
    df_mid = analyzer._compute_indicators(df_mid)
    df_macro = analyzer._compute_indicators(df_macro)

    mid_times = df_mid["time"].values
    macro_times = df_macro["time"].values
    n = len(df_trigger)

    buy_s = np.full(n, -1, dtype=np.int8)
    sell_s = np.full(n, -1, dtype=np.int8)
    adx_a = np.zeros(n)
    macd_a = np.zeros(n)

    if not quiet:
        print(f"Scoring {n - WARMUP} bars…")
    for i in range(WARMUP, n):
        bar = df_trigger.iloc[i]
        atr_i = bar["atr"]
        if atr_i != atr_i or atr_i <= 0:
            continue
        now = bar["time"]
        j_mid = int(np.searchsorted(mid_times, np.datetime64(now), side="right")) - 1
        j_macro = int(np.searchsorted(macro_times, np.datetime64(now), side="right")) - 1
        if j_mid < WARMUP or j_macro < 10:
            continue

        close_i = float(bar["close"])
        sr = analyzer._detect_swing_levels(df_mid.iloc[: j_mid + 1], close_i)
        sc = analyzer.score_conditions(
            df_macro.iloc[j_macro], df_mid.iloc[j_mid], df_mid.iloc[j_mid], bar, close_i,
            analyzer._nearest_level_below(sr.supports, close_i),
            analyzer._nearest_level_above(sr.resistances, close_i),
        )
        buy_s[i], sell_s[i] = sc.buy_score, sc.sell_score
        adx_a[i], macd_a[i] = sc.adx, sc.macd_hist

    valid = np.flatnonzero(buy_s >= 0)
    first_valid = int(valid[0]) if len(valid) else n
    if not quiet:
        if first_valid >= n:
            print("!  No bar had usable higher-timeframe context — nothing to simulate.")
        else:
            usable = (df_trigger["time"].iloc[-1] - df_trigger["time"].iloc[first_valid])
            usable_days = usable.total_seconds() / 86400
            print(f"Tradeable window: {usable_days:.1f} days "
                  f"({n - first_valid} of {n} bars)")
            if usable_days < days * 0.6:
                print(f"!  Requested {days} days but only {usable_days:.0f} are usable —")
                print("   the broker likely caps how far back this timeframe goes.")

    high = df_trigger["high"].to_numpy(dtype=float)
    low = df_trigger["low"].to_numpy(dtype=float)
    return MarketData(
        symbol=symbol, tf_minutes=tf_minutes, spread=spread, usd_per_unit=usd_per_unit,
        lot=lot, times=df_trigger["time"].to_numpy(),
        open_=df_trigger["open"].to_numpy(dtype=float), high=high, low=low,
        close=df_trigger["close"].to_numpy(dtype=float),
        atr=df_trigger["atr"].to_numpy(dtype=float),
        buy_score=buy_s, sell_score=sell_s, adx=adx_a, macd_hist=macd_a,
        median_atr=float(np.nanmedian(df_trigger["atr"].to_numpy(dtype=float))),
        median_range=float(np.nanmedian(high - low)),
        first_valid=first_valid,
    )


def simulate(md: MarketData, p: Params,
             start: int | None = None, end: int | None = None) -> list[SimTrade]:
    """Replay `p` over bars [start, end). Pure function of precomputed data."""
    start = md.first_valid if start is None else max(start, md.first_valid)
    end = len(md.close) if end is None else min(end, len(md.close))
    spread_cost = md.spread * md.usd_per_unit

    trades: list[SimTrade] = []
    t: dict | None = None

    for i in range(start, end):
        hi, lo, cl = md.high[i], md.low[i], md.close[i]

        if t is not None:
            exit_px = None
            reason = ""
            is_buy = t["buy"]

            # Stop before target: assume the adverse move came first
            if is_buy and lo <= t["sl"]:
                exit_px, reason = t["sl"], "SL"
            elif not is_buy and hi >= t["sl"]:
                exit_px, reason = t["sl"], "SL"
            elif is_buy and hi >= t["tp"]:
                exit_px, reason = t["tp"], "TP"
            elif not is_buy and lo <= t["tp"]:
                exit_px, reason = t["tp"], "TP"

            # Trailing floor from the peak as of the PREVIOUS bar
            if exit_px is None and p.trail_activate > 0 and p.trail_distance > 0:
                if t["peak"] >= p.trail_activate:
                    locked = t["peak"] - p.trail_distance
                    if locked > 0:
                        off = (locked + spread_cost) / md.usd_per_unit
                        floor = t["entry"] + off if is_buy else t["entry"] - off
                        if is_buy and lo <= floor:
                            exit_px, reason = floor, "trail"
                        elif not is_buy and hi >= floor:
                            exit_px, reason = floor, "trail"

            # Peak updates only after the exit checks
            if exit_px is None:
                best = hi if is_buy else lo
                move = (best - t["entry"]) if is_buy else (t["entry"] - best)
                t["peak"] = max(t["peak"], move * md.usd_per_unit - spread_cost)

            if exit_px is None and (p.quick_profit > 0 or p.max_loss > 0):
                move = (cl - t["entry"]) if is_buy else (t["entry"] - cl)
                pl = move * md.usd_per_unit - spread_cost
                if p.quick_profit > 0 and pl >= p.quick_profit:
                    exit_px, reason = cl, "quick-profit"
                elif p.max_loss > 0 and pl <= -p.max_loss:
                    exit_px, reason = cl, "max-loss"

            held = i - t["bar"]
            if exit_px is None and p.max_bars and held >= p.max_bars:
                exit_px, reason = cl, "time"

            if exit_px is None:
                continue
            move = (exit_px - t["entry"]) if is_buy else (t["entry"] - exit_px)
            trades.append(SimTrade("BUY" if is_buy else "SELL", t["entry"], exit_px,
                                   move * md.usd_per_unit - spread_cost, held, reason))
            t = None

        if i + 1 >= end:
            break

        bs, ss = md.buy_score[i], md.sell_score[i]
        if bs < 0:
            continue
        if p.min_adx > 0 and md.adx[i] < p.min_adx:
            continue
        mh = md.macd_hist[i]
        if bs >= p.signal_threshold and (not p.require_momentum or mh > 0):
            is_buy = True
        elif ss >= p.signal_threshold and (not p.require_momentum or mh < 0):
            is_buy = False
        else:
            continue

        entry = md.open_[i + 1]
        sl_dist = md.atr[i] * p.sl_atr_mult
        tp_dist = sl_dist * p.rr_ratio
        t = {
            "buy": is_buy, "entry": entry,
            "sl": entry - sl_dist if is_buy else entry + sl_dist,
            "tp": entry + tp_dist if is_buy else entry - tp_dist,
            "bar": i + 1, "peak": 0.0,
        }

    return trades


def stats(trades: list[SimTrade]) -> dict:
    if not trades:
        return {"trades": 0, "net": 0.0, "pf": 0.0, "win_rate": 0.0,
                "max_dd": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "avg_bars": 0.0}
    wins = [t.profit_usd for t in trades if t.profit_usd > 0]
    losses = [t.profit_usd for t in trades if t.profit_usd <= 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    eq = peak = dd = 0.0
    for t in trades:
        eq += t.profit_usd
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {
        "trades": len(trades),
        "net": eq,
        "pf": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "win_rate": len(wins) / len(trades) * 100,
        "max_dd": dd,
        "avg_win": gross_w / len(wins) if wins else 0.0,
        "avg_loss": -gross_l / len(losses) if losses else 0.0,
        "avg_bars": sum(t.bars_held for t in trades) / len(trades),
    }


def _report(trades: list[SimTrade], md: MarketData) -> None:
    print("-" * 62)
    s = stats(trades)
    if not trades:
        print("No trades — the entry filters are too strict for this period.")
        print("Try lowering SIGNAL_THRESHOLD or MIN_ADX, then re-run.")
        return

    # Measure the window trades could actually occur in, not the whole fetch
    span_days = max(
        (md.times[-1] - md.times[min(md.first_valid, len(md.times) - 1)])
        / np.timedelta64(1, "D"), 1e-9
    )
    pf = s["pf"]
    print(f"Period tested   : {span_days:.1f} days")
    print(f"Trades          : {s['trades']}  ({s['trades']/span_days:.1f}/day)")
    print(f"Win rate        : {s['win_rate']:.1f}%")
    print(f"Avg win / loss  : ${s['avg_win']:.2f} / ${s['avg_loss']:.2f}")
    print(f"Profit factor   : {pf:.2f}" if pf != float("inf") else "Profit factor   : inf")
    print(f"Max drawdown    : -${s['max_dd']:.2f}")
    print(f"Avg hold        : {s['avg_bars']*md.tf_minutes:.0f} min")
    print("-" * 62)
    print(f"NET RESULT      : ${s['net']:+.2f}   → "
          f"{'PROFITABLE' if s['net'] > 0 else 'LOSING'}")
    print("-" * 62)

    by: dict[str, list[float]] = {}
    for t in trades:
        by.setdefault(t.reason, []).append(t.profit_usd)
    print("Exits:")
    for reason, pls in sorted(by.items(), key=lambda kv: -len(kv[1])):
        print(f"  {reason:<14} {len(pls):>4} trades   ${sum(pls):+9.2f}")

    print("\nReal results will be worse: no slippage, requotes, variable spread")
    print("or swap are modelled. Treat anything under ~1.3 profit factor as noise.")


def run_backtest(symbol: str, days: int, spread_override: float | None = None) -> None:
    cfg = load_config()
    md = load_market(symbol, days, spread_override, cfg)
    p = Params.from_config(cfg, md.tf_minutes)

    print(f"Entry : {p.signal_threshold}/7 conditions, ADX>={p.min_adx:.0f}, "
          f"momentum gate {'on' if p.require_momentum else 'off'}")
    print(f"Median {md.tf_minutes}m ATR ${md.median_atr:.2f} | median bar range "
          f"${md.median_range:.2f} | stop {p.sl_atr_mult}xATR = "
          f"${md.median_atr*p.sl_atr_mult:.2f} (${md.median_atr*p.sl_atr_mult*md.usd_per_unit:.2f})")
    if p.trail_activate > 0:
        mv = p.trail_activate / md.usd_per_unit
        print(f"Trail arms after a ${mv:.2f} move = {mv/max(md.median_range,1e-9)*100:.0f}% "
              f"of a median bar — under ~50% is inside the noise")

    _report(simulate(md, p), md)


if __name__ == "__main__":
    if mt5 is None:
        print("MetaTrader5 package not available (Windows only).")
        sys.exit(1)
    _cfg = load_config()
    _symbol = sys.argv[1] if len(sys.argv) > 1 else _cfg.symbols[0]
    _days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    _spread = float(sys.argv[3]) if len(sys.argv) > 3 else None
    run_backtest(_symbol, _days, _spread)
