"""
Backtester — replays your CURRENT .env settings over historical MT5 data
so you can see what a setup would actually have done before risking money.

Run:  python backtest.py                 (uses SYMBOLS from .env, 14 days)
      python backtest.py BTCUSD 30       (symbol, days)

Models spread on every trade, the ATR stop, the take-profit, the trailing
stop and the time/quick-profit/max-loss exits — i.e. the same exit plan the
live bot runs. Indicators are computed once over the series (they are all
backward-looking, so this is identical to recomputing per bar, just far
faster) and every signal is evaluated on a COMPLETED bar with entry filled
at the NEXT bar's open, so there is no lookahead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from analyzer import Signal, TechnicalAnalyzer
from config import load_config
from mt5_connector import MT5Connector

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


@dataclass
class SimTrade:
    direction: str
    entry_price: float
    exit_price: float
    profit_usd: float
    bars_held: int
    reason: str


def _tf_constants(cfg) -> tuple[int, int, int, int]:
    """(macro, mid, trigger, trigger_minutes) matching the bot's active mode."""
    if cfg.balanced_mode:
        return mt5.TIMEFRAME_H1, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5, 5
    if cfg.scalp_mode or cfg.turbo_mode:
        return mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M1, 1
    return mt5.TIMEFRAME_D1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_M15, 15


def run_backtest(symbol: str, days: int, spread_override: float | None = None) -> None:
    cfg = load_config()
    analyzer = TechnicalAnalyzer(cfg.indicators, cfg.risk)
    connector = MT5Connector(cfg.mt5)
    connector.connect()

    tf_macro, tf_mid, tf_trigger, tf_minutes = _tf_constants(cfg)
    bars_needed = int(days * 24 * 60 / tf_minutes) + 300

    print(f"Fetching {symbol}: {bars_needed} bars of {tf_minutes}m data…")
    df_trigger = connector.get_ohlcv(symbol, tf_trigger, min(bars_needed, 50_000))
    df_mid = connector.get_ohlcv(symbol, tf_mid, 5_000)
    df_macro = connector.get_ohlcv(symbol, tf_macro, 5_000)

    info = connector.get_symbol_info(symbol)
    connector_market_was_open = connector.is_market_open(symbol)
    connector.disconnect()

    live_spread = info.ask - info.bid
    spread = spread_override if spread_override is not None else live_spread
    if spread_override is not None:
        print(f"!  Using spread override ${spread:.2f} (live tick shows ${live_spread:.2f})")
    elif not connector_market_was_open:
        print(f"!  Market appears CLOSED — the ${live_spread:.2f} spread is a weekend/holiday")
        print("   figure and is much wider than normal. Pass a realistic spread as the")
        print(f"   3rd argument, e.g.  python backtest.py {symbol} {days} 0.30")

    lot = cfg.fixed_lots.get(symbol)
    if lot is None:
        lot = info.volume_min
        print(f"!  No {symbol.upper()}_LOT in .env — using broker minimum {lot}")
    usd_per_unit = lot * info.trade_contract_size

    print(f"Lot {lot} | spread ${spread:.2f} (costs ${spread*usd_per_unit:.2f}/trade) "
          f"| ${usd_per_unit:.2f} per $1 move")
    print(f"Entry : {cfg.risk.signal_threshold}/7 conditions, ADX>={cfg.risk.min_adx:.0f}, "
          f"momentum gate {'on' if cfg.risk.require_momentum else 'off'}")
    exits = [f"SL={cfg.risk.sl_atr_multiplier}xATR", f"TP=1:{cfg.risk.rr_ratio}"]
    if cfg.trail_activate_usd > 0 and cfg.trail_distance_usd > 0:
        exits.append(f"trail ${cfg.trail_activate_usd:.0f}/${cfg.trail_distance_usd:.0f}")
    if cfg.quick_profit_usd > 0:
        exits.append(f"quick ${cfg.quick_profit_usd:.0f}")
    if cfg.max_loss_usd > 0:
        exits.append(f"maxloss ${cfg.max_loss_usd:.0f}")
    print(f"Exits : {', '.join(exits)}")
    print("-" * 62)

    # ── Precompute indicators once (all are backward-looking → no lookahead) ──
    print("Computing indicators…")
    df_trigger = analyzer._compute_indicators(df_trigger)
    df_mid = analyzer._compute_indicators(df_mid)
    df_macro = analyzer._compute_indicators(df_macro)

    mid_times = df_mid["time"].values
    macro_times = df_macro["time"].values

    # Sanity check: how big is a typical bar and stop, in price and in dollars?
    med_atr = float(df_trigger["atr"].median())
    med_range = float((df_trigger["high"] - df_trigger["low"]).median())
    print(f"Median {tf_minutes}m ATR ${med_atr:.2f} | median bar range ${med_range:.2f} "
          f"| stop {cfg.risk.sl_atr_multiplier}xATR = ${med_atr*cfg.risk.sl_atr_multiplier:.2f} "
          f"(${med_atr*cfg.risk.sl_atr_multiplier*usd_per_unit:.2f})")
    if cfg.trail_activate_usd > 0:
        trail_move = cfg.trail_activate_usd / usd_per_unit
        print(f"Trail arms after a ${trail_move:.2f} move — that is "
              f"{trail_move/max(med_range,1e-9)*100:.0f}% of a median bar's range")

    # Bars held before a forced time exit (0 = disabled)
    max_bars = 0
    if cfg.max_trade_age_seconds > 0:
        max_bars = max(1, cfg.max_trade_age_seconds // (tf_minutes * 60))
    elif cfg.max_trade_age_minutes > 0:
        max_bars = max(1, cfg.max_trade_age_minutes // tf_minutes)

    warmup = 260
    trades: list[SimTrade] = []
    trade: dict | None = None
    total = len(df_trigger)

    print(f"Simulating {total - warmup} bars…")
    for i in range(warmup, total):
        if (i - warmup) % 2000 == 0 and i > warmup:
            print(f"  …{i - warmup}/{total - warmup} bars, {len(trades)} trades")

        bar = df_trigger.iloc[i]

        # ── Manage an open position ─────────────────────────────────────────
        if trade is not None:
            exit_px = None
            reason = ""
            is_buy = trade["dir"] == "BUY"

            # Stop first (pessimistic: assume the adverse move happened first)
            if is_buy and bar["low"] <= trade["sl"]:
                exit_px, reason = trade["sl"], "SL"
            elif not is_buy and bar["high"] >= trade["sl"]:
                exit_px, reason = trade["sl"], "SL"
            elif is_buy and bar["high"] >= trade["tp"]:
                exit_px, reason = trade["tp"], "TP"
            elif not is_buy and bar["low"] <= trade["tp"]:
                exit_px, reason = trade["tp"], "TP"

            # Trailing stop. The floor comes from the peak as of the END of the
            # PREVIOUS bar. Using this bar's high to raise the peak and its low
            # to trigger the exit would assume an intrabar order we cannot know,
            # and manufactures fake profit on every wide bar.
            if exit_px is None and cfg.trail_activate_usd > 0 and cfg.trail_distance_usd > 0:
                if trade["peak"] >= cfg.trail_activate_usd:
                    locked = trade["peak"] - cfg.trail_distance_usd
                    if locked > 0:
                        offset = (locked + spread * usd_per_unit) / usd_per_unit
                        floor_px = (trade["entry"] + offset) if is_buy else (trade["entry"] - offset)
                        if is_buy and bar["low"] <= floor_px:
                            exit_px, reason = floor_px, "trail"
                        elif not is_buy and bar["high"] >= floor_px:
                            exit_px, reason = floor_px, "trail"

            # Update the peak AFTER the exit checks, so it only affects later bars
            if exit_px is None:
                best_px = bar["high"] if is_buy else bar["low"]
                best_move = (best_px - trade["entry"]) if is_buy else (trade["entry"] - best_px)
                trade["peak"] = max(trade["peak"], best_move * usd_per_unit - spread * usd_per_unit)

            if exit_px is None:
                move = ((bar["close"] - trade["entry"]) if is_buy
                        else (trade["entry"] - bar["close"]))
                pl = move * usd_per_unit - spread * usd_per_unit
                if cfg.quick_profit_usd > 0 and pl >= cfg.quick_profit_usd:
                    exit_px, reason = bar["close"], "quick-profit"
                elif cfg.max_loss_usd > 0 and pl <= -cfg.max_loss_usd:
                    exit_px, reason = bar["close"], "max-loss"

            held = i - trade["bar"]
            if exit_px is None and max_bars and held >= max_bars:
                exit_px, reason = bar["close"], "time"

            if exit_px is not None:
                move = ((exit_px - trade["entry"]) if is_buy else (trade["entry"] - exit_px))
                profit = move * usd_per_unit - spread * usd_per_unit
                trades.append(SimTrade(trade["dir"], trade["entry"], exit_px,
                                       profit, held, reason))
                trade = None
            else:
                continue

        if i + 1 >= total:
            break

        # ── Evaluate a signal on this COMPLETED bar ─────────────────────────
        now = bar["time"]
        j_mid = int(np.searchsorted(mid_times, np.datetime64(now), side="right")) - 1
        j_macro = int(np.searchsorted(macro_times, np.datetime64(now), side="right")) - 1
        if j_mid < warmup or j_macro < 10:
            continue

        close = float(bar["close"])
        atr = float(bar["atr"]) if bar["atr"] == bar["atr"] else 0.0
        if atr <= 0:
            continue

        sr = analyzer._detect_swing_levels(df_mid.iloc[: j_mid + 1], close)
        near_sup = analyzer._nearest_level_below(sr.supports, close)
        near_res = analyzer._nearest_level_above(sr.resistances, close)

        mid_row = df_mid.iloc[j_mid]
        macro_row = df_macro.iloc[j_macro]
        signal, _ = analyzer._classify_signal(
            symbol=symbol, d1=macro_row, h4=mid_row, h1=mid_row, m15=bar,
            close=close, nearest_support=near_sup, nearest_resistance=near_res,
        )
        if signal == Signal.HOLD:
            continue

        # Fill at the NEXT bar's open — we cannot trade on a bar we just closed
        entry = float(df_trigger.iloc[i + 1]["open"])
        sl_dist = atr * cfg.risk.sl_atr_multiplier
        tp_dist = sl_dist * cfg.risk.rr_ratio
        is_buy = signal == Signal.BUY
        trade = {
            "dir": signal.value,
            "entry": entry,
            "sl": entry - sl_dist if is_buy else entry + sl_dist,
            "tp": entry + tp_dist if is_buy else entry - tp_dist,
            "bar": i + 1,
            "peak": 0.0,
        }

    _report(trades, df_trigger, warmup, tf_minutes)


def _report(trades: list[SimTrade], df, warmup: int, tf_minutes: int) -> None:
    print("-" * 62)
    if not trades:
        print("No trades taken — the entry filters are too strict for this period.")
        print("Try lowering SIGNAL_THRESHOLD or MIN_ADX, then re-run.")
        return

    wins = [t for t in trades if t.profit_usd > 0]
    losses = [t for t in trades if t.profit_usd <= 0]
    net = sum(t.profit_usd for t in trades)
    gross_win = sum(t.profit_usd for t in wins)
    gross_loss = abs(sum(t.profit_usd for t in losses))

    equity = peak = max_dd = 0.0
    for t in trades:
        equity += t.profit_usd
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    span_days = max(
        (df.iloc[-1]["time"] - df.iloc[warmup]["time"]).total_seconds() / 86400, 1e-9
    )

    print(f"Period tested   : {span_days:.1f} days")
    print(f"Trades          : {len(trades)}  ({len(trades)/span_days:.1f}/day)")
    print(f"Win rate        : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win         : ${gross_win/len(wins):.2f}" if wins else "Avg win         : n/a")
    print(f"Avg loss        : -${gross_loss/len(losses):.2f}" if losses else "Avg loss        : n/a")
    print(f"Profit factor   : {gross_win/gross_loss:.2f}" if gross_loss > 0 else "Profit factor   : inf (no losses)")
    print(f"Max drawdown    : -${max_dd:.2f}")
    print(f"Avg hold        : {sum(t.bars_held for t in trades)/len(trades)*tf_minutes:.0f} min")
    print("-" * 62)
    print(f"NET RESULT      : ${net:+.2f}   → {'PROFITABLE' if net > 0 else 'LOSING'}")
    print("-" * 62)

    by_reason: dict[str, list[float]] = {}
    for t in trades:
        by_reason.setdefault(t.reason, []).append(t.profit_usd)
    print("Exits:")
    for reason, pls in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"  {reason:<14} {len(pls):>4} trades   ${sum(pls):+9.2f}")

    print("\nReal results will be worse: this ignores slippage, requotes,")
    print("variable spread and swap. Treat a small profit here as break-even.")


if __name__ == "__main__":
    if mt5 is None:
        print("MetaTrader5 package not available (Windows only).")
        sys.exit(1)

    _cfg = load_config()
    _symbol = sys.argv[1] if len(sys.argv) > 1 else _cfg.symbols[0]
    _days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    _spread = float(sys.argv[3]) if len(sys.argv) > 3 else None
    run_backtest(_symbol, _days, _spread)
