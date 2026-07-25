"""
Backtester — replays your CURRENT .env settings over historical MT5 data
so you can see what a setup would actually have done before risking money.

Run:  python backtest.py                 (uses SYMBOLS from .env, 14 days)
      python backtest.py BTCUSD 30       (symbol, days)

It simulates spread cost on every trade, which is the single biggest reason
scalping setups that look good on paper lose money live.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pandas as pd

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


def run_backtest(symbol: str, days: int) -> None:
    cfg = load_config()
    analyzer = TechnicalAnalyzer(cfg.indicators, cfg.risk)
    connector = MT5Connector(cfg.mt5)
    connector.connect()

    tf_macro, tf_mid, tf_trigger, tf_minutes = _tf_constants(cfg)
    bars_needed = int(days * 24 * 60 / tf_minutes) + 300  # +warmup

    print(f"Fetching {symbol}: {bars_needed} bars of {tf_minutes}m data…")
    df_trigger = connector.get_ohlcv(symbol, tf_trigger, min(bars_needed, 50_000))
    df_mid = connector.get_ohlcv(symbol, tf_mid, 5_000)
    df_macro = connector.get_ohlcv(symbol, tf_macro, 5_000)

    info = connector.get_symbol_info(symbol)
    spread = info.ask - info.bid
    lot = cfg.fixed_lots.get(symbol, info.volume_min)
    usd_per_unit = lot * info.trade_contract_size

    print(f"Lot {lot} | spread ${spread:.2f} | ${usd_per_unit:.2f} per $1 move")
    print(f"Settings: threshold={cfg.risk.signal_threshold}/7  ADX>={cfg.risk.min_adx:.0f}  "
          f"SL={cfg.risk.sl_atr_multiplier}xATR  RR=1:{cfg.risk.rr_ratio}")
    if cfg.quick_profit_usd > 0:
        print(f"          quick-profit ${cfg.quick_profit_usd:.2f}", end="")
    if cfg.max_loss_usd > 0:
        print(f"  max-loss ${cfg.max_loss_usd:.2f}", end="")
    print("\n" + "-" * 62)

    warmup = 260
    trades: list[SimTrade] = []
    open_trade: dict | None = None

    for i in range(warmup, len(df_trigger)):
        bar = df_trigger.iloc[i]
        now = bar["time"]

        # ── Manage an open position on this bar ──────────────────────────────
        if open_trade:
            hit = None
            if open_trade["dir"] == "BUY":
                if bar["low"] <= open_trade["sl"]:
                    exit_px, hit = open_trade["sl"], "SL"
                elif bar["high"] >= open_trade["tp"]:
                    exit_px, hit = open_trade["tp"], "TP"
            else:
                if bar["high"] >= open_trade["sl"]:
                    exit_px, hit = open_trade["sl"], "SL"
                elif bar["low"] <= open_trade["tp"]:
                    exit_px, hit = open_trade["tp"], "TP"

            # Running P/L at this bar's close (for $ based exits)
            if not hit:
                move = ((bar["close"] - open_trade["entry"]) if open_trade["dir"] == "BUY"
                        else (open_trade["entry"] - bar["close"]))
                pl = move * usd_per_unit - spread * usd_per_unit
                if cfg.quick_profit_usd > 0 and pl >= cfg.quick_profit_usd:
                    exit_px, hit = bar["close"], "quick-profit"
                elif cfg.max_loss_usd > 0 and pl <= -cfg.max_loss_usd:
                    exit_px, hit = bar["close"], "max-loss"

            bars_held = i - open_trade["bar"]
            max_bars = 0
            if cfg.max_trade_age_seconds > 0:
                max_bars = max(1, cfg.max_trade_age_seconds // (tf_minutes * 60))
            elif cfg.max_trade_age_minutes > 0:
                max_bars = max(1, cfg.max_trade_age_minutes // tf_minutes)
            if not hit and max_bars and bars_held >= max_bars:
                exit_px, hit = bar["close"], "time"

            if hit:
                move = ((exit_px - open_trade["entry"]) if open_trade["dir"] == "BUY"
                        else (open_trade["entry"] - exit_px))
                profit = move * usd_per_unit - spread * usd_per_unit  # spread charged once
                trades.append(SimTrade(open_trade["dir"], open_trade["entry"],
                                       exit_px, profit, bars_held, hit))
                open_trade = None

        if open_trade:
            continue

        # ── Look for a new entry (no lookahead: slice everything up to `now`) ─
        hist_trigger = df_trigger.iloc[: i + 1]
        hist_mid = df_mid[df_mid["time"] <= now]
        hist_macro = df_macro[df_macro["time"] <= now]
        if len(hist_mid) < warmup or len(hist_macro) < warmup:
            continue

        result = analyzer.analyze(symbol, hist_macro, hist_mid, hist_mid, hist_trigger)
        if result.signal == Signal.HOLD or result.atr <= 0:
            continue

        sl_dist = result.atr * cfg.risk.sl_atr_multiplier
        tp_dist = sl_dist * cfg.risk.rr_ratio
        entry = bar["close"]
        open_trade = {
            "dir": result.signal.value,
            "entry": entry,
            "sl": entry - sl_dist if result.signal == Signal.BUY else entry + sl_dist,
            "tp": entry + tp_dist if result.signal == Signal.BUY else entry - tp_dist,
            "bar": i,
        }

    connector.disconnect()
    _report(trades, df_trigger, tf_minutes)


def _report(trades: list[SimTrade], df, tf_minutes: int) -> None:
    if not trades:
        print("No trades taken. Filters may be too strict for this period.")
        return

    wins = [t for t in trades if t.profit_usd > 0]
    losses = [t for t in trades if t.profit_usd <= 0]
    total = sum(t.profit_usd for t in trades)
    gross_win = sum(t.profit_usd for t in wins)
    gross_loss = abs(sum(t.profit_usd for t in losses))

    # Max drawdown on the running equity curve
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for t in trades:
        equity += t.profit_usd
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    span_days = (df.iloc[-1]["time"] - df.iloc[260]["time"]).total_seconds() / 86400

    print(f"Period tested   : {span_days:.1f} days")
    print(f"Trades          : {len(trades)}  ({len(trades)/max(span_days,1):.1f}/day)")
    print(f"Win rate        : {len(wins)/len(trades)*100:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win / loss  : ${gross_win/max(len(wins),1):.2f} / -${gross_loss/max(len(losses),1):.2f}")
    print(f"Profit factor   : {gross_win/gross_loss:.2f}" if gross_loss else "Profit factor   : ∞")
    print(f"Max drawdown    : -${max_dd:.2f}")
    print(f"Avg hold        : {sum(t.bars_held for t in trades)/len(trades)*tf_minutes:.0f} min")
    print("-" * 62)
    verdict = "PROFITABLE" if total > 0 else "LOSING"
    print(f"NET RESULT      : ${total:+.2f}   → {verdict}")
    print("-" * 62)

    by_reason: dict[str, list[float]] = {}
    for t in trades:
        by_reason.setdefault(t.reason, []).append(t.profit_usd)
    print("Exits:")
    for reason, pls in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"  {reason:<14} {len(pls):>4} trades   ${sum(pls):+9.2f}")

    print("\nNote: real results will be worse — this ignores slippage, "
          "requotes and variable spread.")


if __name__ == "__main__":
    if mt5 is None:
        print("MetaTrader5 package not available (Windows only).")
        sys.exit(1)

    cfg = load_config()
    symbol = sys.argv[1] if len(sys.argv) > 1 else cfg.symbols[0]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    run_backtest(symbol, days)
