"""
Main orchestrator. Runs everything in a single asyncio event loop:
- APScheduler (15-min scan cycle + 30-sec position monitor)
- Telegram Application (polling)
- TradingEngine (bridges async event loop to synchronous MT5 calls)

Run: python bot.py
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import signal
import sys
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from analyzer import AnalysisResult, TechnicalAnalyzer
from config import AppConfig, load_config
from mt5_connector import AccountInfo, MT5Connector, OrderError, OrderType

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from risk_manager import RiskManager
from telegram_bot import TradingBotTelegram

logger = logging.getLogger(__name__)


def setup_logging(level: str, log_file: str) -> None:
    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
    ]
    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
        )
    except Exception:
        pass
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt, handlers=handlers)


class TradingEngine:
    """
    Coordinates analysis → risk calculation → order placement.
    All public methods are async. MT5 calls are offloaded to a thread executor
    via asyncio.to_thread, serialized by _mt5_lock to prevent concurrent access.
    """

    def __init__(
        self,
        config: AppConfig,
        connector: MT5Connector,
        analyzer: TechnicalAnalyzer,
        risk_manager: RiskManager,
    ):
        self._config = config
        self._connector = connector
        self.analyzer = analyzer
        self._risk_manager = risk_manager
        self._mt5_lock = asyncio.Lock()
        self.trading_enabled = True
        self.symbols = config.symbols
        self._known_position_tickets: set[int] = set()
        self._position_open_times: dict[int, datetime] = {}  # bot-opened ticket → UTC open time
        self._peak_profit: dict[int, float] = {}             # ticket → best profit seen (for trailing)
        self._last_scan: datetime | None = None
        self._next_scan: datetime | None = None
        self._tg_bot: TradingBotTelegram | None = None

    def set_telegram(self, tg_bot: TradingBotTelegram) -> None:
        self._tg_bot = tg_bot

    # ── MT5 bridge ──────────────────────────────────────────────────────────

    async def _mt5(self, fn, *args, **kwargs):
        """Run a synchronous MT5 call in a thread, serialized by lock."""
        async with self._mt5_lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def _active_symbols(self) -> list[str]:
        """
        Symbols that are actually tradeable right now.
        Falls back to WEEKEND_SYMBOLS (e.g. crypto) when the main markets are shut,
        so the bot keeps working on weekends and holidays.
        """
        if not self._config.skip_closed_markets:
            return self.symbols

        open_main = []
        for sym in self.symbols:
            if await self._mt5(self._connector.is_market_open, sym):
                open_main.append(sym)

        if open_main:
            return open_main

        fallback = []
        for sym in self._config.weekend_symbols:
            if await self._mt5(self._connector.is_market_open, sym):
                fallback.append(sym)

        if fallback:
            logger.info("Main markets closed — trading weekend symbols: %s", ", ".join(fallback))
        return fallback

    def _within_trading_hours(self) -> bool:
        """True if the current UTC hour is inside the configured trading window."""
        start = self._config.trade_start_hour
        end = self._config.trade_end_hour
        if start == end or (start == 0 and end >= 24):
            return True  # always on
        hour = datetime.utcnow().hour
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end  # window wraps past midnight

    # ── Core analysis ────────────────────────────────────────────────────────

    async def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Fetch 4 timeframes and run multi-timeframe analysis."""
        if self._config.balanced_mode:
            # Balanced scalp: H1 macro trend, M15 intermediate, M5 trigger.
            # Quick trades (M5 bars) but on a timeframe where the move dwarfs the spread.
            tf_h1 = mt5.TIMEFRAME_H1 if mt5 else 16385
            tf_m15 = mt5.TIMEFRAME_M15 if mt5 else 15
            tf_m5 = mt5.TIMEFRAME_M5 if mt5 else 5

            df_macro = await self._mt5(self._connector.get_ohlcv, symbol, tf_h1, 300)
            df_mid = await self._mt5(self._connector.get_ohlcv, symbol, tf_m15, 300)
            df_trigger = await self._mt5(self._connector.get_ohlcv, symbol, tf_m5, 300)
            return self.analyzer.analyze(symbol, df_macro, df_mid, df_mid, df_trigger,
                                         use_forming_bar=False)

        if self._config.scalp_mode:
            # Scalp mode: M15 macro trend, M5 intermediate, M1 trigger.
            # The analyzer slots are timeframe-agnostic, so we feed faster charts
            # into the same D1/H4/H1/M15 positions. ATR then comes from M1 → tight stops.
            tf_m15 = mt5.TIMEFRAME_M15 if mt5 else 15
            tf_m5 = mt5.TIMEFRAME_M5 if mt5 else 5
            tf_m1 = mt5.TIMEFRAME_M1 if mt5 else 1

            df_macro = await self._mt5(self._connector.get_ohlcv, symbol, tf_m15, 300)
            df_mid = await self._mt5(self._connector.get_ohlcv, symbol, tf_m5, 300)
            df_trigger = await self._mt5(self._connector.get_ohlcv, symbol, tf_m1, 300)
            return self.analyzer.analyze(symbol, df_macro, df_mid, df_mid, df_trigger,
                                         use_forming_bar=self._config.turbo_mode)

        tf_d1 = mt5.TIMEFRAME_D1 if mt5 else 16408
        tf_h4 = mt5.TIMEFRAME_H4 if mt5 else 16388
        tf_h1 = mt5.TIMEFRAME_H1 if mt5 else 16385
        tf_m15 = mt5.TIMEFRAME_M15 if mt5 else 15

        df_d1 = await self._mt5(self._connector.get_ohlcv, symbol, tf_d1, 300)
        df_h4 = await self._mt5(self._connector.get_ohlcv, symbol, tf_h4, 250)
        df_h1 = await self._mt5(self._connector.get_ohlcv, symbol, tf_h1, 200)
        df_m15 = await self._mt5(self._connector.get_ohlcv, symbol, tf_m15, 200)

        return self.analyzer.analyze(symbol, df_d1, df_h4, df_h1, df_m15,
                                     use_forming_bar=self._config.turbo_mode)

    # ── Trade execution ──────────────────────────────────────────────────────

    async def execute_trade(self, analysis: AnalysisResult) -> bool:
        """
        Place a market order for the given analysis result.
        Returns True if a trade was opened.
        """
        account = await self._mt5(self._connector.get_account_info)
        symbol_info = await self._mt5(self._connector.get_symbol_info, analysis.symbol)

        fixed_lot = self._config.fixed_lots.get(analysis.symbol)
        params = self._risk_manager.calculate_trade_parameters(analysis, account, symbol_info, fixed_lot=fixed_lot)
        if params is None:
            logger.warning("%s: RiskManager returned None — trade skipped (ATR=%.5g, equity=%.2f, vol_min=%.3f, contract=%.1f)",
                           analysis.symbol, analysis.atr, account.equity,
                           symbol_info.volume_min, symbol_info.trade_contract_size)
            if self._tg_bot:
                await self._tg_bot.send_message_to_all(
                    f"⚠️ *Trade skipped: {analysis.symbol}*\n"
                    f"Signal: `{analysis.signal.value}` | ATR: `{analysis.atr:.5g}`\n"
                    f"Equity: `{account.equity:.2f}` | Min lot: `{symbol_info.volume_min}` "
                    f"| Contract: `{symbol_info.trade_contract_size}`\n"
                    f"_Risk parameters could not be met — see bot logs_"
                )
            return False

        try:
            result = await self._mt5(
                self._connector.place_market_order,
                params.symbol,
                params.order_type,
                params.volume,
                params.sl_price,
                params.tp_price,
                f"Bot {analysis.signal.value} {analysis.trend_strength.label}",
            )
        except OrderError as e:
            logger.error("Order failed for %s: %s (retcode=%d)", analysis.symbol, e, e.retcode)
            if self._tg_bot:
                await self._tg_bot.send_error_alert(
                    str(e), context_str=f"{analysis.symbol} {analysis.signal.value}"
                )
            return False

        # Track open time locally for the time-based exit (avoids MT5 server-time offsets)
        self._position_open_times[result.ticket] = datetime.utcnow()

        if self._tg_bot:
            await self._tg_bot.send_trade_opened(result, params)

        return True

    # ── Scheduled jobs ───────────────────────────────────────────────────────

    async def run_scan_cycle(self) -> None:
        """Scan cycle: analyze all symbols and trade on signals."""
        self._last_scan = datetime.utcnow()
        self._next_scan = self._last_scan + timedelta(seconds=self._config.scan_interval_seconds)
        logger.info("=== Scan cycle started at %s ===", self._last_scan.strftime("%H:%M:%S UTC"))

        if not self.trading_enabled:
            logger.info("Trading is paused — skipping cycle")
            return

        if not self._within_trading_hours():
            logger.info("Outside trading hours (%02d:00–%02d:00 UTC) — no new trades",
                        self._config.trade_start_hour, self._config.trade_end_hour)
            return

        # Verify MT5 connection
        connected = await self._mt5(self._connector.is_connected)
        if not connected:
            logger.warning("MT5 not connected — attempting reconnect")
            try:
                await self._mt5(self._connector.reconnect)
            except Exception as e:
                logger.error("Reconnect failed: %s", e)
                if self._tg_bot:
                    await self._tg_bot.send_error_alert(f"MT5 reconnect failed: {e}")
                return

        analysis_results: list[AnalysisResult] = []
        trades_opened = 0

        active = await self._active_symbols()
        if not active:
            logger.info("No markets open right now (weekend/holiday) — skipping cycle")
            return

        for symbol in active:
            try:
                logger.info("Analyzing %s…", symbol)
                analysis = await self.analyze_symbol(symbol)
                analysis_results.append(analysis)

                logger.info(
                    "%s signal=%s trend=%s RSI=%.1f ADX=%.1f",
                    symbol, analysis.signal.value,
                    analysis.trend_strength.label,
                    analysis.rsi, analysis.adx,
                )

                if analysis.signal.value == "HOLD":
                    continue

                # Check existing positions on this symbol (case-insensitive — Axi uses lowercase symbols)
                all_positions = await self._mt5(self._connector.get_open_positions)
                existing = [p for p in all_positions if p.symbol.upper() == symbol.upper()]
                if len(existing) >= self._config.max_positions_per_symbol:
                    logger.info("%s: %d position(s) open (max %d) — skip",
                                symbol, len(existing), self._config.max_positions_per_symbol)
                    continue

                opened = await self.execute_trade(analysis)
                if opened:
                    trades_opened += 1

            except Exception as e:
                logger.error("Error processing %s: %s", symbol, e, exc_info=True)
                if self._tg_bot:
                    await self._tg_bot.send_error_alert(str(e), context_str=symbol)

            # Brief pause between symbols to avoid MT5 rate issues
            await asyncio.sleep(1)

        if self._tg_bot and analysis_results:
            await self._tg_bot.send_scan_summary(analysis_results, trades_opened)

        logger.info("=== Scan cycle complete. Trades opened: %d ===", trades_opened)

    async def monitor_closed_positions(self) -> None:
        """
        30-second job: detect positions that closed since the last check
        (hit SL or TP) and send Telegram alerts.
        """
        try:
            current_positions = await self._mt5(self._connector.get_open_positions)
            current_tickets = {p.ticket for p in current_positions}

            # Effective max age in seconds (seconds setting wins if set, else minutes)
            max_age_seconds = self._config.max_trade_age_seconds
            if max_age_seconds <= 0 and self._config.max_trade_age_minutes > 0:
                max_age_seconds = self._config.max_trade_age_minutes * 60

            quick_profit = self._config.quick_profit_usd
            # Percentage-of-equity target overrides the fixed $ target when set
            if self._config.quick_profit_percent > 0 and current_positions:
                try:
                    acct = await self._mt5(self._connector.get_account_info)
                    quick_profit = acct.equity * self._config.quick_profit_percent / 100
                except Exception as e:
                    logger.error("Could not fetch equity for %%-profit target: %s", e)
            now = datetime.utcnow()

            for p in current_positions:
                # Quick-profit exit: grab profit the moment it appears (turbo)
                if quick_profit > 0 and p.profit >= quick_profit:
                    logger.info("Ticket %d (%s) hit quick-profit $%.2f — closing (P/L %.2f)",
                                p.ticket, p.symbol, quick_profit, p.profit)
                    try:
                        await self._mt5(self._connector.close_position, p.ticket)
                        self._position_open_times.pop(p.ticket, None)
                        if self._tg_bot:
                            await self._tg_bot.send_message_to_all(
                                f"💰 *Quick profit: {p.symbol}*\n"
                                f"Ticket `{p.ticket}` closed at `{p.profit:+.2f}`"
                            )
                    except Exception as e:
                        logger.error("Quick-profit exit failed for ticket %d: %s", p.ticket, e)
                    continue

                # Hard loss cap: close immediately if loss reaches the cap
                if self._config.max_loss_usd > 0 and p.profit <= -self._config.max_loss_usd:
                    logger.info("Ticket %d (%s) hit max-loss -$%.2f — closing (P/L %.2f)",
                                p.ticket, p.symbol, self._config.max_loss_usd, p.profit)
                    try:
                        await self._mt5(self._connector.close_position, p.ticket)
                        self._position_open_times.pop(p.ticket, None)
                        if self._tg_bot:
                            await self._tg_bot.send_message_to_all(
                                f"🛑 *Stop-out: {p.symbol}*\n"
                                f"Ticket `{p.ticket}` closed at `{p.profit:+.2f}` (loss cap)"
                            )
                    except Exception as e:
                        logger.error("Max-loss exit failed for ticket %d: %s", p.ticket, e)
                    continue

                # Trailing stop: once in profit, ratchet the SL up behind the peak
                if self._config.trail_activate_usd > 0 and self._config.trail_distance_usd > 0:
                    try:
                        await self._update_trailing_stop(p)
                    except Exception as e:
                        logger.error("Trailing stop failed for ticket %d: %s", p.ticket, e)

                # Time-based exit: close bot trades that exceeded max age
                if max_age_seconds > 0:
                    opened = self._position_open_times.get(p.ticket)
                    if opened and (now - opened) > timedelta(seconds=max_age_seconds):
                        logger.info("Ticket %d (%s) older than %ds — time exit (P/L %.2f)",
                                    p.ticket, p.symbol, max_age_seconds, p.profit)
                        try:
                            await self._mt5(self._connector.close_position, p.ticket)
                            self._position_open_times.pop(p.ticket, None)
                            if self._tg_bot:
                                await self._tg_bot.send_message_to_all(
                                    f"⏱ *Time exit: {p.symbol}*\n"
                                    f"Ticket `{p.ticket}` held > {max_age_seconds}s\n"
                                    f"P/L: `{p.profit:+.2f}`"
                                )
                        except Exception as e:
                            logger.error("Time exit failed for ticket %d: %s", p.ticket, e)

            # Find tickets that were known but are now gone
            closed_tickets = self._known_position_tickets - current_tickets

            if closed_tickets:
                now = datetime.utcnow()
                deals = await self._mt5(
                    self._connector.get_historical_deals,
                    now - timedelta(hours=1),
                    now,
                )
                for deal in deals:
                    if deal.position_id in closed_tickets and self._tg_bot:
                        await self._tg_bot.send_trade_closed(deal)

            self._known_position_tickets = current_tickets
            # Prune open-time entries for tickets that no longer exist
            self._position_open_times = {
                t: v for t, v in self._position_open_times.items() if t in current_tickets
            }
            self._peak_profit = {
                t: v for t, v in self._peak_profit.items() if t in current_tickets
            }
        except Exception as e:
            logger.error("Position monitor error: %s", e)

    async def _update_trailing_stop(self, p) -> None:
        """
        Move the SL to lock in profit once the trade is far enough ahead.
        Locks (peak_profit - trail_distance) dollars, and never moves the SL
        backwards. Converts dollars to a price offset via contract size.
        """
        peak = max(self._peak_profit.get(p.ticket, 0.0), p.profit)
        self._peak_profit[p.ticket] = peak

        if peak < self._config.trail_activate_usd:
            return  # not far enough ahead yet

        locked_usd = peak - self._config.trail_distance_usd
        if locked_usd <= 0:
            return

        info = await self._mt5(self._connector.get_symbol_info, p.symbol)
        usd_per_price_unit = p.volume * info.trade_contract_size
        if usd_per_price_unit <= 0:
            return

        offset = locked_usd / usd_per_price_unit
        if p.order_type == OrderType.BUY:
            new_sl = round(p.open_price + offset, info.digits)
            better = new_sl > p.sl  # only ever ratchet upward
        else:
            new_sl = round(p.open_price - offset, info.digits)
            better = p.sl == 0 or new_sl < p.sl

        if not better:
            return

        await self._mt5(self._connector.modify_position_sl_tp, p.ticket, new_sl, p.tp)
        logger.info("Ticket %d (%s): trailing SL → %.5g (locks $%.2f, peak $%.2f)",
                    p.ticket, p.symbol, new_sl, locked_usd, peak)

    # ── Public interface for Telegram handlers ───────────────────────────────

    async def get_status_snapshot(self) -> dict:
        connected = await self._mt5(self._connector.is_connected)
        try:
            acc = await self._mt5(self._connector.get_account_info)
            equity_str = f"{acc.equity:.2f} {acc.currency}"
        except Exception:
            equity_str = "N/A"

        return {
            "trading_enabled": self.trading_enabled,
            "mt5_connected": connected,
            "equity": equity_str,
            "last_scan": self._last_scan.strftime("%Y-%m-%d %H:%M UTC") if self._last_scan else "Never",
            "next_scan": self._next_scan.strftime("%Y-%m-%d %H:%M UTC") if self._next_scan else "N/A",
        }

    async def get_open_positions(self):
        return await self._mt5(self._connector.get_open_positions)

    async def get_account_info(self) -> AccountInfo:
        return await self._mt5(self._connector.get_account_info)


# ── Application entry point ──────────────────────────────────────────────────

async def main() -> None:
    config = load_config()
    setup_logging(config.log_level, config.log_file)
    logger.info("Starting MT5 Trading Bot")

    connector = MT5Connector(config.mt5)
    analyzer = TechnicalAnalyzer(config.indicators, config.risk)
    risk_manager = RiskManager(config.risk)

    # Connect to MT5 before the event loop gets busy
    logger.info("Connecting to MT5…")
    await asyncio.to_thread(connector.connect)

    engine = TradingEngine(config, connector, analyzer, risk_manager)

    tg_bot = TradingBotTelegram(config.telegram, engine)
    application = tg_bot.build_application()
    engine.set_telegram(tg_bot)

    # Seed known positions so we don't flood alerts on startup
    initial_positions = await asyncio.to_thread(connector.get_open_positions)
    engine._known_position_tickets = {p.ticket for p in initial_positions}
    logger.info("Startup: %d open position(s) found", len(initial_positions))

    scheduler = AsyncIOScheduler(timezone=config.timezone)
    scan_seconds = config.scan_interval_seconds
    scheduler.add_job(
        engine.run_scan_cycle,
        trigger=IntervalTrigger(seconds=scan_seconds),
        id="scan_cycle",
        name=f"{scan_seconds}s scan cycle",
        misfire_grace_time=min(60, scan_seconds),
        coalesce=True,
        max_instances=1,
    )
    monitor_seconds = config.monitor_interval_seconds
    scheduler.add_job(
        engine.monitor_closed_positions,
        trigger=IntervalTrigger(seconds=monitor_seconds),
        id="position_monitor",
        name="Position closed monitor",
        misfire_grace_time=min(30, monitor_seconds),
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()
    logger.info("Scheduler started: scan every %ds, monitor every %ds", scan_seconds, monitor_seconds)

    # Shutdown handler
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler

    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)

        # Send startup notification
        if config.balanced_mode:
            if config.trade_start_hour == 0 and config.trade_end_hour >= 24:
                hours = "all day"
            else:
                hours = f"{config.trade_start_hour:02d}:00–{config.trade_end_hour:02d}:00 UTC"
            mode_line = (
                f"⚖️ BALANCED: M5 trigger (H1/M15 trend), ADX≥{config.risk.min_adx:.0f}, "
                f"1 trade/symbol, hours: {hours}\n"
            )
        elif config.turbo_mode:
            mode_line = (
                f"🔥 TURBO: live-bar entries, quick-profit ${config.quick_profit_usd:.0f}, "
                f"time exit {config.max_trade_age_seconds}s, max {config.max_positions_per_symbol} trades\n"
            )
        elif config.scalp_mode:
            mode_line = (
                f"⚡ Scalp mode: M1 trigger, max {config.max_positions_per_symbol} trades/symbol, "
                f"time exit {config.max_trade_age_minutes} min\n"
            )
        else:
            mode_line = ""
        await tg_bot.send_message_to_all(
            f"🚀 *MT5 Bot started*\n"
            f"Symbols: `{', '.join(config.symbols)}`\n"
            f"{mode_line}"
            f"Scanning every {config.scan_interval_seconds} seconds\n"
            f"Max risk per trade: `{config.risk.max_risk_percent}%`"
        )

        # Run immediate first scan
        logger.info("Running initial scan cycle…")
        try:
            await engine.run_scan_cycle()
        except Exception as e:
            logger.error("Initial scan error: %s", e)

        logger.info("Bot running. Press Ctrl+C to stop.")
        await stop_event.wait()

        logger.info("Shutting down…")
        await tg_bot.send_message_to_all("🔴 *Bot shutting down.*")
        await application.updater.stop()
        await application.stop()

    scheduler.shutdown(wait=False)
    await asyncio.to_thread(connector.disconnect)
    logger.info("Bot shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
