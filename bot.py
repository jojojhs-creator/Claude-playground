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

    # ── Core analysis ────────────────────────────────────────────────────────

    async def analyze_symbol(self, symbol: str) -> AnalysisResult:
        """Fetch 4 timeframes and run multi-timeframe analysis."""
        tf_d1 = mt5.TIMEFRAME_D1 if mt5 else 16408
        tf_h4 = mt5.TIMEFRAME_H4 if mt5 else 16388
        tf_h1 = mt5.TIMEFRAME_H1 if mt5 else 16385
        tf_m15 = mt5.TIMEFRAME_M15 if mt5 else 15

        df_d1 = await self._mt5(self._connector.get_ohlcv, symbol, tf_d1, 300)
        df_h4 = await self._mt5(self._connector.get_ohlcv, symbol, tf_h4, 250)
        df_h1 = await self._mt5(self._connector.get_ohlcv, symbol, tf_h1, 200)
        df_m15 = await self._mt5(self._connector.get_ohlcv, symbol, tf_m15, 200)

        return self.analyzer.analyze(symbol, df_d1, df_h4, df_h1, df_m15)

    # ── Trade execution ──────────────────────────────────────────────────────

    async def execute_trade(self, analysis: AnalysisResult) -> bool:
        """
        Place a market order for the given analysis result.
        Returns True if a trade was opened.
        """
        account = await self._mt5(self._connector.get_account_info)
        symbol_info = await self._mt5(self._connector.get_symbol_info, analysis.symbol)

        params = self._risk_manager.calculate_trade_parameters(analysis, account, symbol_info)
        if params is None:
            logger.info("%s: RiskManager returned None — skipping trade", analysis.symbol)
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

        if self._tg_bot:
            await self._tg_bot.send_trade_opened(result, params)

        return True

    # ── Scheduled jobs ───────────────────────────────────────────────────────

    async def run_scan_cycle(self) -> None:
        """15-minute scan cycle: analyze all symbols and trade on signals."""
        self._last_scan = datetime.utcnow()
        self._next_scan = self._last_scan + timedelta(minutes=5)
        logger.info("=== Scan cycle started at %s ===", self._last_scan.strftime("%H:%M:%S UTC"))

        if not self.trading_enabled:
            logger.info("Trading is paused — skipping cycle")
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

        for symbol in self.symbols:
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

                # Check for existing position on this symbol
                existing = await self._mt5(self._connector.get_open_positions, symbol)
                if existing:
                    logger.info("%s: position already open (ticket=%d) — skip",
                                symbol, existing[0].ticket)
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
        except Exception as e:
            logger.error("Position monitor error: %s", e)

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
    scheduler.add_job(
        engine.run_scan_cycle,
        trigger=IntervalTrigger(minutes=5),
        id="scan_cycle",
        name="5-minute scan cycle",
        misfire_grace_time=60,
        coalesce=True,
        max_instances=1,
    )
    scheduler.add_job(
        engine.monitor_closed_positions,
        trigger=IntervalTrigger(seconds=30),
        id="position_monitor",
        name="Position closed monitor",
        misfire_grace_time=30,
        coalesce=True,
        max_instances=1,
    )
    scheduler.start()
    logger.info("Scheduler started: scan every 15 min, monitor every 30 sec")

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
        await tg_bot.send_message_to_all(
            f"🚀 *MT5 Bot started*\n"
            f"Symbols: `{', '.join(config.symbols)}`\n"
            f"Scanning every 5 minutes\n"
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
