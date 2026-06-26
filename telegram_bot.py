"""
Telegram bot handlers and notification sender.
Uses python-telegram-bot v20 async API. All command handlers are async.
Authentication enforced via allowed_chat_ids whitelist.
"""

from __future__ import annotations

import logging
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Callable

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from config import TelegramConfig
from mt5_connector import OrderResult, PositionInfo, DealInfo
from risk_manager import TradeParameters

if TYPE_CHECKING:
    from bot import TradingEngine

logger = logging.getLogger(__name__)


def _auth_required(handler: Callable) -> Callable:
    """Decorator: silently ignore messages from unauthorized chat IDs."""
    @wraps(handler)
    async def wrapper(self: "TradingBotTelegram", update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id not in self._config.allowed_chat_ids:
            logger.warning("Unauthorized access attempt from chat_id=%s", chat_id)
            return
        return await handler(self, update, context)
    return wrapper


class TradingBotTelegram:
    def __init__(self, config: TelegramConfig, engine: "TradingEngine"):
        self._config = config
        self._engine = engine
        self._app: Application | None = None

    def build_application(self) -> Application:
        self._app = (
            ApplicationBuilder()
            .token(self._config.bot_token)
            .build()
        )
        self._app.add_handler(CommandHandler("start", self.cmd_start))
        self._app.add_handler(CommandHandler("status", self.cmd_status))
        self._app.add_handler(CommandHandler("balance", self.cmd_balance))
        self._app.add_handler(CommandHandler("positions", self.cmd_positions))
        self._app.add_handler(CommandHandler("analyze", self.cmd_analyze))
        self._app.add_handler(CommandHandler("run", self.cmd_run))
        self._app.add_handler(CommandHandler("stop", self.cmd_stop))
        self._app.add_handler(CommandHandler("resume", self.cmd_resume))
        self._app.add_handler(CommandHandler("help", self.cmd_help))
        return self._app

    # ── Command handlers ────────────────────────────────────────────────────

    @_auth_required
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 *MT5 Trading Bot online.*\n\nType /help for available commands.",
            parse_mode=ParseMode.MARKDOWN,
        )

    @_auth_required
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "*Available Commands*\n\n"
            "/status — Bot state, MT5 connection, next scan, equity\n"
            "/balance — Account balance / equity / margin\n"
            "/positions — All open positions with P&L\n"
            "/analyze `[SYMBOL]` — On-demand analysis (no trade placed)\n"
            "/run — Force immediate trade scan cycle\n"
            "/stop — Pause automated trading\n"
            "/resume — Resume automated trading\n"
            "/help — This message"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_auth_required
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            snap = await self._engine.get_status_snapshot()
            positions = await self._engine.get_open_positions()
            text = (
                f"*Bot Status*\n"
                f"Trading enabled: `{'✅ Yes' if snap['trading_enabled'] else '🔴 Paused'}`\n"
                f"MT5 connected: `{'✅ Yes' if snap['mt5_connected'] else '❌ No'}`\n"
                f"Open positions: `{len(positions)}`\n"
                f"Equity: `{snap.get('equity', 'N/A')}`\n"
                f"Next scan: `{snap.get('next_scan', 'N/A')}`\n"
                f"Last scan: `{snap.get('last_scan', 'N/A')}`"
            )
        except Exception as e:
            text = f"❌ Error fetching status: {e}"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_auth_required
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            acc = await self._engine.get_account_info()
            text = (
                f"*Account Balance*\n"
                f"Balance: `{acc.balance:.2f} {acc.currency}`\n"
                f"Equity: `{acc.equity:.2f} {acc.currency}`\n"
                f"Margin used: `{acc.margin:.2f} {acc.currency}`\n"
                f"Free margin: `{acc.free_margin:.2f} {acc.currency}`\n"
                f"Margin level: `{acc.margin_level:.1f}%`\n"
                f"Leverage: `1:{acc.leverage}`"
            )
        except Exception as e:
            text = f"❌ Error fetching balance: {e}"
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    @_auth_required
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            positions = await self._engine.get_open_positions()
            if not positions:
                await update.message.reply_text("No open positions.")
                return
            lines = ["*Open Positions*\n"]
            for p in positions:
                pnl_emoji = "🟢" if p.profit >= 0 else "🔴"
                lines.append(
                    f"{pnl_emoji} *{p.symbol}* `{p.order_type.value}`\n"
                    f"  Volume: `{p.volume}` | Open: `{p.open_price:.5g}`\n"
                    f"  SL: `{p.sl:.5g}` | TP: `{p.tp:.5g}`\n"
                    f"  P&L: `{p.profit:+.2f}`\n"
                    f"  Opened: `{p.open_time.strftime('%Y-%m-%d %H:%M')}`\n"
                )
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")

    @_auth_required
    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        symbols = [args[0].upper()] if args else self._engine.symbols
        await update.message.reply_text(f"🔍 Analyzing {', '.join(symbols)}…")
        for symbol in symbols:
            try:
                result = await self._engine.analyze_symbol(symbol)
                report = self._engine.analyzer.build_scan_report([result])
                await update.message.reply_text(report, parse_mode=ParseMode.MARKDOWN)
            except Exception as e:
                await update.message.reply_text(f"❌ Error analyzing {symbol}: {e}")

    @_auth_required
    async def cmd_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("⚡ Forcing immediate scan cycle…")
        try:
            await self._engine.run_scan_cycle()
            await update.message.reply_text("✅ Scan cycle complete.")
        except Exception as e:
            await update.message.reply_text(f"❌ Error during scan: {e}")

    @_auth_required
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._engine.trading_enabled = False
        await update.message.reply_text(
            "🔴 *Trading paused.* Open positions are untouched.\n"
            "Use /resume to re-enable.",
            parse_mode=ParseMode.MARKDOWN,
        )

    @_auth_required
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self._engine.trading_enabled = True
        await update.message.reply_text("✅ *Trading resumed.*", parse_mode=ParseMode.MARKDOWN)

    # ── Outbound alerts ─────────────────────────────────────────────────────

    async def send_trade_opened(self, result: OrderResult, params: TradeParameters):
        direction_emoji = "🟢" if params.order_type.value == "BUY" else "🔴"
        text = (
            f"{direction_emoji} *Trade Opened*\n"
            f"Symbol: `{params.symbol}`\n"
            f"Direction: `{params.order_type.value}`\n"
            f"Volume: `{params.volume} lots`\n"
            f"Entry: `{result.price:.5g}`\n"
            f"Stop Loss: `{params.sl_price:.5g}`\n"
            f"Take Profit: `{params.tp_price:.5g}`\n"
            f"Risk: `${params.risk_amount:.2f}` (`{params.risk_percent:.2f}% equity`)\n"
            f"Ticket: `{result.ticket}`"
        )
        await self.send_message_to_all(text)

    async def send_trade_closed(self, deal: DealInfo):
        pnl_emoji = "💰" if deal.profit >= 0 else "💸"
        text = (
            f"{pnl_emoji} *Trade Closed*\n"
            f"Symbol: `{deal.symbol}`\n"
            f"Volume: `{deal.volume}`\n"
            f"Close Price: `{deal.price:.5g}`\n"
            f"P&L: `{deal.profit:+.2f}`\n"
            f"Time: `{deal.time.strftime('%Y-%m-%d %H:%M UTC')}`"
        )
        await self.send_message_to_all(text)

    async def send_scan_summary(self, symbol_results: list, trades_opened: int):
        """Send a brief scan cycle summary (only if signal found or trade opened)."""
        has_signal = any(r.signal.value != "HOLD" for r in symbol_results)
        if not has_signal and trades_opened == 0:
            return  # silent if nothing to report

        from analyzer import TechnicalAnalyzer
        report = self._engine.analyzer.build_scan_report(symbol_results)
        if trades_opened:
            report = f"⚡ *{trades_opened} trade(s) opened this cycle*\n\n" + report
        await self.send_message_to_all(report)

    async def send_error_alert(self, error: str, context_str: str = ""):
        text = f"⚠️ *Bot Error*\n`{error}`"
        if context_str:
            text += f"\nContext: _{context_str}_"
        await self.send_message_to_all(text)

    async def send_message_to_all(self, text: str, parse_mode: str = ParseMode.MARKDOWN):
        if self._app is None:
            logger.warning("Telegram app not built, cannot send message")
            return
        for chat_id in self._config.allowed_chat_ids:
            try:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode,
                )
            except Exception as e:
                logger.error("Failed to send Telegram message to %s: %s", chat_id, e)
