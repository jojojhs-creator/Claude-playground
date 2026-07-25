"""
MT5 connector — the only module that imports MetaTrader5.
All public methods are synchronous and must be called from a thread executor.
Uses threading.RLock to protect against concurrent MT5 API access.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Callable

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # allows importing on non-Windows for testing

from config import MT5Config

logger = logging.getLogger(__name__)


class MT5Error(Exception):
    pass


class OrderError(Exception):
    def __init__(self, message: str, retcode: int):
        super().__init__(message)
        self.retcode = retcode


class OrderType(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SymbolInfo:
    name: str
    bid: float
    ask: float
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float
    currency_profit: str
    trade_stops_level: int = 0  # broker min stop distance in points
    tick_time: int = 0          # epoch seconds of last tick (0 = unknown)
    trade_mode: int = 0         # 0 = disabled, 4 = full access (mt5.SYMBOL_TRADE_MODE_*)


@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int


@dataclass
class PositionInfo:
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    open_price: float
    sl: float
    tp: float
    profit: float
    open_time: datetime
    comment: str


@dataclass
class OrderResult:
    ticket: int
    symbol: str
    volume: float
    price: float
    order_type: OrderType
    comment: str


@dataclass
class DealInfo:
    ticket: int
    position_id: int
    symbol: str
    volume: float
    price: float
    profit: float
    time: datetime
    comment: str


def _with_reconnect(max_retries: int = 3, delay: float = 5.0):
    """Retry decorator: on MT5Error reconnects and retries up to max_retries times."""
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(self: "MT5Connector", *args, **kwargs):
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(self, *args, **kwargs)
                except MT5Error as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning("MT5Error on %s (attempt %d/%d): %s — reconnecting in %.0fs",
                                       fn.__name__, attempt + 1, max_retries, e, delay)
                        time.sleep(delay)
                        self.reconnect()
            raise last_error
        return wrapper
    return decorator


class MT5Connector:
    MAGIC = 234_001  # identifies orders placed by this bot

    def __init__(self, config: MT5Config):
        self._config = config
        self._lock = threading.RLock()

    def connect(self) -> bool:
        if mt5 is None:
            raise MT5Error("MetaTrader5 package not available (Windows only)")
        with self._lock:
            kwargs: dict = {
                "login": self._config.login,
                "password": self._config.password,
                "server": self._config.server,
            }
            if self._config.path:
                kwargs["path"] = self._config.path

            if not mt5.initialize(**kwargs):
                err = mt5.last_error()
                raise MT5Error(f"MT5 initialize failed: {err}")

            if not mt5.login(self._config.login, self._config.password, self._config.server):
                err = mt5.last_error()
                raise MT5Error(f"MT5 login failed: {err}")

            info = mt5.account_info()
            if info is None:
                raise MT5Error("Connected but could not fetch account info")

            logger.info("MT5 connected: account=%d server=%s balance=%.2f %s",
                        info.login, info.server, info.balance, info.currency)
            return True

    def disconnect(self) -> None:
        if mt5 is None:
            return
        with self._lock:
            mt5.shutdown()
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        if mt5 is None:
            return False
        with self._lock:
            info = mt5.terminal_info()
            return info is not None and info.connected

    def reconnect(self) -> bool:
        logger.info("Attempting MT5 reconnect...")
        self.disconnect()
        time.sleep(2)
        return self.connect()

    @_with_reconnect()
    def get_ohlcv(self, symbol: str, timeframe: int, count: int):
        """Returns a pandas DataFrame with columns: time, open, high, low, close, tick_volume."""
        import pandas as pd
        with self._lock:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            err = mt5.last_error() if mt5 else "N/A"
            raise MT5Error(f"Failed to fetch OHLCV for {symbol}: {err}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    @_with_reconnect()
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        with self._lock:
            info = mt5.symbol_info(symbol)
            if info is None:
                raise MT5Error(f"Symbol not found: {symbol}")
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise MT5Error(f"Could not fetch tick for {symbol}")
        return SymbolInfo(
            name=symbol,
            bid=tick.bid,
            ask=tick.ask,
            point=info.point,
            digits=info.digits,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
            currency_profit=info.currency_profit,
            trade_stops_level=int(info.trade_stops_level),
            tick_time=int(getattr(tick, "time", 0) or 0),
            trade_mode=int(getattr(info, "trade_mode", 0) or 0),
        )

    def is_market_open(self, symbol: str, max_tick_age_seconds: int = 300) -> bool:
        """
        True if the symbol looks tradeable right now.
        A market that is closed (weekend, holiday) stops producing ticks, so a
        stale last-tick time is the most reliable cross-broker signal.
        """
        try:
            info = self.get_symbol_info(symbol)
        except MT5Error:
            return False
        if info.trade_mode == 0:  # SYMBOL_TRADE_MODE_DISABLED
            return False
        if info.tick_time <= 0:
            return False
        age = time.time() - info.tick_time
        return age <= max_tick_age_seconds

    @_with_reconnect()
    def get_account_info(self) -> AccountInfo:
        with self._lock:
            info = mt5.account_info()
        if info is None:
            raise MT5Error("Failed to fetch account info")
        margin_level = info.margin_level if info.margin > 0 else 0.0
        return AccountInfo(
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.margin_free,
            margin_level=margin_level,
            currency=info.currency,
            leverage=info.leverage,
        )

    @_with_reconnect()
    def get_open_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        with self._lock:
            if symbol:
                raw = mt5.positions_get(symbol=symbol)
            else:
                raw = mt5.positions_get()
        if raw is None:
            return []
        return [
            PositionInfo(
                ticket=p.ticket,
                symbol=p.symbol,
                order_type=OrderType.BUY if p.type == 0 else OrderType.SELL,
                volume=p.volume,
                open_price=p.price_open,
                sl=p.sl,
                tp=p.tp,
                profit=p.profit,
                open_time=datetime.fromtimestamp(p.time),
                comment=p.comment,
            )
            for p in raw
        ]

    @_with_reconnect()
    def place_market_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> OrderResult:
        if mt5 is None:
            raise MT5Error("MetaTrader5 not available")

        with self._lock:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise MT5Error(f"Cannot fetch tick for {symbol}")

            if order_type == OrderType.BUY:
                price = tick.ask
                mt5_type = mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                mt5_type = mt5.ORDER_TYPE_SELL

            # Enforce broker minimum stop distance (trade_stops_level)
            sym_info = mt5.symbol_info(symbol)
            if sym_info and sym_info.trade_stops_level > 0:
                min_dist = (sym_info.trade_stops_level + 10) * sym_info.point
                if order_type == OrderType.BUY:
                    if sl > 0 and (price - sl) < min_dist:
                        sl = round(price - min_dist, sym_info.digits)
                        logger.info("SL adjusted to meet min stop distance: %.5g", sl)
                    if tp > 0 and (tp - price) < min_dist:
                        tp = round(price + min_dist, sym_info.digits)
                        logger.info("TP adjusted to meet min stop distance: %.5g", tp)
                else:
                    if sl > 0 and (sl - price) < min_dist:
                        sl = round(price + min_dist, sym_info.digits)
                        logger.info("SL adjusted to meet min stop distance: %.5g", sl)
                    if tp > 0 and (price - tp) < min_dist:
                        tp = round(price - min_dist, sym_info.digits)
                        logger.info("TP adjusted to meet min stop distance: %.5g", tp)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 30,
                "magic": self.MAGIC,
                "comment": comment[:31],  # MT5 truncates at 31 chars
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None:
                err = mt5.last_error()
                raise OrderError(f"order_send returned None for {symbol}: {err}", retcode=-1)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                # Try FOK filling as fallback
                request["type_filling"] = mt5.ORDER_FILLING_FOK
                result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                retcode = result.retcode if result else -1
                comment_out = result.comment if result else "None"
                raise OrderError(
                    f"Order failed for {symbol} {order_type.value}: retcode={retcode} {comment_out}",
                    retcode=retcode,
                )

        logger.info("Order placed: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f ticket=%d",
                    symbol, order_type.value, volume, result.price, sl, tp, result.order)
        return OrderResult(
            ticket=result.order,
            symbol=symbol,
            volume=volume,
            price=result.price,
            order_type=order_type,
            comment=comment,
        )

    @_with_reconnect()
    def close_position(self, ticket: int) -> OrderResult:
        if mt5 is None:
            raise MT5Error("MetaTrader5 not available")

        with self._lock:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                raise MT5Error(f"Position {ticket} not found")
            pos = positions[0]

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                raise MT5Error(f"Cannot fetch tick for {pos.symbol}")

            if pos.type == 0:  # BUY → close with SELL
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 30,
                "magic": self.MAGIC,
                "comment": "Bot close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else -1
            raise OrderError(f"Close failed for ticket {ticket}: retcode={retcode}", retcode=retcode)

        order_type = OrderType.SELL if pos.type == 0 else OrderType.BUY
        return OrderResult(
            ticket=result.order,
            symbol=pos.symbol,
            volume=pos.volume,
            price=result.price,
            order_type=order_type,
            comment="Bot close",
        )

    @_with_reconnect()
    def modify_position_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        if mt5 is None:
            raise MT5Error("MetaTrader5 not available")
        with self._lock:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                raise MT5Error(f"Position {ticket} not found for modify")
            pos = positions[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp,
            }
            result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            retcode = result.retcode if result else -1
            raise OrderError(f"Modify SL/TP failed for {ticket}: retcode={retcode}", retcode=retcode)
        return True

    @_with_reconnect()
    def get_historical_deals(self, from_dt: datetime, to_dt: datetime) -> list[DealInfo]:
        if mt5 is None:
            return []
        with self._lock:
            deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            return []
        return [
            DealInfo(
                ticket=d.ticket,
                position_id=d.position_id,
                symbol=d.symbol,
                volume=d.volume,
                price=d.price,
                profit=d.profit,
                time=datetime.fromtimestamp(d.time),
                comment=d.comment,
            )
            for d in deals
            if d.entry == 1  # entry=1 means deal out (position closed)
        ]
