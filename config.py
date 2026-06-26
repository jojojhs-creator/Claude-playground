"""
Configuration loader. Reads .env and exposes typed dataclasses.
Call load_config() once at startup; treat the result as an immutable singleton.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


class ConfigError(Exception):
    pass


@dataclass
class MT5Config:
    login: int
    password: str
    server: str
    path: str | None


@dataclass
class TelegramConfig:
    bot_token: str
    allowed_chat_ids: set[int]


@dataclass
class RiskConfig:
    max_risk_percent: float
    trend_weak_threshold: float
    trend_strong_threshold: float
    tp_multiplier_weak: float
    tp_multiplier_medium: float
    tp_multiplier_strong: float


@dataclass
class IndicatorConfig:
    ema_fast: int
    ema_slow: int
    ema_100: int
    sma_200: int
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    atr_period: int
    adx_period: int
    sr_lookback: int


@dataclass
class AppConfig:
    mt5: MT5Config
    telegram: TelegramConfig
    risk: RiskConfig
    indicators: IndicatorConfig
    symbols: list[str]
    timezone: str
    log_level: str
    log_file: str
    fixed_lots: dict = field(default_factory=dict)  # symbol → fixed lot size


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        raise ConfigError(f"Missing required environment variable: {key}")
    return val


def _int(key: str, default: int | None = None) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required integer env var: {key}")
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got: {raw!r}")


def _float(key: str, default: float | None = None) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required float env var: {key}")
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a float, got: {raw!r}")


def load_config(env_file: str = ".env") -> AppConfig:
    load_dotenv(env_file, override=False)

    login_str = _require("MT5_LOGIN")
    try:
        login = int(login_str)
    except ValueError:
        raise ConfigError(f"MT5_LOGIN must be a number, got: {login_str!r}")

    path_raw = os.getenv("MT5_PATH", "").strip()
    mt5_path = path_raw if path_raw else None

    chat_ids_raw = _require("TELEGRAM_ALLOWED_CHAT_IDS")
    try:
        allowed_chat_ids = {int(cid.strip()) for cid in chat_ids_raw.split(",") if cid.strip()}
    except ValueError:
        raise ConfigError(f"TELEGRAM_ALLOWED_CHAT_IDS must be comma-separated integers, got: {chat_ids_raw!r}")

    symbols_raw = os.getenv("SYMBOLS", "XAUUSD,BTCUSD").strip()
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]
    if not symbols:
        raise ConfigError("SYMBOLS must contain at least one symbol")

    max_risk = _float("MAX_RISK_PERCENT", 2.0)
    if not (0 < max_risk <= 10):
        raise ConfigError(f"MAX_RISK_PERCENT must be between 0 and 10, got: {max_risk}")

    # Per-symbol fixed lot sizes (e.g. XAUUSD_LOT=0.01, BTCUSD_LOT=0.03)
    fixed_lots: dict[str, float] = {}
    for sym in symbols:
        lot_raw = os.getenv(f"{sym.upper()}_LOT", "").strip()
        if lot_raw:
            try:
                lot_val = float(lot_raw)
                if lot_val > 0:
                    fixed_lots[sym] = lot_val
            except ValueError:
                raise ConfigError(f"{sym}_LOT must be a float, got: {lot_raw!r}")

    return AppConfig(
        mt5=MT5Config(
            login=login,
            password=os.getenv("MT5_PASSWORD", ""),
            server=_require("MT5_SERVER"),
            path=mt5_path,
        ),
        telegram=TelegramConfig(
            bot_token=_require("TELEGRAM_BOT_TOKEN"),
            allowed_chat_ids=allowed_chat_ids,
        ),
        risk=RiskConfig(
            max_risk_percent=max_risk,
            trend_weak_threshold=_float("TREND_WEAK_THRESHOLD", 25.0),
            trend_strong_threshold=_float("TREND_STRONG_THRESHOLD", 50.0),
            tp_multiplier_weak=_float("TP_MULTIPLIER_WEAK", 1.5),
            tp_multiplier_medium=_float("TP_MULTIPLIER_MEDIUM", 2.5),
            tp_multiplier_strong=_float("TP_MULTIPLIER_STRONG", 3.5),
        ),
        indicators=IndicatorConfig(
            ema_fast=_int("EMA_FAST", 20),
            ema_slow=_int("EMA_SLOW", 50),
            ema_100=_int("EMA_100", 100),
            sma_200=_int("SMA_200", 200),
            rsi_period=_int("RSI_PERIOD", 14),
            macd_fast=_int("MACD_FAST", 12),
            macd_slow=_int("MACD_SLOW", 26),
            macd_signal=_int("MACD_SIGNAL", 9),
            atr_period=_int("ATR_PERIOD", 14),
            adx_period=_int("ADX_PERIOD", 14),
            sr_lookback=_int("SR_LOOKBACK", 50),
        ),
        symbols=symbols,
        timezone=os.getenv("TIMEZONE", "UTC").strip() or "UTC",
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        log_file=os.getenv("LOG_FILE", "logs/trading_bot.log").strip() or "logs/trading_bot.log",
        fixed_lots=fixed_lots,
    )
