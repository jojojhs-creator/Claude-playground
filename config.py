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
    sl_atr_multiplier: float   # fraction of ATR used for SL distance (e.g. 0.5 for scalping)
    rr_ratio: float            # TP = SL distance × rr_ratio (e.g. 3.0 for 1:3)


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
    scalp_mode: bool = False            # True → M15/M5/M1 timeframes instead of D1/H4/H1/M15
    scan_interval_seconds: int = 300    # how often the scan cycle runs
    max_positions_per_symbol: int = 1   # concurrent positions allowed per symbol
    max_trade_age_minutes: int = 0      # close bot trades older than this (0 = disabled)


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

    scalp_mode = os.getenv("SCALP_MODE", "false").strip().lower() in ("1", "true", "yes")

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
            sl_atr_multiplier=_float("SL_ATR_MULTIPLIER", 0.5),
            rr_ratio=_float("RR_RATIO", 3.0),
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
        scalp_mode=scalp_mode,
        scan_interval_seconds=_int("SCAN_INTERVAL_SECONDS", 60 if scalp_mode else 300),
        max_positions_per_symbol=_int("MAX_POSITIONS_PER_SYMBOL", 3 if scalp_mode else 1),
        max_trade_age_minutes=_int("MAX_TRADE_AGE_MINUTES", 15 if scalp_mode else 0),
    )
