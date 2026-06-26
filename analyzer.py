"""
Technical analysis engine. Pure computation — no MT5 calls, no Telegram calls.
Uses last completed bar (iloc[-2]) to avoid signal bias from forming candles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import pandas_ta as ta

from config import IndicatorConfig, RiskConfig

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TrendStrength:
    adx: float
    label: str          # "weak" | "medium" | "strong"
    tp_multiplier: float


@dataclass
class SRLevels:
    supports: list[float] = field(default_factory=list)
    resistances: list[float] = field(default_factory=list)


@dataclass
class AnalysisResult:
    symbol: str
    timestamp: datetime
    signal: Signal
    trend_strength: TrendStrength
    atr: float
    close: float
    ema20: float
    ema50: float
    ema100: float
    sma200: float
    rsi: float
    macd_hist: float
    adx: float
    sr_levels: SRLevels
    nearest_support: float | None
    nearest_resistance: float | None
    rationale: str


class TechnicalAnalyzer:
    # Minimum bars required for indicator warm-up (SMA200 needs 200+ bars)
    MIN_BARS = 250

    def __init__(self, config: IndicatorConfig, risk_config: RiskConfig):
        self._cfg = config
        self._risk = risk_config

    def analyze(
        self,
        symbol: str,
        df_d1: pd.DataFrame,
        df_h4: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_m15: pd.DataFrame,
    ) -> AnalysisResult:
        """
        Multi-timeframe analysis.
        D1: macro trend direction
        H4: intermediate trend + S/R detection
        H1: entry timing confirmation
        M15: signal trigger
        """
        df_d1 = self._compute_indicators(df_d1)
        df_h4 = self._compute_indicators(df_h4)
        df_h1 = self._compute_indicators(df_h1)
        df_m15 = self._compute_indicators(df_m15)

        # Use last COMPLETED bar (iloc[-2]) on each timeframe
        d1 = df_d1.iloc[-2]
        h4 = df_h4.iloc[-2]
        h1 = df_h1.iloc[-2]
        m15 = df_m15.iloc[-2]

        close = float(m15["close"])
        atr = float(h1.get("atr", m15.get("atr", 0)))
        adx = float(m15.get("adx", 0))
        trend_strength = self._classify_trend_strength(adx)

        sr_levels = self._detect_swing_levels(df_h4, close)

        nearest_support = self._nearest_level_below(sr_levels.supports, close)
        nearest_resistance = self._nearest_level_above(sr_levels.resistances, close)

        signal, rationale = self._classify_signal(
            symbol=symbol,
            d1=d1, h4=h4, h1=h1, m15=m15,
            close=close,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

        return AnalysisResult(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            signal=signal,
            trend_strength=trend_strength,
            atr=atr,
            close=close,
            ema20=float(m15.get("ema20", 0)),
            ema50=float(m15.get("ema50", 0)),
            ema100=float(m15.get("ema100", 0)),
            sma200=float(m15.get("sma200", 0)),
            rsi=float(m15.get("rsi", 50)),
            macd_hist=float(m15.get("macd_hist", 0)),
            adx=adx,
            sr_levels=sr_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            rationale=rationale,
        )

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators. Returns the df with added columns."""
        if len(df) < 50:
            logger.warning("DataFrame has only %d rows — indicator warm-up may be poor", len(df))

        cfg = self._cfg
        df = df.copy()

        # Moving averages
        df["ema20"] = ta.ema(df["close"], length=cfg.ema_fast)
        df["ema50"] = ta.ema(df["close"], length=cfg.ema_slow)
        df["ema100"] = ta.ema(df["close"], length=cfg.ema_100)
        df["sma200"] = ta.sma(df["close"], length=cfg.sma_200)

        # Momentum
        df["rsi"] = ta.rsi(df["close"], length=cfg.rsi_period)

        # MACD
        macd_df = ta.macd(
            df["close"],
            fast=cfg.macd_fast,
            slow=cfg.macd_slow,
            signal=cfg.macd_signal,
        )
        if macd_df is not None:
            hist_col = [c for c in macd_df.columns if "h" in c.lower()]
            sig_col = [c for c in macd_df.columns if "s" in c.lower() and "ma" in c.lower()]
            macd_col = [c for c in macd_df.columns if c not in hist_col and c not in sig_col]
            if hist_col:
                df["macd_hist"] = macd_df[hist_col[0]]
            if sig_col:
                df["macd_signal"] = macd_df[sig_col[0]]
            if macd_col:
                df["macd"] = macd_df[macd_col[0]]

        # Volatility / trend
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_period)

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=cfg.adx_period)
        if adx_df is not None:
            adx_col = [c for c in adx_df.columns if c.upper().startswith("ADX")]
            dmp_col = [c for c in adx_df.columns if "DMP" in c.upper()]
            dmn_col = [c for c in adx_df.columns if "DMN" in c.upper()]
            if adx_col:
                df["adx"] = adx_df[adx_col[0]]
            if dmp_col:
                df["dmp"] = adx_df[dmp_col[0]]
            if dmn_col:
                df["dmn"] = adx_df[dmn_col[0]]

        return df

    def _classify_signal(
        self,
        symbol: str,
        d1, h4, h1, m15,
        close: float,
        nearest_support: float | None,
        nearest_resistance: float | None,
    ) -> tuple[Signal, str]:
        """
        Multi-timeframe signal classification.
        D1 + H4 must agree on trend; M15 fires the trigger.
        """
        def _val(row, key: str, default=0.0):
            v = row.get(key, default)
            return default if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

        # D1 trend direction
        d1_ema20 = _val(d1, "ema20")
        d1_ema50 = _val(d1, "ema50")
        d1_bullish = d1_ema20 > d1_ema50
        d1_bearish = d1_ema20 < d1_ema50

        # H4 intermediate trend
        h4_ema20 = _val(h4, "ema20")
        h4_ema50 = _val(h4, "ema50")
        h4_bullish = h4_ema20 > h4_ema50
        h4_bearish = h4_ema20 < h4_ema50

        # M15 signal trigger values
        m15_ema20 = _val(m15, "ema20")
        m15_ema50 = _val(m15, "ema50")
        m15_ema100 = _val(m15, "ema100")
        m15_sma200 = _val(m15, "sma200")
        m15_rsi = _val(m15, "rsi", 50.0)
        m15_macd_hist = _val(m15, "macd_hist")

        # Price relative to major MAs on M15
        above_sma200 = close > m15_sma200 if m15_sma200 else None
        ma_bull_stack = m15_ema20 > m15_ema50 > m15_ema100
        ma_bear_stack = m15_ema20 < m15_ema50 < m15_ema100

        # Proximity filter: skip if price is already sitting at a key level
        at_resistance = (
            nearest_resistance is not None
            and abs(close - nearest_resistance) / close < 0.003  # within 0.3%
        )
        at_support = (
            nearest_support is not None
            and abs(close - nearest_support) / close < 0.003
        )

        # BUY conditions (need 5 out of 7)
        buy_conditions = [
            d1_bullish,
            h4_bullish,
            above_sma200 is True,
            ma_bull_stack,
            40 < m15_rsi < 75,
            m15_macd_hist > 0,
            not at_resistance,
        ]

        # SELL conditions (need 5 out of 7)
        sell_conditions = [
            d1_bearish,
            h4_bearish,
            above_sma200 is False,
            ma_bear_stack,
            25 < m15_rsi < 60,
            m15_macd_hist < 0,
            not at_support,
        ]

        buy_score = sum(1 for c in buy_conditions if c)
        sell_score = sum(1 for c in sell_conditions if c)

        SIGNAL_THRESHOLD = 5  # out of 7 conditions required

        if buy_score >= SIGNAL_THRESHOLD:
            rationale = (
                f"BUY signal — D1/H4 bullish, M15 MA stack bullish, "
                f"RSI={m15_rsi:.1f}, MACD hist={m15_macd_hist:.4f}, "
                f"price above SMA200"
            )
            return Signal.BUY, rationale

        if sell_score >= SIGNAL_THRESHOLD:
            rationale = (
                f"SELL signal — D1/H4 bearish, M15 MA stack bearish, "
                f"RSI={m15_rsi:.1f}, MACD hist={m15_macd_hist:.4f}, "
                f"price below SMA200"
            )
            return Signal.SELL, rationale

        # Partial match — log why we're holding
        if buy_score >= 5:
            missing = [
                "D1 bullish" if not d1_bullish else None,
                "H4 bullish" if not h4_bullish else None,
                "above SMA200" if above_sma200 is not True else None,
                "MA stack bullish" if not ma_bull_stack else None,
                "RSI 45-70" if not (45 < m15_rsi < 70) else None,
                "MACD hist >0" if not (m15_macd_hist > 0) else None,
                "not at resistance" if at_resistance else None,
            ]
            missing_str = ", ".join(m for m in missing if m)
            rationale = f"Near BUY ({buy_score}/7) — missing: {missing_str}"
        elif sell_score >= 5:
            missing = [
                "D1 bearish" if not d1_bearish else None,
                "H4 bearish" if not h4_bearish else None,
                "below SMA200" if above_sma200 is not False else None,
                "MA stack bearish" if not ma_bear_stack else None,
                "RSI 30-55" if not (30 < m15_rsi < 55) else None,
                "MACD hist <0" if not (m15_macd_hist < 0) else None,
                "not at support" if at_support else None,
            ]
            missing_str = ", ".join(m for m in missing if m)
            rationale = f"Near SELL ({sell_score}/7) — missing: {missing_str}"
        else:
            rationale = (
                f"HOLD — BUY score {buy_score}/7, SELL score {sell_score}/7. "
                f"D1={'bull' if d1_bullish else 'bear'}, "
                f"RSI={m15_rsi:.1f}, MACD={m15_macd_hist:.4f}"
            )

        return Signal.HOLD, rationale

    def _classify_trend_strength(self, adx: float) -> TrendStrength:
        if adx >= self._risk.trend_strong_threshold:
            return TrendStrength(adx=adx, label="strong", tp_multiplier=self._risk.tp_multiplier_strong)
        elif adx >= self._risk.trend_weak_threshold:
            return TrendStrength(adx=adx, label="medium", tp_multiplier=self._risk.tp_multiplier_medium)
        else:
            return TrendStrength(adx=adx, label="weak", tp_multiplier=self._risk.tp_multiplier_weak)

    def _detect_swing_levels(self, df_h4: pd.DataFrame, close: float) -> SRLevels:
        """
        Detect support and resistance from H4 OHLCV data.
        Uses swing highs/lows and daily pivot points.
        """
        lookback = self._cfg.sr_lookback
        df = df_h4.tail(lookback).copy().reset_index(drop=True)

        supports: list[float] = []
        resistances: list[float] = []

        # Swing highs and lows (3-bar pattern)
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        for i in range(2, n - 2):
            # Swing high: higher than 2 bars each side
            if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and \
               highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
                resistances.append(float(highs[i]))

            # Swing low: lower than 2 bars each side
            if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and \
               lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
                supports.append(float(lows[i]))

        # Daily pivot points from the last completed H4 sequence
        if len(df) >= 6:  # need at least 6 H4 bars for one day
            last_day = df.tail(6)
            H = float(last_day["high"].max())
            L = float(last_day["low"].min())
            C = float(last_day["close"].iloc[-1])
            P = (H + L + C) / 3
            R1 = 2 * P - L
            R2 = P + (H - L)
            S1 = 2 * P - H
            S2 = P - (H - L)
            resistances.extend([R1, R2])
            supports.extend([S1, S2])

        # Cluster nearby levels (within 0.3% of each other → take mean)
        supports = self._cluster_levels(supports, close, threshold_pct=0.003)
        resistances = self._cluster_levels(resistances, close, threshold_pct=0.003)

        # Filter: supports below current price, resistances above
        supports = sorted([s for s in supports if s < close], reverse=True)
        resistances = sorted([r for r in resistances if r > close])

        return SRLevels(supports=supports[:5], resistances=resistances[:5])

    def _cluster_levels(self, levels: list[float], reference: float, threshold_pct: float) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: list[list[float]] = []
        current_cluster: list[float] = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / reference <= threshold_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)

        return [float(np.mean(c)) for c in clusters]

    def _nearest_level_below(self, levels: list[float], price: float) -> float | None:
        below = [l for l in levels if l < price]
        return max(below) if below else None

    def _nearest_level_above(self, levels: list[float], price: float) -> float | None:
        above = [l for l in levels if l > price]
        return min(above) if above else None

    def build_scan_report(self, results: list[AnalysisResult]) -> str:
        """Format a Telegram-ready Markdown summary of all analyzed symbols."""
        lines = ["*📊 Market Scan Report*", f"_{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_", ""]
        for r in results:
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(r.signal.value, "⚪")
            lines.append(f"{signal_emoji} *{r.symbol}* — `{r.signal.value}`")
            lines.append(f"  Close: `{r.close:.5g}` | ATR: `{r.atr:.5g}`")
            lines.append(f"  EMA20: `{r.ema20:.5g}` | EMA50: `{r.ema50:.5g}` | SMA200: `{r.sma200:.5g}`")
            lines.append(f"  RSI: `{r.rsi:.1f}` | MACD hist: `{r.macd_hist:.4f}` | ADX: `{r.adx:.1f}` ({r.trend_strength.label})")
            if r.nearest_resistance:
                lines.append(f"  Resistance: `{r.nearest_resistance:.5g}`")
            if r.nearest_support:
                lines.append(f"  Support: `{r.nearest_support:.5g}`")
            lines.append(f"  _{r.rationale}_")
            lines.append("")
        return "\n".join(lines)
