"""
Risk manager: position sizing, stop-loss, and take-profit calculation.
Enforces the 2% maximum equity risk per trade.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from analyzer import AnalysisResult, Signal, SRLevels
from config import RiskConfig
from mt5_connector import AccountInfo, OrderType, SymbolInfo

logger = logging.getLogger(__name__)


@dataclass
class TradeParameters:
    symbol: str
    order_type: OrderType
    volume: float           # lots, validated and rounded to volume_step
    entry_price: float
    sl_price: float         # absolute price level
    tp_price: float         # absolute price level (S/R-adjusted)
    sl_distance: float      # price units between entry and SL
    tp_distance: float      # price units between entry and TP
    risk_amount: float      # USD at risk
    risk_percent: float     # actual % of equity risked


class RiskManager:
    def __init__(self, config: RiskConfig):
        self._cfg = config

    def calculate_trade_parameters(
        self,
        analysis: AnalysisResult,
        account: AccountInfo,
        symbol_info: SymbolInfo,
        fixed_lot: float | None = None,
    ) -> TradeParameters | None:
        """
        Full risk calculation pipeline.
        Returns None if signal is HOLD or risk cannot be managed safely.
        If fixed_lot is provided, skips dynamic lot sizing.
        """
        if analysis.signal == Signal.HOLD:
            return None

        if analysis.atr <= 0 or (isinstance(analysis.atr, float) and analysis.atr != analysis.atr):
            logger.warning("%s: ATR is zero/NaN, skipping trade", analysis.symbol)
            return None

        order_type = OrderType.BUY if analysis.signal == Signal.BUY else OrderType.SELL
        entry_price = symbol_info.ask if order_type == OrderType.BUY else symbol_info.bid

        if fixed_lot is not None:
            # Resolve actual lot first so SL cap uses the real lot, not min_lot
            step = symbol_info.volume_step
            volume = math.floor(fixed_lot / step) * step
            volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
            volume = round(volume, 8)
            logger.info("%s: using fixed lot size %.2f", analysis.symbol, volume)
        else:
            volume = None  # resolved after SL calculation

        sl_price, sl_distance = self._calculate_sl(
            order_type=order_type,
            entry_price=entry_price,
            atr=analysis.atr,
            equity=account.equity,
            sr_levels=analysis.sr_levels,
            digits=symbol_info.digits,
            symbol_info=symbol_info,
            actual_lot=volume,  # None → uses min_lot cap; fixed → uses real lot cap
        )

        if sl_distance <= 0 or sl_distance != sl_distance:  # NaN check
            logger.warning("%s: SL distance is zero/NaN, skipping", analysis.symbol)
            return None

        if volume is None:
            volume = self._calculate_lot_size(
                sl_distance=sl_distance,
                equity=account.equity,
                symbol_info=symbol_info,
            )

        if volume is None:
            return None

        tp_price, tp_distance = self._calculate_tp(
            order_type=order_type,
            entry_price=entry_price,
            atr=analysis.atr,
            trend_strength=analysis.trend_strength,
            sr_levels=analysis.sr_levels,
            digits=symbol_info.digits,
        )

        # Compute actual risk for logging
        unit_value = symbol_info.trade_contract_size  # USD per 1 price-unit move per lot
        risk_amount = volume * sl_distance * unit_value
        risk_percent = (risk_amount / account.equity) * 100

        logger.info(
            "%s %s: vol=%.2f SL=%.5g TP=%.5g risk=$%.2f (%.2f%% equity)",
            analysis.symbol, order_type.value, volume, sl_price, tp_price,
            risk_amount, risk_percent,
        )

        return TradeParameters(
            symbol=analysis.symbol,
            order_type=order_type,
            volume=volume,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
        )

    def _calculate_sl(
        self,
        order_type: OrderType,
        entry_price: float,
        atr: float,
        equity: float,
        sr_levels: SRLevels,
        digits: int,
        symbol_info: SymbolInfo,
        actual_lot: float | None = None,
    ) -> tuple[float, float]:
        """
        Returns (sl_price, sl_distance_in_price_units).
        Raw SL = 1× ATR from entry. Adjusted to nearest S/R if closer.
        Hard cap: SL cannot risk more than 2% of equity at the given lot size.
        """
        raw_sl_distance = atr * 1.0

        # Adjust SL toward nearest S/R structure if it is tighter than ATR
        if order_type == OrderType.BUY and sr_levels.supports:
            supports_below = [s for s in sr_levels.supports if s < entry_price]
            if supports_below:
                nearest_support = max(supports_below)
                structure_distance = entry_price - nearest_support
                if 0 < structure_distance < raw_sl_distance:
                    raw_sl_distance = structure_distance

        elif order_type == OrderType.SELL and sr_levels.resistances:
            resistances_above = [r for r in sr_levels.resistances if r > entry_price]
            if resistances_above:
                nearest_resistance = min(resistances_above)
                structure_distance = nearest_resistance - entry_price
                if 0 < structure_distance < raw_sl_distance:
                    raw_sl_distance = structure_distance

        # Cap SL so that loss at SL ≤ 2% equity.
        # Use actual_lot if known (fixed lot), otherwise fall back to min_lot.
        unit_value = symbol_info.trade_contract_size
        lot_for_cap = actual_lot if actual_lot and actual_lot > 0 else symbol_info.volume_min
        max_risk_usd = equity * (self._cfg.max_risk_percent / 100)
        if unit_value > 0 and lot_for_cap > 0:
            max_sl_distance = max_risk_usd / (unit_value * lot_for_cap)
            sl_distance = min(raw_sl_distance, max_sl_distance)
        else:
            sl_distance = raw_sl_distance

        logger.info("%s %s: ATR=%.5g raw_sl_dist=%.5g final_sl_dist=%.5g sl=%.5g lot=%.3f contract=%.2f",
                    symbol_info.name, order_type.value, atr, raw_sl_distance, sl_distance,
                    entry_price + sl_distance if order_type.value == "SELL" else entry_price - sl_distance,
                    lot_for_cap, unit_value)

        if order_type == OrderType.BUY:
            sl_price = round(entry_price - sl_distance, digits)
        else:
            sl_price = round(entry_price + sl_distance, digits)

        return sl_price, sl_distance

    def _calculate_tp(
        self,
        order_type: OrderType,
        entry_price: float,
        atr: float,
        trend_strength,
        sr_levels: SRLevels,
        digits: int,
    ) -> tuple[float, float]:
        """
        Returns (tp_price, tp_distance).
        Base TP = ATR × trend multiplier. Snaps to nearest S/R if within ±15%.
        """
        raw_tp_distance = atr * trend_strength.tp_multiplier

        if order_type == OrderType.BUY:
            raw_tp_price = entry_price + raw_tp_distance
            # Snap to nearest resistance within ±15% of raw TP
            candidate = self._snap_to_sr(raw_tp_price, sr_levels.resistances, tolerance=0.15)
            tp_price = candidate if candidate else raw_tp_price
        else:
            raw_tp_price = entry_price - raw_tp_distance
            candidate = self._snap_to_sr(raw_tp_price, sr_levels.supports, tolerance=0.15)
            tp_price = candidate if candidate else raw_tp_price

        tp_price = round(tp_price, digits)
        tp_distance = abs(tp_price - entry_price)
        return tp_price, tp_distance

    def _snap_to_sr(
        self, target: float, levels: list[float], tolerance: float
    ) -> float | None:
        """
        Return the S/R level closest to target if within tolerance (fraction of target).
        Returns None if no level is close enough.
        """
        best: float | None = None
        best_dist = float("inf")
        for level in levels:
            dist = abs(level - target)
            if dist / target <= tolerance and dist < best_dist:
                best = level
                best_dist = dist
        return best

    def _calculate_lot_size(
        self,
        sl_distance: float,
        equity: float,
        symbol_info: SymbolInfo,
    ) -> float | None:
        """
        Lot size so that loss at SL equals max_risk_percent% of equity.
        Clamps to broker min/max and rounds down to volume_step.
        Returns None if even minimum lot exceeds 2× the risk cap.
        """
        unit_value = symbol_info.trade_contract_size  # e.g., 100 for XAUUSD
        max_risk_usd = equity * (self._cfg.max_risk_percent / 100)

        raw_lot = max_risk_usd / (sl_distance * unit_value)

        # Round DOWN to volume_step
        step = symbol_info.volume_step
        lot = math.floor(raw_lot / step) * step
        lot = round(lot, 10)  # eliminate float precision artifacts

        # Clamp to broker limits
        if lot < symbol_info.volume_min:
            # Check if minimum lot is tolerable (≤ 2× risk cap)
            risk_at_min = symbol_info.volume_min * sl_distance * unit_value
            if risk_at_min > max_risk_usd * 2:
                logger.warning(
                    "%s: lot at min (%.2f) would risk $%.2f (%.2f%% equity) — skipping trade",
                    symbol_info.name, symbol_info.volume_min, risk_at_min,
                    risk_at_min / equity * 100,
                )
                return None
            lot = symbol_info.volume_min
            logger.info("%s: lot floored to broker minimum %.2f", symbol_info.name, lot)

        lot = min(lot, symbol_info.volume_max)
        lot = round(lot, 8)
        return lot
