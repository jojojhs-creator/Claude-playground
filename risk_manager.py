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
    ) -> TradeParameters | None:
        """
        Full risk calculation pipeline.
        Returns None if signal is HOLD or risk cannot be managed safely.
        """
        if analysis.signal == Signal.HOLD:
            return None

        if analysis.atr <= 0:
            logger.warning("%s: ATR is zero or negative, skipping trade", analysis.symbol)
            return None

        order_type = OrderType.BUY if analysis.signal == Signal.BUY else OrderType.SELL
        entry_price = symbol_info.ask if order_type == OrderType.BUY else symbol_info.bid

        sl_price, sl_distance = self._calculate_sl(
            order_type=order_type,
            entry_price=entry_price,
            atr=analysis.atr,
            equity=account.equity,
            sr_levels=analysis.sr_levels,
            digits=symbol_info.digits,
            symbol_info=symbol_info,
        )

        if sl_distance <= 0:
            logger.warning("%s: SL distance is zero, skipping", analysis.symbol)
            return None

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
    ) -> tuple[float, float]:
        """
        Returns (sl_price, sl_distance_in_price_units).
        Raw SL = 1× ATR from entry. Adjusted to nearest S/R if closer.
        Hard cap: SL cannot risk more than 2% of equity at minimum lot.
        """
        raw_sl_distance = atr * 1.0

        # Adjust SL to nearest S/R level if it gives a natural structure level
        if order_type == OrderType.BUY and sr_levels.supports:
            nearest_support = max(s for s in sr_levels.supports if s < entry_price)
            structure_distance = entry_price - nearest_support
            # Use structure SL if it's tighter than ATR-based (better risk/reward)
            if 0 < structure_distance < raw_sl_distance:
                raw_sl_distance = structure_distance
                logger.debug("%s BUY: using structure SL at %.5g (vs ATR-based %.5g)",
                             symbol_info.name, nearest_support, entry_price - atr)

        elif order_type == OrderType.SELL and sr_levels.resistances:
            nearest_resistance = min(r for r in sr_levels.resistances if r > entry_price)
            structure_distance = nearest_resistance - entry_price
            if 0 < structure_distance < raw_sl_distance:
                raw_sl_distance = structure_distance

        # Cap SL at 2% of equity: max USD risk at min lot → max SL distance
        unit_value = symbol_info.trade_contract_size
        min_lot = symbol_info.volume_min
        max_risk_usd = equity * (self._cfg.max_risk_percent / 100)
        max_sl_distance_for_min_lot = max_risk_usd / (unit_value * min_lot)
        sl_distance = min(raw_sl_distance, max_sl_distance_for_min_lot)

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
