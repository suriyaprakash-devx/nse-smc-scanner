from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from config import settings
from indicators import IndicatorSnapshot
from smc_engine import SMCSnapshot

@dataclass
class SignalResult:
    symbol: str
    direction: str  # Strictly "BUY" or "SELL"
    entry_price: float
    current_price: float
    buy_score: float
    sell_score: float
    confluence_factors: List[str]
    setup_name: str
    volume_surge: bool
    volume_ratio: float

class ConfluenceSignalEngine:
    """Multi-factor confluence decision engine combining Technical Indicators, Price Action, SMC, and Volume.
    STRICT RULE: NEVER outputs WAIT, WATCH, HOLD, or NEUTRAL. Strictly outputs BUY or SELL.
    """

    def __init__(self, volume_threshold: float = settings.DEFAULT_VOLUME_MULTIPLIER):
        self.volume_threshold = volume_threshold

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot
    ) -> SignalResult:
        buy_score = 0.0
        sell_score = 0.0
        confluences: List[str] = []

        # =========================================================
        # 1. TECHNICAL INDICATORS SCORING
        # =========================================================
        # EMA Confluence
        if current_price > indicators.ema_9 > indicators.ema_21 > indicators.ema_50:
            buy_score += 2.5
            confluences.append("EMA Alignment: Price > EMA 9 > 21 > 50 (Bullish Momentum)")
        elif current_price < indicators.ema_9 < indicators.ema_21 < indicators.ema_50:
            sell_score += 2.5
            confluences.append("EMA Alignment: Price < EMA 9 < 21 < 50 (Bearish Trend)")
        elif current_price > indicators.ema_21:
            buy_score += 1.0
        else:
            sell_score += 1.0

        # VWAP
        if current_price >= indicators.vwap:
            buy_score += 1.5
            confluences.append(f"VWAP: Trading above intraday VWAP (₹{indicators.vwap:.2f})")
        else:
            sell_score += 1.5
            confluences.append(f"VWAP: Trading below intraday VWAP (₹{indicators.vwap:.2f})")

        # RSI
        if indicators.rsi >= 55:
            buy_score += 1.5
            confluences.append(f"RSI: Bullish momentum (RSI {indicators.rsi:.1f} > 50)")
        elif indicators.rsi <= 45:
            sell_score += 1.5
            confluences.append(f"RSI: Bearish momentum (RSI {indicators.rsi:.1f} < 50)")
        else:
            if indicators.rsi >= 50:
                buy_score += 0.5
            else:
                sell_score += 0.5

        # MACD
        if indicators.macd > indicators.macd_signal and indicators.macd_hist > 0:
            buy_score += 1.5
            confluences.append("MACD: Positive histogram expansion")
        elif indicators.macd < indicators.macd_signal and indicators.macd_hist < 0:
            sell_score += 1.5
            confluences.append("MACD: Negative histogram contraction")

        # ADX Trend Strength
        if indicators.adx >= 20:
            if indicators.plus_di > indicators.minus_di:
                buy_score += 1.2
                confluences.append(f"ADX: Strong bullish trend (+DI {indicators.plus_di:.1f} > -DI)")
            else:
                sell_score += 1.2
                confluences.append(f"ADX: Strong bearish trend (-DI {indicators.minus_di:.1f} > +DI)")

        # Bollinger Bands
        if current_price > indicators.bb_middle:
            buy_score += 0.8
        else:
            sell_score += 0.8

        # Momentum
        if indicators.momentum > 0:
            buy_score += 0.8
        else:
            sell_score += 0.8

        # =========================================================
        # 2. PRICE ACTION SCORING
        # =========================================================
        if indicators.is_bullish_candle:
            buy_score += 1.0
        else:
            sell_score += 1.0

        if indicators.is_engulfing_bullish:
            buy_score += 2.0
            confluences.append("Price Action: Bullish Engulfing candle pattern")
        elif indicators.is_engulfing_bearish:
            sell_score += 2.0
            confluences.append("Price Action: Bearish Engulfing candle pattern")

        if indicators.is_breakout_20:
            buy_score += 2.0
            confluences.append("Price Action: 20-candle range upward breakout")
        elif indicators.is_breakdown_20:
            sell_score += 2.0
            confluences.append("Price Action: 20-candle range downward breakdown")

        if indicators.lower_wick_pct >= 40.0:
            buy_score += 1.2
            confluences.append(f"Price Action: Strong lower wick buyer rejection ({indicators.lower_wick_pct:.0f}%)")
        elif indicators.upper_wick_pct >= 40.0:
            sell_score += 1.2
            confluences.append(f"Price Action: Strong upper wick seller rejection ({indicators.upper_wick_pct:.0f}%)")

        # =========================================================
        # 3. SMART MONEY CONCEPTS (SMC) SCORING
        # =========================================================
        # Market Structure
        if smc.trend == "BULLISH":
            buy_score += 2.5
            confluences.append(f"SMC Structure: {smc.market_structure}")
        elif smc.trend == "BEARISH":
            sell_score += 2.5
            confluences.append(f"SMC Structure: {smc.market_structure}")

        # BOS / CHOCH Break
        if smc.bos_choch_event.get("detected"):
            direction = smc.bos_choch_event.get("direction")
            ev_type = smc.bos_choch_event.get("type")
            if direction == "BULLISH":
                buy_score += 3.0
                confluences.append(f"SMC Break: Bullish {ev_type} confirmed")
            elif direction == "BEARISH":
                sell_score += 3.0
                confluences.append(f"SMC Break: Bearish {ev_type} confirmed")

        # Liquidity Sweeps
        if smc.sweep_event.get("detected"):
            s_dir = smc.sweep_event.get("direction")
            if s_dir == "BULLISH":
                buy_score += 2.5
                confluences.append("SMC Liquidity: Sell-Side Liquidity (SSL) swept & rejected")
            elif s_dir == "BEARISH":
                sell_score += 2.5
                confluences.append("SMC Liquidity: Buy-Side Liquidity (BSL) swept & rejected")

        # Premium / Discount Zone
        zone = smc.dealing_range.get("zone", "")
        if zone == "DISCOUNT":
            buy_score += 1.5
            confluences.append(f"SMC Zone: In Discount ({smc.dealing_range.get('pct', 0)}% of dealing range)")
        elif zone == "PREMIUM":
            sell_score += 1.5
            confluences.append(f"SMC Zone: In Premium ({smc.dealing_range.get('pct', 0)}% of dealing range)")

        # Order Block proximity
        for ob in smc.active_order_blocks[:3]:
            if ob.ob_type == "BULLISH" and current_price >= ob.bottom and current_price <= ob.top * 1.01:
                buy_score += 2.0
                confluences.append(f"SMC Order Block: Supported by Bullish OB [₹{ob.bottom:.2f} - ₹{ob.top:.2f}]")
                break
            elif ob.ob_type == "BEARISH" and current_price <= ob.top and current_price >= ob.bottom * 0.99:
                sell_score += 2.0
                confluences.append(f"SMC Order Block: Pressured by Bearish OB [₹{ob.bottom:.2f} - ₹{ob.top:.2f}]")
                break

        # Previous Day Levels (PDH / PDL)
        if smc.pdh_pdl_status == "ABOVE_PDH":
            buy_score += 1.5
            confluences.append(f"PDH/PDL: Above Previous Day High (₹{smc.pdh:.2f})")
        elif smc.pdh_pdl_status == "BELOW_PDL":
            sell_score += 1.5
            confluences.append(f"PDH/PDL: Below Previous Day Low (₹{smc.pdl:.2f})")

        # =========================================================
        # 4. VOLUME CONFIRMATION MULTIPLIER
        # =========================================================
        volume_surge = indicators.volume_ratio >= self.volume_threshold
        if volume_surge:
            confluences.append(f"Volume Surge: {indicators.volume_ratio:.2f}x of 20-MA (Strong Confirmation)")
            # Boost the winning side
            if buy_score > sell_score:
                buy_score *= 1.25
            else:
                sell_score *= 1.25

        # =========================================================
        # 5. STRICT BINARY RESOLUTION (BUY or SELL ONLY)
        # =========================================================
        # Strict rule: NEVER WAIT, WATCH, HOLD, or NEUTRAL
        if buy_score >= sell_score:
            direction = "BUY"
            setup_name = "Bullish Confluence Expansion"
        else:
            direction = "SELL"
            setup_name = "Bearish Confluence Distribution"

        # Calculate optimal Entry Price based on setup
        entry_price = self._calculate_entry_price(direction, current_price, indicators, smc)

        return SignalResult(
            symbol=symbol,
            direction=direction,
            entry_price=round(entry_price, 2),
            current_price=round(current_price, 2),
            buy_score=round(buy_score, 1),
            sell_score=round(sell_score, 1),
            confluence_factors=confluences,
            setup_name=setup_name,
            volume_surge=volume_surge,
            volume_ratio=round(indicators.volume_ratio, 2)
        )

    def _calculate_entry_price(
        self, direction: str, current_price: float, indicators: IndicatorSnapshot, smc: SMCSnapshot
    ) -> float:
        """Calculates precise valid entry price based on market structure and indicator confluences."""
        if direction == "BUY":
            # If price is slightly above VWAP or 9 EMA, ideal pullback entry is near EMA 9 / VWAP / OB top
            candidates = [current_price]
            if current_price > indicators.vwap > (current_price * 0.985):
                candidates.append(indicators.vwap)
            if current_price > indicators.ema_9 > (current_price * 0.99):
                candidates.append(indicators.ema_9)
            # Check bullish OB
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BULLISH" and (current_price * 0.98) <= ob.top <= current_price:
                    candidates.append(ob.top)
            # Retest price or current price
            return min(candidates) if len(candidates) > 1 else current_price

        else:
            # SELL / Short entry
            candidates = [current_price]
            if current_price < indicators.vwap < (current_price * 1.015):
                candidates.append(indicators.vwap)
            if current_price < indicators.ema_9 < (current_price * 1.01):
                candidates.append(indicators.ema_9)
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BEARISH" and (current_price * 1.02) >= ob.bottom >= current_price:
                    candidates.append(ob.bottom)
            return max(candidates) if len(candidates) > 1 else current_price

signal_engine = ConfluenceSignalEngine()
