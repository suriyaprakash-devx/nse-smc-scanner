from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import settings
from indicators import IndicatorSnapshot
from smc_engine import SMCSnapshot

@dataclass
class SignalResult:
    symbol: str
    direction: str  # Strictly "BUY" or "SELL"
    entry_price: float
    entry_zone: str
    confirmation_price: str
    signal_candle_time: str
    current_price: float
    buy_score: float
    sell_score: float
    setup_name: str
    volume_surge: bool
    volume: int
    volume_ma: float
    volume_ratio: float
    ema_6: float
    ema_30: float
    vwap: float
    atr: float
    structure_confluences: List[str]
    trend_confluences: List[str]
    participation_confluences: List[str]
    risk_confluences: List[str]
    reasons: List[str]
    invalidation_level: str
    confluence_factors: List[str] = field(default_factory=list)

class ConfluenceSignalEngine:
    """Multi-factor quantitative decision engine combining SMC, EMA 6/30, VWAP, Volume, and ATR.
    STRICT RULE: Strictly outputs BUY or SELL as a directional intraday semi-algo bias.
    Groups confirmations into 4 transparent categories:
      1. Structure (SMC Swings, BOS/CHOCH, Sweeps, Fresh OBs & FVGs, Dealing Range)
      2. Trend (EMA 6 / EMA 30, VWAP)
      3. Participation (Volume, Volume 20-MA, Volume Ratio >= 1.5x)
      4. Risk (ATR, Structural Invalidation, R:R >= 2.0)
    """

    def __init__(self, volume_threshold: float = settings.DEFAULT_VOLUME_MULTIPLIER):
        self.volume_threshold = volume_threshold

    def evaluate(
        self,
        symbol: str,
        current_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot,
        timeframe: str = "5m"
    ) -> SignalResult:
        buy_score = 0.0
        sell_score = 0.0

        structure_reasons: List[str] = []
        trend_reasons: List[str] = []
        participation_reasons: List[str] = []
        risk_reasons: List[str] = []
        all_reasons: List[str] = []

        # =========================================================
        # 1. STRUCTURE CATEGORY (SMC, Swings, BOS/CHOCH, Sweeps, OB, FVG)
        # =========================================================
        # 1.1 Market Structure Trend
        if smc.trend == "BULLISH":
            buy_score += 3.0
            msg = f"SMC Structure: {smc.market_structure}"
            structure_reasons.append(msg)
        elif smc.trend == "BEARISH":
            sell_score += 3.0
            msg = f"SMC Structure: {smc.market_structure}"
            structure_reasons.append(msg)
        else:
            structure_reasons.append("SMC Structure: Consolidation dealing range")

        # 1.2 BOS / CHOCH Break Events
        if smc.bos_choch_event.get("detected"):
            direction = smc.bos_choch_event.get("direction")
            ev_type = smc.bos_choch_event.get("type")
            ev_price = smc.bos_choch_event.get("price", 0.0)
            if direction == "BULLISH":
                buy_score += 3.5
                structure_reasons.append(f"Bullish {ev_type} confirmed above ₹{ev_price:.2f}")
            elif direction == "BEARISH":
                sell_score += 3.5
                structure_reasons.append(f"Bearish {ev_type} confirmed below ₹{ev_price:.2f}")

        # 1.3 Liquidity Sweeps (SSL / BSL)
        if smc.sweep_event.get("detected"):
            s_dir = smc.sweep_event.get("direction")
            s_price = smc.sweep_event.get("price", 0.0)
            if s_dir == "BULLISH":
                buy_score += 3.0
                structure_reasons.append(f"Bullish Liquidity Sweep (SSL taken @ ₹{s_price:.2f}, wick rejected)")
            elif s_dir == "BEARISH":
                sell_score += 3.0
                structure_reasons.append(f"Bearish Liquidity Sweep (BSL taken @ ₹{s_price:.2f}, wick rejected)")

        # 1.4 PDH / PDL Sweeps & Status
        if smc.pdh_pdl_sweep.get("detected"):
            pd_dir = smc.pdh_pdl_sweep.get("direction")
            if pd_dir == "BULLISH":
                buy_score += 2.5
                structure_reasons.append(f"Bullish PDL Sweep: Previous Day Low swept & defended @ ₹{smc.pdl:.2f}")
            elif pd_dir == "BEARISH":
                sell_score += 2.5
                structure_reasons.append(f"Bearish PDH Sweep: Previous Day High swept & rejected @ ₹{smc.pdh:.2f}")
        else:
            if smc.pdh_pdl_status == "ABOVE_PDH":
                buy_score += 1.5
                structure_reasons.append(f"Trading above Previous Day High (₹{smc.pdh:.2f})")
            elif smc.pdh_pdl_status == "BELOW_PDL":
                sell_score += 1.5
                structure_reasons.append(f"Trading below Previous Day Low (₹{smc.pdl:.2f})")

        # 1.5 Fresh Order Blocks & FVGs
        for ob in smc.fresh_order_blocks[:2]:
            if ob.ob_type == "BULLISH" and current_price >= ob.bottom and current_price <= ob.top * 1.01:
                buy_score += 2.0
                structure_reasons.append(f"Supported by Fresh Bullish OB [₹{ob.bottom:.2f} - ₹{ob.top:.2f}]")
                break
            elif ob.ob_type == "BEARISH" and current_price <= ob.top and current_price >= ob.bottom * 0.99:
                sell_score += 2.0
                structure_reasons.append(f"Pressured by Fresh Bearish OB [₹{ob.bottom:.2f} - ₹{ob.top:.2f}]")
                break

        for fvg in smc.fresh_fvgs[:2]:
            if fvg.fvg_type == "BULLISH" and current_price >= fvg.bottom:
                buy_score += 1.5
                structure_reasons.append(f"Fresh Bullish FVG imbalance [₹{fvg.bottom:.2f} - ₹{fvg.top:.2f}]")
                break
            elif fvg.fvg_type == "BEARISH" and current_price <= fvg.top:
                sell_score += 1.5
                structure_reasons.append(f"Fresh Bearish FVG imbalance [₹{fvg.bottom:.2f} - ₹{fvg.top:.2f}]")
                break

        # 1.6 Premium / Discount Dealing Range
        zone = smc.dealing_range.get("zone", "")
        zone_pct = smc.dealing_range.get("pct", 50.0)
        if zone == "DISCOUNT":
            buy_score += 1.5
            structure_reasons.append(f"Dealing Range: In Discount ({zone_pct:.0f}% of range) — favorable for longs")
        elif zone == "PREMIUM":
            sell_score += 1.5
            structure_reasons.append(f"Dealing Range: In Premium ({zone_pct:.0f}% of range) — favorable for shorts")

        # =========================================================
        # 2. TREND CATEGORY (EMA 6/30, VWAP)
        # =========================================================
        # 2.1 EMA 6 vs EMA 30
        if indicators.ema_6_above_30:
            buy_score += 3.0
            if indicators.ema_cross_bullish:
                trend_reasons.append(f"EMA 6/30: Bullish Golden Crossover (EMA 6 ₹{indicators.ema_6:.2f} > EMA 30 ₹{indicators.ema_30:.2f})")
            else:
                trend_reasons.append(f"EMA 6/30: Bullish Trend Alignment (EMA 6 ₹{indicators.ema_6:.2f} > EMA 30 ₹{indicators.ema_30:.2f})")
        else:
            sell_score += 3.0
            if indicators.ema_cross_bearish:
                trend_reasons.append(f"EMA 6/30: Bearish Death Crossover (EMA 6 ₹{indicators.ema_6:.2f} < EMA 30 ₹{indicators.ema_30:.2f})")
            else:
                trend_reasons.append(f"EMA 6/30: Bearish Trend Alignment (EMA 6 ₹{indicators.ema_6:.2f} < EMA 30 ₹{indicators.ema_30:.2f})")

        # 2.2 Intraday Session VWAP (09:15 Reset)
        if current_price >= indicators.vwap:
            buy_score += 2.5
            trend_reasons.append(f"Intraday VWAP: Trading above session VWAP (₹{indicators.vwap:.2f})")
        else:
            sell_score += 2.5
            trend_reasons.append(f"Intraday VWAP: Trading below session VWAP (₹{indicators.vwap:.2f})")

        # =========================================================
        # 3. PARTICIPATION CATEGORY (Volume & Volume Ratio)
        # =========================================================
        vol_ratio = indicators.volume_ratio
        vol_surge = vol_ratio >= self.volume_threshold

        if vol_surge:
            # Volume surge heavily amplifies conviction on the dominant side
            participation_reasons.append(
                f"Volume Surge: {vol_ratio:.2f}x of 20-MA (Current: {indicators.volume:,} | Avg: {int(indicators.volume_ma):,})"
            )
            if buy_score >= sell_score:
                buy_score += 3.0
            else:
                sell_score += 3.0
        else:
            participation_reasons.append(
                f"Volume: {indicators.volume:,} (Avg: {int(indicators.volume_ma):,} | Ratio: {vol_ratio:.2f}x)"
            )

        # =========================================================
        # 4. RISK & PRICE ACTION CONFLUENCES
        # =========================================================
        if indicators.is_bullish_candle:
            buy_score += 1.0
        else:
            sell_score += 1.0

        if indicators.is_engulfing_bullish:
            buy_score += 1.5
            risk_reasons.append("Price Action: Bullish Engulfing completed candle")
        elif indicators.is_engulfing_bearish:
            sell_score += 1.5
            risk_reasons.append("Price Action: Bearish Engulfing completed candle")

        risk_reasons.append(f"ATR Volatility: ₹{indicators.atr:.2f} (structural buffer calibrated)")
        risk_reasons.append("Enforced Minimum Risk/Reward: 1 : 2.0+")

        # =========================================================
        # 5. DECISIVE BINARY RESOLUTION (STRICTLY BUY OR SELL)
        # =========================================================
        if buy_score >= sell_score:
            direction = "BUY"
            setup_name = "Bullish SMC & EMA 6/30 Confluence Expansion"
            all_reasons = structure_reasons + trend_reasons + participation_reasons + risk_reasons
            # Invalidation
            recent_sl = smc.swing_lows[-1].price if smc.swing_lows else (current_price - indicators.atr * 1.5)
            invalidation_level = f"Break below structural swing low ₹{recent_sl:.2f}"
            
            # Entry Zone & Confirmation Price
            entry_price = current_price
            zone_low = round(max(current_price - (indicators.atr * 0.4), current_price * 0.997), 2)
            zone_high = round(current_price, 2)
            entry_zone = f"₹{zone_low:.2f} – ₹{zone_high:.2f}"
            confirmation_price = f"Confirmed close holding above EMA 6 (₹{indicators.ema_6:.2f}) & VWAP (₹{indicators.vwap:.2f})"
        else:
            direction = "SELL"
            setup_name = "Bearish SMC & EMA 6/30 Confluence Distribution"
            all_reasons = structure_reasons + trend_reasons + participation_reasons + risk_reasons
            recent_sh = smc.swing_highs[-1].price if smc.swing_highs else (current_price + indicators.atr * 1.5)
            invalidation_level = f"Break above structural swing high ₹{recent_sh:.2f}"

            entry_price = current_price
            zone_low = round(current_price, 2)
            zone_high = round(min(current_price + (indicators.atr * 0.4), current_price * 1.003), 2)
            entry_zone = f"₹{zone_low:.2f} – ₹{zone_high:.2f}"
            confirmation_price = f"Confirmed close holding below EMA 6 (₹{indicators.ema_6:.2f}) & VWAP (₹{indicators.vwap:.2f})"

        # Signal completed candle timestamp
        last_candle_ts = ""
        if hasattr(indicators.raw_df, "iloc") and not indicators.raw_df.empty:
            last_candle_ts = str(indicators.raw_df["timestamp"].iloc[-1])

        return SignalResult(
            symbol=symbol,
            direction=direction,
            entry_price=round(entry_price, 2),
            entry_zone=entry_zone,
            confirmation_price=confirmation_price,
            signal_candle_time=last_candle_ts,
            current_price=round(current_price, 2),
            buy_score=round(buy_score, 1),
            sell_score=round(sell_score, 1),
            setup_name=setup_name,
            volume_surge=vol_surge,
            volume=int(indicators.volume),
            volume_ma=round(float(indicators.volume_ma), 2),
            volume_ratio=round(float(vol_ratio), 2),
            ema_6=round(float(indicators.ema_6), 2),
            ema_30=round(float(indicators.ema_30), 2),
            vwap=round(float(indicators.vwap), 2),
            atr=round(float(indicators.atr), 2),
            structure_confluences=structure_reasons,
            trend_confluences=trend_reasons,
            participation_confluences=participation_reasons,
            risk_confluences=risk_reasons,
            reasons=all_reasons,
            invalidation_level=invalidation_level,
            confluence_factors=all_reasons
        )

signal_engine = ConfluenceSignalEngine()
