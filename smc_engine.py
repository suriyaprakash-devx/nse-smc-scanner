from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class SwingPoint:
    index: int
    timestamp: str
    price: float
    point_type: str  # "HIGH" or "LOW"
    structure_tag: str = ""  # "HH", "HL", "LH", "LL"

@dataclass
class LiquidityLevel:
    price: float
    level_type: str  # "BSL" (Buy-side) or "SSL" (Sell-side)
    is_equal: bool = False  # Equal Highs (EQH) or Equal Lows (EQL)
    timestamp: str = ""

@dataclass
class FairValueGap:
    top: float
    bottom: float
    fvg_type: str  # "BULLISH" or "BEARISH"
    created_at_index: int
    timestamp: str
    mitigated: bool = False

@dataclass
class OrderBlock:
    top: float
    bottom: float
    ob_type: str  # "BULLISH" or "BEARISH"
    candle_index: int
    timestamp: str
    mitigated: bool = False

@dataclass
class SMCSnapshot:
    trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    market_structure: str  # e.g., "BULLISH (Higher Highs & Higher Lows)"
    bos_choch_event: Dict[str, Any]
    sweep_event: Dict[str, Any]
    active_order_blocks: List[OrderBlock]
    active_fvgs: List[FairValueGap]
    dealing_range: Dict[str, Any]
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    pdh: float
    pdl: float
    pdh_pdl_status: str  # "ABOVE_PDH", "BELOW_PDL", "INSIDE_PD_RANGE"

class SMCEngine:
    """Smart Money Concepts (SMC) quantitative engine.
    Strictly deterministic and non-repainting: operates strictly on completed candles.
    """

    def __init__(self, swing_window: int = 3, eq_tolerance: float = 0.0015):
        self.swing_window = swing_window
        self.eq_tolerance = eq_tolerance

    def analyze(self, df: pd.DataFrame, pdh: Optional[float] = None, pdl: Optional[float] = None) -> SMCSnapshot:
        """Executes full SMC analysis pipeline on closed candle data."""
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        opens = df["open"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        current_price = closes[-1]

        # 1. Detect Swing Highs and Swing Lows
        swing_highs, swing_lows = self._detect_swings(df)
        self._classify_structures(swing_highs, swing_lows)

        # 2. Determine Dealing Range & Premium / Discount Zone
        dealing_range = self._calculate_dealing_range(swing_highs, swing_lows, current_price, highs, lows)

        # 3. Liquidity Levels and Sweeps
        liquidity_levels = self._find_liquidity_levels(swing_highs, swing_lows)
        sweep_event = self._detect_liquidity_sweeps(df, liquidity_levels)

        # 4. Fair Value Gaps (FVG)
        fvgs = self._detect_fvgs(df)

        # 5. Structure Breaks: BOS & CHOCH
        structure_event = self._detect_structure_breaks(df, swing_highs, swing_lows)

        # 6. Order Blocks
        order_blocks = self._detect_order_blocks(df, structure_event)

        # 7. Overall Market Structure & Trend
        trend, ms_label = self._evaluate_trend(swing_highs, swing_lows, structure_event)

        # 8. PDH / PDL status
        if pdh is None or pdh <= 0:
            # Estimate from previous session candles if not provided
            pdh = float(np.max(highs[:-20])) if n > 25 else float(np.max(highs))
        if pdl is None or pdl <= 0:
            pdl = float(np.min(lows[:-20])) if n > 25 else float(np.min(lows))

        if current_price > pdh:
            pdh_pdl_status = "ABOVE_PDH"
        elif current_price < pdl:
            pdh_pdl_status = "BELOW_PDL"
        else:
            pdh_pdl_status = "INSIDE_PD_RANGE"

        return SMCSnapshot(
            trend=trend,
            market_structure=ms_label,
            bos_choch_event=structure_event,
            sweep_event=sweep_event,
            active_order_blocks=order_blocks,
            active_fvgs=fvgs,
            dealing_range=dealing_range,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            pdh=round(pdh, 2),
            pdl=round(pdl, 2),
            pdh_pdl_status=pdh_pdl_status
        )

    def _detect_swings(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Identifies Swing Highs and Swing Lows using fractal window."""
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        w = self.swing_window

        swing_highs: List[SwingPoint] = []
        swing_lows: List[SwingPoint] = []

        for i in range(w, n - w):
            # Swing High
            is_sh = True
            for k in range(1, w + 1):
                if highs[i] <= highs[i - k] or highs[i] <= highs[i + k]:
                    is_sh = False
                    break
            if is_sh:
                swing_highs.append(SwingPoint(
                    index=i,
                    timestamp=str(timestamps[i]),
                    price=float(highs[i]),
                    point_type="HIGH"
                ))

            # Swing Low
            is_sl = True
            for k in range(1, w + 1):
                if lows[i] >= lows[i - k] or lows[i] >= lows[i + k]:
                    is_sl = False
                    break
            if is_sl:
                swing_lows.append(SwingPoint(
                    index=i,
                    timestamp=str(timestamps[i]),
                    price=float(lows[i]),
                    point_type="LOW"
                ))

        return swing_highs, swing_lows

    def _classify_structures(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]):
        """Classifies swings into Higher Highs (HH), Lower Highs (LH), Higher Lows (HL), Lower Lows (LL)."""
        for i in range(1, len(swing_highs)):
            curr = swing_highs[i]
            prev = swing_highs[i - 1]
            curr.structure_tag = "HH" if curr.price > prev.price else "LH"

        for i in range(1, len(swing_lows)):
            curr = swing_lows[i]
            prev = swing_lows[i - 1]
            curr.structure_tag = "HL" if curr.price > prev.price else "LL"

    def _calculate_dealing_range(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint], current_price: float,
        highs: np.ndarray, lows: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates dealing range and Premium / Discount equilibrium."""
        if swing_highs and swing_lows:
            range_high = max(sp.price for sp in swing_highs[-3:])
            range_low = min(sp.price for sp in swing_lows[-3:])
        else:
            range_high = float(np.max(highs))
            range_low = float(np.min(lows))

        if range_high <= range_low:
            range_high = current_price * 1.01
            range_low = current_price * 0.99

        eq = range_low + (range_high - range_low) * 0.5
        pct = ((current_price - range_low) / (range_high - range_low)) * 100.0
        pct = max(0.0, min(100.0, pct))
        zone = "DISCOUNT" if current_price < eq else "PREMIUM"

        return {
            "high": round(range_high, 2),
            "low": round(range_low, 2),
            "eq": round(eq, 2),
            "pct": round(pct, 1),
            "zone": zone
        }

    def _find_liquidity_levels(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> List[LiquidityLevel]:
        """Identifies BSL and SSL liquidity levels and checks for Equal Highs / Lows."""
        levels: List[LiquidityLevel] = []
        for sh in swing_highs[-4:]:
            is_eqh = any(
                sh.index != other.index and abs(sh.price - other.price) / sh.price <= self.eq_tolerance
                for other in swing_highs[-4:]
            )
            levels.append(LiquidityLevel(price=sh.price, level_type="BSL", is_equal=is_eqh, timestamp=sh.timestamp))

        for sl in swing_lows[-4:]:
            is_eql = any(
                sl.index != other.index and abs(sl.price - other.price) / sl.price <= self.eq_tolerance
                for other in swing_lows[-4:]
            )
            levels.append(LiquidityLevel(price=sl.price, level_type="SSL", is_equal=is_eql, timestamp=sl.timestamp))

        return levels

    def _detect_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Detects 3-candle Fair Value Gaps (Bullish & Bearish imbalances)."""
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        fvgs: List[FairValueGap] = []

        start_idx = max(2, n - 25)
        for i in range(start_idx, n):
            # Bullish FVG: Low of candle i > High of candle i-2
            if lows[i] > highs[i - 2]:
                gap_top = lows[i]
                gap_bottom = highs[i - 2]
                mitigated = any(lows[k] <= gap_bottom for k in range(i + 1, n))
                fvgs.append(FairValueGap(
                    top=round(gap_top, 2),
                    bottom=round(gap_bottom, 2),
                    fvg_type="BULLISH",
                    created_at_index=i,
                    timestamp=timestamps[i],
                    mitigated=mitigated
                ))
            # Bearish FVG: High of candle i < Low of candle i-2
            elif highs[i] < lows[i - 2]:
                gap_top = lows[i - 2]
                gap_bottom = highs[i]
                mitigated = any(highs[k] >= gap_top for k in range(i + 1, n))
                fvgs.append(FairValueGap(
                    top=round(gap_top, 2),
                    bottom=round(gap_bottom, 2),
                    fvg_type="BEARISH",
                    created_at_index=i,
                    timestamp=timestamps[i],
                    mitigated=mitigated
                ))
        return fvgs

    def _detect_structure_breaks(
        self, df: pd.DataFrame, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> Dict[str, Any]:
        """Checks for Bullish/Bearish BOS or CHOCH on candle body closes."""
        closes = df["close"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)

        event = {"detected": False, "type": "None", "direction": "NEUTRAL", "price": 0.0, "details": "No break"}
        if not swing_highs or not swing_lows:
            return event

        last_sh = swing_highs[-1]
        last_sl = swing_lows[-1]

        # Scan recent 8 candles
        for i in range(max(0, n - 8), n):
            c_close = closes[i]
            if c_close > last_sh.price and i > last_sh.index:
                is_choch = last_sh.structure_tag == "LH"
                return {
                    "detected": True,
                    "type": "CHOCH" if is_choch else "BOS",
                    "direction": "BULLISH",
                    "price": round(last_sh.price, 2),
                    "details": f"Bullish {'CHOCH' if is_choch else 'BOS'} @ ₹{c_close:.2f} above swing high ₹{last_sh.price:.2f}",
                    "index": i,
                    "timestamp": timestamps[i]
                }
            elif c_close < last_sl.price and i > last_sl.index:
                is_choch = last_sl.structure_tag == "HL"
                return {
                    "detected": True,
                    "type": "CHOCH" if is_choch else "BOS",
                    "direction": "BEARISH",
                    "price": round(last_sl.price, 2),
                    "details": f"Bearish {'CHOCH' if is_choch else 'BOS'} @ ₹{c_close:.2f} below swing low ₹{last_sl.price:.2f}",
                    "index": i,
                    "timestamp": timestamps[i]
                }
        return event

    def _detect_liquidity_sweeps(self, df: pd.DataFrame, liquidity_levels: List[LiquidityLevel]) -> Dict[str, Any]:
        """Detects Liquidity Sweep / Grab: wick breaches level, but body closes inside."""
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)

        event = {"detected": False, "type": "None", "direction": "NEUTRAL", "price": 0.0, "details": "No sweep"}

        for i in range(max(0, n - 8), n):
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]

            for lvl in liquidity_levels:
                # SSL sweep: wick dips below SSL, candle closes above SSL -> Bullish Sweep
                if lvl.level_type == "SSL" and c_low < lvl.price and c_close > lvl.price:
                    tag = "Equal Lows (EQL)" if lvl.is_equal else "Sell-Side Liquidity (SSL)"
                    return {
                        "detected": True,
                        "type": "SSL_SWEEP",
                        "direction": "BULLISH",
                        "price": round(lvl.price, 2),
                        "details": f"Bullish {tag} swept @ ₹{lvl.price:.2f} (wick low ₹{c_low:.2f})",
                        "timestamp": timestamps[i]
                    }
                # BSL sweep: wick peaks above BSL, candle closes below BSL -> Bearish Sweep
                elif lvl.level_type == "BSL" and c_high > lvl.price and c_close < lvl.price:
                    tag = "Equal Highs (EQH)" if lvl.is_equal else "Buy-Side Liquidity (BSL)"
                    return {
                        "detected": True,
                        "type": "BSL_SWEEP",
                        "direction": "BEARISH",
                        "price": round(lvl.price, 2),
                        "details": f"Bearish {tag} swept @ ₹{lvl.price:.2f} (wick high ₹{c_high:.2f})",
                        "timestamp": timestamps[i]
                    }
        return event

    def _detect_order_blocks(self, df: pd.DataFrame, structure_event: Dict[str, Any]) -> List[OrderBlock]:
        """Identifies active Bullish and Bearish Order Blocks."""
        opens = df["open"].values.astype(float)
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        obs: List[OrderBlock] = []

        # Find Bullish Order Block (down candle before upward displacement)
        for i in range(max(0, n - 18), n - 1):
            if closes[i] < opens[i]:  # Bearish candle
                if closes[min(n - 1, i + 2)] > highs[i]:
                    mitigated = any(closes[k] < lows[i] for k in range(i + 1, n))
                    if not mitigated:
                        obs.append(OrderBlock(
                            top=round(highs[i], 2),
                            bottom=round(lows[i], 2),
                            ob_type="BULLISH",
                            candle_index=i,
                            timestamp=timestamps[i],
                            mitigated=False
                        ))

        # Find Bearish Order Block (up candle before downward displacement)
        for i in range(max(0, n - 18), n - 1):
            if closes[i] > opens[i]:  # Bullish candle
                if closes[min(n - 1, i + 2)] < lows[i]:
                    mitigated = any(closes[k] > highs[i] for k in range(i + 1, n))
                    if not mitigated:
                        obs.append(OrderBlock(
                            top=round(highs[i], 2),
                            bottom=round(lows[i], 2),
                            ob_type="BEARISH",
                            candle_index=i,
                            timestamp=timestamps[i],
                            mitigated=False
                        ))

        return obs

    def _evaluate_trend(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint], structure_event: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Determines market trend based on swing sequence and structure breaks."""
        if structure_event.get("detected"):
            direction = structure_event.get("direction")
            ev_type = structure_event.get("type")
            if direction == "BULLISH":
                return "BULLISH", f"Bullish Structure ({ev_type})"
            elif direction == "BEARISH":
                return "BEARISH", f"Bearish Structure ({ev_type})"

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            last_sh = swing_highs[-1]
            last_sl = swing_lows[-1]
            if last_sh.structure_tag == "HH" and last_sl.structure_tag == "HL":
                return "BULLISH", "Bullish (Higher Highs & Higher Lows)"
            elif last_sh.structure_tag == "LH" and last_sl.structure_tag == "LL":
                return "BEARISH", "Bearish (Lower Highs & Lower Lows)"

        return "SIDEWAYS", "Consolidation / Range"

smc_engine = SMCEngine()
