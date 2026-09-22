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
    is_protected: bool = False  # Strong high/low that led to a BOS

@dataclass
class LiquidityLevel:
    price: float
    level_type: str  # "BSL" (Buy-side) or "SSL" (Sell-side) or "PDH" or "PDL"
    is_equal: bool = False  # Equal Highs (EQH) or Equal Lows (EQL)
    timestamp: str = ""
    swept: bool = False

@dataclass
class FairValueGap:
    top: float
    bottom: float
    fvg_type: str  # "BULLISH" or "BEARISH"
    created_at_index: int
    timestamp: str
    mitigated: bool = False
    is_fresh: bool = True

@dataclass
class OrderBlock:
    top: float
    bottom: float
    ob_type: str  # "BULLISH" or "BEARISH"
    candle_index: int
    timestamp: str
    mitigated: bool = False
    is_fresh: bool = True
    volume_surge: bool = False

@dataclass
class SMCSnapshot:
    trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    market_structure: str  # e.g., "Bullish (Higher Highs & Higher Lows)"
    bos_choch_event: Dict[str, Any]
    sweep_event: Dict[str, Any]
    active_order_blocks: List[OrderBlock]
    fresh_order_blocks: List[OrderBlock]
    active_fvgs: List[FairValueGap]
    fresh_fvgs: List[FairValueGap]
    dealing_range: Dict[str, Any]
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    protected_highs: List[SwingPoint]
    protected_lows: List[SwingPoint]
    pdh: float
    pdl: float
    pdh_pdl_status: str  # "ABOVE_PDH", "BELOW_PDL", "INSIDE_PD_RANGE"
    pdh_pdl_sweep: Dict[str, Any]

class SMCEngine:
    """Smart Money Concepts (SMC) quantitative engine.
    Strictly deterministic and non-repainting: operates strictly on completed candles.
    Tracks confirmed swings, protected levels, BOS/CHOCH, sweeps (including PDH/PDL),
    fresh/mitigated Order Blocks and FVGs, and dealing ranges.
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
        volumes = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        current_price = closes[-1]

        # 1. Detect Swing Highs and Swing Lows
        swing_highs, swing_lows = self._detect_swings(df)
        self._classify_structures(swing_highs, swing_lows)

        # 2. Determine Dealing Range & Premium / Discount Zone
        dealing_range = self._calculate_dealing_range(swing_highs, swing_lows, current_price, highs, lows)

        # 3. Structure Breaks: BOS & CHOCH and mark protected swings
        structure_event = self._detect_structure_breaks(df, swing_highs, swing_lows)
        protected_highs, protected_lows = self._identify_protected_swings(swing_highs, swing_lows, structure_event)

        # 4. Liquidity Levels and Sweeps (including PDH and PDL)
        liquidity_levels = self._find_liquidity_levels(swing_highs, swing_lows)
        sweep_event = self._detect_liquidity_sweeps(df, liquidity_levels)
        pdh_pdl_sweep = self._detect_pd_sweeps(df, pdh, pdl)

        # 5. Fair Value Gaps (FVG) - Fresh vs Mitigated
        all_fvgs = self._detect_fvgs(df)
        fresh_fvgs = [f for f in all_fvgs if not f.mitigated]

        # 6. Order Blocks - Fresh vs Mitigated
        all_obs = self._detect_order_blocks(df, structure_event)
        fresh_obs = [ob for ob in all_obs if not ob.mitigated]

        # 7. Overall Market Structure & Trend
        trend, ms_label = self._evaluate_trend(swing_highs, swing_lows, structure_event)

        # 8. PDH / PDL status
        actual_pdh = float(pdh) if (pdh is not None and pdh > 0) else float(np.max(highs[:-15]) if n > 20 else np.max(highs))
        actual_pdl = float(pdl) if (pdl is not None and pdl > 0) else float(np.min(lows[:-15]) if n > 20 else np.min(lows))

        if current_price > actual_pdh:
            pdh_pdl_status = "ABOVE_PDH"
        elif current_price < actual_pdl:
            pdh_pdl_status = "BELOW_PDL"
        else:
            pdh_pdl_status = "INSIDE_PD_RANGE"

        return SMCSnapshot(
            trend=trend,
            market_structure=ms_label,
            bos_choch_event=structure_event,
            sweep_event=sweep_event,
            active_order_blocks=all_obs,
            fresh_order_blocks=fresh_obs,
            active_fvgs=all_fvgs,
            fresh_fvgs=fresh_fvgs,
            dealing_range=dealing_range,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            protected_highs=protected_highs,
            protected_lows=protected_lows,
            pdh=round(actual_pdh, 2),
            pdl=round(actual_pdl, 2),
            pdh_pdl_status=pdh_pdl_status,
            pdh_pdl_sweep=pdh_pdl_sweep
        )

    def _detect_swings(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Identifies confirmed Swing Highs and Swing Lows using fractal window.
        Operates strictly up to n - w - 1 to ensure zero future bias.
        """
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

    def _identify_protected_swings(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint], structure_event: Dict[str, Any]
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """Identifies strong protected highs and lows that created confirmed breaks."""
        protected_highs = []
        protected_lows = []

        if structure_event.get("detected"):
            direction = structure_event.get("direction")
            if direction == "BULLISH" and swing_lows:
                # The lowest swing low before the break is protected
                p_low = swing_lows[-1]
                p_low.is_protected = True
                protected_lows.append(p_low)
            elif direction == "BEARISH" and swing_highs:
                # The highest swing high before the break is protected
                p_high = swing_highs[-1]
                p_high.is_protected = True
                protected_highs.append(p_high)

        return protected_highs, protected_lows

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
        """Detects Fair Value Gaps and monitors whether they remain fresh or have been mitigated."""
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        fvgs: List[FairValueGap] = []

        start_idx = max(2, n - 35)
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
                    mitigated=mitigated,
                    is_fresh=not mitigated
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
                    mitigated=mitigated,
                    is_fresh=not mitigated
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

        # Scan recent completed candles (last 10)
        for i in range(max(0, n - 10), n):
            c_close = closes[i]
            if c_close > last_sh.price and i > last_sh.index:
                is_choch = last_sh.structure_tag == "LH"
                return {
                    "detected": True,
                    "type": "CHOCH" if is_choch else "BOS",
                    "direction": "BULLISH",
                    "price": round(last_sh.price, 2),
                    "details": f"Bullish {'CHOCH' if is_choch else 'BOS'} close @ ₹{c_close:.2f} above swing high ₹{last_sh.price:.2f}",
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
                    "details": f"Bearish {'CHOCH' if is_choch else 'BOS'} close @ ₹{c_close:.2f} below swing low ₹{last_sl.price:.2f}",
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

    def _detect_pd_sweeps(self, df: pd.DataFrame, pdh: Optional[float], pdl: Optional[float]) -> Dict[str, Any]:
        """Detects liquidity sweeps of genuine Previous Day High or Low."""
        event = {"detected": False, "type": "None", "direction": "NEUTRAL", "price": 0.0, "details": "None"}
        if pdh is None or pdl is None or pdh <= 0 or pdl <= 0:
            return event

        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)

        for i in range(max(0, n - 8), n):
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]

            # PDH Sweep: Wick breaks above PDH, but body closes below PDH -> Bearish Rejection
            if c_high > pdh and c_close < pdh:
                return {
                    "detected": True,
                    "type": "PDH_SWEEP",
                    "direction": "BEARISH",
                    "price": round(pdh, 2),
                    "details": f"PDH Swept & Rejected @ ₹{pdh:.2f} (wick high ₹{c_high:.2f})",
                    "timestamp": timestamps[i]
                }
            # PDL Sweep: Wick breaks below PDL, but body closes above PDL -> Bullish Rejection
            elif c_low < pdl and c_close > pdl:
                return {
                    "detected": True,
                    "type": "PDL_SWEEP",
                    "direction": "BULLISH",
                    "price": round(pdl, 2),
                    "details": f"PDL Swept & Rejected @ ₹{pdl:.2f} (wick low ₹{c_low:.2f})",
                    "timestamp": timestamps[i]
                }
        return event

    def _detect_order_blocks(self, df: pd.DataFrame, structure_event: Dict[str, Any]) -> List[OrderBlock]:
        """Identifies active Bullish and Bearish Order Blocks with mitigation tracking."""
        opens = df["open"].values.astype(float)
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        timestamps = df["timestamp"].astype(str).values
        n = len(df)
        obs: List[OrderBlock] = []

        # Find Bullish Order Block (down candle before upward displacement)
        for i in range(max(0, n - 25), n - 1):
            if closes[i] < opens[i]:  # Bearish candle
                # Check for subsequent upward displacement
                if closes[min(n - 1, i + 2)] > highs[i]:
                    # Mitigated if subsequent candle closes below the OB bottom
                    mitigated = any(closes[k] < lows[i] for k in range(i + 1, n))
                    obs.append(OrderBlock(
                        top=round(highs[i], 2),
                        bottom=round(lows[i], 2),
                        ob_type="BULLISH",
                        candle_index=i,
                        timestamp=timestamps[i],
                        mitigated=mitigated,
                        is_fresh=not mitigated
                    ))

        # Find Bearish Order Block (up candle before downward displacement)
        for i in range(max(0, n - 25), n - 1):
            if closes[i] > opens[i]:  # Bullish candle
                if closes[min(n - 1, i + 2)] < lows[i]:
                    mitigated = any(closes[k] > highs[i] for k in range(i + 1, n))
                    obs.append(OrderBlock(
                        top=round(highs[i], 2),
                        bottom=round(lows[i], 2),
                        ob_type="BEARISH",
                        candle_index=i,
                        timestamp=timestamps[i],
                        mitigated=mitigated,
                        is_fresh=not mitigated
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
