from dataclasses import dataclass
from typing import List, Optional, Tuple
from indicators import IndicatorSnapshot
from smc_engine import SMCSnapshot

@dataclass
class TargetProfile:
    tp1: float
    tp1_name: str
    tp1_distance_pts: float
    tp1_distance_pct: float
    tp2: float
    tp2_name: str
    tp2_distance_pts: float
    tp2_distance_pct: float
    tp3: Optional[float] = None
    tp3_name: Optional[str] = None
    tp3_distance_pts: Optional[float] = None
    tp3_distance_pct: Optional[float] = None
    # Primary target for R:R calculation
    target_price: float = 0.0
    target_reference_name: str = ""
    disclaimer: str = "Technical analysis estimate — not guaranteed."

class TargetEngine:
    """Calculates multi-level structural targets:
    TP1 = Nearest meaningful liquidity/structure (swing high/low, fresh OB, or FVG)
    TP2 = Next major liquidity/structure (major BSL/SSL, PDH/PDL, dealing range boundary)
    TP3 = Extended Fibonacci expansion / runner target
    Targets are calibrated realistically for intraday trading.
    """

    def calculate_targets(
        self,
        direction: str,
        entry_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot
    ) -> TargetProfile:
        atr = max(indicators.atr, entry_price * 0.002)
        min_move = max(atr * 0.5, entry_price * 0.003)

        if direction == "BUY":
            candidates: List[Tuple[float, str]] = []

            # 1. Recent confirmed swing highs above entry
            for sh in sorted(smc.swing_highs, key=lambda s: s.price):
                if sh.price > entry_price + min_move:
                    candidates.append((sh.price, f"Swing High (₹{sh.price:.2f})"))

            # 2. Fresh Bearish Order Block boundary
            for ob in smc.fresh_order_blocks:
                if ob.ob_type == "BEARISH" and ob.bottom > entry_price + min_move:
                    candidates.append((ob.bottom, f"Fresh Bearish OB Resistance (₹{ob.bottom:.2f})"))

            # 3. Fresh Bearish FVG boundary
            for fvg in smc.fresh_fvgs:
                if fvg.fvg_type == "BEARISH" and fvg.bottom > entry_price + min_move:
                    candidates.append((fvg.bottom, f"Fresh Bearish FVG Inflow (₹{fvg.bottom:.2f})"))

            # 4. Previous Day High (PDH) if above entry
            if smc.pdh > entry_price + min_move:
                candidates.append((smc.pdh, f"Previous Day High (₹{smc.pdh:.2f})"))

            # 5. Dealing Range High
            dr_high = smc.dealing_range.get("high", 0.0)
            if dr_high > entry_price + min_move:
                candidates.append((dr_high, f"Dealing Range High (₹{dr_high:.2f})"))

            # Remove duplicates and sort ascending
            unique_candidates = []
            seen_prices = set()
            for p, name in sorted(candidates, key=lambda x: x[0]):
                rounded_p = round(p, 2)
                if rounded_p not in seen_prices and rounded_p <= entry_price * 1.08:
                    seen_prices.add(rounded_p)
                    unique_candidates.append((rounded_p, name))

            # Determine TP1, TP2, TP3
            fallback_tp1 = round(entry_price + (atr * 1.8), 2)
            fallback_tp2 = round(entry_price + (atr * 3.2), 2)
            fallback_tp3 = round(entry_price + (atr * 4.5), 2)

            if not unique_candidates:
                tp1, tp1_name = fallback_tp1, "ATR Volatility Expansion (TP1)"
                tp2, tp2_name = fallback_tp2, "ATR Extended Expansion (TP2)"
                tp3, tp3_name = fallback_tp3, "ATR Runner Target (TP3)"
            elif len(unique_candidates) == 1:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = max(fallback_tp2, tp1 + atr), "Next Major Structural Level (TP2)"
                tp3, tp3_name = tp2 + atr, "Extended Target (TP3)"
            elif len(unique_candidates) == 2:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = unique_candidates[1]
                tp3, tp3_name = max(fallback_tp3, tp2 + atr), "Extended Structural Expansion (TP3)"
            else:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = unique_candidates[1]
                tp3, tp3_name = unique_candidates[2]

            tp1_dist_pts = round(tp1 - entry_price, 2)
            tp1_dist_pct = round((tp1_dist_pts / entry_price) * 100.0, 2)
            tp2_dist_pts = round(tp2 - entry_price, 2)
            tp2_dist_pct = round((tp2_dist_pts / entry_price) * 100.0, 2)
            tp3_dist_pts = round(tp3 - entry_price, 2)
            tp3_dist_pct = round((tp3_dist_pts / entry_price) * 100.0, 2)

        else:
            # SELL / SHORT Targets
            candidates: List[Tuple[float, str]] = []

            # 1. Recent confirmed swing lows below entry
            for sl in sorted(smc.swing_lows, key=lambda s: s.price, reverse=True):
                if sl.price < entry_price - min_move:
                    candidates.append((sl.price, f"Swing Low (₹{sl.price:.2f})"))

            # 2. Fresh Bullish Order Block boundary
            for ob in smc.fresh_order_blocks:
                if ob.ob_type == "BULLISH" and ob.top < entry_price - min_move:
                    candidates.append((ob.top, f"Fresh Bullish OB Support (₹{ob.top:.2f})"))

            # 3. Fresh Bullish FVG boundary
            for fvg in smc.fresh_fvgs:
                if fvg.fvg_type == "BULLISH" and fvg.top < entry_price - min_move:
                    candidates.append((fvg.top, f"Fresh Bullish FVG Inflow (₹{fvg.top:.2f})"))

            # 4. Previous Day Low (PDL) if below entry
            if smc.pdl < entry_price - min_move:
                candidates.append((smc.pdl, f"Previous Day Low (₹{smc.pdl:.2f})"))

            # 5. Dealing Range Low
            dr_low = smc.dealing_range.get("low", 0.0)
            if dr_low < entry_price - min_move:
                candidates.append((dr_low, f"Dealing Range Low (₹{dr_low:.2f})"))

            # Remove duplicates and sort descending (closest below entry first)
            unique_candidates = []
            seen_prices = set()
            for p, name in sorted(candidates, key=lambda x: x[0], reverse=True):
                rounded_p = round(p, 2)
                if rounded_p not in seen_prices and rounded_p >= entry_price * 0.92:
                    seen_prices.add(rounded_p)
                    unique_candidates.append((rounded_p, name))

            fallback_tp1 = round(max(1.0, entry_price - (atr * 1.8)), 2)
            fallback_tp2 = round(max(1.0, entry_price - (atr * 3.2)), 2)
            fallback_tp3 = round(max(1.0, entry_price - (atr * 4.5)), 2)

            if not unique_candidates:
                tp1, tp1_name = fallback_tp1, "ATR Volatility Expansion (TP1)"
                tp2, tp2_name = fallback_tp2, "ATR Extended Expansion (TP2)"
                tp3, tp3_name = fallback_tp3, "ATR Runner Target (TP3)"
            elif len(unique_candidates) == 1:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = min(fallback_tp2, tp1 - atr), "Next Major Structural Level (TP2)"
                tp3, tp3_name = max(1.0, tp2 - atr), "Extended Target (TP3)"
            elif len(unique_candidates) == 2:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = unique_candidates[1]
                tp3, tp3_name = min(fallback_tp3, tp2 - atr), "Extended Structural Expansion (TP3)"
            else:
                tp1, tp1_name = unique_candidates[0]
                tp2, tp2_name = unique_candidates[1]
                tp3, tp3_name = unique_candidates[2]

            tp1_dist_pts = round(entry_price - tp1, 2)
            tp1_dist_pct = round(-(tp1_dist_pts / entry_price) * 100.0, 2)
            tp2_dist_pts = round(entry_price - tp2, 2)
            tp2_dist_pct = round(-(tp2_dist_pts / entry_price) * 100.0, 2)
            tp3_dist_pts = round(entry_price - tp3, 2)
            tp3_dist_pct = round(-(tp3_dist_pts / entry_price) * 100.0, 2)

        # For backwards-compatibility and primary R:R evaluation, target_price defaults to TP2
        return TargetProfile(
            tp1=tp1,
            tp1_name=tp1_name,
            tp1_distance_pts=tp1_dist_pts,
            tp1_distance_pct=tp1_dist_pct,
            tp2=tp2,
            tp2_name=tp2_name,
            tp2_distance_pts=tp2_dist_pts,
            tp2_distance_pct=tp2_dist_pct,
            tp3=tp3,
            tp3_name=tp3_name,
            tp3_distance_pts=tp3_dist_pts,
            tp3_distance_pct=tp3_dist_pct,
            target_price=tp2,
            target_reference_name=tp2_name
        )

    def calculate_target(
        self,
        direction: str,
        entry_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot
    ) -> TargetProfile:
        """Alias for calculate_targets to ensure backwards compatibility."""
        return self.calculate_targets(direction, entry_price, indicators, smc)

target_engine = TargetEngine()
