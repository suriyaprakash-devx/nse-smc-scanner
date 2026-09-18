from dataclasses import dataclass
from typing import List, Optional
from indicators import IndicatorSnapshot
from smc_engine import SMCSnapshot

@dataclass
class TargetProfile:
    target_price: float
    target_distance_pts: float
    target_distance_pct: float
    target_reference_name: str
    disclaimer: str = "Technical analysis estimate — not guaranteed."

class TargetEngine:
    """Calculates structural Maximum Target based on market structure, liquidity pools, order blocks, and ATR."""

    def calculate_target(
        self,
        direction: str,
        entry_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot
    ) -> TargetProfile:
        atr = indicators.atr
        min_move = max(atr * 1.5, entry_price * 0.008)  # Minimum meaningful structural move

        if direction == "BUY":
            candidates = []
            labels = []

            # 1. Buy-Side Liquidity (BSL / EQH) above entry
            bsl_levels = [sh.price for sh in smc.swing_highs if sh.price > (entry_price + min_move)]
            if bsl_levels:
                max_sh = max(bsl_levels)
                candidates.append(max_sh)
                labels.append(f"Major Swing High / BSL (₹{max_sh:.2f})")

            # 2. Bearish Order Block boundary
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BEARISH" and ob.bottom > (entry_price + min_move):
                    candidates.append(ob.bottom)
                    labels.append(f"Bearish Order Block Resistance (₹{ob.bottom:.2f})")
                    break

            # 3. Unmitigated Bearish FVG
            for fvg in smc.active_fvgs:
                if fvg.fvg_type == "BEARISH" and not fvg.mitigated and fvg.bottom > (entry_price + min_move):
                    candidates.append(fvg.bottom)
                    labels.append(f"Unmitigated FVG Level (₹{fvg.bottom:.2f})")
                    break

            # 4. Dealing Range High / Fibonacci Extension (1.272 / 1.618)
            dr_high = smc.dealing_range.get("high", entry_price)
            dr_low = smc.dealing_range.get("low", entry_price)
            dr_span = max(dr_high - dr_low, entry_price * 0.01)
            fib_target = dr_high + (dr_span * 0.618)
            if fib_target > entry_price + min_move:
                candidates.append(fib_target)
                labels.append(f"Fibonacci 1.618 Extension (₹{fib_target:.2f})")

            # 5. Fallback ATR Structural expansion
            fallback_target = entry_price + (atr * 3.0)
            if not candidates:
                target = fallback_target
                ref_label = "ATR Volatility Expansion Target"
            else:
                # Pick the furthest structural target that is within reasonable realistic bound (max 15%)
                capped_candidates = [(c, l) for c, l in zip(candidates, labels) if c <= entry_price * 1.15]
                if capped_candidates:
                    target, ref_label = max(capped_candidates, key=lambda x: x[0])
                else:
                    target = fallback_target
                    ref_label = "ATR Volatility Expansion Target"

            # Ensure target is strictly above entry by at least min_move
            if target <= entry_price + min_move:
                target = entry_price + min_move
                ref_label = "Structural Minimum Upside Target"

            pts = target - entry_price
            pct = (pts / entry_price) * 100.0

        else:
            # SELL / Short Target
            candidates = []
            labels = []

            # 1. Sell-Side Liquidity (SSL / EQL) below entry
            ssl_levels = [sl.price for sl in smc.swing_lows if sl.price < (entry_price - min_move)]
            if ssl_levels:
                min_sl = min(ssl_levels)
                candidates.append(min_sl)
                labels.append(f"Major Swing Low / SSL (₹{min_sl:.2f})")

            # 2. Bullish Order Block support
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BULLISH" and ob.top < (entry_price - min_move):
                    candidates.append(ob.top)
                    labels.append(f"Bullish Order Block Support (₹{ob.top:.2f})")
                    break

            # 3. Unmitigated Bullish FVG
            for fvg in smc.active_fvgs:
                if fvg.fvg_type == "BULLISH" and not fvg.mitigated and fvg.top < (entry_price - min_move):
                    candidates.append(fvg.top)
                    labels.append(f"Unmitigated FVG Level (₹{fvg.top:.2f})")
                    break

            # 4. Fibonacci Extension downside
            dr_high = smc.dealing_range.get("high", entry_price)
            dr_low = smc.dealing_range.get("low", entry_price)
            dr_span = max(dr_high - dr_low, entry_price * 0.01)
            fib_target = max(entry_price * 0.5, dr_low - (dr_span * 0.618))
            if fib_target < entry_price - min_move:
                candidates.append(fib_target)
                labels.append(f"Fibonacci 1.618 Downside Extension (₹{fib_target:.2f})")

            # 5. Fallback ATR Structural contraction
            fallback_target = max(1.0, entry_price - (atr * 3.0))
            if not candidates:
                target = fallback_target
                ref_label = "ATR Volatility Expansion Target"
            else:
                capped_candidates = [(c, l) for c, l in zip(candidates, labels) if c >= entry_price * 0.85]
                if capped_candidates:
                    target, ref_label = min(capped_candidates, key=lambda x: x[0])
                else:
                    target = fallback_target
                    ref_label = "ATR Volatility Expansion Target"

            # Ensure target is strictly below entry by at least min_move
            if target >= entry_price - min_move:
                target = max(1.0, entry_price - min_move)
                ref_label = "Structural Minimum Downside Target"

            pts = entry_price - target
            pct = -(pts / entry_price) * 100.0

        return TargetProfile(
            target_price=round(target, 2),
            target_distance_pts=round(pts, 2),
            target_distance_pct=round(pct, 2),
            target_reference_name=ref_label,
            disclaimer="Technical analysis estimate — not guaranteed."
        )

target_engine = TargetEngine()
