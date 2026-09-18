from dataclasses import dataclass
from typing import Optional, List
from config import settings
from indicators import IndicatorSnapshot
from smc_engine import SMCSnapshot

@dataclass
class RiskProfile:
    stop_loss: float
    risk_points: float
    risk_pct: float
    invalidation_level_name: str
    risk_reward_ratio: float  # e.g., 2.0 for 1:2.0

class RiskEngine:
    """Computes dynamic structural Stop Loss and Risk/Reward based on SMC invalidation levels and ATR buffers."""

    def __init__(self, atr_multiplier: float = 1.0):
        self.atr_multiplier = atr_multiplier

    def calculate_stop_loss(
        self,
        direction: str,
        entry_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot,
        target_price: float
    ) -> RiskProfile:
        atr_buffer = max(indicators.atr * self.atr_multiplier, entry_price * 0.003)

        if direction == "BUY":
            candidate_sls = []
            invalidation_names = []

            # 1. Recent confirmed swing low
            if smc.swing_lows:
                recent_sl = smc.swing_lows[-1].price
                if recent_sl < entry_price:
                    candidate_sls.append(recent_sl - (atr_buffer * 0.5))
                    invalidation_names.append(f"Swing Low (₹{recent_sl:.2f})")

            # 2. Bullish Order Block low
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BULLISH" and ob.bottom < entry_price:
                    candidate_sls.append(ob.bottom - (atr_buffer * 0.3))
                    invalidation_names.append(f"Bullish OB Low (₹{ob.bottom:.2f})")
                    break

            # 3. Liquidity sweep low
            if smc.sweep_event.get("detected") and smc.sweep_event.get("direction") == "BULLISH":
                sweep_lvl = smc.sweep_event.get("price", 0.0)
                if 0 < sweep_lvl < entry_price:
                    candidate_sls.append(sweep_lvl - atr_buffer)
                    invalidation_names.append(f"SSL Sweep Low (₹{sweep_lvl:.2f})")

            # 4. ATR buffer floor
            fallback_sl = entry_price - (indicators.atr * 1.5)
            if not candidate_sls:
                sl = fallback_sl
                invalidation_tag = "ATR Dynamic Buffer"
            else:
                # Pick the highest structural invalidation point that is below entry to preserve tight risk
                valid_pairs = [(s, name) for s, name in zip(candidate_sls, invalidation_names) if s < entry_price]
                if valid_pairs:
                    # Choose tightest valid structural pivot
                    sl, invalidation_tag = max(valid_pairs, key=lambda x: x[0])
                else:
                    sl = fallback_sl
                    invalidation_tag = "ATR Dynamic Buffer"

            # Enforce absolute safety clamp: SL must be strictly below entry price (at least 0.4% and at most 5%)
            min_sl_dist = entry_price * 0.004
            max_sl_dist = entry_price * 0.05
            if (entry_price - sl) < min_sl_dist:
                sl = entry_price - min_sl_dist
            elif (entry_price - sl) > max_sl_dist:
                sl = entry_price - max_sl_dist

            risk_pts = entry_price - sl
            risk_pct = (risk_pts / entry_price) * 100.0

            reward_pts = max(0.0, target_price - entry_price)
            rr = (reward_pts / risk_pts) if risk_pts > 0 else 2.0

        else:
            # SELL / Short
            candidate_sls = []
            invalidation_names = []

            # 1. Recent confirmed swing high
            if smc.swing_highs:
                recent_sh = smc.swing_highs[-1].price
                if recent_sh > entry_price:
                    candidate_sls.append(recent_sh + (atr_buffer * 0.5))
                    invalidation_names.append(f"Swing High (₹{recent_sh:.2f})")

            # 2. Bearish Order Block high
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BEARISH" and ob.top > entry_price:
                    candidate_sls.append(ob.top + (atr_buffer * 0.3))
                    invalidation_names.append(f"Bearish OB High (₹{ob.top:.2f})")
                    break

            # 3. Liquidity sweep high
            if smc.sweep_event.get("detected") and smc.sweep_event.get("direction") == "BEARISH":
                sweep_lvl = smc.sweep_event.get("price", 0.0)
                if sweep_lvl > entry_price:
                    candidate_sls.append(sweep_lvl + atr_buffer)
                    invalidation_names.append(f"BSL Sweep High (₹{sweep_lvl:.2f})")

            # 4. ATR buffer ceiling
            fallback_sl = entry_price + (indicators.atr * 1.5)
            if not candidate_sls:
                sl = fallback_sl
                invalidation_tag = "ATR Dynamic Buffer"
            else:
                valid_pairs = [(s, name) for s, name in zip(candidate_sls, invalidation_names) if s > entry_price]
                if valid_pairs:
                    sl, invalidation_tag = min(valid_pairs, key=lambda x: x[0])
                else:
                    sl = fallback_sl
                    invalidation_tag = "ATR Dynamic Buffer"

            # Enforce safety clamp
            min_sl_dist = entry_price * 0.004
            max_sl_dist = entry_price * 0.05
            if (sl - entry_price) < min_sl_dist:
                sl = entry_price + min_sl_dist
            elif (sl - entry_price) > max_sl_dist:
                sl = entry_price + max_sl_dist

            risk_pts = sl - entry_price
            risk_pct = (risk_pts / entry_price) * 100.0

            reward_pts = max(0.0, entry_price - target_price)
            rr = (reward_pts / risk_pts) if risk_pts > 0 else 2.0

        return RiskProfile(
            stop_loss=round(sl, 2),
            risk_points=round(risk_pts, 2),
            risk_pct=round(risk_pct, 2),
            invalidation_level_name=invalidation_tag,
            risk_reward_ratio=round(rr, 1)
        )

risk_engine = RiskEngine()
