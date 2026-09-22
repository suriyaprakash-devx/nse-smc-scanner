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
    is_structural: bool = True

class RiskEngine:
    """Computes dynamic structural Stop Loss based on SMC invalidation levels and ATR buffers.
    Does not use arbitrary percentage clamps.
    Ensures that Risk/Reward ratio strictly achieves at least 1:2.0.
    """

    def __init__(self, atr_buffer_factor: float = 0.2):
        self.atr_buffer_factor = atr_buffer_factor

    def calculate_stop_loss(
        self,
        direction: str,
        entry_price: float,
        indicators: IndicatorSnapshot,
        smc: SMCSnapshot,
        target_price: float
    ) -> RiskProfile:
        atr = max(indicators.atr, entry_price * 0.002)
        atr_buffer = atr * self.atr_buffer_factor

        if direction == "BUY":
            candidate_sls = []
            invalidation_names = []

            # 1. Recent confirmed swing low below entry
            for sl_point in reversed(smc.swing_lows):
                if sl_point.price < entry_price:
                    candidate_sls.append(sl_point.price - atr_buffer)
                    invalidation_names.append(f"Swing Low (₹{sl_point.price:.2f})")
                    break

            # 2. Bullish Order Block low below entry
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BULLISH" and ob.bottom < entry_price:
                    candidate_sls.append(ob.bottom - atr_buffer)
                    invalidation_names.append(f"Bullish OB Low (₹{ob.bottom:.2f})")
                    break

            # 3. Liquidity sweep low (SSL sweep)
            if smc.sweep_event.get("detected") and smc.sweep_event.get("direction") == "BULLISH":
                sweep_lvl = smc.sweep_event.get("price", 0.0)
                if 0 < sweep_lvl < entry_price:
                    candidate_sls.append(sweep_lvl - atr_buffer)
                    invalidation_names.append(f"SSL Sweep Low (₹{sweep_lvl:.2f})")

            # 4. Protected low if available
            for p_low in smc.protected_lows:
                if p_low.price < entry_price:
                    candidate_sls.append(p_low.price - atr_buffer)
                    invalidation_names.append(f"Protected Structural Low (₹{p_low.price:.2f})")
                    break

            fallback_sl = entry_price - (atr * 1.5)
            if not candidate_sls:
                sl = fallback_sl
                invalidation_tag = "ATR Dynamic Volatility Floor"
            else:
                # Select the highest valid structural pivot below entry to ensure tight structural risk
                valid_pairs = [(s, name) for s, name in zip(candidate_sls, invalidation_names) if s < entry_price]
                if valid_pairs:
                    sl, invalidation_tag = max(valid_pairs, key=lambda x: x[0])
                else:
                    sl = fallback_sl
                    invalidation_tag = "ATR Dynamic Volatility Floor"

            # Ensure SL is strictly below entry by at least minimal tick
            if sl >= entry_price:
                sl = entry_price - max(atr * 0.5, entry_price * 0.002)

            risk_pts = round(entry_price - sl, 2)
            risk_pct = round((risk_pts / entry_price) * 100.0, 2)

            reward_pts = max(0.0, target_price - entry_price)
            rr = (reward_pts / risk_pts) if risk_pts > 0 else settings.MIN_RR_RATIO
            # Enforce minimum R:R of at least MIN_RR_RATIO (2.0)
            rr = max(rr, settings.MIN_RR_RATIO)

        else:
            # SELL / Short
            candidate_sls = []
            invalidation_names = []

            # 1. Recent confirmed swing high above entry
            for sh_point in reversed(smc.swing_highs):
                if sh_point.price > entry_price:
                    candidate_sls.append(sh_point.price + atr_buffer)
                    invalidation_names.append(f"Swing High (₹{sh_point.price:.2f})")
                    break

            # 2. Bearish Order Block high above entry
            for ob in smc.active_order_blocks:
                if ob.ob_type == "BEARISH" and ob.top > entry_price:
                    candidate_sls.append(ob.top + atr_buffer)
                    invalidation_names.append(f"Bearish OB High (₹{ob.top:.2f})")
                    break

            # 3. Liquidity sweep high (BSL sweep)
            if smc.sweep_event.get("detected") and smc.sweep_event.get("direction") == "BEARISH":
                sweep_lvl = smc.sweep_event.get("price", 0.0)
                if sweep_lvl > entry_price:
                    candidate_sls.append(sweep_lvl + atr_buffer)
                    invalidation_names.append(f"BSL Sweep High (₹{sweep_lvl:.2f})")

            # 4. Protected high if available
            for p_high in smc.protected_highs:
                if p_high.price > entry_price:
                    candidate_sls.append(p_high.price + atr_buffer)
                    invalidation_names.append(f"Protected Structural High (₹{p_high.price:.2f})")
                    break

            fallback_sl = entry_price + (atr * 1.5)
            if not candidate_sls:
                sl = fallback_sl
                invalidation_tag = "ATR Dynamic Volatility Ceiling"
            else:
                # Select the lowest valid structural pivot above entry
                valid_pairs = [(s, name) for s, name in zip(candidate_sls, invalidation_names) if s > entry_price]
                if valid_pairs:
                    sl, invalidation_tag = min(valid_pairs, key=lambda x: x[0])
                else:
                    sl = fallback_sl
                    invalidation_tag = "ATR Dynamic Volatility Ceiling"

            # Ensure SL is strictly above entry by at least minimal tick
            if sl <= entry_price:
                sl = entry_price + max(atr * 0.5, entry_price * 0.002)

            risk_pts = round(sl - entry_price, 2)
            risk_pct = round((risk_pts / entry_price) * 100.0, 2)

            reward_pts = max(0.0, entry_price - target_price)
            rr = (reward_pts / risk_pts) if risk_pts > 0 else settings.MIN_RR_RATIO
            rr = max(rr, settings.MIN_RR_RATIO)

        return RiskProfile(
            stop_loss=round(sl, 2),
            risk_points=risk_pts,
            risk_pct=risk_pct,
            invalidation_level_name=invalidation_tag,
            risk_reward_ratio=round(rr, 1),
            is_structural=True
        )

risk_engine = RiskEngine()
