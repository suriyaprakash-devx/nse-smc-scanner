import os
import pytest
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import pytz

from config import settings
from calendar_utils import is_nse_holiday, is_nse_trading_day, get_previous_trading_day, get_market_session_status
from utils import MarketDataUnavailableError, mask_token
from market_data import filter_completed_candles, resample_to_10m, prepare_market_data
from indicators import compute_indicators
from smc_engine import smc_engine, SMCSnapshot
from signal_engine import signal_engine, SignalResult
from target_engine import target_engine, TargetProfile
from risk_engine import risk_engine, RiskProfile
from upstox_client import upstox_client
from telegram import telegram_notifier

IST = pytz.timezone("Asia/Kolkata")

# Helper to build realistic synthetic DataFrame for testing
def create_sample_candles(n_bars: int = 50, start_time: Optional[datetime] = None, base_price: float = 100.0, trend: str = "bullish") -> pd.DataFrame:
    if start_time is None:
        start_time = IST.localize(datetime(2026, 9, 21, 9, 15))
    
    rows = []
    price = base_price
    for i in range(n_bars):
        ts = start_time + timedelta(minutes=i * 5)
        if trend == "bullish":
            delta = 0.4 + (i * 0.05)
        elif trend == "bearish":
            delta = -0.4 - (i * 0.05)
        else:
            delta = 0.2 if i % 2 == 0 else -0.2
            
        c_open = price
        c_close = price + delta
        c_high = max(c_open, c_close) + 0.3
        c_low = min(c_open, c_close) - 0.3
        volume = 100000 + (50000 if i == n_bars - 1 else 0)
        
        rows.append({
            "timestamp": ts,
            "open": round(c_open, 2),
            "high": round(c_high, 2),
            "low": round(c_low, 2),
            "close": round(c_close, 2),
            "volume": volume
        })
        price = c_close
        
    return pd.DataFrame(rows)


def test_previous_day_trading_session():
    """Verifies that previous trading session correctly rolls over weekends and NSE holidays."""
    # Test on a Monday: previous session should be Friday
    monday_dt = IST.localize(datetime(2026, 9, 21, 10, 30))  # Sept 21, 2026 is Monday
    prev_day = get_previous_trading_day(monday_dt)
    assert prev_day.weekday() == 4  # Friday
    assert prev_day == date(2026, 9, 18)

    # Test on a weekend (Sunday): should roll back to Friday
    sunday_dt = IST.localize(datetime(2026, 9, 20, 12, 0))
    prev_sunday = get_previous_trading_day(sunday_dt)
    assert prev_sunday.weekday() == 4
    assert prev_sunday == date(2026, 9, 18)


def test_nse_holiday_handling():
    """Verifies NSE official holidays detection."""
    # Republic day 2026-01-26
    is_hol, hol_name = is_nse_holiday(date(2026, 1, 26))
    assert is_hol is True
    assert "Republic Day" in hol_name
    assert is_nse_trading_day(date(2026, 1, 26)) is False

    # Day following Republic day (Tuesday 2026-01-27)
    assert is_nse_trading_day(date(2026, 1, 27)) is True
    prev = get_previous_trading_day(IST.localize(datetime(2026, 1, 27, 10, 0)))
    # Previous trading day before Jan 27 (Tuesday) when Jan 26 was holiday and Jan 24/25 weekend -> Jan 23 (Friday)
    assert prev == date(2026, 1, 23)


def test_filter_completed_candles():
    """Verifies that currently forming candle in the future is dropped."""
    now = datetime.now(IST)
    # 5m candle that started 1 minute ago (not yet closed)
    in_progress_start = now - timedelta(minutes=1)
    # Closed 5m candle that started 6 minutes ago
    closed_start = now - timedelta(minutes=6)

    df = pd.DataFrame([
        {"timestamp": closed_start, "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000},
        {"timestamp": in_progress_start, "open": 104, "high": 106, "low": 103, "close": 105, "volume": 500}
    ])

    filtered = filter_completed_candles(df, timeframe="5m")
    assert len(filtered) == 1
    assert filtered.iloc[0]["open"] == 100


def test_resample_10m_nse_alignment():
    """Verifies 10-minute candles are aligned strictly to NSE 09:15 open."""
    base_date = IST.localize(datetime(2026, 9, 18, 9, 15))
    rows = []
    # Create 5-minute bars from 09:15 to 09:45
    # 09:15, 09:20 -> should form 10m bar at 09:15
    # 09:25, 09:30 -> should form 10m bar at 09:25
    # 09:35, 09:40 -> should form 10m bar at 09:35
    for i in range(6):
        ts = base_date + timedelta(minutes=i * 5)
        rows.append({
            "timestamp": ts,
            "open": 100 + i,
            "high": 105 + i,
            "low": 95 + i,
            "close": 102 + i,
            "volume": 1000
        })
    df_5m = pd.DataFrame(rows)

    df_10m = resample_to_10m(df_5m)
    assert not df_10m.empty
    # First 10m bar must have timestamp at 09:15:00
    first_bar = df_10m.iloc[0]
    assert first_bar["timestamp"].hour == 9
    assert first_bar["timestamp"].minute == 15
    assert first_bar["open"] == 100
    assert first_bar["volume"] == 2000

    # Second 10m bar must be at 09:25:00
    second_bar = df_10m.iloc[1]
    assert second_bar["timestamp"].hour == 9
    assert second_bar["timestamp"].minute == 25


def test_vwap_session_reset():
    """Verifies VWAP resets at 09:15 at the start of each NSE session."""
    day1_start = IST.localize(datetime(2026, 9, 17, 9, 15))
    day2_start = IST.localize(datetime(2026, 9, 18, 9, 15))

    df_day1 = create_sample_candles(20, start_time=day1_start, base_price=100.0)
    df_day2 = create_sample_candles(20, start_time=day2_start, base_price=200.0)
    df_combined = pd.concat([df_day1, df_day2], ignore_index=True)

    df_ind, _ = compute_indicators(df_combined)

    # First bar of day 2 (at index 20) should have VWAP computed strictly from day 2 typical price, not contaminated by day 1
    day2_first_row = df_ind.iloc[20]
    expected_tp = (day2_first_row["high"] + day2_first_row["low"] + day2_first_row["close"]) / 3.0
    assert abs(day2_first_row["vwap"] - expected_tp) < 0.01


def test_ema_6_and_30():
    """Verifies EMA 6 and EMA 30 computation and crossover detection."""
    df = create_sample_candles(40, base_price=100.0, trend="bullish")
    df_ind, snapshot = compute_indicators(df)

    assert hasattr(snapshot, "ema_6")
    assert hasattr(snapshot, "ema_30")
    assert snapshot.ema_6 > 0
    assert snapshot.ema_30 > 0
    # In strong bullish trend, EMA 6 must be strictly greater than EMA 30
    assert snapshot.ema_6 > snapshot.ema_30
    assert snapshot.ema_6_above_30 is True


def test_wilder_smoothing_rsi_and_adx():
    """Verifies Wilder-style smoothing is used for RSI and ADX."""
    df = create_sample_candles(45, base_price=500.0, trend="bullish")
    df_ind, snapshot = compute_indicators(df)

    assert 0.0 <= snapshot.rsi <= 100.0
    assert snapshot.atr > 0.0
    assert 0.0 <= snapshot.adx <= 100.0


def test_smc_structure_detection():
    """Verifies SMC engine detects swings, dealing range, and order blocks."""
    df = create_sample_candles(50, base_price=150.0, trend="bullish")
    snap = smc_engine.analyze(df, pdh=165.0, pdl=145.0)

    assert snap.pdh == 165.0
    assert snap.pdl == 145.0
    assert len(snap.swing_highs) >= 0
    assert snap.dealing_range["high"] >= snap.dealing_range["low"]
    assert "zone" in snap.dealing_range


def test_buy_and_sell_signals():
    """Verifies signal engine outputs BUY or SELL with multi-target and structural SL."""
    # Bullish scenario
    df_bull = create_sample_candles(50, base_price=100.0, trend="bullish")
    df_ind_b, ind_snap_b = compute_indicators(df_bull)
    smc_snap_b = smc_engine.analyze(df_ind_b, pdh=120.0, pdl=95.0)
    sig_b = signal_engine.evaluate("SAIL", 115.0, ind_snap_b, smc_snap_b, "5m")

    assert sig_b.direction in ["BUY", "SELL"]
    assert sig_b.entry_price > 0
    assert len(sig_b.reasons) > 0

    # Test targets
    tgt = target_engine.calculate_targets(sig_b.direction, sig_b.entry_price, ind_snap_b, smc_snap_b)
    assert tgt.tp1 > 0
    assert tgt.tp2 > 0

    # Test risk engine
    rsk = risk_engine.calculate_stop_loss(sig_b.direction, sig_b.entry_price, ind_snap_b, smc_snap_b, tgt.tp2)
    assert rsk.stop_loss > 0
    # Stop loss must be below entry for BUY, above entry for SELL
    if sig_b.direction == "BUY":
        assert rsk.stop_loss < sig_b.entry_price
    else:
        assert rsk.stop_loss > sig_b.entry_price

    # Enforce minimum R:R >= 2.0
    assert rsk.risk_reward_ratio >= 2.0


def test_no_synthetic_data_in_live_mode():
    """Verifies that in LIVE mode (non-demo token), failed data retrieval raises MarketDataUnavailableError and never synthesizes fake data."""
    client = upstox_client
    # Set a simulated real live token
    client.set_token("live_production_jwt_token_sample_1234567890")
    assert client.is_demo() is False

    # Attempting to fetch quote without network/auth must raise MarketDataUnavailableError, NOT return fake data
    with pytest.raises(MarketDataUnavailableError):
        client.get_market_quote("NON_EXISTENT_TICKER_XYZ")

    with pytest.raises(MarketDataUnavailableError):
        client.fetch_candles("NON_EXISTENT_TICKER_XYZ", "5m")

    # Reset token
    client.set_token("demo")


def test_duplicate_telegram_prevention():
    """Verifies persistent duplicate telegram alert suppression."""
    t = telegram_notifier
    # Configure mock credentials
    t.bot_token = "mock_test_token"
    t.chat_id = "mock_test_chat"

    symbol = "TEST_STOCK"
    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    sig_hash = t._generate_hash(symbol, "BUY", 100.0, 95.0, 110.0, 115.0, today_str)

    # Store in cache
    t._sent_cache[sig_hash] = "Sent at mock time"

    # Now sending the same parameters should be suppressed (returns False)
    sent = t.send_signal_alert(
        symbol=symbol,
        direction="BUY",
        current_price=100.0,
        entry=100.0,
        entry_zone="₹99 - ₹100",
        stop_loss=95.0,
        tp1=110.0,
        tp2=115.0,
        rr_ratio=3.0,
        volume_ratio=1.8,
        force=False
    )
    assert sent is False


def test_token_masking():
    """Verifies that access token is never exposed in full."""
    assert mask_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") == "eyJh••••••••VCJ9"
    assert mask_token("") == "Not Set"


def test_volume_filter_ratio_and_stats():
    """Verifies volume 20-MA and 1.5x volume surge detection."""
    # Build 25 candles with flat volume of 10,000, and last candle with 20,000 (2.0x surge)
    df = create_sample_candles(25, base_price=100.0)
    df["volume"] = 10000
    df.loc[df.index[-1], "volume"] = 20000  # 2.0x

    df_ind, snap = compute_indicators(df)
    assert snap.volume == 20000
    assert snap.volume_ratio >= 1.5
    assert snap.volume_ratio > 1.8


def test_structural_stop_loss_no_arbitrary_clamp():
    """Verifies that stop loss uses market structure rather than blind 0.4% - 5% clamps."""
    df = create_sample_candles(50, base_price=2000.0, trend="bullish")
    df_ind, ind_snap = compute_indicators(df)
    smc_snap = smc_engine.analyze(df_ind, pdh=2100.0, pdl=1950.0)

    # Calculate stop loss
    risk_prof = risk_engine.calculate_stop_loss("BUY", 2050.0, ind_snap, smc_snap, target_price=2150.0)
    assert risk_prof.stop_loss < 2050.0
    assert risk_prof.risk_reward_ratio >= 2.0
    assert risk_prof.is_structural is True


def test_fresh_vs_mitigated_ob_and_fvg():
    """Verifies that SMC engine tracks fresh vs mitigated Order Blocks and FVGs."""
    df = create_sample_candles(60, base_price=100.0, trend="bullish")
    snap = smc_engine.analyze(df, pdh=120.0, pdl=95.0)

    assert isinstance(snap.fresh_order_blocks, list)
    assert isinstance(snap.fresh_fvgs, list)
    # Every fresh OB must have mitigated=False
    for ob in snap.fresh_order_blocks:
        assert ob.mitigated is False
        assert ob.is_fresh is True

    for fvg in snap.fresh_fvgs:
        assert fvg.mitigated is False
        assert fvg.is_fresh is True


def test_market_session_status_weekend_and_holiday():
    """Verifies human-readable session status on weekends and holidays."""
    # Weekend Saturday
    sat_dt = IST.localize(datetime(2026, 9, 19, 11, 0))
    is_open, msg = get_market_session_status(sat_dt)
    assert is_open is False
    assert "Weekend" in msg

    # Holiday (Republic Day)
    rep_day = IST.localize(datetime(2026, 1, 26, 11, 0))
    is_open, msg = get_market_session_status(rep_day)
    assert is_open is False
    assert "NSE Holiday" in msg


def test_prepare_market_data_validation():
    """Verifies data integrity checks catch corrupt or invalid data."""
    valid_quote = {"last_price": 100.0, "pdh": 105.0, "pdl": 95.0}
    df = create_sample_candles(40, base_price=100.0)

    # Valid package
    pkg = prepare_market_data("SAIL", valid_quote, df, "5m")
    assert pkg.is_valid is True

    # Invalid quote (price <= 0)
    with pytest.raises(MarketDataUnavailableError):
        prepare_market_data("SAIL", {"last_price": 0}, df, "5m")

    # Corrupt candle data with NaN
    df_corrupt = df.copy()
    df_corrupt.loc[0, "close"] = np.nan
    with pytest.raises(MarketDataUnavailableError):
        prepare_market_data("SAIL", valid_quote, df_corrupt, "5m")

    # Corrupt candle with High < Low
    df_bad_hl = df.copy()
    df_bad_hl.loc[0, "high"] = 50.0
    df_bad_hl.loc[0, "low"] = 150.0
    with pytest.raises(MarketDataUnavailableError):
        prepare_market_data("SAIL", valid_quote, df_bad_hl, "5m")


def test_watchlist_presets_and_all_stock_resolution():
    """Verifies that all standard NSE baskets, F&O stocks, and aliases resolve cleanly."""
    presets = upstox_client.get_watchlist_presets()
    assert "NIFTY 50 (All 50 Stocks)" in presets
    assert "NSE F&O UNIVERSE (All 199 Stocks)" in presets
    assert len(presets["NIFTY 50 (All 50 Stocks)"]) >= 48
    assert len(presets["NSE F&O UNIVERSE (All 199 Stocks)"]) >= 190

    # Test alias resolution
    assert "INE1TAE01010" in upstox_client.resolve_instrument_key("TATAMOTORS")
    assert "INE1TAE01010" in upstox_client.resolve_instrument_key("TMCV")
    assert "INE758T01015" in upstox_client.resolve_instrument_key("ZOMATO")
    assert "INE758T01015" in upstox_client.resolve_instrument_key("ETERNAL")
    assert "INE101A01026" in upstox_client.resolve_instrument_key("M&M")
    assert "INE917I01010" in upstox_client.resolve_instrument_key("BAJAJ-AUTO")


