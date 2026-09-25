import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from smc_ml import run_smc_ml, analyze_smc_stock, Settings
from calendar_utils import is_nse_holiday, is_nse_trading_day, get_previous_trading_day
from utils import mask_token, format_currency, format_pct
from market_data import prepare_market_data, filter_completed_candles
from upstox_client import upstox_client

IST = pytz.timezone("Asia/Kolkata")

def create_sample_candles(n_bars: int = 100, base_price: float = 100.0) -> pd.DataFrame:
    start_time = IST.localize(datetime(2026, 9, 25, 9, 15))
    rows = []
    price = base_price
    for i in range(n_bars):
        ts = start_time + timedelta(minutes=i * 5)
        delta = 0.3 * np.sin(i / 5.0) + (0.1 if i % 2 == 0 else -0.1)
        c_open = price
        c_close = price + delta
        c_high = max(c_open, c_close) + 0.4
        c_low = min(c_open, c_close) - 0.4
        volume = 50000 + int(i * 1000)
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


def test_smc_ml_execution():
    df = create_sample_candles(n_bars=80)
    res = run_smc_ml(df)
    assert "bars" in res
    assert "breaks" in res
    assert "signals" in res
    assert "trades" in res
    assert "zones" in res
    assert "pools" in res
    assert len(res["bars"]) == 80
    print("test_smc_ml_execution passed!")


def test_analyze_smc_stock_buy_or_sell_only():
    df = create_sample_candles(n_bars=90)
    snap = analyze_smc_stock("RELIANCE", df)
    assert snap.direction in ["BUY", "SELL"]
    assert snap.entry > 0
    assert snap.stop > 0
    assert snap.tp1 > 0
    assert snap.tp2 > 0
    assert snap.risk_reward > 0
    print(f"test_analyze_smc_stock_buy_or_sell_only passed! Direction: {snap.direction}, Entry: {snap.entry}, SL: {snap.stop}, TP1: {snap.tp1}, TP2: {snap.tp2}")


def test_nifty_200_basket_available():
    presets = upstox_client.get_watchlist_presets()
    assert "NIFTY 200 (Top 200 Stocks)" in presets
    nifty_200 = presets["NIFTY 200 (Top 200 Stocks)"]
    assert len(nifty_200) == 200
    print("test_nifty_200_basket_available passed! Count: 200")


def test_whole_nse_universe_searchable():
    symbols = upstox_client.get_all_symbols()
    assert len(symbols) >= 3000
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "HDFCBANK" in symbols
    print(f"test_whole_nse_universe_searchable passed! Total NSE stocks: {len(symbols)}")


if __name__ == "__main__":
    test_smc_ml_execution()
    test_analyze_smc_stock_buy_or_sell_only()
    test_nifty_200_basket_available()
    test_whole_nse_universe_searchable()
    print("\nALL SMC ML TESTS PASSED SUCCESSFULLY!")
