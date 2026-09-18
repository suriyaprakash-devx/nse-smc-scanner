from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils import get_ist_now

class MarketDataUnavailableError(Exception):
    """Raised when market data is missing, corrupted, stale, or insufficient."""
    pass

@dataclass
class MarketDataPackage:
    symbol: str
    timeframe: str
    quote: dict
    candles: pd.DataFrame
    last_closed_candle: pd.Series
    is_valid: bool = True
    error_message: Optional[str] = None

def filter_completed_candles(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    """Ensures strictly closed candles are evaluated to eliminate look-ahead bias.
    Drops the currently forming candle if its close time is in the future.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])

    df_work = df_work.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate timeframe duration in minutes
    tf_minutes = 5
    if timeframe == "1m":
        tf_minutes = 1
    elif timeframe == "3m":
        tf_minutes = 3
    elif timeframe == "15m":
        tf_minutes = 15
    elif timeframe == "30m":
        tf_minutes = 30
    elif timeframe == "1h":
        tf_minutes = 60

    last_bar_time = df_work["timestamp"].iloc[-1]
    candle_duration = timedelta(minutes=tf_minutes)
    candle_close_time = last_bar_time + candle_duration
    
    now_ist = get_ist_now()
    if last_bar_time.tzinfo is not None:
        now_cmp = now_ist
    else:
        now_cmp = now_ist.replace(tzinfo=None)

    # If the last candle has not yet finished closing, drop it
    if now_cmp < candle_close_time:
        df_work = df_work.iloc[:-1].reset_index(drop=True)

    return df_work

def prepare_market_data(
    symbol: str,
    quote: dict,
    raw_candles: pd.DataFrame,
    timeframe: str = "5m",
    min_candles: int = 30
) -> MarketDataPackage:
    """Strictly validates market data quality.
    Raises MarketDataUnavailableError if data is corrupt, incomplete, or invalid.
    """
    clean_sym = symbol.strip().upper()
    
    # 1. Quote validation
    if not quote or "last_price" not in quote or float(quote.get("last_price", 0)) <= 0:
        raise MarketDataUnavailableError("Market data unavailable — please try again.")

    # 2. Candle DataFrame validation
    if raw_candles is None or raw_candles.empty:
        raise MarketDataUnavailableError("Market data unavailable — please try again.")

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in raw_candles.columns:
            raise MarketDataUnavailableError("Market data unavailable — please try again.")

    # 3. Filter strictly closed candles
    closed_candles = filter_completed_candles(raw_candles, timeframe)
    
    if len(closed_candles) < min_candles:
        # If dropping current candle left fewer than min_candles, check if raw has enough
        if len(raw_candles) >= min_candles:
            closed_candles = raw_candles.iloc[:-1] if len(raw_candles) > 1 else raw_candles
        else:
            raise MarketDataUnavailableError("Market data unavailable — please try again.")

    # 4. Check for invalid or corrupt prices (NaN, <= 0)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        closed_candles[col] = pd.to_numeric(closed_candles[col], errors="coerce")
        if closed_candles[col].isna().any() or (col != "volume" and (closed_candles[col] <= 0).any()):
            raise MarketDataUnavailableError("Market data unavailable — please try again.")

    last_closed = closed_candles.iloc[-1]

    return MarketDataPackage(
        symbol=clean_sym,
        timeframe=timeframe,
        quote=quote,
        candles=closed_candles,
        last_closed_candle=last_closed,
        is_valid=True
    )
