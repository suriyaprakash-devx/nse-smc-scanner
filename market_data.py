from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta, time as dtime
import pandas as pd
import numpy as np

from utils import get_ist_now, get_ist_timezone, MarketDataUnavailableError

@dataclass
class MarketDataPackage:
    symbol: str
    timeframe: str
    quote: dict
    candles: pd.DataFrame
    last_closed_candle: pd.Series
    is_valid: bool = True
    error_message: Optional[str] = None

def get_timeframe_minutes(timeframe: str) -> int:
    """Returns the duration of a timeframe string in minutes."""
    tf = timeframe.lower().strip()
    if tf == "1m":
        return 1
    elif tf == "3m":
        return 3
    elif tf == "5m":
        return 5
    elif tf == "10m":
        return 10
    elif tf == "15m":
        return 15
    elif tf == "30m":
        return 30
    elif tf == "1h":
        return 60
    return 5

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
    
    tf_minutes = get_timeframe_minutes(timeframe)
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

def resample_to_10m(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs 10-minute candles strictly aligned to NSE market open (09:15 IST).
    Bins are:
      09:15–09:24 (bin start: 09:15)
      09:25–09:34 (bin start: 09:25)
      09:35–09:44 (bin start: 09:35)
      ...
    Never creates 09:10-09:19 or 09:20-09:29 candles.
    Only returns completed 10-minute candles.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df_work = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"])

    ist_tz = get_ist_timezone()
    if df_work["timestamp"].dt.tz is not None:
        df_work["timestamp"] = df_work["timestamp"].dt.tz_convert(ist_tz)

    df_work = df_work.sort_values("timestamp").reset_index(drop=True)

    # Group by date and 10-minute bin starting at 09:15
    records = []
    
    # Process day by day to respect session boundaries
    for date_val, group in df_work.groupby(df_work["timestamp"].dt.date):
        group = group.sort_values("timestamp")
        
        # Dictionary of bin_start_dt -> list of rows
        bins: dict = {}
        for _, row in group.iterrows():
            ts = row["timestamp"]
            h = ts.hour
            m = ts.minute
            
            # Minutes relative to 09:15
            market_open_minutes = 9 * 60 + 15
            bar_minutes = h * 60 + m
            
            if bar_minutes < market_open_minutes:
                # Pre-market candle (if present), bin into pre-market
                bin_start = ts.replace(minute=(m // 10) * 10, second=0, microsecond=0)
            else:
                elapsed = bar_minutes - market_open_minutes
                bin_offset = (elapsed // 10) * 10
                bin_h = (market_open_minutes + bin_offset) // 60
                bin_m = (market_open_minutes + bin_offset) % 60
                bin_start = ts.replace(hour=bin_h, minute=bin_m, second=0, microsecond=0)
                
            if bin_start not in bins:
                bins[bin_start] = []
            bins[bin_start].append(row)
            
        for bin_dt, rows in bins.items():
            if not rows:
                continue
            sub_df = pd.DataFrame(rows)
            # Require at least 2 bars of 5m (or 10 bars of 1m) for a solid candle,
            # or if sub_df covers the interval span
            rec = {
                "timestamp": bin_dt,
                "open": float(sub_df["open"].iloc[0]),
                "high": float(sub_df["high"].max()),
                "low": float(sub_df["low"].min()),
                "close": float(sub_df["close"].iloc[-1]),
                "volume": int(sub_df["volume"].sum())
            }
            records.append(rec)

    resampled_df = pd.DataFrame(records)
    if resampled_df.empty:
        return resampled_df

    resampled_df = resampled_df.sort_values("timestamp").reset_index(drop=True)
    # Ensure completed candles only for 10m
    return filter_completed_candles(resampled_df, timeframe="10m")

def prepare_market_data(
    symbol: str,
    quote: dict,
    raw_candles: pd.DataFrame,
    timeframe: str = "5m",
    min_candles: int = 30
) -> MarketDataPackage:
    """Strictly validates market data quality.
    
    Checks:
    - Live Upstox quote validity (price > 0).
    - Correct symbol.
    - Candle data existence & column format.
    - Session-aligned 10m resampling when requested.
    - Dropping of in-progress candle.
    - OHLC integrity: High >= Open, High >= Close, Low <= Open, Low <= Close, High >= Low, Volume >= 0.
    - No duplicate or future timestamps.
    - Sufficient candle count (>= min_candles).
    
    Raises MarketDataUnavailableError if data is corrupt, incomplete, or invalid.
    """
    clean_sym = symbol.strip().upper()

    # 1. Quote validation
    if not quote or "last_price" not in quote or float(quote.get("last_price", 0)) <= 0:
        raise MarketDataUnavailableError(f"Market quote unavailable for {clean_sym}. Please verify API connection.")

    # 2. Candle DataFrame validation
    if raw_candles is None or raw_candles.empty:
        raise MarketDataUnavailableError(f"Candle data unavailable for {clean_sym} on {timeframe} timeframe.")

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in raw_candles.columns:
            raise MarketDataUnavailableError(f"Corrupt candle data: Missing '{col}' column for {clean_sym}.")

    candles = raw_candles.copy()
    if not pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles["timestamp"] = pd.to_datetime(candles["timestamp"])

    # 3. Handle 10m timeframe alignment
    if timeframe == "10m":
        # Check if incoming candles are 5m or 1m and need 10m session alignment
        closed_candles = resample_to_10m(candles)
    else:
        # Filter strictly closed candles
        closed_candles = filter_completed_candles(candles, timeframe)

    if closed_candles.empty or len(closed_candles) < min_candles:
        raise MarketDataUnavailableError(
            f"Insufficient completed candle data for {clean_sym} ({len(closed_candles)}/{min_candles} bars). "
            "Need more historical bars for EMA 30 & SMC analysis."
        )

    # 4. Check for invalid or corrupt prices (NaN, <= 0)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        closed_candles[col] = pd.to_numeric(closed_candles[col], errors="coerce")
        if closed_candles[col].isna().any():
            raise MarketDataUnavailableError(f"Corrupt market data: NaN detected in '{col}' for {clean_sym}.")
        if col != "volume" and (closed_candles[col] <= 0).any():
            raise MarketDataUnavailableError(f"Invalid market data: Zero or negative price in '{col}' for {clean_sym}.")
        if col == "volume" and (closed_candles[col] < 0).any():
            raise MarketDataUnavailableError(f"Invalid market data: Negative volume detected for {clean_sym}.")

    # 5. OHLC logical sanity checks
    invalid_high = (closed_candles["high"] < closed_candles["open"]) | (closed_candles["high"] < closed_candles["close"]) | (closed_candles["high"] < closed_candles["low"])
    if invalid_high.any():
        raise MarketDataUnavailableError(f"Data anomaly: High price is less than Open/Close/Low for {clean_sym}.")

    invalid_low = (closed_candles["low"] > closed_candles["open"]) | (closed_candles["low"] > closed_candles["close"])
    if invalid_low.any():
        raise MarketDataUnavailableError(f"Data anomaly: Low price is greater than Open/Close for {clean_sym}.")

    # 6. Check for duplicate timestamps
    if closed_candles["timestamp"].duplicated().any():
        closed_candles = closed_candles.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    # 7. Check for future timestamps
    now_ist = get_ist_now()
    first_ts = closed_candles["timestamp"].iloc[0]
    now_cmp = now_ist if first_ts.tzinfo is not None else now_ist.replace(tzinfo=None)
    if (closed_candles["timestamp"] > (now_cmp + timedelta(minutes=2))).any():
        raise MarketDataUnavailableError(f"Data anomaly: Future timestamps detected in candles for {clean_sym}.")

    last_closed = closed_candles.iloc[-1]

    return MarketDataPackage(
        symbol=clean_sym,
        timeframe=timeframe,
        quote=quote,
        candles=closed_candles,
        last_closed_candle=last_closed,
        is_valid=True
    )
