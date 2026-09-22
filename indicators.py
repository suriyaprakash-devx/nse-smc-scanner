from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class IndicatorSnapshot:
    ema_6: float
    ema_30: float
    ema_6_above_30: bool
    ema_cross_bullish: bool
    ema_cross_bearish: bool
    vwap: float
    rsi: float
    atr: float
    adx: float
    plus_di: float
    minus_di: float
    volume: int
    volume_ma: float
    volume_ratio: float
    # Secondary indicators for display
    ema_9: float
    ema_21: float
    ema_50: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    momentum: float
    # Price action metrics
    is_bullish_candle: bool
    body_pct: float
    upper_wick_pct: float
    lower_wick_pct: float
    is_engulfing_bullish: bool
    is_engulfing_bearish: bool
    is_momentum_candle: bool
    is_breakout_20: bool
    is_breakdown_20: bool
    raw_df: pd.DataFrame

def compute_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, IndicatorSnapshot]:
    """Computes technical indicators and price action features on strictly closed candles.
    
    Core enhancements:
    - EMA 6 and EMA 30 for agile intraday trend/momentum.
    - True Wilder's smoothing for RSI (14) and ADX/ATR (14).
    - Session-reset VWAP resetting strictly at 09:15 each NSE trading day.
    - Volume 20-MA and volume ratio for mandatory filtering.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_p = df["open"].astype(float)
    volume = df["volume"].astype(float)

    # 1. Core EMAs: EMA 6 & EMA 30
    df["ema_6"] = close.ewm(span=6, adjust=False).mean()
    df["ema_30"] = close.ewm(span=30, adjust=False).mean()
    df["ema_6_above_30"] = df["ema_6"] > df["ema_30"]
    
    # Crossover detection on the latest completed candle
    prev_ema_6 = df["ema_6"].shift(1)
    prev_ema_30 = df["ema_30"].shift(1)
    df["ema_cross_bullish"] = (df["ema_6"] > df["ema_30"]) & (prev_ema_6 <= prev_ema_30)
    df["ema_cross_bearish"] = (df["ema_6"] < df["ema_30"]) & (prev_ema_6 >= prev_ema_30)

    # Additional standard EMAs for multi-timeframe reference
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # 2. VWAP: Strictly resets at 09:15 at the start of every NSE trading session
    # Intraday typical price
    typical_price = (high + low + close) / 3.0
    vol_price = typical_price * volume

    # Extract date for session grouping
    dates = df["timestamp"].dt.date
    df["cum_vol_price"] = vol_price.groupby(dates).cumsum()
    df["cum_vol"] = volume.groupby(dates).cumsum().replace(0, 1.0)
    df["vwap"] = df["cum_vol_price"] / df["cum_vol"]

    # 3. Wilder's Smoothing for RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    
    # Alpha = 1/14 corresponds to Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # 4. Wilder's Smoothing for ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean()

    # 5. Wilder's Smoothing for ADX (14) with +DI, -DI
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    smooth_tr = tr.ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean().replace(0, 1e-9)
    smooth_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean()
    smooth_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean()
    
    df["plus_di"] = 100.0 * (smooth_plus_dm / smooth_tr)
    df["minus_di"] = 100.0 * (smooth_minus_dm / smooth_tr)
    di_sum = (df["plus_di"] + df["minus_di"]).replace(0, 1e-9)
    dx = 100.0 * ((df["plus_di"] - df["minus_di"]).abs() / di_sum)
    df["adx"] = dx.ewm(alpha=1.0 / 14, min_periods=1, adjust=False).mean().fillna(25.0)

    # 6. Volume Moving Average (20) and Ratio
    df["volume_ma"] = volume.rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = volume / df["volume_ma"].replace(0, 1.0)

    # 7. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 8. Bollinger Bands (20, 2 std)
    bb_middle = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std().fillna(0)
    df["bb_middle"] = bb_middle
    df["bb_upper"] = bb_middle + (2.0 * bb_std)
    df["bb_lower"] = bb_middle - (2.0 * bb_std)

    # 9. Momentum (10 bars)
    df["momentum"] = close.diff(periods=min(10, len(df) - 1)).fillna(0.0)

    # 10. Price Action features
    candle_range = (high - low).replace(0, 1e-9)
    body = (close - open_p).abs()
    upper_wick = high - pd.concat([close, open_p], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_p], axis=1).min(axis=1) - low

    df["body_pct"] = (body / candle_range) * 100.0
    df["upper_wick_pct"] = (upper_wick / candle_range) * 100.0
    df["lower_wick_pct"] = (lower_wick / candle_range) * 100.0
    df["is_bullish_candle"] = close > open_p

    # Engulfing patterns
    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    df["is_engulfing_bullish"] = (
        (close > open_p) &
        (prev_close < prev_open) &
        (close >= prev_open) &
        (open_p <= prev_close)
    ).fillna(False)
    
    df["is_engulfing_bearish"] = (
        (close < open_p) &
        (prev_close > prev_open) &
        (close <= prev_open) &
        (open_p >= prev_close)
    ).fillna(False)

    # Breakout / Breakdown of last 20 candles
    rolling_20_high = high.shift(1).rolling(window=20, min_periods=5).max()
    rolling_20_low = low.shift(1).rolling(window=20, min_periods=5).min()
    df["is_breakout_20"] = (close > rolling_20_high).fillna(False)
    df["is_breakdown_20"] = (close < rolling_20_low).fillna(False)

    avg_body = body.rolling(window=14, min_periods=1).mean()
    df["is_momentum_candle"] = (body > (1.4 * avg_body)).fillna(False)

    last_row = df.iloc[-1]

    snapshot = IndicatorSnapshot(
        ema_6=round(float(last_row["ema_6"]), 2),
        ema_30=round(float(last_row["ema_30"]), 2),
        ema_6_above_30=bool(last_row["ema_6_above_30"]),
        ema_cross_bullish=bool(last_row["ema_cross_bullish"]),
        ema_cross_bearish=bool(last_row["ema_cross_bearish"]),
        vwap=round(float(last_row["vwap"]), 2),
        rsi=round(float(last_row["rsi"]), 2),
        atr=round(float(last_row["atr"]), 2),
        adx=round(float(last_row["adx"]), 2),
        plus_di=round(float(last_row["plus_di"]), 2),
        minus_di=round(float(last_row["minus_di"]), 2),
        volume=int(last_row["volume"]),
        volume_ma=round(float(last_row["volume_ma"]), 2),
        volume_ratio=round(float(last_row["volume_ratio"]), 2),
        ema_9=round(float(last_row["ema_9"]), 2),
        ema_21=round(float(last_row["ema_21"]), 2),
        ema_50=round(float(last_row["ema_50"]), 2),
        macd=round(float(last_row["macd"]), 2),
        macd_signal=round(float(last_row["macd_signal"]), 2),
        macd_hist=round(float(last_row["macd_hist"]), 2),
        bb_upper=round(float(last_row["bb_upper"]), 2),
        bb_middle=round(float(last_row["bb_middle"]), 2),
        bb_lower=round(float(last_row["bb_lower"]), 2),
        momentum=round(float(last_row["momentum"]), 2),
        is_bullish_candle=bool(last_row["is_bullish_candle"]),
        body_pct=round(float(last_row["body_pct"]), 2),
        upper_wick_pct=round(float(last_row["upper_wick_pct"]), 2),
        lower_wick_pct=round(float(last_row["lower_wick_pct"]), 2),
        is_engulfing_bullish=bool(last_row["is_engulfing_bullish"]),
        is_engulfing_bearish=bool(last_row["is_engulfing_bearish"]),
        is_momentum_candle=bool(last_row["is_momentum_candle"]),
        is_breakout_20=bool(last_row["is_breakout_20"]),
        is_breakdown_20=bool(last_row["is_breakdown_20"]),
        raw_df=df
    )

    return df, snapshot
