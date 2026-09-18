from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class IndicatorSnapshot:
    ema_9: float
    ema_21: float
    ema_50: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    vwap: float
    atr: float
    adx: float
    plus_di: float
    minus_di: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    volume: int
    volume_ma: float
    volume_ratio: float
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
    """Computes all technical indicators and price action features on closed candles."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    volume = df["volume"]

    # 1. EMAs (9, 21, 50)
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # 2. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    # 3. MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 4. VWAP (Cumulative typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3.0
    cum_vol_price = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, 1)
    df["vwap"] = cum_vol_price / cum_vol

    # 5. ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14, min_periods=1).mean()

    # 6. ADX (14) with +DI, -DI
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), -minus_dm, 0.0)
    tr_smooth = tr.rolling(window=14, min_periods=1).sum().replace(0, 1e-9)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=14, min_periods=1).sum() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=14, min_periods=1).sum() / tr_smooth)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = dx.rolling(window=14, min_periods=1).mean().fillna(25.0)

    # 7. Bollinger Bands (20, 2 std)
    bb_middle = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std().fillna(0)
    df["bb_middle"] = bb_middle
    df["bb_upper"] = bb_middle + (2.0 * bb_std)
    df["bb_lower"] = bb_middle - (2.0 * bb_std)

    # 8. Volume Moving Average (20) and Ratio
    df["volume_ma"] = volume.rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = volume / df["volume_ma"].replace(0, 1)

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

    # Engulfing detection
    prev_open = open_p.shift(1)
    prev_close = close.shift(1)
    is_engulfing_bull = (
        (close > open_p) &
        (prev_close < prev_open) &
        (close >= prev_open) &
        (open_p <= prev_close)
    )
    is_engulfing_bear = (
        (close < open_p) &
        (prev_close > prev_open) &
        (close <= prev_open) &
        (open_p >= prev_close)
    )
    df["is_engulfing_bullish"] = is_engulfing_bull.fillna(False)
    df["is_engulfing_bearish"] = is_engulfing_bear.fillna(False)

    # Breakout / Breakdown of last 20 candles
    rolling_20_high = high.shift(1).rolling(window=20, min_periods=5).max()
    rolling_20_low = low.shift(1).rolling(window=20, min_periods=5).min()
    df["is_breakout_20"] = (close > rolling_20_high).fillna(False)
    df["is_breakdown_20"] = (close < rolling_20_low).fillna(False)

    avg_body = body.rolling(window=14, min_periods=1).mean()
    df["is_momentum_candle"] = (body > (1.4 * avg_body)).fillna(False)

    last_row = df.iloc[-1]

    snapshot = IndicatorSnapshot(
        ema_9=round(float(last_row["ema_9"]), 2),
        ema_21=round(float(last_row["ema_21"]), 2),
        ema_50=round(float(last_row["ema_50"]), 2),
        rsi=round(float(last_row["rsi"]), 2),
        macd=round(float(last_row["macd"]), 2),
        macd_signal=round(float(last_row["macd_signal"]), 2),
        macd_hist=round(float(last_row["macd_hist"]), 2),
        vwap=round(float(last_row["vwap"]), 2),
        atr=round(float(last_row["atr"]), 2),
        adx=round(float(last_row["adx"]), 2),
        plus_di=round(float(last_row["plus_di"]), 2),
        minus_di=round(float(last_row["minus_di"]), 2),
        bb_upper=round(float(last_row["bb_upper"]), 2),
        bb_middle=round(float(last_row["bb_middle"]), 2),
        bb_lower=round(float(last_row["bb_lower"]), 2),
        volume=int(last_row["volume"]),
        volume_ma=round(float(last_row["volume_ma"]), 2),
        volume_ratio=round(float(last_row["volume_ratio"]), 2),
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
