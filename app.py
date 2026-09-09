"""
NSE SWING SMART MONEY CONCEPTS (SMC) SCANNER
============================================

TIMEFRAMES
----------
HTF       : 1D
ENTRY     : 60M
HOLD      : Configurable, default 5 calendar days

SMC
---
- Daily market bias
- Daily stock structure
- Liquidity sweep
- BOS
- CHOCH
- Displacement
- Fair Value Gap
- Order Block
- Volume confirmation
- Confluence scoring

RISK
----
- Structural SL
- Maximum SL %
- TP1 = 1.5R
- TP2 = configurable RR
- Signal expiry
- Persistent SQLite database

BROKER
------
Upstox historical candle API

IMPORTANT
---------
This is an analysis/alert scanner.
It DOES NOT place orders automatically.
"""

import os
import time
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "NSE SWING SMC SCANNER"

API_BASE = "https://api.upstox.com"
NSE_INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)

IST = timezone(timedelta(hours=5, minutes=30))

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

DEFAULT_DB = "swing_scanner.db"

# Do not assume broker rate limits are unlimited.
# Keep this conservative.
RATE_LIMIT_CALLS_PER_SECOND = 8

SESSION = threading.local()


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


def market_datetime(d, hour, minute):
    return datetime(
        d.year,
        d.month,
        d.day,
        hour,
        minute,
        tzinfo=IST,
    )


def market_is_open():
    now = now_ist()

    if now.weekday() >= 5:
        return False

    return market_datetime(
        now.date(),
        MARKET_OPEN_HOUR,
        MARKET_OPEN_MINUTE,
    ) <= now <= market_datetime(
        now.date(),
        MARKET_CLOSE_HOUR,
        MARKET_CLOSE_MINUTE,
    )


def safe_float(value, default=0.0):
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def pct_change(a, b):
    if b == 0:
        return 0
    return ((a - b) / b) * 100


# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    def __init__(self, rate=8):
        self.rate = rate
        self.lock = threading.Lock()
        self.calls = []

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()

                self.calls = [
                    x for x in self.calls
                    if now - x < 1
                ]

                if len(self.calls) < self.rate:
                    self.calls.append(now)
                    return

            time.sleep(0.02)


RATE_LIMITER = RateLimiter(RATE_LIMIT_CALLS_PER_SECOND)


# ============================================================
# HTTP SESSION
# ============================================================

def get_session():

    if not hasattr(SESSION, "session"):

        SESSION.session = requests.Session()

        SESSION.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "NSE-Swing-SMC-Scanner/1.0",
        })

    return SESSION.session


# ============================================================
# UPSTOX CLIENT
# ============================================================

class UpstoxClient:

    def __init__(self, access_token):

        self.access_token = access_token.strip()

        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def request(self, method, url, **kwargs):

        RATE_LIMITER.wait()

        headers = kwargs.pop("headers", {})
        merged_headers = dict(self.headers)
        merged_headers.update(headers)

        response = get_session().request(
            method,
            url,
            headers=merged_headers,
            timeout=20,
            **kwargs,
        )

        if response.status_code == 429:
            time.sleep(1)
            response = get_session().request(
                method,
                url,
                headers=merged_headers,
                timeout=20,
                **kwargs,
            )

        response.raise_for_status()

        return response

    def validate_connection(self):

        url = f"{API_BASE}/v2/user/profile"

        response = self.request("GET", url)

        return response.json()

    def get_candles(
        self,
        instrument_key,
        interval,
        to_date,
        from_date,
    ):

        encoded_key = quote(
            instrument_key,
            safe=""
        )

        url = (
            f"{API_BASE}/v3/historical-candle/"
            f"{encoded_key}/"
            f"{interval}/"
            f"{to_date}/"
            f"{from_date}"
        )

        response = self.request("GET", url)

        payload = response.json()

        data = payload.get("data", {})

        candles = data.get("candles", [])

        if not candles:
            return pd.DataFrame()

        rows = []

        for c in candles:

            if len(c) < 6:
                continue

            rows.append({
                "timestamp": c[0],
                "open": safe_float(c[1]),
                "high": safe_float(c[2]),
                "low": safe_float(c[3]),
                "close": safe_float(c[4]),
                "volume": safe_float(c[5]),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        ).dt.tz_convert(IST)

        df = df.sort_values("timestamp")
        df = df.drop_duplicates("timestamp")
        df = df.reset_index(drop=True)

        return df


# ============================================================
# NSE INSTRUMENTS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_nse_instruments():

    response = requests.get(
        NSE_INSTRUMENT_URL,
        timeout=30,
    )

    response.raise_for_status()

    import gzip
    import json

    raw = gzip.decompress(response.content)

    data = json.loads(raw.decode("utf-8"))

    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_nse_equities():

    data = download_nse_instruments()

    rows = []

    for item in data:

        if (
            item.get("segment") == "NSE_EQ"
            and item.get("instrument_type") == "EQ"
            and item.get("exchange") == "NSE"
        ):

            symbol = item.get("trading_symbol")
            key = item.get("instrument_key")

            if symbol and key:

                rows.append({
                    "symbol": symbol,
                    "instrument_key": key,
                    "name": item.get("name", symbol),
                })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.drop_duplicates("symbol")
    df = df.sort_values("symbol")

    return df.reset_index(drop=True)


# ============================================================
# COMPLETED CANDLES
# ============================================================

def filter_completed_intraday(df, minutes):

    if df.empty:
        return df

    current = now_ist()

    result = df.copy()

    result["end_time"] = (
        result["timestamp"]
        + pd.Timedelta(minutes=minutes)
    )

    result = result[
        result["end_time"] <= current
    ]

    return result.drop(
        columns=["end_time"]
    ).reset_index(drop=True)


def filter_completed_daily(df):

    if df.empty:
        return df

    current = now_ist()

    result = df.copy()

    # During market hours the current daily candle is incomplete.
    if current.time() < datetime(
        1900,
        1,
        1,
        15,
        30,
    ).time():

        result = result[
            result["timestamp"].dt.date < current.date()
        ]

    return result.reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False,
    ).mean()


def add_emas(df):

    df = df.copy()

    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)

    return df


# ============================================================
# SWING DETECTION
# ============================================================

def find_swings(df, length=3):

    if len(df) < (length * 2 + 5):
        return [], []

    highs = []
    lows = []

    high_values = df["high"].values
    low_values = df["low"].values

    for i in range(
        length,
        len(df) - length
    ):

        left_highs = high_values[
            i - length:i
        ]

        right_highs = high_values[
            i + 1:i + length + 1
        ]

        left_lows = low_values[
            i - length:i
        ]

        right_lows = low_values[
            i + 1:i + length + 1
        ]

        if (
            high_values[i] >= left_highs.max()
            and
            high_values[i] > right_highs.max()
        ):
            highs.append({
                "index": i,
                "price": high_values[i],
                "timestamp": df.iloc[i]["timestamp"],
            })

        if (
            low_values[i] <= left_lows.min()
            and
            low_values[i] < right_lows.min()
        ):
            lows.append({
                "index": i,
                "price": low_values[i],
                "timestamp": df.iloc[i]["timestamp"],
            })

    return highs, lows


# ============================================================
# STRUCTURE
# ============================================================

def analyze_structure(
    df,
    swing_length=3,
):

    highs, lows = find_swings(
        df,
        swing_length,
    )

    if len(highs) < 2 or len(lows) < 2:

        return {
            "trend": "NEUTRAL",
            "bos": False,
            "choch": False,
            "last_swing_high": None,
            "last_swing_low": None,
        }

    last_high = highs[-1]
    previous_high = highs[-2]

    last_low = lows[-1]
    previous_low = lows[-2]

    current_close = df.iloc[-1]["close"]

    higher_high = (
        last_high["price"]
        > previous_high["price"]
    )

    higher_low = (
        last_low["price"]
        > previous_low["price"]
    )

    lower_high = (
        last_high["price"]
        < previous_high["price"]
    )

    lower_low = (
        last_low["price"]
        < previous_low["price"]
    )

    trend = "NEUTRAL"

    if higher_high and higher_low:
        trend = "BULLISH"

    elif lower_high and lower_low:
        trend = "BEARISH"

    bos = False
    choch = False

    if trend == "BULLISH":

        bos = (
            current_close
            > last_high["price"]
        )

    elif trend == "BEARISH":

        bos = (
            current_close
            < last_low["price"]
        )

    # Detect possible reversal structure.
    if lower_high and lower_low:

        choch = (
            current_close
            > previous_high["price"]
        )

    elif higher_high and higher_low:

        choch = (
            current_close
            < previous_low["price"]
        )

    return {
        "trend": trend,
        "bos": bos,
        "choch": choch,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    df,
    swing_length=3,
    lookback=15,
):

    highs, lows = find_swings(
        df,
        swing_length,
    )

    if not highs or not lows:
        return None

    latest = df.iloc[-1]

    recent_highs = [
        x for x in highs
        if x["index"] < len(df) - 1
        and x["index"] >= max(
            0,
            len(df) - 1 - lookback
        )
    ]

    recent_lows = [
        x for x in lows
        if x["index"] < len(df) - 1
        and x["index"] >= max(
            0,
            len(df) - 1 - lookback
        )
    ]

    bullish_sweep = False
    bearish_sweep = False

    sweep_low = None
    sweep_high = None

    if recent_lows:

        reference = recent_lows[-1]

        bullish_sweep = (
            latest["low"]
            < reference["price"]
            and
            latest["close"]
            > reference["price"]
        )

        if bullish_sweep:
            sweep_low = latest["low"]

    if recent_highs:

        reference = recent_highs[-1]

        bearish_sweep = (
            latest["high"]
            > reference["price"]
            and
            latest["close"]
            < reference["price"]
        )

        if bearish_sweep:
            sweep_high = latest["high"]

    if bullish_sweep:

        return {
            "direction": "BUY",
            "price": sweep_low,
            "timestamp": latest["timestamp"],
        }

    if bearish_sweep:

        return {
            "direction": "SELL",
            "price": sweep_high,
            "timestamp": latest["timestamp"],
        }

    return None


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(
    df,
    multiplier=1.5,
    lookback=5,
):

    if len(df) < lookback + 2:
        return False, None

    body = (
        df["close"] - df["open"]
    ).abs()

    current_body = body.iloc[-1]

    average_body = body.iloc[
        -lookback - 1:-1
    ].mean()

    if average_body <= 0:
        return False, None

    ratio = (
        current_body
        / average_body
    )

    bullish = (
        df.iloc[-1]["close"]
        > df.iloc[-1]["open"]
        and
        ratio >= multiplier
    )

    bearish = (
        df.iloc[-1]["close"]
        < df.iloc[-1]["open"]
        and
        ratio >= multiplier
    )

    if bullish:
        return True, "BUY"

    if bearish:
        return True, "SELL"

    return False, None


# ============================================================
# FVG
# ============================================================

def detect_fvg(
    df,
    max_age=30,
):

    if len(df) < 3:
        return None

    start = max(
        2,
        len(df) - max_age - 3
    )

    for i in range(
        len(df) - 1,
        start - 1,
        -1
    ):

        first = df.iloc[i - 2]
        middle = df.iloc[i - 1]
        third = df.iloc[i]

        # Bullish FVG
        if first["high"] < third["low"]:

            return {
                "direction": "BUY",
                "low": first["high"],
                "high": third["low"],
                "index": i,
                "timestamp": third["timestamp"],
            }

        # Bearish FVG
        if first["low"] > third["high"]:

            return {
                "direction": "SELL",
                "low": third["high"],
                "high": first["low"],
                "index": i,
                "timestamp": third["timestamp"],
            }

    return None


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_block(
    df,
    direction,
    lookback=20,
):

    if len(df) < 5:
        return None

    start = max(
        1,
        len(df) - lookback
    )

    for i in range(
        len(df) - 2,
        start - 1,
        -1
    ):

        candle = df.iloc[i]
        next_candle = df.iloc[i + 1]

        if direction == "BUY":

            # Last bearish candle before bullish expansion
            if (
                candle["close"]
                < candle["open"]
                and
                next_candle["close"]
                > candle["high"]
            ):

                return {
                    "direction": "BUY",
                    "low": candle["low"],
                    "high": candle["high"],
                    "timestamp": candle["timestamp"],
                    "index": i,
                }

        elif direction == "SELL":

            # Last bullish candle before bearish expansion
            if (
                candle["close"]
                > candle["open"]
                and
                next_candle["close"]
                < candle["low"]
            ):

                return {
                    "direction": "SELL",
                    "low": candle["low"],
                    "high": candle["high"],
                    "timestamp": candle["timestamp"],
                    "index": i,
                }

    return None


# ============================================================
# ZONE INTERACTION
# ============================================================

def price_inside_zone(
    price,
    zone,
):

    if not zone:
        return False

    return (
        zone["low"]
        <= price
        <= zone["high"]
    )


def zone_distance_pct(
    price,
    zone,
):

    if not zone:
        return 999

    if price_inside_zone(price, zone):
        return 0

    if price < zone["low"]:

        return (
            (zone["low"] - price)
            / price
            * 100
        )

    return (
        (price - zone["high"])
        / price
        * 100
    )


# ============================================================
# VOLUME
# ============================================================

def daily_volume_ratio(
    df,
    lookback=20,
):

    if len(df) < lookback + 1:
        return 0

    current_volume = df.iloc[-1]["volume"]

    average_volume = df.iloc[
        -lookback - 1:-1
    ]["volume"].mean()

    if average_volume <= 0:
        return 0

    return (
        current_volume
        / average_volume
    )


# ============================================================
# NIFTY DAILY BIAS
# ============================================================

def evaluate_nifty_bias(
    client,
    nifty_instrument_key,
):

    try:

        end = today_ist()
        start = end - timedelta(days=180)

        df = client.get_candles(
            nifty_instrument_key,
            "days",
            end.strftime("%Y-%m-%d"),
            start.strftime("%Y-%m-%d"),
        )

        df = filter_completed_daily(df)

        if len(df) < 60:
            return "NEUTRAL"

        df = add_emas(df)

        last = df.iloc[-1]

        if (
            last["close"] > last["ema20"]
            and
            last["ema20"] > last["ema50"]
        ):
            return "BULLISH"

        if (
            last["close"] < last["ema20"]
            and
            last["ema20"] < last["ema50"]
        ):
            return "BEARISH"

        return "NEUTRAL"

    except Exception:
        return "NEUTRAL"


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    htf_alignment,
    sweep,
    displacement,
    structure,
    zone,
    volume_ratio,
):

    score = 0

    components = {}

    # HTF alignment
    if htf_alignment:
        score += 20
        components["HTF Alignment"] = 20
    else:
        components["HTF Alignment"] = 0

    # Liquidity
    if sweep:
        score += 20
        components["Liquidity Sweep"] = 20
    else:
        components["Liquidity Sweep"] = 0

    # Displacement
    if displacement:
        score += 15
        components["Displacement"] = 15
    else:
        components["Displacement"] = 0

    # Structure
    if structure:
        score += 20
        components["BOS / CHOCH"] = 20
    else:
        components["BOS / CHOCH"] = 0

    # Zone
    if zone:
        score += 15
        components["FVG / OB"] = 15
    else:
        components["FVG / OB"] = 0

    # Volume
    if volume_ratio >= 2:
        score += 10
        components["Volume"] = 10

    elif volume_ratio >= 1.5:
        score += 8
        components["Volume"] = 8

    elif volume_ratio >= 1.2:
        score += 5
        components["Volume"] = 5

    else:
        components["Volume"] = 0

    return score, components


# ============================================================
# SL / TARGETS
# ============================================================

def calculate_trade_levels(
    direction,
    entry,
    daily_df,
    hourly_df,
    sweep,
    max_sl_pct,
    target_rr,
):

    recent = hourly_df.tail(20)

    if direction == "BUY":

        structural_low = recent["low"].min()

        if sweep:
            structural_low = min(
                structural_low,
                sweep["price"]
            )

        sl = structural_low * 0.9975

        risk = entry - sl

        if risk <= 0:
            return None

        sl_pct = (
            risk / entry
        ) * 100

        if sl_pct > max_sl_pct:
            return None

        tp1 = entry + risk * 1.5
        tp2 = entry + risk * target_rr

    else:

        structural_high = recent["high"].max()

        if sweep:
            structural_high = max(
                structural_high,
                sweep["price"]
            )

        sl = structural_high * 1.0025

        risk = sl - entry

        if risk <= 0:
            return None

        sl_pct = (
            risk / entry
        ) * 100

        if sl_pct > max_sl_pct:
            return None

        tp1 = entry - risk * 1.5
        tp2 = entry - risk * target_rr

    rr = abs(tp2 - entry) / risk

    return {
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "risk": round(risk, 2),
        "sl_pct": round(sl_pct, 2),
        "rr": round(rr, 2),
    }


# ============================================================
# SIGNAL EVALUATION
# ============================================================

def evaluate_swing_setup(
    symbol,
    daily_df,
    hourly_df,
    nifty_bias,
    settings,
):

    if daily_df.empty or hourly_df.empty:
        return None

    if len(daily_df) < 80:
        return None

    if len(hourly_df) < 50:
        return None

    daily_df = add_emas(daily_df)

    daily_structure = analyze_structure(
        daily_df,
        settings["daily_swing_length"],
    )

    hourly_structure = analyze_structure(
        hourly_df,
        settings["hourly_swing_length"],
    )

    daily_trend = daily_structure["trend"]

    if daily_trend not in [
        "BULLISH",
        "BEARISH",
    ]:
        return None

    # Optional Nifty filter
    if settings["use_nifty_bias"]:

        if nifty_bias == "BULLISH" and daily_trend != "BULLISH":
            return None

        if nifty_bias == "BEARISH" and daily_trend != "BEARISH":
            return None

    direction = (
        "BUY"
        if daily_trend == "BULLISH"
        else "SELL"
    )

    # --------------------------------------------------------
    # DAILY LIQUIDITY
    # --------------------------------------------------------

    daily_sweep = detect_liquidity_sweep(
        daily_df,
        settings["daily_swing_length"],
        settings["daily_sweep_lookback"],
    )

    if not daily_sweep:
        return None

    if daily_sweep["direction"] != direction:
        return None

    # --------------------------------------------------------
    # 60M STRUCTURE
    # --------------------------------------------------------

    hourly_sweep = detect_liquidity_sweep(
        hourly_df,
        settings["hourly_swing_length"],
        settings["hourly_sweep_lookback"],
    )

    # A 60M sweep is preferred.
    # Daily sweep remains mandatory.
    if hourly_sweep:
        if hourly_sweep["direction"] != direction:
            hourly_sweep = None

    # --------------------------------------------------------
    # DISPLACEMENT
    # --------------------------------------------------------

    displacement_found, displacement_direction = (
        detect_displacement(
            hourly_df,
            settings["displacement_multiplier"],
        )
    )

    if not displacement_found:
        return None

    if displacement_direction != direction:
        return None

    # --------------------------------------------------------
    # BOS / CHOCH
    # --------------------------------------------------------

    structure_break = (
        hourly_structure["bos"]
        or hourly_structure["choch"]
    )

    if not structure_break:
        return None

    # --------------------------------------------------------
    # FVG / ORDER BLOCK
    # --------------------------------------------------------

    fvg = detect_fvg(
        hourly_df,
        settings["zone_age"],
    )

    ob = detect_order_block(
        hourly_df,
        direction,
        settings["zone_age"],
    )

    latest_price = hourly_df.iloc[-1]["close"]

    zone = None

    if fvg and fvg["direction"] == direction:
        zone = fvg

    if ob and ob["direction"] == direction:

        if zone is None:
            zone = ob

        else:

            # Prefer the closer zone
            if (
                zone_distance_pct(
                    latest_price,
                    ob
                )
                <
                zone_distance_pct(
                    latest_price,
                    zone
                )
            ):
                zone = ob

    if zone is None:
        return None

    # Require price near zone.
    distance = zone_distance_pct(
        latest_price,
        zone,
    )

    if distance > settings["max_zone_distance_pct"]:
        return None

    # --------------------------------------------------------
    # DAILY VOLUME
    # --------------------------------------------------------

    volume_ratio = daily_volume_ratio(
        daily_df,
        settings["volume_lookback"],
    )

    if volume_ratio < settings["min_volume_ratio"]:
        return None

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    htf_alignment = (
        daily_trend == direction
    )

    score, components = calculate_score(
        htf_alignment=htf_alignment,
        sweep=True,
        displacement=True,
        structure=True,
        zone=True,
        volume_ratio=volume_ratio,
    )

    if score < settings["min_score"]:
        return None

    # --------------------------------------------------------
    # ENTRY
    # --------------------------------------------------------

    entry = latest_price

    levels = calculate_trade_levels(
        direction=direction,
        entry=entry,
        daily_df=daily_df,
        hourly_df=hourly_df,
        sweep=daily_sweep,
        max_sl_pct=settings["max_sl_pct"],
        target_rr=settings["target_rr"],
    )

    if levels is None:
        return None

    if levels["rr"] < settings["min_rr"]:
        return None

    signal_time = now_ist()

    return {
        "symbol": symbol,
        "direction": direction,

        "entry": levels["entry"],
        "sl": levels["sl"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],

        "risk": levels["risk"],
        "sl_pct": levels["sl_pct"],
        "rr": levels["rr"],

        "score": score,

        "daily_bias": daily_trend,
        "nifty_bias": nifty_bias,

        "daily_sweep": True,
        "hourly_sweep": bool(hourly_sweep),

        "bos": hourly_structure["bos"],
        "choch": hourly_structure["choch"],

        "displacement": True,

        "fvg": bool(
            fvg
            and fvg["direction"] == direction
        ),

        "order_block": bool(
            ob
            and ob["direction"] == direction
        ),

        "volume_ratio": round(
            volume_ratio,
            2
        ),

        "zone_type": (
            "FVG"
            if fvg
            and fvg["direction"] == direction
            else "ORDER BLOCK"
        ),

        "candle_time": hourly_df.iloc[-1][
            "timestamp"
        ].isoformat(),

        "signal_time": signal_time.isoformat(),

        "status": "ACTIVE",

        "reason": (
            f"{direction} setup: "
            f"Daily {daily_trend}, "
            f"liquidity sweep + "
            f"60M structure break + "
            f"displacement + "
            f"FVG/OB + "
            f"volume confirmation"
        ),

        "components": components,
    }


# ============================================================
# SQLITE DATABASE
# ============================================================

def init_database(db_path):

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry REAL,
            sl REAL,
            tp1 REAL,
            tp2 REAL,
            risk REAL,
            sl_pct REAL,
            rr REAL,
            score INTEGER,
            daily_bias TEXT,
            nifty_bias TEXT,
            volume_ratio REAL,
            zone_type TEXT,
            candle_time TEXT,
            signal_time TEXT,
            status TEXT,
            reason TEXT,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(symbol, direction, candle_time)
        )
    """)

    conn.commit()

    return conn


def save_signal(conn, signal):

    now = now_ist().isoformat()

    conn.execute("""
        INSERT OR REPLACE INTO signals (
            symbol,
            direction,
            entry,
            sl,
            tp1,
            tp2,
            risk,
            sl_pct,
            rr,
            score,
            daily_bias,
            nifty_bias,
            volume_ratio,
            zone_type,
            candle_time,
            signal_time,
            status,
            reason,
            created_at,
            updated_at
        )
        VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, (
        signal["symbol"],
        signal["direction"],
        signal["entry"],
        signal["sl"],
        signal["tp1"],
        signal["tp2"],
        signal["risk"],
        signal["sl_pct"],
        signal["rr"],
        signal["score"],
        signal["daily_bias"],
        signal["nifty_bias"],
        signal["volume_ratio"],
        signal["zone_type"],
        signal["candle_time"],
        signal["signal_time"],
        signal["status"],
        signal["reason"],
        now,
        now,
    ))

    conn.commit()


def load_active_signals(
    conn,
    max_days=5,
):

    cutoff = (
        now_ist()
        - timedelta(days=max_days)
    ).isoformat()

    rows = conn.execute("""
        SELECT
            symbol,
            direction,
            entry,
            sl,
            tp1,
            tp2,
            risk,
            sl_pct,
            rr,
            score,
            daily_bias,
            nifty_bias,
            volume_ratio,
            zone_type,
            candle_time,
            signal_time,
            status,
            reason
        FROM signals
        WHERE status = 'ACTIVE'
        AND signal_time >= ?
        ORDER BY score DESC
    """, (cutoff,)).fetchall()

    columns = [
        "symbol",
        "direction",
        "entry",
        "sl",
        "tp1",
        "tp2",
        "risk",
        "sl_pct",
        "rr",
        "score",
        "daily_bias",
        "nifty_bias",
        "volume_ratio",
        "zone_type",
        "candle_time",
        "signal_time",
        "status",
        "reason",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def expire_old_signals(
    conn,
    max_days=5,
):

    cutoff = (
        now_ist()
        - timedelta(days=max_days)
    ).isoformat()

    conn.execute("""
        UPDATE signals
        SET status = 'EXPIRED',
            updated_at = ?
        WHERE status = 'ACTIVE'
        AND signal_time < ?
    """, (
        now_ist().isoformat(),
        cutoff,
    ))

    conn.commit()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    bot_token,
    chat_id,
    signal,
):

    if not bot_token or not chat_id:
        return False

    message = f"""
📈 NSE SWING SMC SIGNAL

Symbol: {signal['symbol']}
Direction: {signal['direction']}

Entry: ₹{signal['entry']}
SL: ₹{signal['sl']}

TP1: ₹{signal['tp1']}
TP2: ₹{signal['tp2']}

RR: {signal['rr']}R
Score: {signal['score']}/100

Daily Bias: {signal['daily_bias']}
Nifty Bias: {signal['nifty_bias']}

Volume: {signal['volume_ratio']}x

Zone: {signal['zone_type']}

SMC:
✓ Liquidity Sweep
✓ BOS / CHOCH
✓ Displacement
✓ FVG / OB
✓ Volume

Holding period:
Up to configured swing expiry.

⚠️ Analysis only. Manage risk manually.
"""

    url = (
        f"https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
            },
            timeout=15,
        )

        return response.ok

    except Exception:
        return False


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_symbol(
    client,
    symbol,
    instrument_key,
    nifty_bias,
    settings,
):

    try:

        end = today_ist()

        # Daily history
        daily_start = (
            end
            - timedelta(days=240)
        )

        daily_df = client.get_candles(
            instrument_key,
            "days",
            end.strftime("%Y-%m-%d"),
            daily_start.strftime("%Y-%m-%d"),
        )

        daily_df = filter_completed_daily(
            daily_df
        )

        if daily_df.empty:
            return None

        latest_price = daily_df.iloc[-1]["close"]

        if latest_price < settings["min_price"]:
            return None

        # 60 minute history
        hourly_start = (
            end
            - timedelta(days=30)
        )

        hourly_df = client.get_candles(
            instrument_key,
            "minutes/60",
            end.strftime("%Y-%m-%d"),
            hourly_start.strftime("%Y-%m-%d"),
        )

        hourly_df = filter_completed_intraday(
            hourly_df,
            60,
        )

        if hourly_df.empty:
            return None

        return evaluate_swing_setup(
            symbol=symbol,
            daily_df=daily_df,
            hourly_df=hourly_df,
            nifty_bias=nifty_bias,
            settings=settings,
        )

    except Exception:
        return None


# ============================================================
# UNIVERSE
# ============================================================

def select_universe(
    instruments,
    universe_type,
    custom_symbols,
):

    if instruments.empty:
        return instruments

    if universe_type == "NIFTY50":

        # Keep this list editable.
        nifty50 = {
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "ICICIBANK",
            "INFY",
            "ITC",
            "SBIN",
            "BHARTIARTL",
            "LT",
            "HINDUNILVR",
            "AXISBANK",
            "KOTAKBANK",
            "MARUTI",
            "M&M",
            "SUNPHARMA",
            "TITAN",
            "BAJFINANCE",
            "HCLTECH",
            "ADANIENT",
            "NTPC",
            "TATASTEEL",
            "ONGC",
            "POWERGRID",
            "COALINDIA",
            "WIPRO",
            "ULTRACEMCO",
            "NESTLEIND",
            "ASIANPAINT",
            "TECHM",
            "JSWSTEEL",
        }

        return instruments[
            instruments["symbol"].isin(
                nifty50
            )
        ].copy()

    if universe_type == "CUSTOM":

        symbols = {
            x.strip().upper()
            for x in custom_symbols.split(",")
            if x.strip()
        }

        return instruments[
            instruments["symbol"].isin(symbols)
        ].copy()

    if universe_type.startswith("TOP"):

        number = int(
            universe_type.replace(
                "TOP ",
                ""
            )
        )

        # IMPORTANT:
        # This is an API-efficient universe selection.
        # It does not falsely claim to be turnover-ranked.
        #
        # For true dynamic turnover ranking,
        # use Upstox market-quote/full-market data
        # as a separate first-stage scanner.

        return instruments.head(number).copy()

    return instruments.copy()


# ============================================================
# SCAN MARKET
# ============================================================

def scan_market(
    client,
    universe,
    nifty_bias,
    settings,
    progress_callback=None,
):

    signals = []

    total = len(universe)

    if total == 0:
        return signals

    workers = settings["max_threads"]

    completed = 0

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {}

        for _, row in universe.iterrows():

            future = executor.submit(
                scan_symbol,
                client,
                row["symbol"],
                row["instrument_key"],
                nifty_bias,
                settings,
            )

            futures[future] = row["symbol"]

        for future in as_completed(futures):

            completed += 1

            try:

                result = future.result()

                if result:
                    signals.append(result)

            except Exception:
                pass

            if progress_callback:

                progress_callback(
                    completed,
                    total,
                )

    signals.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return signals


# ============================================================
# CHART
# ============================================================

def create_chart(
    df,
    signal,
):

    data = df.tail(100).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="60M",
        )
    )

    fig.add_hline(
        y=signal["entry"],
        annotation_text="ENTRY",
    )

    fig.add_hline(
        y=signal["sl"],
        annotation_text="SL",
    )

    fig.add_hline(
        y=signal["tp1"],
        annotation_text="TP1",
    )

    fig.add_hline(
        y=signal["tp2"],
        annotation_text="TP2",
    )

    fig.update_layout(
        height=650,
        xaxis_rangeslider_visible=False,
        title=f"{signal['symbol']} - 60M Swing Setup",
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================

st.title("📈 NSE Swing SMC Scanner")

st.caption(
    "Daily HTF + 60M SMC confirmation | "
    "No automatic order placement"
)

with st.sidebar:

    st.header("🔐 Upstox")

    access_token = st.text_input(
        "Upstox Access Token",
        type="password",
    )

    st.header("🌐 Universe")

    universe_type = st.selectbox(
        "Stock Universe",
        [
            "TOP 2000",
            "TOP 1000",
            "TOP 500",
            "NIFTY50",
            "ALL NSE",
            "CUSTOM",
        ],
    )

    custom_symbols = st.text_input(
        "Custom symbols",
        placeholder="RELIANCE,TCS,INFY",
    )

    st.header("📊 Swing Settings")

    min_price = st.number_input(
        "Minimum stock price ₹",
        min_value=1.0,
        value=25.0,
    )

    daily_swing_length = st.slider(
        "Daily swing length",
        2,
        7,
        3,
    )

    hourly_swing_length = st.slider(
        "60M swing length",
        2,
        7,
        3,
    )

    daily_sweep_lookback = st.slider(
        "Daily sweep lookback",
        5,
        30,
        15,
    )

    hourly_sweep_lookback = st.slider(
        "60M sweep lookback",
        5,
        40,
        15,
    )

    displacement_multiplier = st.slider(
        "Displacement multiplier",
        1.1,
        3.0,
        1.5,
        0.1,
    )

    zone_age = st.slider(
        "Zone age 60M candles",
        5,
        60,
        30,
    )

    max_zone_distance_pct = st.slider(
        "Maximum zone distance %",
        0.1,
        5.0,
        1.0,
        0.1,
    )

    st.header("📈 Volume")

    volume_lookback = st.slider(
        "Daily volume lookback",
        10,
        50,
        20,
    )

    min_volume_ratio = st.slider(
        "Minimum daily volume",
        1.0,
        3.0,
        1.5,
        0.1,
    )

    st.header("🎯 Risk")

    max_sl_pct = st.slider(
        "Maximum SL %",
        0.5,
        6.0,
        3.0,
        0.25,
    )

    min_rr = st.slider(
        "Minimum RR",
        1.5,
        5.0,
        2.0,
        0.25,
    )

    target_rr = st.slider(
        "TP2 RR",
        2.0,
        6.0,
        2.5,
        0.25,
    )

    min_score = st.slider(
        "Minimum score",
        50,
        100,
        80,
    )

    max_hold_days = st.slider(
        "Maximum signal age / holding window",
        1,
        15,
        5,
    )

    st.header("🌍 Market Filter")

    use_nifty_bias = st.checkbox(
        "Use Nifty Daily Bias",
        value=True,
    )

    st.header("⚙️ Performance")

    max_threads = st.slider(
        "Scanner threads",
        2,
        16,
        8,
    )

    st.header("📱 Telegram")

    telegram_token = st.text_input(
        "Bot Token",
        type="password",
    )

    telegram_chat_id = st.text_input(
        "Chat ID",
    )

    auto_refresh = st.checkbox(
        "Auto refresh",
        value=True,
    )


# ============================================================
# SETTINGS
# ============================================================

settings = {
    "min_price": min_price,

    "daily_swing_length":
        daily_swing_length,

    "hourly_swing_length":
        hourly_swing_length,

    "daily_sweep_lookback":
        daily_sweep_lookback,

    "hourly_sweep_lookback":
        hourly_sweep_lookback,

    "displacement_multiplier":
        displacement_multiplier,

    "zone_age":
        zone_age,

    "max_zone_distance_pct":
        max_zone_distance_pct,

    "volume_lookback":
        volume_lookback,

    "min_volume_ratio":
        min_volume_ratio,

    "max_sl_pct":
        max_sl_pct,

    "min_rr":
        min_rr,

    "target_rr":
        target_rr,

    "min_score":
        min_score,

    "max_hold_days":
        max_hold_days,

    "use_nifty_bias":
        use_nifty_bias,

    "max_threads":
        max_threads,
}


# ============================================================
# AUTO REFRESH
# ============================================================

if (
    auto_refresh
    and st_autorefresh is not None
):

    st_autorefresh(
        interval=15 * 60 * 1000,
        key="swing_scanner_refresh",
    )


# ============================================================
# SESSION STATE
# ============================================================

if "connected" not in st.session_state:
    st.session_state.connected = False

if "client" not in st.session_state:
    st.session_state.client = None

if "instruments" not in st.session_state:
    st.session_state.instruments = None

if "last_scan" not in st.session_state:
    st.session_state.last_scan = None

if "signals" not in st.session_state:
    st.session_state.signals = []


# ============================================================
# DATABASE
# ============================================================

db_path = DEFAULT_DB

conn = init_database(db_path)

expire_old_signals(
    conn,
    max_hold_days,
)


# ============================================================
# CONNECTION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    connect_clicked = st.button(
        "🔌 Connect Upstox",
        use_container_width=True,
    )

if connect_clicked:

    if not access_token:

        st.error(
            "Enter your Upstox access token."
        )

    else:

        try:

            client = UpstoxClient(
                access_token
            )

            profile = (
                client.validate_connection()
            )

            st.session_state.client = client
            st.session_state.connected = True

            st.success(
                "Upstox connection successful."
            )

        except Exception as e:

            st.session_state.connected = False

            st.error(
                f"Connection failed: {e}"
            )


# ============================================================
# LOAD INSTRUMENTS
# ============================================================

if st.session_state.connected:

    if st.session_state.instruments is None:

        with st.spinner(
            "Loading NSE instruments..."
        ):

            try:

                st.session_state.instruments = (
                    get_nse_equities()
                )

            except Exception as e:

                st.error(
                    f"Failed to load NSE instruments: {e}"
                )

    instruments = (
        st.session_state.instruments
    )

else:

    instruments = None


# ============================================================
# STATUS
# ============================================================

with col2:

    if market_is_open():

        st.metric(
            "Market",
            "OPEN",
        )

    else:

        st.metric(
            "Market",
            "CLOSED",
        )


with col3:

    active_df = load_active_signals(
        conn,
        max_hold_days,
    )

    st.metric(
        "Active Swing Signals",
        len(active_df),
    )


# ============================================================
# SCAN BUTTON
# ============================================================

if instruments is not None:

    universe = select_universe(
        instruments,
        universe_type,
        custom_symbols,
    )

    st.info(
        f"Universe selected: "
        f"{len(universe)} stocks"
    )

    scan_clicked = st.button(
        "🚀 SCAN SWING MARKET",
        type="primary",
        use_container_width=True,
    )

    if scan_clicked:

        client = st.session_state.client

        # ----------------------------------------------------
        # Find NIFTY instrument
        # ----------------------------------------------------

        nifty_candidates = instruments[
            instruments["symbol"].isin([
                "NIFTY",
                "NIFTY50",
                "NIFTY 50",
            ])
        ]

        if not nifty_candidates.empty:

            nifty_key = (
                nifty_candidates.iloc[0][
                    "instrument_key"
                ]
            )

            nifty_bias = (
                evaluate_nifty_bias(
                    client,
                    nifty_key,
                )
            )

        else:

            # If index instrument isn't found,
            # don't block stock analysis.
            nifty_bias = "NEUTRAL"

        st.write(
            f"### Nifty Daily Bias: `{nifty_bias}`"
        )

        progress = st.progress(0)

        status_text = st.empty()

        def update_progress(done, total):

            if total > 0:

                progress.progress(
                    done / total
                )

            status_text.write(
                f"Scanning {done}/{total}"
            )

        start_time = time.time()

        signals = scan_market(
            client=client,
            universe=universe,
            nifty_bias=nifty_bias,
            settings=settings,
            progress_callback=update_progress,
        )

        elapsed = (
            time.time()
            - start_time
        )

        # Save signals
        new_alerts = []

        for signal in signals:

            # Check whether this exact
            # signal already exists.
            existing = conn.execute("""
                SELECT id
                FROM signals
                WHERE symbol = ?
                AND direction = ?
                AND candle_time = ?
            """, (
                signal["symbol"],
                signal["direction"],
                signal["candle_time"],
            )).fetchone()

            if existing is None:

                save_signal(
                    conn,
                    signal,
                )

                new_alerts.append(
                    signal
                )

            else:

                # Update existing signal
                save_signal(
                    conn,
                    signal,
                )

        # Telegram only for genuinely new signals
        if (
            telegram_token
            and telegram_chat_id
        ):

            for signal in new_alerts:

                send_telegram(
                    telegram_token,
                    telegram_chat_id,
                    signal,
                )

        st.session_state.last_scan = now_ist()

        st.session_state.signals = (
            signals
        )

        st.success(
            f"Scan completed in "
            f"{elapsed:.1f} seconds. "
            f"Found {len(signals)} setups."
        )


# ============================================================
# ACTIVE SIGNALS
# ============================================================

st.divider()

st.header("🔥 Active Swing Signals")

active_df = load_active_signals(
    conn,
    max_hold_days,
)

if active_df.empty:

    st.info(
        "No active swing setups currently."
    )

else:

    display_df = active_df.copy()

    display_df["Entry"] = (
        display_df["entry"]
        .round(2)
    )

    display_df["SL"] = (
        display_df["sl"]
        .round(2)
    )

    display_df["TP1"] = (
        display_df["tp1"]
        .round(2)
    )

    display_df["TP2"] = (
        display_df["tp2"]
        .round(2)
    )

    display_df["RR"] = (
        display_df["rr"]
        .round(2)
    )

    display_df["Score"] = (
        display_df["score"]
    )

    display_df["Volume"] = (
        display_df["volume_ratio"]
        .round(2)
    )

    show_columns = [
        "symbol",
        "direction",
        "Entry",
        "SL",
        "TP1",
        "TP2",
        "RR",
        "Score",
        "daily_bias",
        "Volume",
        "zone_type",
        "signal_time",
    ]

    st.dataframe(
        display_df[show_columns],
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TOP SIGNALS
# ============================================================

if not active_df.empty:

    st.divider()

    st.header("🏆 Best Setups")

    top_signals = active_df.head(10)

    cols = st.columns(
        min(3, len(top_signals))
    )

    for i, (_, row) in enumerate(
        top_signals.iterrows()
    ):

        with cols[
            i % len(cols)
        ]:

            direction_icon = (
                "🟢"
                if row["direction"] == "BUY"
                else "🔴"
            )

            st.subheader(
                f"{direction_icon} "
                f"{row['symbol']}"
            )

            st.write(
                f"**Direction:** "
                f"{row['direction']}"
            )

            st.write(
                f"**Entry:** ₹{row['entry']:.2f}"
            )

            st.write(
                f"**SL:** ₹{row['sl']:.2f}"
            )

            st.write(
                f"**TP1:** ₹{row['tp1']:.2f}"
            )

            st.write(
                f"**TP2:** ₹{row['tp2']:.2f}"
            )

            st.write(
                f"**RR:** {row['rr']:.2f}R"
            )

            st.write(
                f"**Score:** "
                f"{row['score']}/100"
            )

            st.write(
                f"**Volume:** "
                f"{row['volume_ratio']:.2f}x"
            )

            st.write(
                f"**Zone:** "
                f"{row['zone_type']}"
            )


# ============================================================
# CHART SELECTOR
# ============================================================

if not active_df.empty:

    st.divider()

    st.header("📊 Setup Chart")

    selected_symbol = st.selectbox(
        "Select stock",
        active_df["symbol"].tolist(),
    )

    selected_rows = active_df[
        active_df["symbol"]
        == selected_symbol
    ]

    if not selected_rows.empty:

        selected = (
            selected_rows.iloc[0]
        )

        client = (
            st.session_state.client
        )

        row = instruments[
            instruments["symbol"]
            == selected_symbol
        ]

        if not row.empty:

            key = row.iloc[0][
                "instrument_key"
            ]

            try:

                end = today_ist()

                start = (
                    end
                    - timedelta(days=30)
                )

                chart_df = client.get_candles(
                    key,
                    "minutes/60",
                    end.strftime("%Y-%m-%d"),
                    start.strftime("%Y-%m-%d"),
                )

                chart_df = (
                    filter_completed_intraday(
                        chart_df,
                        60,
                    )
                )

                if not chart_df.empty:

                    chart_signal = {
                        "symbol":
                            selected["symbol"],

                        "entry":
                            selected["entry"],

                        "sl":
                            selected["sl"],

                        "tp1":
                            selected["tp1"],

                        "tp2":
                            selected["tp2"],
                    }

                    fig = create_chart(
                        chart_df,
                        chart_signal,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

            except Exception as e:

                st.warning(
                    f"Chart unavailable: {e}"
                )


# ============================================================
# SCANNER INFORMATION
# ============================================================

st.divider()

st.header("ℹ️ Scanner Logic")

info_cols = st.columns(4)

with info_cols[0]:

    st.metric(
        "HTF",
        "Daily",
    )

with info_cols[1]:

    st.metric(
        "Entry",
        "60 Minutes",
    )

with info_cols[2]:

    st.metric(
        "Min Volume",
        f"{min_volume_ratio:.1f}x",
    )

with info_cols[3]:

    st.metric(
        "Target",
        f"{target_rr:.1f}R",
    )

st.markdown("""
### BUY logic

**Daily**
- Bullish market structure
- Daily liquidity sweep
- Nifty bullish alignment when enabled

**60M**
- Bullish displacement
- BOS or CHOCH
- FVG / Order Block
- Price near the zone

**Confirmation**
- Daily volume >= configured threshold
- Risk within maximum SL
- RR above minimum

### SELL logic

The inverse conditions are used:

- Daily bearish structure
- Buy-side liquidity sweep
- Bearish 60M displacement
- Bearish BOS / CHOCH
- Bearish FVG / Order Block
- Volume confirmation
- Risk/RR validation
""")


# ============================================================
# LAST SCAN
# ============================================================

if st.session_state.last_scan:

    st.caption(
        "Last scan: "
        + st.session_state.last_scan.strftime(
            "%d-%m-%Y %H:%M:%S"
        )
        + " IST"
    )

st.caption(
    "Swing scanner is for analysis and alerts only. "
    "It does not place orders."
)                            
