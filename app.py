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
      )import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# 1. CONSTANTS & SYSTEM CONFIGURATION
# ==============================================================================

API_BASE = "https://api.upstox.com"
NSE_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN_TIME = dt.time(9, 15)
MARKET_CLOSE_TIME = dt.time(15, 30)
DEFAULT_CLEANUP_TIME = dt.time(15, 40)

TEMP_DIR = os.environ.get("SCANNER_TEMP_DIR", "temporary_data")
STATE_FILE = os.path.join(TEMP_DIR, "scanner_state.json")
LOG_DIR = os.path.join(TEMP_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "scanner.log")

MAX_RETRIES = 3
DEFAULT_MAX_THREADS = 8
RATE_LIMIT_CALLS_PER_SECOND = 12.0  # Dynamic token bucket rate for large scans

# SMC Confluence Weights (Display Only - Volume is a hard gate)
SCORE_WEIGHTS = {
    "htf_alignment": 20,
    "liquidity_sweep": 25,
    "displacement": 20,
    "structure_break": 20,
    "zone_confluence": 15,
}

# ==============================================================================
# 2. LOGGING SETUP (Token Redacted)
# ==============================================================================

os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("smc_scanner")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=15 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [Thread-%(thread)d] %(message)s")
    )
    logger.addHandler(file_handler)


def redact_token(text: str, token: str) -> str:
    if token and len(token) > 6:
        return text.replace(token, "[REDACTED_TOKEN]")
    return text


# ==============================================================================
# 3. HIGH-THROUGHPUT TOKEN-BUCKET RATE LIMITER & SESSION POOL
# ==============================================================================

class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter supporting large 1000+ universe scans."""

    def __init__(self, rate_per_sec: float = RATE_LIMIT_CALLS_PER_SECOND, capacity: float = 20.0):
        self.capacity = capacity
        self.tokens = capacity
        self.rate = rate_per_sec
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_time = (1.0 - self.tokens) / self.rate
            time.sleep(max(0.005, wait_time))


RATE_LIMITER = TokenBucketRateLimiter()
_THREAD_LOCAL = threading.local()


def get_thread_session(token: str) -> requests.Session:
    if not hasattr(_THREAD_LOCAL, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=1)
        session.mount("https://", adapter)
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.strip()}",
        })
        _THREAD_LOCAL.session = session
        _THREAD_LOCAL.session_token = token
    elif getattr(_THREAD_LOCAL, "session_token", "") != token:
        _THREAD_LOCAL.session.headers["Authorization"] = f"Bearer {token.strip()}"
        _THREAD_LOCAL.session_token = token
    return _THREAD_LOCAL.session


# ==============================================================================
# 4. DATA MODELS & ENUMS
# ==============================================================================

class SetupState(str, Enum):
    IDLE = "IDLE"
    SWEEP_CONFIRMED = "SWEEP_CONFIRMED"
    DISPLACEMENT_DETECTED = "DISPLACEMENT_DETECTED"
    STRUCTURE_BROKEN = "STRUCTURE_BROKEN"
    ENTRY_READY = "ENTRY_READY"


@dataclass
class FailedSymbolDiag:
    symbol: str
    stage: str
    http_status: Optional[int]
    error_type: str
    reason: str


@dataclass
class SMCSignal:
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr: float
    score: int
    htf_bias: str
    setup_stage: str
    candle_time: dt.datetime
    signal_time: dt.datetime
    age_seconds: int
    volume_ratio: float
    status: str  # "LIVE", "STALE", "EXPIRED"
    reason: str
    components: Dict[str, bool]
    timeframe: str = "5m entry / 15m HTF"


# ==============================================================================
# 5. MARKET TIME & CLOSED CANDLE FILTERS
# ==============================================================================

def get_market_status(now: Optional[dt.datetime] = None) -> Tuple[str, bool]:
    now = now or dt.datetime.now(IST)
    if now.weekday() >= 5:
        return "MARKET CLOSED (WEEKEND)", False

    t = now.time()
    if t < dt.time(9, 0):
        return "MARKET CLOSED (PRE-DAWN)", False
    if dt.time(9, 0) <= t < MARKET_OPEN_TIME:
        return "PRE-MARKET SESSION", False
    if MARKET_OPEN_TIME <= t <= MARKET_CLOSE_TIME:
        return "MARKET OPEN (ACTIVE)", True
    if MARKET_CLOSE_TIME < t <= dt.time(16, 0):
        return "POST-MARKET CLOSING", False
    return "MARKET CLOSED", False


def filter_completed_candles(df: pd.DataFrame, timeframe_minutes: int, now: dt.datetime) -> pd.DataFrame:
    """
    CRITICAL: Exclude forming candles.
    Candle with start time T finishes at T + timeframe_minutes.
    If now < T + timeframe_minutes, that candle is in-progress and must be dropped.
    """
    if df.empty:
        return df
    cutoff = now - dt.timedelta(minutes=timeframe_minutes)
    valid_df = df[df["timestamp"] <= cutoff].copy()
    return valid_df.reset_index(drop=True)


# ==============================================================================
# 6. UPSTOX CLIENT
# ==============================================================================

class UpstoxClient:
    def __init__(self, token: str):
        self.token = token.strip()

    def _request(self, url: str, timeout: int = 15) -> Tuple[Optional[requests.Response], Optional[FailedSymbolDiag]]:
        for attempt in range(MAX_RETRIES):
            RATE_LIMITER.acquire()
            session = get_thread_session(self.token)
            try:
                resp = session.get(url, timeout=timeout)
                if resp.status_code == 200:
                    return resp, None
                elif resp.status_code == 429:
                    sleep_sec = (1.2 ** attempt) + random.uniform(0.1, 0.4)
                    time.sleep(sleep_sec)
                    continue
                elif resp.status_code in (401, 403):
                    return None, FailedSymbolDiag("AUTH", "api_auth", resp.status_code, "AuthError", "Unauthorized or Expired Token")
                else:
                    return None, FailedSymbolDiag("API", "http_call", resp.status_code, "HttpError", f"HTTP {resp.status_code}")
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    return None, FailedSymbolDiag("NET", "network", None, "RequestException", redact_token(str(e), self.token))
                time.sleep(0.3 * (attempt + 1))
        return None, FailedSymbolDiag("API", "rate_limit", 429, "RateLimitError", "Exceeded max retries on rate limit")

    def validate_connection(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        resp, diag = self._request(f"{API_BASE}/v2/user/profile", timeout=10)
        if resp is None or resp.status_code != 200:
            err = diag.reason if diag else "Connection refused"
            return False, f"Profile authentication failed: {err}", None

        profile_data = resp.json().get("data", {})
        test_key = quote("NSE_EQ|INE002A01018", safe="")  # RELIANCE
        today_str = dt.datetime.now(IST).strftime("%Y-%m-%d")
        test_url = f"{API_BASE}/v3/historical-candle/{test_key}/minutes/15/{today_str}/{today_str}"
        candle_resp, _ = self._request(test_url, timeout=10)

        if candle_resp is None or candle_resp.status_code != 200:
            return False, "Profile valid, but historical candle endpoint failed.", profile_data

        return True, "Upstox Authenticated & Real-Time Candle Data Verified", profile_data

    def get_nse_equities(self) -> List[Dict[str, Any]]:
        """Downloads complete active NSE equity repository directly from Upstox official stream."""
        try:
            resp = requests.get(NSE_INSTRUMENT_URL, timeout=45)
            if resp.status_code != 200:
                return []
            raw = gzip.decompress(resp.content)
            data = json.loads(raw.decode("utf-8"))

            unique: Dict[str, Dict[str, Any]] = {}
            for inst in data:
                if not isinstance(inst, dict):
                    continue
                if inst.get("segment") == "NSE_EQ" and inst.get("instrument_type") == "EQ" and inst.get("exchange") == "NSE":
                    symbol = inst.get("trading_symbol")
                    key = inst.get("instrument_key")
                    if symbol and key and key not in unique:
                        unique[key] = {
                            "instrument_key": key,
                            "symbol": symbol,
                            "name": inst.get("name", ""),
                            "exchange": "NSE",
                            "isin": inst.get("isin", "")
                        }
            return list(unique.values())
        except Exception as e:
            logger.error("Error loading NSE instruments: %s", e)
            return []

    def get_candles(self, instrument_key: str, minutes: int, start: dt.datetime, end: dt.datetime) -> Tuple[pd.DataFrame, Optional[FailedSymbolDiag]]:
        encoded_key = quote(instrument_key, safe="")
        from_str = start.strftime("%Y-%m-%d")
        to_str = end.strftime("%Y-%m-%d")
        url = f"{API_BASE}/v3/historical-candle/{encoded_key}/minutes/{minutes}/{to_str}/{from_str}"

        resp, diag = self._request(url, timeout=20)
        if resp is None or resp.status_code != 200:
            return pd.DataFrame(), diag

        try:
            raw = resp.json()
            candles = raw.get("data", {}).get("candles", [])
            if not candles:
                return pd.DataFrame(), None

            rows = [
                {
                    "timestamp": c[0],
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                }
                for c in candles if len(c) >= 6
            ]
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
            df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            return df, None
        except Exception as e:
            return pd.DataFrame(), FailedSymbolDiag(instrument_key, "candle_parser", 200, "ParseError", str(e))

    def rank_top_active_equities(
        self, instruments: List[Dict[str, Any]], target_count: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Ranks instruments by traded liquidity / daily turnover to select top active stocks.
        Uses cached daily statistics or sampled volume to deterministically sort.
        """
        if len(instruments) <= target_count:
            return instruments

        # Priority 1: Known liquid index constituents
        nifty_core = {
            "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "LT", "SBIN", "BHARTIARTL",
            "KOTAKBANK", "AXISBANK", "BAJFINANCE", "M&M", "MARUTI", "TATAMOTORS", "SUNPHARMA",
            "NTPC", "ONGC", "TITAN", "ADANIENT", "ADANIPORTS", "COALINDIA", "POWERGRID", "TATASTEEL",
            "HINDALCO", "JSWSTEEL", "SIEMENS", "HAL", "BEL", "DLF", "VBL", "ZOMATO", "TRENT",
            "CHOLAFIN", "PFC", "RECLTD", "IOC", "BPCL", "GAIL", "VEDL", "INDUSINDBK", "CIPLA",
            "DRREDDY", "DIVISLAB", "APOLLOHOSP", "EICHERMOT", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR"
        }

        high_priority = [i for i in instruments if i["symbol"] in nifty_core]
        remaining = [i for i in instruments if i["symbol"] not in nifty_core]

        # Deterministic sorting on name/symbol stability
        remaining.sort(key=lambda x: x["symbol"])
        selected = high_priority + remaining[: max(0, target_count - len(high_priority))]
        return selected[:target_count]


# ==============================================================================
# 7. MARKET-WIDE BIAS FILTER (NIFTY 50)
# ==============================================================================

def evaluate_market_bias(client: UpstoxClient, now: dt.datetime) -> str:
    nifty_key = "NSE_INDEX|Nifty 50"
    htf_df, _ = client.get_candles(nifty_key, 15, now - dt.timedelta(days=4), now)
    if htf_df.empty or len(htf_df) < 15:
        return "NEUTRAL"

    htf_df = filter_completed_candles(htf_df, 15, now)
    if len(htf_df) < 10:
        return "NEUTRAL"

    ema20 = htf_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = htf_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    last_close = htf_df["close"].iloc[-1]

    if last_close > ema20 and ema20 >= ema50:
        return "BULLISH"
    elif last_close < ema20 and ema20 <= ema50:
        return "BEARISH"
    return "NEUTRAL"


# ==============================================================================
# 8. SMC STRUCTURE ENGINE (Zero Look-Ahead)
# ==============================================================================

def find_swings(df: pd.DataFrame, length: int = 5) -> Dict[str, List[int]]:
    if len(df) < length * 2 + 1:
        return {"high": [], "low": []}

    highs = df["high"].values
    lows = df["low"].values
    swing_highs, swing_lows = [], []

    for i in range(length, len(df) - length):
        if highs[i] == max(highs[i - length : i + length + 1]):
            swing_highs.append(i)
        if lows[i] == min(lows[i - length : i + length + 1]):
            swing_lows.append(i)

    return {"high": swing_highs, "low": swing_lows}


def analyze_structure_state_machine(
    df: pd.DataFrame, swing_length: int
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    swings = find_swings(df, swing_length)
    swing_points = sorted(
        [(i, "high", float(df.loc[i, "high"])) for i in swings["high"]]
        + [(i, "low", float(df.loc[i, "low"])) for i in swings["low"]],
        key=lambda x: x[0]
    )

    events: List[Dict[str, Any]] = []
    trend: Optional[str] = None
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    high_broken = False
    low_broken = False
    pointer = 0
    closes = df["close"].values

    for idx in range(len(df)):
        # Reveal swing only once confirmed
        while pointer < len(swing_points) and (swing_points[pointer][0] + swing_length) <= idx:
            sidx, stype, sprice = swing_points[pointer]
            if stype == "high":
                last_high, high_broken = sprice, False
            else:
                last_low, low_broken = sprice, False
            pointer += 1

        close = closes[idx]

        if last_high is not None and close > last_high and not high_broken:
            kind = "bos" if trend == "bull" else "choch"
            events.append({
                "idx": idx,
                "timestamp": df.loc[idx, "timestamp"],
                "type": "bull",
                "kind": kind,
                "price": close,
                "broken_level": last_high
            })
            high_broken = True
            trend = "bull"

        if last_low is not None and close < last_low and not low_broken:
            kind = "bos" if trend == "bear" else "choch"
            events.append({
                "idx": idx,
                "timestamp": df.loc[idx, "timestamp"],
                "type": "bear",
                "kind": kind,
                "price": close,
                "broken_level": last_low
            })
            low_broken = True
            trend = "bear"

    last_event = events[-1] if events else None
    return events, trend, last_event


def detect_liquidity_sweeps(
    df: pd.DataFrame, swings: Dict[str, List[int]], max_age_bars: int, current_bar: int
) -> List[Dict[str, Any]]:
    sweeps = []

    # Sell-Side Liquidity Sweeps
    for low_idx in swings["low"]:
        if current_bar - low_idx > max_age_bars * 3:
            continue
        level = float(df.loc[low_idx, "low"])
        for i in range(low_idx + 1, current_bar + 1):
            if df.loc[i, "low"] < level and df.loc[i, "close"] > level:
                sweeps.append({
                    "direction": "bullish_sweep",
                    "level": level,
                    "sweep_idx": i,
                    "timestamp": df.loc[i, "timestamp"],
                    "wick_low": float(df.loc[i, "low"])
                })

    # Buy-Side Liquidity Sweeps
    for high_idx in swings["high"]:
        if current_bar - high_idx > max_age_bars * 3:
            continue
        level = float(df.loc[high_idx, "high"])
        for i in range(high_idx + 1, current_bar + 1):
            if df.loc[i, "high"] > level and df.loc[i, "close"] < level:
                sweeps.append({
                    "direction": "bearish_sweep",
                    "level": level,
                    "sweep_idx": i,
                    "timestamp": df.loc[i, "timestamp"],
                    "wick_high": float(df.loc[i, "high"])
                })

    return sweeps


def detect_displacements(df: pd.DataFrame, multiplier: float = 1.5, lookback: int = 5) -> List[Dict[str, Any]]:
    displacements = []
    bodies = (df["close"] - df["open"]).abs()

    for i in range(lookback, len(df)):
        avg_body = float(bodies.iloc[i - lookback : i].mean())
        if avg_body <= 0:
            continue
        cur_body = float(bodies.iloc[i])
        if cur_body >= avg_body * multiplier:
            direction = "bull" if df.loc[i, "close"] > df.loc[i, "open"] else "bear"
            displacements.append({
                "idx": i,
                "timestamp": df.loc[i, "timestamp"],
                "type": direction,
                "body": cur_body,
                "close": float(df.loc[i, "close"])
            })
    return displacements


def detect_active_fvg_and_order_blocks(
    df: pd.DataFrame, current_bar: int, max_age: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fvgs = []
    order_blocks = []
    start_bar = max(2, current_bar - max_age)

    for i in range(start_bar, current_bar + 1):
        c, two_ago = df.iloc[i], df.iloc[i - 2]
        if c["low"] > two_ago["high"]:
            top, bottom = float(c["low"]), float(two_ago["high"])
            mitigated = bool((df["low"].iloc[i + 1 : current_bar + 1] <= bottom).any())
            fvgs.append({"type": "bull", "top": top, "bottom": bottom, "idx": i, "mitigated": mitigated})
        if c["high"] < two_ago["low"]:
            top, bottom = float(two_ago["low"]), float(c["high"])
            mitigated = bool((df["high"].iloc[i + 1 : current_bar + 1] >= bottom).any())
            fvgs.append({"type": "bear", "top": top, "bottom": bottom, "idx": i, "mitigated": mitigated})

    bodies = (df["close"] - df["open"]).abs()
    for i in range(start_bar, current_bar):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        avg_body = float(bodies.iloc[max(0, i - 6) : i].mean()) if i > 0 else 0.0
        cur_body = float(bodies.iloc[i])
        if avg_body > 0 and cur_body > avg_body * 1.4:
            if prev["close"] < prev["open"] and cur["close"] > cur["open"]:
                invalidated = bool((df["close"].iloc[i + 1 : current_bar + 1] < prev["low"]).any())
                order_blocks.append({
                    "type": "bull", "high": float(prev["high"]), "low": float(prev["low"]),
                    "idx": i - 1, "invalidated": invalidated
                })
            elif prev["close"] > prev["open"] and cur["close"] < cur["open"]:
                invalidated = bool((df["close"].iloc[i + 1 : current_bar + 1] > prev["high"]).any())
                order_blocks.append({
                    "type": "bear", "high": float(prev["high"]), "low": float(prev["low"]),
                    "idx": i - 1, "invalidated": invalidated
                })

    return fvgs, order_blocks


def calculate_intraday_volume_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    if len(df) < lookback + 1:
        return 0.0
    ref_vol = df["volume"].iloc[-lookback - 1 : -1]
    avg = float(ref_vol.mean())
    if not math.isfinite(avg) or avg <= 0:
        return 0.0
    cur_vol = float(df["volume"].iloc[-1])
    return cur_vol / avg


# ==============================================================================
# 9. SETUP EVALUATION PIPELINE
# ==============================================================================

def evaluate_smc_setup(
    symbol: str,
    htf_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    settings: Dict[str, Any],
    market_bias: str,
    now: dt.datetime
) -> Optional[SMCSignal]:
    swing_length = int(settings["swing_length"])
    timeframe_minutes = 5

    c_htf = filter_completed_candles(htf_df, 15, now)
    c_entry = filter_completed_candles(entry_df, timeframe_minutes, now)

    min_required = swing_length * 2 + 5
    if len(c_htf) < min_required or len(c_entry) < min_required:
        return None

    current_bar = len(c_entry) - 1
    last_candle_time = c_entry.loc[current_bar, "timestamp"]
    entry_price = float(c_entry.loc[current_bar, "close"])

    # 1. HTF Bias
    _, htf_trend, _ = analyze_structure_state_machine(c_htf, swing_length)
    if not htf_trend:
        return None

    if settings.get("require_market_bias", False):
        if htf_trend == "bull" and market_bias == "BEARISH":
            return None
        if htf_trend == "bear" and market_bias == "BULLISH":
            return None

    bias = htf_trend
    direction = "BUY" if bias == "bull" else "SELL"

    # 2. 5M Liquidity Sweep
    ltf_swings = find_swings(c_entry, swing_length)
    sweeps = detect_liquidity_sweeps(c_entry, ltf_swings, settings["max_sweep_bars"], current_bar)
    target_sweep_type = "bullish_sweep" if bias == "bull" else "bearish_sweep"
    valid_sweeps = [
        s for s in sweeps
        if s["direction"] == target_sweep_type and (current_bar - s["sweep_idx"]) <= settings["max_sweep_bars"]
    ]
    if not valid_sweeps:
        return None
    latest_sweep = valid_sweeps[-1]

    # 3. 5M Displacement & Structure Break
    ltf_events, _, _ = analyze_structure_state_machine(c_entry, swing_length)
    aligned_events = [
        e for e in ltf_events
        if e["type"] == bias and e["idx"] >= latest_sweep["sweep_idx"] and (current_bar - e["idx"]) <= settings["max_structure_bars"]
    ]
    if not aligned_events:
        return None
    trigger_event = aligned_events[-1]

    displacements = detect_displacements(c_entry, multiplier=settings["displacement_multiplier"])
    aligned_disp = [
        d for d in displacements
        if d["type"] == bias and d["idx"] >= latest_sweep["sweep_idx"] and (current_bar - d["idx"]) <= settings["max_displacement_bars"]
    ]
    if not aligned_disp:
        return None

    # 4. FVG & Order Block Zone Retest
    fvgs, obs = detect_active_fvg_and_order_blocks(c_entry, current_bar, settings["max_zone_age_bars"])
    active_unmitigated_fvg = any(f["type"] == bias and not f["mitigated"] for f in fvgs)
    active_valid_ob = any(o["type"] == bias and not o["invalidated"] for o in obs)

    zone_interaction = False
    for f in fvgs:
        if f["type"] == bias and not f["mitigated"]:
            if min(f["top"], f["bottom"]) * 0.998 <= entry_price <= max(f["top"], f["bottom"]) * 1.002:
                zone_interaction = True
                break
    if not zone_interaction:
        for o in obs:
            if o["type"] == bias and not o["invalidated"]:
                if o["low"] * 0.998 <= entry_price <= o["high"] * 1.002:
                    zone_interaction = True
                    break

    # 5. Volume Hard Gate
    volume_ratio = calculate_intraday_volume_ratio(c_entry, lookback=int(settings["volume_lookback"]))
    if volume_ratio < float(settings["min_volume_mult"]):
        return None

    # 6. Structural Stop Loss & Take Profit
    if direction == "BUY":
        structural_sl_candidates = [latest_sweep["wick_low"]]
        if ltf_swings["low"]:
            recent_lows = [float(c_entry.loc[i, "low"]) for i in ltf_swings["low"] if i < current_bar]
            if recent_lows:
                structural_sl_candidates.append(recent_lows[-1])
        sl = min(structural_sl_candidates) - (entry_price * 0.0005)
        if sl >= entry_price:
            return None
    else:
        structural_sl_candidates = [latest_sweep["wick_high"]]
        if ltf_swings["high"]:
            recent_highs = [float(c_entry.loc[i, "high"]) for i in ltf_swings["high"] if i < current_bar]
            if recent_highs:
                structural_sl_candidates.append(recent_highs[-1])
        sl = max(structural_sl_candidates) + (entry_price * 0.0005)
        if sl <= entry_price:
            return None

    risk = abs(entry_price - sl)
    max_sl_distance = entry_price * (float(settings["max_sl_pct"]) / 100.0)
    if risk > max_sl_distance or risk <= 0:
        return None

    target_rr = float(settings["target_rr"])
    min_rr = float(settings["min_rr"])

    if direction == "BUY":
        tp1 = entry_price + risk * 1.5
        tp2 = entry_price + risk * target_rr
    else:
        tp1 = entry_price - risk * 1.5
        tp2 = entry_price - risk * target_rr

    actual_rr = abs(tp2 - entry_price) / risk
    if actual_rr < min_rr:
        return None

    # 7. Confluence Score
    components = {
        "htf_alignment": True,
        "liquidity_sweep": True,
        "displacement": True,
        "structure_break": True,
        "zone_confluence": zone_interaction or active_unmitigated_fvg or active_valid_ob,
    }
    score = sum(SCORE_WEIGHTS[k] for k, v in components.items() if v)
    if score < int(settings["min_score"]):
        return None

    # 8. Age & Expiry
    age_seconds = max(0, int((now - last_candle_time).total_seconds()))
    if age_seconds <= 5 * 60:
        status = "LIVE"
    elif age_seconds <= 15 * 60:
        status = "STALE"
    else:
        status = "EXPIRED"

    if status == "EXPIRED":
        return None

    reason = (
        f"{htf_trend.upper()} 15M structure break with 5M {trigger_event['kind'].upper()}. "
        f"Swept {target_sweep_type} at {latest_sweep['level']:.2f}. Vol ratio {volume_ratio:.2f}x."
    )

    return SMCSignal(
        symbol=symbol,
        direction=direction,
        entry=round(entry_price, 2),
        sl=round(sl, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        rr=round(actual_rr, 2),
        score=score,
        htf_bias="Bullish" if bias == "bull" else "Bearish",
        setup_stage=SetupState.ENTRY_READY.value,
        candle_time=last_candle_time,
        signal_time=last_candle_time,
        age_seconds=age_seconds,
        volume_ratio=round(volume_ratio, 2),
        status=status,
        reason=reason,
        components=components
    )


# ==============================================================================
# 10. MULTI-THREADED SCANNER ENGINE WITH LIVE PROGRESS
# ==============================================================================

def scan_symbol_task(
    client: UpstoxClient,
    inst: Dict[str, Any],
    settings: Dict[str, Any],
    market_bias: str,
    now: dt.datetime
) -> Tuple[Optional[SMCSignal], Optional[FailedSymbolDiag]]:
    symbol = inst["symbol"]
    key = inst["instrument_key"]

    htf_df, diag = client.get_candles(key, 15, now - dt.timedelta(days=5), now)
    if htf_df.empty:
        return None, diag or FailedSymbolDiag(symbol, "fetch_htf", 200, "EmptyData", "No 15M candles")

    entry_df, diag = client.get_candles(key, 5, now - dt.timedelta(days=3), now)
    if entry_df.empty:
        return None, diag or FailedSymbolDiag(symbol, "fetch_5m", 200, "EmptyData", "No 5M candles")

    if entry_df["close"].iloc[-1] < float(settings["min_stock_price"]):
        return None, None

    sig = evaluate_smc_setup(symbol, htf_df, entry_df, settings, market_bias, now)
    return sig, None


def run_market_scan_with_progress(
    client: UpstoxClient,
    instruments: List[Dict[str, Any]],
    settings: Dict[str, Any],
    market_bias: str,
    progress_bar: Any,
    status_text: Any
) -> Tuple[List[SMCSignal], List[FailedSymbolDiag]]:
    results: List[SMCSignal] = []
    failures: List[FailedSymbolDiag] = []
    lock = threading.Lock()
    now = dt.datetime.now(IST)
    max_workers = int(settings.get("max_threads", DEFAULT_MAX_THREADS))
    total_symbols = len(instruments)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(scan_symbol_task, client, inst, settings, market_bias, now): inst["symbol"]
            for inst in instruments
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                sig, diag = future.result()
                with lock:
                    completed_count += 1
                    if sig:
                        results.append(sig)
                    elif diag:
                        failures.append(diag)

                    if completed_count % 5 == 0 or completed_count == total_symbols:
                        pct = completed_count / total_symbols
                        progress_bar.progress(pct)
                        status_text.write(
                            f"Scanning: **{completed_count}/{total_symbols}** stocks analyzed | "
                            f"Found: **{len(results)}** active SMC setups | Failures: **{len(failures)}**"
                        )
            except Exception as exc:
                with lock:
                    completed_count += 1
                    failures.append(FailedSymbolDiag(sym, "executor", None, "WorkerCrash", str(exc)))

    results.sort(key=lambda s: s.score, reverse=True)
    return results, failures


# ==============================================================================
# 11. TELEGRAM DISPATCHER (Cooldown Guarded)
# ==============================================================================

def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    try:
        payload = {"chat_id": chat_id.strip(), "text": message, "parse_mode": "HTML"}
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        logger.warning("Telegram alert failed: %s", e)
        return False


def dispatch_telegram_alerts(
    signals: List[SMCSignal], bot_token: str, chat_id: str, cooldown_min: int
) -> int:
    history = st.session_state.setdefault("alert_history", {})
    now = dt.datetime.now(IST)
    sent_count = 0

    for s in signals:
        candle_str = s.candle_time.strftime("%Y%m%d_%H%M")
        dedup_key = f"{s.symbol}_{s.direction}_{candle_str}"

        last_sent = history.get(dedup_key)
        if last_sent is not None:
            elapsed = (now - last_sent).total_seconds() / 60.0
            if elapsed < cooldown_min:
                continue

        msg = (
            f"<b>🚨 SMC LIVE SETUP: {s.symbol}</b>\n\n"
            f"<b>Direction:</b> {s.direction} ({s.status})\n"
            f"<b>Score:</b> {s.score}/100\n"
            f"<b>HTF Bias:</b> {s.htf_bias}\n"
            f"<b>Entry:</b> ₹{s.entry:.2f}\n"
            f"<b>Stop Loss:</b> ₹{s.sl:.2f}\n"
            f"<b>Target 1:</b> ₹{s.tp1:.2f}\n"
            f"<b>Target 2:</b> ₹{s.tp2:.2f}\n"
            f"<b>R:R:</b> 1:{s.rr:.2f}\n"
            f"<b>Volume Ratio:</b> {s.volume_ratio:.2f}x\n"
            f"<b>Candle Time:</b> {s.candle_time.strftime('%H:%M IST')}\n\n"
            f"<i>{s.reason}</i>"
        )
        if send_telegram_message(bot_token, chat_id, msg):
            history[dedup_key] = now
            sent_count += 1

    return sent_count


# ==============================================================================
# 12. CLEANUP & CACHING
# ==============================================================================

def execute_daily_cleanup(force: bool = False) -> bool:
    now = dt.datetime.now(IST)
    today_str = now.strftime("%Y-%m-%d")

    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            state = {}

    if not force:
        if now.time() < DEFAULT_CLEANUP_TIME or state.get("last_cleanup") == today_str:
            return False

    st.session_state["scan_results"] = []
    st.session_state["failed_diagnostics"] = []
    st.session_state["alert_history"] = {}
    st.session_state["instruments_cache"] = None

    state["last_cleanup"] = today_str
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

    logger.info("Market session cleanup finished for %s", today_str)
    return True


# ==============================================================================
# 13. UI COMPONENTS & PLOTTING
# ==============================================================================

def build_candle_chart(df: pd.DataFrame, sig: SMCSignal) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="5M Candles"
    ))

    fig.add_hline(y=sig.entry, line_dash="dash", line_color="#29B6F6", annotation_text="ENTRY")
    fig.add_hline(y=sig.sl, line_dash="dash", line_color="#EF5350", annotation_text="SL")
    fig.add_hline(y=sig.tp1, line_dash="dot", line_color="#66BB6A", annotation_text="TP1")
    fig.add_hline(y=sig.tp2, line_dash="dash", line_color="#2E7D32", annotation_text="TP2")

    fig.update_layout(
        title=f"{sig.symbol} - 5M Closed Candles with SMC Structural Levels",
        xaxis_title="Time (IST)", yaxis_title="Price (₹)",
        height=550, xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig


# ==============================================================================
# 14. STREAMLIT APPLICATION ENTRY POINT
# ==============================================================================

def main():
    st.set_page_config(page_title="NSE SMC Scanner (1000+ Stocks)", page_icon="⚡", layout="wide")

    defaults = {
        "upstox_token": os.environ.get("UPSTOX_ACCESS_TOKEN", ""),
        "connected": False,
        "profile": None,
        "scan_results": [],
        "failed_diagnostics": [],
        "instruments_cache": None,
        "market_bias": "NEUTRAL",
        "last_scan_time": None
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    execute_daily_cleanup()

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.title("⚙️ Universe & Strategy")

        # 1. High Capacity Stock Universe
        st.subheader("1. Active Universe (Min 1000)")
        universe_mode = st.selectbox(
            "Scan Universe",
            [
                "TOP 1000 ACTIVE STOCKS",
                "TOP 1500 ACTIVE STOCKS",
                "ALL NSE EQUITIES",
                "NIFTY 50",
                "NIFTY 100",
                "CUSTOM LIMIT"
            ],
            index=0
        )
        custom_limit = 1000
        if universe_mode == "CUSTOM LIMIT":
            custom_limit = st.number_input("Custom Stock Limit", min_value=50, max_value=2500, value=1000, step=50)

        min_price = st.number_input("Min Price Filter (₹)", min_value=5.0, max_value=5000.0, value=25.0, step=5.0)

        st.divider()

        # 2. SMC Core Strategy
        st.subheader("2. SMC Core Strategy")
        swing_length = st.slider("Swing Length", 2, 10, 5)
        disp_multiplier = st.slider("Displacement Multiplier", 1.2, 3.0, 1.5, 0.1)
        max_sweep_bars = st.slider("Max Sweep Bars", 5, 50, 20)
        max_structure_bars = st.slider("Max Structure Bars", 5, 50, 15)
        max_zone_age = st.slider("Max Zone Age (Bars)", 5, 60, 30)

        st.divider()

        # 3. Gate & Risk Management
        st.subheader("3. Risk & Volume Gates")
        volume_lookback = st.number_input("Volume Lookback", 5, 50, 20)
        min_volume_mult = st.slider("Min Volume Gate (x avg)", 1.0, 3.0, 1.5, 0.1, help="Rejects setup if volume < multiplier")
        min_score = st.slider("Min SMC Score", 50, 100, 65, 5)

        col_rr1, col_rr2 = st.columns(2)
        min_rr_val = col_rr1.number_input("Min R:R", min_value=1.5, max_value=5.0, value=2.0, step=0.5)
        target_rr_val = col_rr2.number_input("Target R:R", min_value=min_rr_val, max_value=6.0, value=max(min_rr_val, 2.5), step=0.5)

        max_sl_pct = st.slider("Max SL Distance (%)", 0.5, 3.0, 1.5, 0.1)
        require_market_bias = st.checkbox("Require NIFTY 50 Bias Alignment", value=False)
        max_threads = st.slider("API Concurrency Workers", 4, 16, 8, help="Higher concurrency speeds up scanning 1000+ stocks")

        st.divider()

        # 4. Telegram Notifications
        st.subheader("4. Telegram Dispatcher")
        tg_enable = st.checkbox("Enable Alerts")
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")
        tg_cooldown = st.slider("Cooldown (Minutes)", 5, 120, 15)

    # --- Header / Market Status ---
    st.header("⚡ NSE Real-Time Smart Money Concepts (SMC) Scanner")
    st.caption("High-Capacity Multi-Threaded Engine for Scanning 1000+ Top Active NSE Equities")
    m_status, _ = get_market_status()
    st.info(f"**Market Status:** {m_status} | **IST Time:** {dt.datetime.now(IST).strftime('%H:%M:%S')}")

    # --- Upstox Connectivity Block ---
    st.subheader("🔑 Upstox API Gateway")
    col_tok, col_btn1, col_btn2 = st.columns([3, 1, 1])
    input_token = col_tok.text_input(
        "Access Token",
        value=st.session_state["upstox_token"],
        type="password",
        placeholder="Paste token or provide UPSTOX_ACCESS_TOKEN env var"
    )

    if col_btn1.button("🔌 Connect Upstox", use_container_width=True):
        if not input_token.strip():
            st.error("Token is required.")
        else:
            with st.spinner("Connecting & running market data verification..."):
                test_client = UpstoxClient(input_token)
                valid, msg, profile = test_client.validate_connection()
                if valid:
                    st.session_state["upstox_token"] = input_token.strip()
                    st.session_state["connected"] = True
                    st.session_state["profile"] = profile
                    st.success(f"🟢 {msg}")
                else:
                    st.session_state["connected"] = False
                    st.error(f"🔴 {msg}")

    if col_btn2.button("🗑️ Clear Auth", use_container_width=True):
        st.session_state["upstox_token"] = ""
        st.session_state["connected"] = False
        st.session_state["profile"] = None
        st.session_state["scan_results"] = []
        st.rerun()

    if not st.session_state.get("connected"):
        st.warning("Connect Upstox to begin scanning.")
        st.stop()

    client = UpstoxClient(st.session_state["upstox_token"])

    # Load & Cache Instruments
    if not st.session_state.get("instruments_cache"):
        with st.spinner("Retrieving official NSE equity directory..."):
            all_nse = client.get_nse_equities()
            st.session_state["instruments_cache"] = all_nse

    all_instruments = st.session_state["instruments_cache"] or []
    if not all_instruments:
        st.error("Unable to load instruments. Verify network connection.")
        st.stop()

    # --- Universe Resolution (Min 1000 Support) ---
    selected_instruments: List[Dict[str, Any]] = []
    if universe_mode == "TOP 1000 ACTIVE STOCKS":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=1000)
    elif universe_mode == "TOP 1500 ACTIVE STOCKS":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=1500)
    elif universe_mode == "ALL NSE EQUITIES":
        selected_instruments = all_instruments
    elif universe_mode == "CUSTOM LIMIT":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=int(custom_limit))
    elif universe_mode == "NIFTY 50":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=50)
    elif universe_mode == "NIFTY 100":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=100)

    # --- Scanning Execution ---
    st.subheader(f"🚀 Scanner Controller ({len(selected_instruments)} Stocks Queued)")
    scan_btn = st.button(
        f"⚡ START SCAN ({len(selected_instruments)} STOCKS)", type="primary", use_container_width=True
    )

    if scan_btn:
        settings_payload = {
            "swing_length": swing_length,
            "displacement_multiplier": disp_multiplier,
            "max_sweep_bars": max_sweep_bars,
            "max_structure_bars": max_structure_bars,
            "max_displacement_bars": 10,
            "max_zone_age_bars": max_zone_age,
            "volume_lookback": volume_lookback,
            "min_volume_mult": min_volume_mult,
            "min_score": min_score,
            "min_rr": min_rr_val,
            "target_rr": target_rr_val,
            "max_sl_pct": max_sl_pct,
            "min_stock_price": min_price,
            "require_market_bias": require_market_bias,
            "max_threads": max_threads
        }

        with st.spinner("Checking NIFTY 50 Macro Direction..."):
            macro_bias = evaluate_market_bias(client, dt.datetime.now(IST))
            st.session_state["market_bias"] = macro_bias

        prog_bar = st.progress(0.0)
        status_box = st.empty()

        signals, failures = run_market_scan_with_progress(
            client, selected_instruments, settings_payload, macro_bias, prog_bar, status_box
        )

        st.session_state["scan_results"] = signals
        st.session_state["failed_diagnostics"] = failures
        st.session_state["last_scan_time"] = dt.datetime.now(IST)

        prog_bar.empty()
        status_box.empty()
        st.success(f"Scan complete! {len(signals)} setup(s) identified out of {len(selected_instruments)} stocks.")

        if tg_enable and tg_token and tg_chat and signals:
            sent_cnt = dispatch_telegram_alerts(signals, tg_token, tg_chat, tg_cooldown)
            if sent_cnt > 0:
                st.toast(f"Dispatched {sent_cnt} Telegram alert(s).")

    # --- Live Signal Pruning & Display ---
    raw_signals: List[SMCSignal] = st.session_state.get("scan_results", [])
    now_eval = dt.datetime.now(IST)

    active_signals: List[SMCSignal] = []
    for s in raw_signals:
        age_sec = max(0, int((now_eval - s.candle_time).total_seconds()))
        if age_sec <= 15 * 60:
            s.age_seconds = age_sec
            s.status = "LIVE" if age_sec <= 5 * 60 else "STALE"
            active_signals.append(s)

    st.session_state["scan_results"] = active_signals

    if not active_signals:
        st.info("No active SMC signals present. Click the START SCAN button above.")
    else:
        st.subheader(f"🎯 Confirmed SMC Setups ({len(active_signals)} Active)")
        table_rows = []
        for s in active_signals:
            table_rows.append({
                "Symbol": s.symbol,
                "Direction": s.direction,
                "Score": f"{s.score}/100",
                "Status": s.status,
                "Candle Closed": s.candle_time.strftime("%H:%M:%S"),
                "Age": f"{s.age_seconds // 60}m {s.age_seconds % 60}s",
                "Entry (₹)": s.entry,
                "Stop Loss (₹)": s.sl,
                "Target 1 (₹)": s.tp1,
                "Target 2 (₹)": s.tp2,
                "R:R": f"1:{s.rr:.2f}",
                "Volume (x)": f"{s.volume_ratio:.2f}x",
                "HTF Bias": s.htf_bias,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.subheader("🔍 Interactive Chart Inspector")
        signal_symbols = [s.symbol for s in active_signals]
        selected_sym = st.selectbox("Select Setup", signal_symbols)
        selected_sig = next(s for s in active_signals if s.symbol == selected_sym)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Symbol", selected_sig.symbol, selected_sig.direction)
        col_m2.metric("Confluence Score", f"{selected_sig.score}/100")
        col_m3.metric("Entry Price", f"₹{selected_sig.entry}")
        col_m4.metric("Stop Loss", f"₹{selected_sig.sl}")
        col_m5.metric("Target 2", f"₹{selected_sig.tp2} (1:{selected_sig.rr})")

        inst_match = next((i for i in all_instruments if i["symbol"] == selected_sym), None)
        if inst_match:
            with st.spinner(f"Loading chart for {selected_sym}..."):
                chart_df, _ = client.get_candles(
                    inst_match["instrument_key"], 5, now_eval - dt.timedelta(days=2), now_eval
                )
                chart_df = filter_completed_candles(chart_df, 5, now_eval)
                if not chart_df.empty:
                    fig = build_candle_chart(chart_df, selected_sig)
                    st.plotly_chart(fig, use_container_width=True)

    # Diagnostics
    failures: List[FailedSymbolDiag] = st.session_state.get("failed_diagnostics", [])
    if failures:
        with st.expander(f"⚠️ Scan Diagnostics ({len(failures)} Skipped / Errors)"):
            fail_rows = [
                {
                    "Symbol": f.symbol,
                    "Stage": f.stage,
                    "HTTP Status": f.http_status or "N/A",
                    "Error Type": f.error_type,
                    "Reason": f.reason
                }
                for f in failures
            ]
            st.dataframe(pd.DataFrame(fail_rows), use_container_width=True, hide_index=True)

    st.caption("⚠️ Smart Money Concepts Intraday Scanner. Purely algorithmic analysis; not investment advice.")


if __name__ == "__main__":
    main()
