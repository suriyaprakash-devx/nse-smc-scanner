"""
================================================================================
NSE INTRADAY PURE SMC SCANNER - STRUCTURAL PRODUCTION BUILD
================================================================================

PURE SMC:
    15M CONFIRMED STRUCTURE
        ↓
    HTF BIAS
        ↓
    5M LIQUIDITY SWEEP
        ↓
    DISPLACEMENT
        ↓
    BOS / CHOCH
        ↓
    FVG / ORDER BLOCK
        ↓
    PRICE LEAVES ZONE
        ↓
    GENUINE ZONE RETEST
        ↓
    REJECTION / CONFIRMATION
        ↓
    VOLUME GATE
        ↓
    STRUCTURAL SL
        ↓
    RR VALIDATION
        ↓
    BUY / SELL SIGNAL

IMPORTANT:
- No EMA
- No VWAP
- No MACD
- No RSI
- No Supertrend
- No automatic order placement
- Closed candles only
- Confirmed swings only
- No look-ahead in live decision
- FVG/OB tied to displacement/structure sequence
- Genuine retest required
- Sweep must happen before displacement
- Displacement must happen before BOS/CHOCH
- BOS/CHOCH must happen before zone retest
================================================================================
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import logging.handlers
import math
import os
import random
import threading
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

API_BASE = "https://api.upstox.com"

NSE_INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = dt.time(9, 15)
MARKET_CLOSE = dt.time(15, 30)
CLEANUP_TIME = dt.time(15, 40)

TEMP_DIR = os.environ.get("SCANNER_TEMP_DIR", "temporary_data")
STATE_FILE = os.path.join(TEMP_DIR, "scanner_state.json")

LOG_DIR = os.path.join(TEMP_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "scanner.log")

MAX_RETRIES = 3

DEFAULT_THREADS = 8

RATE_LIMIT = 12.0


# =============================================================================
# 2. LOGGING
# =============================================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("PURE_SMC_SCANNER")

if not logger.handlers:

    logger.setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=15 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] "
            "[Thread-%(thread)d] %(message)s"
        )
    )

    logger.addHandler(handler)


# =============================================================================
# 3. RATE LIMITER
# =============================================================================

class TokenBucket:

    def __init__(
        self,
        rate: float = RATE_LIMIT,
        capacity: float = 20.0,
    ):

        self.rate = rate
        self.capacity = capacity

        self.tokens = capacity

        self.last_update = time.monotonic()

        self.lock = threading.Lock()

    def acquire(self):

        while True:

            with self.lock:

                now = time.monotonic()

                elapsed = now - self.last_update

                self.last_update = now

                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate,
                )

                if self.tokens >= 1:

                    self.tokens -= 1

                    return

                wait = (1 - self.tokens) / self.rate

            time.sleep(max(wait, 0.005))


RATE_LIMITER = TokenBucket()

THREAD_LOCAL = threading.local()


def get_session(token: str) -> requests.Session:

    existing_token = getattr(
        THREAD_LOCAL,
        "token",
        None,
    )

    if not hasattr(THREAD_LOCAL, "session"):

        session = requests.Session()

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=25,
            pool_maxsize=25,
            max_retries=1,
        )

        session.mount("https://", adapter)

        THREAD_LOCAL.session = session

        THREAD_LOCAL.token = token

    elif existing_token != token:

        THREAD_LOCAL.token = token

    THREAD_LOCAL.session.headers.update(
        {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token.strip()}",
        }
    )

    return THREAD_LOCAL.session


# =============================================================================
# 4. ENUMS
# =============================================================================

class SetupStage(str, Enum):

    IDLE = "IDLE"

    SWEEP_CONFIRMED = "SWEEP_CONFIRMED"

    DISPLACEMENT_DETECTED = "DISPLACEMENT_DETECTED"

    STRUCTURE_BROKEN = "STRUCTURE_BROKEN"

    ZONE_CREATED = "ZONE_CREATED"

    ZONE_LEFT = "ZONE_LEFT"

    RETEST_CONFIRMED = "RETEST_CONFIRMED"

    ENTRY_READY = "ENTRY_READY"


# =============================================================================
# 5. DATA MODELS
# =============================================================================

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

    direction: str

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

    status: str

    reason: str

    components: Dict[str, bool]

    sweep_level: float

    sweep_time: dt.datetime

    displacement_time: dt.datetime

    structure_time: dt.datetime

    zone_type: str

    zone_top: float

    zone_bottom: float

    timeframe: str = "5M Entry / 15M HTF"


# =============================================================================
# 6. TIME
# =============================================================================

def now_ist() -> dt.datetime:

    return dt.datetime.now(IST)


def is_market_open(now: Optional[dt.datetime] = None) -> bool:

    now = now or now_ist()

    if now.weekday() >= 5:
        return False

    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def filter_completed_candles(
    df: pd.DataFrame,
    timeframe_minutes: int,
    now: dt.datetime,
) -> pd.DataFrame:

    if df.empty:
        return df

    data = df.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    ).dt.tz_convert(IST)

    candle_end = (
        data["timestamp"]
        + dt.timedelta(minutes=timeframe_minutes)
    )

    data = data[candle_end <= now]

    return (
        data
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# =============================================================================
# 7. UPSTOX CLIENT
# =============================================================================

class UpstoxClient:

    def __init__(self, token: str):

        self.token = token.strip()

    def request(
        self,
        url: str,
        timeout: int = 20,
    ) -> Tuple[
        Optional[requests.Response],
        Optional[FailedSymbolDiag],
    ]:

        for attempt in range(MAX_RETRIES):

            RATE_LIMITER.acquire()

            session = get_session(self.token)

            try:

                response = session.get(
                    url,
                    timeout=timeout,
                )

                if response.status_code == 200:

                    return response, None

                if response.status_code == 429:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:

                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = 1.5
                    else:

                        delay = (
                            1.5 ** attempt
                            + random.uniform(0.1, 0.4)
                        )

                    time.sleep(delay)

                    continue

                if response.status_code in (401, 403):

                    return None, FailedSymbolDiag(
                        "AUTH",
                        "API_AUTH",
                        response.status_code,
                        "AuthError",
                        "Access token expired or unauthorized.",
                    )

                return None, FailedSymbolDiag(
                    "API",
                    "HTTP",
                    response.status_code,
                    "HTTPError",
                    f"HTTP {response.status_code}",
                )

            except requests.RequestException as exc:

                if attempt == MAX_RETRIES - 1:

                    return None, FailedSymbolDiag(
                        "NETWORK",
                        "NETWORK",
                        None,
                        "RequestException",
                        str(exc),
                    )

                time.sleep(
                    0.5 * (attempt + 1)
                )

        return None, FailedSymbolDiag(
            "API",
            "RATE_LIMIT",
            429,
            "RateLimitError",
            "Maximum retries exceeded.",
        )

    # -------------------------------------------------------------------------
    # AUTH
    # -------------------------------------------------------------------------

    def validate_connection(
        self,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:

        response, diag = self.request(
            f"{API_BASE}/v2/user/profile",
            timeout=10,
        )

        if response is None:

            return (
                False,
                diag.reason if diag else "Authentication failed.",
                None,
            )

        try:

            profile = response.json().get(
                "data",
                {},
            )

        except Exception:

            profile = {}

        return (
            True,
            "Upstox authentication successful.",
            profile,
        )

    # -------------------------------------------------------------------------
    # NSE EQUITIES
    # -------------------------------------------------------------------------

    def get_nse_equities(
        self,
    ) -> List[Dict[str, Any]]:

        try:

            response = requests.get(
                NSE_INSTRUMENT_URL,
                timeout=45,
            )

            response.raise_for_status()

            raw = gzip.decompress(
                response.content
            )

            data = json.loads(
                raw.decode("utf-8")
            )

            result = {}

            for item in data:

                if not isinstance(item, dict):
                    continue

                if item.get("segment") != "NSE_EQ":
                    continue

                if item.get("instrument_type") != "EQ":
                    continue

                if item.get("exchange") != "NSE":
                    continue

                symbol = item.get(
                    "trading_symbol"
                )

                key = item.get(
                    "instrument_key"
                )

                if not symbol or not key:
                    continue

                result[key] = {
                    "symbol": symbol,
                    "instrument_key": key,
                    "name": item.get(
                        "name",
                        "",
                    ),
                    "isin": item.get(
                        "isin",
                        "",
                    ),
                }

            return list(result.values())

        except Exception as exc:

            logger.exception(
                "NSE instrument download failed: %s",
                exc,
            )

            return []

    # -------------------------------------------------------------------------
    # HISTORICAL CANDLES
    # -------------------------------------------------------------------------

    def get_candles(
        self,
        instrument_key: str,
        minutes: int,
        start: dt.datetime,
        end: dt.datetime,
    ) -> Tuple[
        pd.DataFrame,
        Optional[FailedSymbolDiag],
    ]:

        encoded = quote(
            instrument_key,
            safe="",
        )

        from_date = start.strftime(
            "%Y-%m-%d"
        )

        to_date = end.strftime(
            "%Y-%m-%d"
        )

        url = (
            f"{API_BASE}/v3/historical-candle/"
            f"{encoded}/minutes/{minutes}/"
            f"{to_date}/{from_date}"
        )

        response, diag = self.request(
            url,
            timeout=20,
        )

        if response is None:

            return pd.DataFrame(), diag

        try:

            payload = response.json()

            candles = (
                payload
                .get("data", {})
                .get("candles", [])
            )

            rows = []

            for candle in candles:

                if len(candle) < 6:
                    continue

                rows.append(
                    {
                        "timestamp": candle[0],
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
                )

            if not rows:

                return (
                    pd.DataFrame(),
                    None,
                )

            df = pd.DataFrame(rows)

            df["timestamp"] = (
                pd.to_datetime(
                    df["timestamp"],
                    utc=True,
                )
                .dt
                .tz_convert(IST)
            )

            df = (
                df
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
                .drop_duplicates(
                    "timestamp"
                )
                .sort_values(
                    "timestamp"
                )
                .reset_index(drop=True)
            )

            return df, None

        except Exception as exc:

            return (
                pd.DataFrame(),
                FailedSymbolDiag(
                    instrument_key,
                    "CANDLE_PARSE",
                    200,
                    "ParseError",
                    str(exc),
                ),
            )


# =============================================================================
# 8. CONFIRMED SWINGS
# =============================================================================

def confirmed_swings(
    df: pd.DataFrame,
    length: int,
    current_bar: Optional[int] = None,
) -> Dict[str, List[int]]:

    if current_bar is None:
        current_bar = len(df) - 1

    highs = df["high"].to_numpy()

    lows = df["low"].to_numpy()

    swing_highs = []

    swing_lows = []

    last_index = min(
        len(df) - length - 1,
        current_bar - length,
    )

    if last_index < length:
        return {
            "high": [],
            "low": [],
        }

    for i in range(
        length,
        last_index + 1,
    ):

        high_window = highs[
            i - length:
            i + length + 1
        ]

        low_window = lows[
            i - length:
            i + length + 1
        ]

        center_high = highs[i]

        center_low = lows[i]

        if (
            center_high == high_window.max()
            and np.sum(
                high_window == center_high
            ) == 1
        ):

            swing_highs.append(i)

        if (
            center_low == low_window.min()
            and np.sum(
                low_window == center_low
            ) == 1
        ):

            swing_lows.append(i)

    return {
        "high": swing_highs,
        "low": swing_lows,
    }


# =============================================================================
# 9. STRUCTURE ENGINE
# =============================================================================

def structure_events(
    df: pd.DataFrame,
    swing_length: int,
    current_bar: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]],
    Optional[str],
]:

    if current_bar is None:
        current_bar = len(df) - 1

    swings = confirmed_swings(
        df,
        swing_length,
        current_bar,
    )

    swing_points = []

    for idx in swings["high"]:

        swing_points.append(
            (
                idx,
                "high",
                float(df.loc[idx, "high"]),
            )
        )

    for idx in swings["low"]:

        swing_points.append(
            (
                idx,
                "low",
                float(df.loc[idx, "low"]),
            )
        )

    swing_points.sort(
        key=lambda x: x[0]
    )

    last_high = None
    last_low = None

    last_high_idx = None
    last_low_idx = None

    broken_high = set()
    broken_low = set()

    trend = None

    events = []

    pointer = 0

    closes = df["close"].to_numpy()

    for idx in range(
        current_bar + 1
    ):

        # -------------------------------------------------------------
        # ONLY REVEAL SWING AFTER CONFIRMATION
        # -------------------------------------------------------------

        while (
            pointer < len(swing_points)
            and swing_points[pointer][0]
            + swing_length
            <= idx
        ):

            sidx, kind, price = (
                swing_points[pointer]
            )

            if kind == "high":

                last_high = price
                last_high_idx = sidx

            else:

                last_low = price
                last_low_idx = sidx

            pointer += 1

        close = float(
            closes[idx]
        )

        # -------------------------------------------------------------
        # BULLISH STRUCTURE BREAK
        # -------------------------------------------------------------

        if (
            last_high is not None
            and last_high_idx is not None
            and close > last_high
            and last_high_idx not in broken_high
        ):

            previous_trend = trend

            event_kind = (
                "BOS"
                if previous_trend == "bull"
                else "CHOCH"
                if previous_trend == "bear"
                else "BOS"
            )

            events.append(
                {
                    "idx": idx,
                    "timestamp": df.loc[
                        idx,
                        "timestamp",
                    ],
                    "type": "bull",
                    "kind": event_kind,
                    "broken_level": last_high,
                    "broken_swing_idx": last_high_idx,
                    "close": close,
                }
            )

            broken_high.add(
                last_high_idx
            )

            trend = "bull"

        # -------------------------------------------------------------
        # BEARISH STRUCTURE BREAK
        # -------------------------------------------------------------

        if (
            last_low is not None
            and last_low_idx is not None
            and close < last_low
            and last_low_idx not in broken_low
        ):

            previous_trend = trend

            event_kind = (
                "BOS"
                if previous_trend == "bear"
                else "CHOCH"
                if previous_trend == "bull"
                else "BOS"
            )

            events.append(
                {
                    "idx": idx,
                    "timestamp": df.loc[
                        idx,
                        "timestamp",
                    ],
                    "type": "bear",
                    "kind": event_kind,
                    "broken_level": last_low,
                    "broken_swing_idx": last_low_idx,
                    "close": close,
                }
            )

            broken_low.add(
                last_low_idx
            )

            trend = "bear"

    return events, trend


# =============================================================================
# 10. LIQUIDITY SWEEP
# =============================================================================

def detect_latest_sweep(
    df: pd.DataFrame,
    swing_length: int,
    direction: str,
    current_bar: int,
    max_age: int,
) -> Optional[Dict[str, Any]]:

    swings = confirmed_swings(
        df,
        swing_length,
        current_bar,
    )

    candidates = (
        swings["low"]
        if direction == "bull"
        else swings["high"]
    )

    candidates = sorted(
        candidates,
        reverse=True,
    )

    current_time = df.loc[
        current_bar,
        "timestamp",
    ]

    for swing_idx in candidates:

        # -------------------------------------------------------------
        # SWING MUST ALREADY BE CONFIRMED
        # -------------------------------------------------------------

        confirmation_idx = (
            swing_idx + swing_length
        )

        if confirmation_idx >= current_bar:
            continue

        if (
            current_bar - swing_idx
            > max_age * 3
        ):
            continue

        level = float(
            df.loc[
                swing_idx,
                "low" if direction == "bull"
                else "high",
            ]
        )

        # -------------------------------------------------------------
        # SEARCH ONLY AFTER CONFIRMATION
        # -------------------------------------------------------------

        start = confirmation_idx + 1

        for i in range(
            start,
            current_bar + 1,
        ):

            high = float(
                df.loc[i, "high"]
            )

            low = float(
                df.loc[i, "low"]
            )

            close = float(
                df.loc[i, "close"]
            )

            if direction == "bull":

                # SELL-SIDE LIQUIDITY TAKEN
                # THEN CLOSES BACK ABOVE LEVEL

                if (
                    low < level
                    and close > level
                ):

                    return {
                        "direction": "bull",
                        "swing_idx": swing_idx,
                        "sweep_idx": i,
                        "level": level,
                        "wick": low,
                        "timestamp": df.loc[
                            i,
                            "timestamp",
                        ],
                    }

            else:

                # BUY-SIDE LIQUIDITY TAKEN
                # THEN CLOSES BACK BELOW LEVEL

                if (
                    high > level
                    and close < level
                ):

                    return {
                        "direction": "bear",
                        "swing_idx": swing_idx,
                        "sweep_idx": i,
                        "level": level,
                        "wick": high,
                        "timestamp": df.loc[
                            i,
                            "timestamp",
                        ],
                    }

    return None


# =============================================================================
# 11. DISPLACEMENT
# =============================================================================

def detect_displacement_after(
    df: pd.DataFrame,
    start_idx: int,
    direction: str,
    multiplier: float,
    lookback: int = 5,
    max_bars: int = 10,
) -> Optional[Dict[str, Any]]:

    bodies = (
        df["close"] -
        df["open"]
    ).abs()

    end_idx = min(
        len(df) - 1,
        start_idx + max_bars,
    )

    for i in range(
        start_idx + 1,
        end_idx + 1,
    ):

        if i < lookback:
            continue

        average_body = float(
            bodies.iloc[
                i - lookback:i
            ].mean()
        )

        current_body = float(
            bodies.iloc[i]
        )

        if average_body <= 0:
            continue

        bullish = (
            df.loc[i, "close"]
            > df.loc[i, "open"]
        )

        bearish = (
            df.loc[i, "close"]
            < df.loc[i, "open"]
        )

        correct_direction = (
            bullish
            if direction == "bull"
            else bearish
        )

        if (
            correct_direction
            and current_body
            >= average_body * multiplier
        ):

            return {
                "idx": i,
                "timestamp": df.loc[
                    i,
                    "timestamp",
                ],
                "body": current_body,
                "average_body": average_body,
                "ratio": current_body
                / average_body,
            }

    return None


# =============================================================================
# 12. FVG
# =============================================================================

def detect_fvg_created_by_sequence(
    df: pd.DataFrame,
    displacement_idx: int,
    structure_idx: int,
    direction: str,
) -> Optional[Dict[str, Any]]:

    start = max(
        2,
        displacement_idx,
    )

    end = min(
        structure_idx + 2,
        len(df) - 1,
    )

    for i in range(
        start,
        end + 1,
    ):

        candle = df.iloc[i]

        two_back = df.iloc[i - 2]

        # -------------------------------------------------------------
        # BULLISH FVG
        # -------------------------------------------------------------

        if direction == "bull":

            if (
                float(candle["low"])
                > float(two_back["high"])
            ):

                return {
                    "type": "FVG",
                    "direction": "bull",
                    "top": float(
                        candle["low"]
                    ),
                    "bottom": float(
                        two_back["high"]
                    ),
                    "created_idx": i,
                }

        # -------------------------------------------------------------
        # BEARISH FVG
        # -------------------------------------------------------------

        else:

            if (
                float(candle["high"])
                < float(two_back["low"])
            ):

                return {
                    "type": "FVG",
                    "direction": "bear",
                    "top": float(
                        two_back["low"]
                    ),
                    "bottom": float(
                        candle["high"]
                    ),
                    "created_idx": i,
                }

    return None


# =============================================================================
# 13. ORDER BLOCK
# =============================================================================

def detect_order_block(
    df: pd.DataFrame,
    displacement_idx: int,
    direction: str,
) -> Optional[Dict[str, Any]]:

    # Look backward for the LAST opposite candle
    # immediately before displacement.

    start = max(
        0,
        displacement_idx - 6,
    )

    for i in range(
        displacement_idx - 1,
        start - 1,
        -1,
    ):

        candle = df.iloc[i]

        bullish_candle = (
            candle["close"]
            > candle["open"]
        )

        bearish_candle = (
            candle["close"]
            < candle["open"]
        )

        if direction == "bull":

            if bearish_candle:

                return {
                    "type": "OB",
                    "direction": "bull",
                    "top": float(
                        candle["high"]
                    ),
                    "bottom": float(
                        candle["low"]
                    ),
                    "created_idx": i,
                }

        else:

            if bullish_candle:

                return {
                    "type": "OB",
                    "direction": "bear",
                    "top": float(
                        candle["high"]
                    ),
                    "bottom": float(
                        candle["low"]
                    ),
                    "created_idx": i,
                }

    return None


# =============================================================================
# 14. ZONE INVALIDATION
# =============================================================================

def zone_invalidated(
    df: pd.DataFrame,
    zone: Dict[str, Any],
    current_bar: int,
) -> bool:

    direction = zone["direction"]

    created = zone["created_idx"]

    top = float(zone["top"])

    bottom = float(zone["bottom"])

    if created >= current_bar:
        return False

    future = df.iloc[
        created + 1:
        current_bar + 1
    ]

    if future.empty:
        return False

    if direction == "bull":

        # Strong close below entire zone
        return bool(
            (
                future["close"]
                < bottom
            ).any()
        )

    else:

        # Strong close above entire zone
        return bool(
            (
                future["close"]
                > top
            ).any()
        )


# =============================================================================
# 15. TRUE ZONE RETEST
# =============================================================================

def find_true_zone_retest(
    df: pd.DataFrame,
    zone: Dict[str, Any],
    current_bar: int,
    direction: str,
    max_zone_age: int,
    tolerance: float = 0.001,
) -> Optional[Dict[str, Any]]:

    created = int(
        zone["created_idx"]
    )

    top = float(
        zone["top"]
    )

    bottom = float(
        zone["bottom"]
    )

    # -------------------------------------------------------------
    # Zone must have existed before current candle
    # -------------------------------------------------------------

    if created >= current_bar:
        return None

    if (
        current_bar - created
        > max_zone_age
    ):
        return None

    # -------------------------------------------------------------
    # INVALIDATION
    # -------------------------------------------------------------

    if zone_invalidated(
        df,
        zone,
        current_bar,
    ):

        return None

    # -------------------------------------------------------------
    # CRITICAL:
    # PRICE MUST LEAVE THE ZONE FIRST
    # -------------------------------------------------------------

    left_zone = False

    leave_idx = None

    for i in range(
        created + 1,
        current_bar,
    ):

        close = float(
            df.loc[i, "close"]
        )

        if direction == "bull":

            if close > top * (
                1 + tolerance
            ):

                left_zone = True
                leave_idx = i
                break

        else:

            if close < bottom * (
                1 - tolerance
            ):

                left_zone = True
                leave_idx = i
                break

    if not left_zone:
        return None

    # -------------------------------------------------------------
    # NOW SEARCH FOR RETURN INTO ZONE
    # -------------------------------------------------------------

    for i in range(
        leave_idx + 1,
        current_bar + 1,
    ):

        high = float(
            df.loc[i, "high"]
        )

        low = float(
            df.loc[i, "low"]
        )

        close = float(
            df.loc[i, "close"]
        )

        open_price = float(
            df.loc[i, "open"]
        )

        overlaps = (
            low <= top
            and high >= bottom
        )

        if not overlaps:
            continue

        # ---------------------------------------------------------
        # BULLISH REJECTION
        # ---------------------------------------------------------

        if direction == "bull":

            close_back_above = (
                close >= bottom
            )

            bullish_close = (
                close > open_price
            )

            # Rejection should finish in/above zone,
            # preferably bullish.

            if (
                close_back_above
                and bullish_close
            ):

                return {
                    "idx": i,
                    "timestamp": df.loc[
                        i,
                        "timestamp",
                    ],
                    "close": close,
                    "zone_top": top,
                    "zone_bottom": bottom,
                }

        # ---------------------------------------------------------
        # BEARISH REJECTION
        # ---------------------------------------------------------

        else:

            close_back_below = (
                close <= top
            )

            bearish_close = (
                close < open_price
            )

            if (
                close_back_below
                and bearish_close
            ):

                return {
                    "idx": i,
                    "timestamp": df.loc[
                        i,
                        "timestamp",
                    ],
                    "close": close,
                    "zone_top": top,
                    "zone_bottom": bottom,
                }

    return None


# =============================================================================
# 16. VOLUME
# =============================================================================

def volume_ratio(
    df: pd.DataFrame,
    current_idx: int,
    lookback: int,
) -> float:

    if current_idx < lookback:
        return 0.0

    previous = df["volume"].iloc[
        current_idx - lookback:
        current_idx
    ]

    average = float(
        previous.mean()
    )

    if (
        not math.isfinite(average)
        or average <= 0
    ):
        return 0.0

    current = float(
        df.loc[
            current_idx,
            "volume",
        ]
    )

    return current / average


# =============================================================================
# 17. STRUCTURAL SL
# =============================================================================

def calculate_structural_sl(
    df: pd.DataFrame,
    direction: str,
    sweep: Dict[str, Any],
    zone: Dict[str, Any],
    retest_idx: int,
) -> Optional[float]:

    entry = float(
        df.loc[
            retest_idx,
            "close",
        ]
    )

    buffer = entry * 0.0005

    if direction == "bull":

        # Most relevant invalidation is below
        # sweep extreme and zone.

        candidates = [
            float(sweep["wick"]),
            float(zone["bottom"]),
        ]

        sl = min(candidates) - buffer

        if sl >= entry:
            return None

        return sl

    else:

        candidates = [
            float(sweep["wick"]),
            float(zone["top"]),
        ]

        sl = max(candidates) + buffer

        if sl <= entry:
            return None

        return sl


# =============================================================================
# 18. SMC QUALITY SCORE
# =============================================================================

def calculate_quality_score(
    direction: str,
    sweep: Dict[str, Any],
    displacement: Dict[str, Any],
    structure: Dict[str, Any],
    zone_type: str,
    volume: float,
) -> int:

    score = 0

    # -------------------------------------------------------------
    # HTF alignment
    # -------------------------------------------------------------

    score += 20

    # -------------------------------------------------------------
    # Liquidity sweep
    # -------------------------------------------------------------

    score += 20

    # -------------------------------------------------------------
    # Displacement strength
    # -------------------------------------------------------------

    ratio = float(
        displacement["ratio"]
    )

    if ratio >= 2.0:
        score += 20

    elif ratio >= 1.5:
        score += 15

    else:
        score += 10

    # -------------------------------------------------------------
    # BOS / CHOCH
    # -------------------------------------------------------------

    if structure["kind"] == "CHOCH":
        score += 20
    else:
        score += 15

    # -------------------------------------------------------------
    # Zone
    # -------------------------------------------------------------

    if zone_type == "FVG":
        score += 15

    elif zone_type == "OB":
        score += 12

    # -------------------------------------------------------------
    # Volume
    # -------------------------------------------------------------

    if volume >= 2.5:
        score += 5

    elif volume >= 2.0:
        score += 4

    elif volume >= 1.5:
        score += 3

    return min(
        100,
        int(score),
    )


# =============================================================================
# 19. HTF BIAS
# =============================================================================

def get_htf_bias(
    df: pd.DataFrame,
    swing_length: int,
) -> Optional[str]:

    df = df.reset_index(
        drop=True
    )

    if len(df) < (
        swing_length * 2 + 10
    ):
        return None

    _, trend = structure_events(
        df,
        swing_length,
    )

    return trend


# =============================================================================
# 20. COMPLETE PURE SMC SETUP
# =============================================================================

def evaluate_smc_setup(
    symbol: str,
    htf_df: pd.DataFrame,
    ltf_df: pd.DataFrame,
    settings: Dict[str, Any],
    now: dt.datetime,
) -> Optional[SMCSignal]:

    swing_length = int(
        settings["swing_length"]
    )

    htf = filter_completed_candles(
        htf_df,
        15,
        now,
    )

    ltf = filter_completed_candles(
        ltf_df,
        5,
        now,
    )

    if htf.empty or ltf.empty:
        return None

    minimum = (
        swing_length * 2 + 10
    )

    if (
        len(htf) < minimum
        or len(ltf) < minimum
    ):
        return None

    current_bar = len(ltf) - 1

    # =================================================================
    # 1. HTF STRUCTURE
    # =================================================================

    htf_bias = get_htf_bias(
        htf,
        swing_length,
    )

    if htf_bias not in (
        "bull",
        "bear",
    ):
        return None

    direction = htf_bias

    # =================================================================
    # 2. LTF LIQUIDITY SWEEP
    # =================================================================

    sweep = detect_latest_sweep(
        ltf,
        swing_length,
        direction,
        current_bar,
        int(settings["max_sweep_bars"]),
    )

    if sweep is None:
        return None

    sweep_idx = int(
        sweep["sweep_idx"]
    )

    # The sweep must not be the current candle
    # if there is no subsequent sequence.
    if sweep_idx >= current_bar:
        return None

    # =================================================================
    # 3. DISPLACEMENT
    # =================================================================

    displacement = detect_displacement_after(
        ltf,
        sweep_idx,
        direction,
        float(
            settings[
                "displacement_multiplier"
            ]
        ),
        lookback=5,
        max_bars=int(
            settings[
                "max_displacement_bars"
            ]
        ),
    )

    if displacement is None:
        return None

    displacement_idx = int(
        displacement["idx"]
    )

    if displacement_idx >= current_bar:
        return None

    # =================================================================
    # 4. STRUCTURE BREAK
    # =================================================================

    events, _ = structure_events(
        ltf,
        swing_length,
        current_bar,
    )

    structure_candidates = [
        e
        for e in events
        if e["type"] == direction
        and displacement_idx
        <= e["idx"]
        <= min(
            current_bar,
            displacement_idx
            + int(
                settings[
                    "max_structure_bars"
                ]
            ),
        )
    ]

    if not structure_candidates:
        return None

    # Latest valid structure event
    structure = structure_candidates[-1]

    structure_idx = int(
        structure["idx"]
    )

    if structure_idx >= current_bar:
        return None

    # =================================================================
    # 5. CREATE FVG
    # =================================================================

    fvg = detect_fvg_created_by_sequence(
        ltf,
        displacement_idx,
        structure_idx,
        direction,
    )

    # =================================================================
    # 6. CREATE ORDER BLOCK
    # =================================================================

    ob = detect_order_block(
        ltf,
        displacement_idx,
        direction,
    )

    # =================================================================
    # 7. SELECT BEST ZONE
    # =================================================================

    zones = []

    if fvg is not None:
        zones.append(fvg)

    if ob is not None:
        zones.append(ob)

    if not zones:
        return None

    # Prefer FVG because it is directly price-imbalance based.
    zones.sort(
        key=lambda z: (
            0
            if z["type"] == "FVG"
            else 1
        )
    )

    selected_zone = None
    retest = None

    # =================================================================
    # 8. TRUE RETEST
    # =================================================================

    for zone in zones:

        result = find_true_zone_retest(
            ltf,
            zone,
            current_bar,
            direction,
            int(
                settings[
                    "max_zone_age_bars"
                ]
            ),
            tolerance=0.001,
        )

        if result is not None:

            selected_zone = zone
            retest = result

            break

    if (
        selected_zone is None
        or retest is None
    ):
        return None

    retest_idx = int(
        retest["idx"]
    )

    # =================================================================
    # 9. VOLUME HARD GATE
    # =================================================================

    vol = volume_ratio(
        ltf,
        retest_idx,
        int(
            settings[
                "volume_lookback"
            ]
        ),
    )

    if vol < float(
        settings[
            "min_volume_mult"
        ]
    ):
        return None

    # =================================================================
    # 10. ENTRY
    # =================================================================

    entry = float(
        ltf.loc[
            retest_idx,
            "close",
        ]
    )

    # =================================================================
    # 11. STRUCTURAL STOP
    # =================================================================

    sl = calculate_structural_sl(
        ltf,
        direction,
        sweep,
        selected_zone,
        retest_idx,
    )

    if sl is None:
        return None

    risk = abs(
        entry - sl
    )

    if risk <= 0:
        return None

    max_sl = (
        entry
        * float(
            settings["max_sl_pct"]
        )
        / 100
    )

    if risk > max_sl:
        return None

    # =================================================================
    # 12. TARGETS
    # =================================================================

    target_rr = float(
        settings["target_rr"]
    )

    min_rr = float(
        settings["min_rr"]
    )

    if direction == "bull":

        tp1 = entry + (
            risk * 1.5
        )

        tp2 = entry + (
            risk * target_rr
        )

    else:

        tp1 = entry - (
            risk * 1.5
        )

        tp2 = entry - (
            risk * target_rr
        )

    actual_rr = (
        abs(tp2 - entry)
        / risk
    )

    if actual_rr < min_rr:
        return None

    # =================================================================
    # 13. QUALITY SCORE
    # =================================================================

    score = calculate_quality_score(
        direction,
        sweep,
        displacement,
        structure,
        selected_zone["type"],
        vol,
    )

    if score < int(
        settings["min_score"]
    ):
        return None

    # =================================================================
    # 14. SIGNAL AGE
    # =================================================================

    candle_time = ltf.loc[
        retest_idx,
        "timestamp",
    ]

    age_seconds = max(
        0,
        int(
            (
                now - candle_time
            ).total_seconds()
        ),
    )

    if age_seconds <= 300:

        status = "LIVE"

    elif age_seconds <= 900:

        status = "STALE"

    else:

        return None

    # =================================================================
    # 15. COMPONENTS
    # =================================================================

    components = {
        "htf_structure": True,
        "liquidity_sweep": True,
        "displacement": True,
        "bos_choch": True,
        "fvg_or_ob": True,
        "zone_left": True,
        "zone_retest": True,
        "rejection": True,
        "volume_gate": True,
        "rr_valid": True,
    }

    structure_label = (
        structure["kind"]
    )

    zone_name = selected_zone[
        "type"
    ]

    reason = (
        f"{'BULLISH' if direction == 'bull' else 'BEARISH'} "
        f"15M structure → "
        f"{'SELL-SIDE' if direction == 'bull' else 'BUY-SIDE'} "
        f"liquidity sweep → "
        f"{displacement['ratio']:.2f}x displacement → "
        f"5M {structure_label} → "
        f"{zone_name} creation → "
        f"zone departure → genuine retest → "
        f"rejection. "
        f"Volume {vol:.2f}x."
    )

    # =================================================================
    # 16. FINAL SIGNAL
    # =================================================================

    return SMCSignal(
        symbol=symbol,
        direction=(
            "BUY"
            if direction == "bull"
            else "SELL"
        ),
        entry=round(entry, 2),
        sl=round(sl, 2),
        tp1=round(tp1, 2),
        tp2=round(tp2, 2),
        rr=round(actual_rr, 2),
        score=score,
        htf_bias=(
            "Bullish"
            if direction == "bull"
            else "Bearish"
        ),
        setup_stage=(
            SetupStage.ENTRY_READY.value
        ),
        candle_time=candle_time,
        signal_time=candle_time,
        age_seconds=age_seconds,
        volume_ratio=round(
            vol,
            2,
        ),
        status=status,
        reason=reason,
        components=components,
        sweep_level=float(
            sweep["level"]
        ),
        sweep_time=sweep[
            "timestamp"
        ],
        displacement_time=displacement[
            "timestamp"
        ],
        structure_time=structure[
            "timestamp"
        ],
        zone_type=zone_name,
        zone_top=float(
            selected_zone["top"]
        ),
        zone_bottom=float(
            selected_zone["bottom"]
        ),
    )


# =============================================================================
# 21. SYMBOL SCANNER
# =============================================================================

def scan_symbol(
    client: UpstoxClient,
    instrument: Dict[str, Any],
    settings: Dict[str, Any],
    now: dt.datetime,
) -> Tuple[
    Optional[SMCSignal],
    Optional[FailedSymbolDiag],
]:

    symbol = instrument["symbol"]

    key = instrument[
        "instrument_key"
    ]

    try:

        # -------------------------------------------------------------
        # 15M
        # -------------------------------------------------------------

        htf_df, diag = client.get_candles(
            key,
            15,
            now - dt.timedelta(
                days=5
            ),
            now,
        )

        if htf_df.empty:

            return (
                None,
                diag or FailedSymbolDiag(
                    symbol,
                    "HTF_DATA",
                    200,
                    "EmptyData",
                    "No 15M candles.",
                ),
            )

        # -------------------------------------------------------------
        # 5M
        # -------------------------------------------------------------

        ltf_df, diag = client.get_candles(
            key,
            5,
            now - dt.timedelta(
                days=3
            ),
            now,
        )

        if ltf_df.empty:

            return (
                None,
                diag or FailedSymbolDiag(
                    symbol,
                    "LTF_DATA",
                    200,
                    "EmptyData",
                    "No 5M candles.",
                ),
            )

        ltf_closed = filter_completed_candles(
            ltf_df,
            5,
            now,
        )

        if ltf_closed.empty:
            return None, None

        latest_price = float(
            ltf_closed["close"].iloc[-1]
        )

        if latest_price < float(
            settings["min_stock_price"]
        ):
            return None, None

        signal = evaluate_smc_setup(
            symbol,
            htf_df,
            ltf_df,
            settings,
            now,
        )

        return signal, None

    except Exception as exc:

        logger.exception(
            "Scanner error for %s",
            symbol,
        )

        return (
            None,
            FailedSymbolDiag(
                symbol,
                "SCAN",
                None,
                "WorkerError",
                str(exc),
            ),
        )


# =============================================================================
# 22. MARKET SCANNER
# =============================================================================

def run_market_scan(
    client: UpstoxClient,
    instruments: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> Tuple[
    List[SMCSignal],
    List[FailedSymbolDiag],
]:

    results = []

    failures = []

    total = len(
        instruments
    )

    completed = 0

    now = now_ist()

    workers = int(
        settings.get(
            "max_threads",
            DEFAULT_THREADS,
        )
    )

    progress = st.progress(
        0.0
    )

    status = st.empty()

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = {
            executor.submit(
                scan_symbol,
                client,
                instrument,
                settings,
                now,
            ): instrument[
                "symbol"
            ]
            for instrument in instruments
        }

        for future in as_completed(
            futures
        ):

            symbol = futures[
                future
            ]

            completed += 1

            try:

                signal, diag = (
                    future.result()
                )

                if signal:
                    results.append(
                        signal
                    )

                if diag:
                    failures.append(
                        diag
                    )

            except Exception as exc:

                failures.append(
                    FailedSymbolDiag(
                        symbol,
                        "EXECUTOR",
                        None,
                        "WorkerCrash",
                        str(exc),
                    )
                )

            progress.progress(
                completed / max(
                    total,
                    1,
                )
            )

            if (
                completed % 10 == 0
                or completed == total
            ):

                status.write(
                    f"Scanning "
                    f"**{completed}/{total}** | "
                    f"Signals: "
                    f"**{len(results)}** | "
                    f"Failures: "
                    f"**{len(failures)}**"
                )

    progress.empty()

    status.empty()

    # Strongest setups first
    results.sort(
        key=lambda x: (
            x.score,
            x.volume_ratio,
            x.rr,
        ),
        reverse=True,
    )

    return (
        results,
        failures,
    )


# =============================================================================
# 23. UNIVERSE
# =============================================================================

def select_universe(
    instruments: List[Dict[str, Any]],
    mode: str,
    limit: int,
) -> List[Dict[str, Any]]:

    if mode == "ALL NSE EQUITIES":

        return instruments

    if mode == "CUSTOM LIMIT":

        return instruments[
            :min(
                limit,
                len(instruments),
            )
        ]

    # IMPORTANT:
    # The instrument file alone does NOT contain
    # reliable live turnover ranking.
    #
    # Therefore we do not falsely call alphabetic
    # selection "TOP ACTIVE".
    #
    # Priority list only provides deterministic
    # high-liquidity coverage.

    priority_symbols = {
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "INFY",
        "TCS",
        "ITC",
        "SBIN",
        "LT",
        "BHARTIARTL",
        "AXISBANK",
        "KOTAKBANK",
        "BAJFINANCE",
        "M&M",
        "MARUTI",
        "TATAMOTORS",
        "SUNPHARMA",
        "NTPC",
        "ONGC",
        "TITAN",
        "ADANIENT",
        "ADANIPORTS",
        "POWERGRID",
        "TATASTEEL",
        "HINDALCO",
        "JSWSTEEL",
        "BEL",
        "HAL",
        "TRENT",
        "DLF",
        "ZOMATO",
        "PFC",
        "RECLTD",
        "IOC",
        "BPCL",
        "GAIL",
        "VEDL",
        "INDUSINDBK",
        "CIPLA",
        "DRREDDY",
        "DIVISLAB",
        "APOLLOHOSP",
        "EICHERMOT",
        "BAJAJ-AUTO",
        "HEROMOTOCO",
        "TVSMOTOR",
    }

    priority = [
        x
        for x in instruments
        if x["symbol"]
        in priority_symbols
    ]

    remaining = [
        x
        for x in instruments
        if x["symbol"]
        not in priority_symbols
    ]

    remaining.sort(
        key=lambda x: x["symbol"]
    )

    if mode == "NIFTY 50":

        return priority[:50]

    if mode == "NIFTY 100":

        return priority[:100]

    if mode == "TOP 1500 PRIORITY":

        return (
            priority
            + remaining[
                :max(
                    0,
                    1500 - len(priority),
                )
            ]
        )[:1500]

    if mode == "TOP 2000 PRIORITY":

        return (
            priority
            + remaining[
                :max(
                    0,
                    2000 - len(priority),
                )
            ]
        )[:2000]

    return instruments


# =============================================================================
# 24. TELEGRAM
# =============================================================================

def send_telegram(
    token: str,
    chat_id: str,
    signal: SMCSignal,
) -> bool:

    if not token or not chat_id:
        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{token.strip()}/sendMessage"
    )

    message = (
        f"<b>🚨 PURE SMC SIGNAL</b>\n\n"

        f"<b>Symbol:</b> {signal.symbol}\n"
        f"<b>Direction:</b> {signal.direction}\n"
        f"<b>Score:</b> {signal.score}/100\n"
        f"<b>HTF:</b> {signal.htf_bias}\n\n"

        f"<b>Entry:</b> ₹{signal.entry:.2f}\n"
        f"<b>SL:</b> ₹{signal.sl:.2f}\n"
        f"<b>TP1:</b> ₹{signal.tp1:.2f}\n"
        f"<b>TP2:</b> ₹{signal.tp2:.2f}\n"
        f"<b>RR:</b> 1:{signal.rr:.2f}\n\n"

        f"<b>Volume:</b> "
        f"{signal.volume_ratio:.2f}x\n"

        f"<b>Zone:</b> "
        f"{signal.zone_type}\n"

        f"<b>Sweep:</b> "
        f"{signal.sweep_level:.2f}\n"

        f"<b>Candle:</b> "
        f"{signal.candle_time.strftime('%H:%M IST')}\n\n"

        f"<i>{signal.reason}</i>"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id": chat_id.strip(),
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )

        return response.status_code == 200

    except Exception as exc:

        logger.warning(
            "Telegram error: %s",
            exc,
        )

        return False


# =============================================================================
# 25. STATE
# =============================================================================

def load_state() -> Dict[str, Any]:

    os.makedirs(
        TEMP_DIR,
        exist_ok=True,
    )

    if not os.path.exists(
        STATE_FILE
    ):

        return {}

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_state(
    state: Dict[str, Any],
):

    os.makedirs(
        TEMP_DIR,
        exist_ok=True,
    )

    temporary = (
        STATE_FILE
        + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            state,
            file,
            indent=2,
        )

    os.replace(
        temporary,
        STATE_FILE,
    )


# =============================================================================
# 26. DAILY CLEANUP
# =============================================================================

def daily_cleanup():

    now = now_ist()

    if now.time() < CLEANUP_TIME:
        return

    state = load_state()

    today = now.strftime(
        "%Y-%m-%d"
    )

    if state.get(
        "last_cleanup"
    ) == today:

        return

    # Clear Streamlit data
    for key in [
        "scan_results",
        "failed_diagnostics",
        "alert_history",
    ]:

        st.session_state[
            key
        ] = []

    # Delete temporary generated files,
    # but keep scanner log and state file.

    if os.path.exists(
        TEMP_DIR
    ):

        for filename in os.listdir(
            TEMP_DIR
        ):

            path = os.path.join(
                TEMP_DIR,
                filename,
            )

            if (
                os.path.isfile(path)
                and path != STATE_FILE
            ):

                try:
                    os.remove(path)
                except Exception:
                    pass

            elif (
                os.path.isdir(path)
                and path != LOG_DIR
            ):

                import shutil

                try:
                    shutil.rmtree(
                        path
                    )
                except Exception:
                    pass

    state[
        "last_cleanup"
    ] = today

    save_state(
        state
    )

    logger.info(
        "Daily cleanup completed: %s",
        today,
    )


# =============================================================================
# 27. CHART
# =============================================================================

def build_chart(
    df: pd.DataFrame,
    signal: SMCSignal,
) -> go.Figure:

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="5M",
        )
    )

    fig.add_hline(
        y=signal.entry,
        line_dash="dash",
        annotation_text="ENTRY",
    )

    fig.add_hline(
        y=signal.sl,
        line_dash="dash",
        annotation_text="SL",
    )

    fig.add_hline(
        y=signal.tp1,
        line_dash="dot",
        annotation_text="TP1",
    )

    fig.add_hline(
        y=signal.tp2,
        line_dash="dash",
        annotation_text="TP2",
    )

    # Sweep level

    fig.add_hline(
        y=signal.sweep_level,
        line_dash="dot",
        annotation_text="LIQUIDITY",
    )

    # Zone

    fig.add_hrect(
        y0=signal.zone_bottom,
        y1=signal.zone_top,
        opacity=0.20,
        annotation_text=signal.zone_type,
    )

    fig.update_layout(
        title=(
            f"{signal.symbol} | "
            f"Pure SMC 5M / 15M"
        ),
        xaxis_title="Time IST",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    return fig


# =============================================================================
# 28. STREAMLIT
# =============================================================================

def main():

    st.set_page_config(
        page_title="Pure SMC NSE Scanner",
        page_icon="⚡",
        layout="wide",
    )

    defaults = {
        "upstox_token": os.environ.get(
            "UPSTOX_ACCESS_TOKEN",
            "",
        ),
        "connected": False,
        "profile": None,
        "scan_results": [],
        "failed_diagnostics": [],
        "instruments_cache": None,
        "last_scanned_candle": None,
        "last_scan_time": None,
        "alert_history": {},
    }

    for key, value in defaults.items():

        st.session_state.setdefault(
            key,
            value,
        )

    daily_cleanup()

    # =================================================================
    # SIDEBAR
    # =================================================================

    with st.sidebar:

        st.title(
            "⚡ Pure SMC Settings"
        )

        st.subheader(
            "Universe"
        )

        universe_mode = st.selectbox(
            "Universe",
            [
                "TOP 2000 PRIORITY",
                "TOP 1500 PRIORITY",
                "ALL NSE EQUITIES",
                "NIFTY 50",
                "NIFTY 100",
                "CUSTOM LIMIT",
            ],
            index=0,
        )

        custom_limit = 2000

        if universe_mode == "CUSTOM LIMIT":

            custom_limit = st.number_input(
                "Stock Limit",
                min_value=50,
                max_value=3000,
                value=2000,
                step=50,
            )

        min_price = st.number_input(
            "Minimum Stock Price ₹",
            min_value=1.0,
            max_value=5000.0,
            value=25.0,
            step=5.0,
        )

        st.divider()

        # -------------------------------------------------------------
        # SMC
        # -------------------------------------------------------------

        st.subheader(
            "SMC Structure"
        )

        swing_length = st.slider(
            "Confirmed Swing Length",
            2,
            10,
            5,
        )

        displacement_multiplier = st.slider(
            "Displacement Strength",
            1.2,
            3.0,
            1.5,
            0.1,
        )

        max_sweep_bars = st.slider(
            "Maximum Sweep Age",
            5,
            50,
            20,
        )

        max_displacement_bars = st.slider(
            "Sweep → Displacement",
            2,
            20,
            10,
        )

        max_structure_bars = st.slider(
            "Displacement → BOS/CHOCH",
            2,
            30,
            15,
        )

        max_zone_age = st.slider(
            "Maximum Zone Age",
            5,
            60,
            30,
        )

        st.divider()

        # -------------------------------------------------------------
        # VOLUME / RISK
        # -------------------------------------------------------------

        st.subheader(
            "Confirmation Gates"
        )

        volume_lookback = st.number_input(
            "Volume Lookback",
            min_value=5,
            max_value=50,
            value=20,
        )

        min_volume = st.slider(
            "Minimum Volume",
            1.0,
            4.0,
            1.5,
            0.1,
        )

        min_score = st.slider(
            "Minimum SMC Quality",
            50,
            100,
            70,
            5,
        )

        max_sl_pct = st.slider(
            "Maximum SL %",
            0.5,
            3.0,
            1.5,
            0.1,
        )

        min_rr = st.number_input(
            "Minimum R:R",
            min_value=1.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
        )

        target_rr = st.number_input(
            "Target R:R",
            min_value=2.0,
            max_value=6.0,
            value=2.5,
            step=0.5,
        )

        max_threads = st.slider(
            "API Workers",
            4,
            16,
            8,
        )

        st.divider()

        # -------------------------------------------------------------
        # TELEGRAM
        # -------------------------------------------------------------

        st.subheader(
            "Telegram"
        )

        telegram_enabled = st.checkbox(
            "Enable Telegram",
            value=False,
        )

        telegram_token = st.text_input(
            "Bot Token",
            type="password",
        )

        telegram_chat = st.text_input(
            "Chat ID",
        )

    # =================================================================
    # HEADER
    # =================================================================

    st.title(
        "⚡ NSE Pure Smart Money Concepts Scanner"
    )

    st.caption(
        "15M HTF + 5M LTF | "
        "Liquidity Sweep → Displacement → "
        "BOS/CHOCH → FVG/OB Retest"
    )

    current_time = now_ist()

    if current_time.weekday() >= 5:

        st.info(
            "🔴 NSE market closed — weekend."
        )

    elif current_time.time() < MARKET_OPEN:

        st.info(
            "🕘 Waiting for NSE market open."
        )

    elif current_time.time() > MARKET_CLOSE:

        st.info(
            "🔴 NSE market closed."
        )

    else:

        st.success(
            f"🟢 NSE market active | "
            f"{current_time.strftime('%H:%M:%S')} IST"
        )

    # =================================================================
    # UPSTOX
    # =================================================================

    st.subheader(
        "🔑 Upstox Connection"
    )

    token_col, button_col = st.columns(
        [4, 1]
    )

    token_input = token_col.text_input(
        "Access Token",
        value=st.session_state[
            "upstox_token"
        ],
        type="password",
    )

    if button_col.button(
        "Connect",
        width="stretch",
    ):

        if not token_input.strip():

            st.error(
                "Access token required."
            )

        else:

            with st.spinner(
                "Validating Upstox..."
            ):

                client_test = UpstoxClient(
                    token_input
                )

                valid, message, profile = (
                    client_test.validate_connection()
                )

                if valid:

                    st.session_state[
                        "upstox_token"
                    ] = token_input.strip()

                    st.session_state[
                        "connected"
                    ] = True

                    st.session_state[
                        "profile"
                    ] = profile

                    st.success(
                        message
                    )

                else:

                    st.session_state[
                        "connected"
                    ] = False

                    st.error(
                        message
                    )

    if not st.session_state.get(
        "connected"
    ):

        st.warning(
            "Connect Upstox to start scanning."
        )

        st.stop()

    client = UpstoxClient(
        st.session_state[
            "upstox_token"
        ]
    )

    # =================================================================
    # INSTRUMENTS
    # =================================================================

    if not st.session_state.get(
        "instruments_cache"
    ):

        with st.spinner(
            "Loading NSE equity universe..."
        ):

            instruments = (
                client.get_nse_equities()
            )

            st.session_state[
                "instruments_cache"
            ] = instruments

    instruments = (
        st.session_state[
            "instruments_cache"
        ]
        or []
    )

    if not instruments:

        st.error(
            "Unable to load NSE instruments."
        )

        st.stop()

    # =================================================================
    # UNIVERSE
    # =================================================================

    selected = select_universe(
        instruments,
        universe_mode,
        int(custom_limit),
    )

    st.info(
        f"Universe selected: "
        f"**{len(selected)} stocks**"
    )

    # =================================================================
    # AUTO REFRESH
    # =================================================================

    st_autorefresh(
        interval=5 * 60 * 1000,
        key="smc_auto_refresh",
    )

    # =================================================================
    # CANDLE KEY
    # =================================================================

    now = now_ist()

    minute_bucket = (
        now.minute // 5
    ) * 5

    candle_start = now.replace(
        minute=minute_bucket,
        second=0,
        microsecond=0,
    )

    last_closed = (
        candle_start
        - dt.timedelta(minutes=5)
    )

    candle_key = (
        last_closed.strftime(
            "%Y-%m-%d %H:%M"
        )
    )

    market_open = (
        MARKET_OPEN
        <= now.time()
        <= MARKET_CLOSE
    )

    last_scanned = st.session_state.get(
        "last_scanned_candle"
    )

    should_scan = (
        market_open
        and candle_key != last_scanned
    )

    # =================================================================
    # SCAN
    # =================================================================

    if should_scan:

        settings = {
            "swing_length": swing_length,
            "displacement_multiplier": displacement_multiplier,
            "max_sweep_bars": max_sweep_bars,
            "max_displacement_bars": max_displacement_bars,
            "max_structure_bars": max_structure_bars,
            "max_zone_age_bars": max_zone_age,
            "volume_lookback": volume_lookback,
            "min_volume_mult": min_volume,
            "min_score": min_score,
            "min_rr": min_rr,
            "target_rr": target_rr,
            "max_sl_pct": max_sl_pct,
            "min_stock_price": min_price,
            "max_threads": max_threads,
        }

        st.subheader(
            f"🔎 Scanning completed candle "
            f"{candle_key}"
        )

        # IMPORTANT:
        # Do not mark candle as scanned before success.

        scan_start = time.monotonic()

        signals, failures = (
            run_market_scan(
                client,
                selected,
                settings,
            )
        )

        scan_duration = (
            time.monotonic()
            - scan_start
        )

        # Save results only after scan completed

        st.session_state[
            "scan_results"
        ] = signals

        st.session_state[
            "failed_diagnostics"
        ] = failures

        st.session_state[
            "last_scan_time"
        ] = now

        st.session_state[
            "last_scanned_candle"
        ] = candle_key

        state = load_state()

        state[
            "last_scanned_candle"
        ] = candle_key

        state[
            "last_scan_time"
        ] = now.isoformat()

        save_state(
            state
        )

        st.success(
            f"✅ Scan completed | "
            f"{len(signals)} signals | "
            f"{scan_duration:.1f}s"
        )

        # -------------------------------------------------------------
        # TELEGRAM
        # -------------------------------------------------------------

        if (
            telegram_enabled
            and telegram_token
            and telegram_chat
        ):

            history = st.session_state.setdefault(
                "alert_history",
                {},
            )

            sent = 0

            for signal in signals:

                key = (
                    f"{signal.symbol}_"
                    f"{signal.direction}_"
                    f"{signal.candle_time}"
                )

                if key in history:
                    continue

                if send_telegram(
                    telegram_token,
                    telegram_chat,
                    signal,
                ):

                    history[key] = (
                        now.isoformat()
                    )

                    sent += 1

            if sent:

                st.toast(
                    f"📲 {sent} Telegram alerts sent."
                )

    # =================================================================
    # DISPLAY
    # =================================================================

    raw_signals = (
        st.session_state.get(
            "scan_results",
            [],
        )
    )

    current = now_ist()

    active = []

    for signal in raw_signals:

        age = max(
            0,
            int(
                (
                    current
                    - signal.candle_time
                ).total_seconds()
            ),
        )

        if age <= 15 * 60:

            signal.age_seconds = age

            signal.status = (
                "LIVE"
                if age <= 5 * 60
                else "STALE"
            )

            active.append(
                signal
            )

    st.session_state[
        "scan_results"
    ] = active

    # =================================================================
    # SIGNAL TABLE
    # =================================================================

    if not active:

        st.info(
            "No confirmed SMC setup currently active."
        )

    else:

        st.subheader(
            f"🎯 Confirmed SMC Setups "
            f"({len(active)})"
        )

        rows = []

        for s in active:

            rows.append(
                {
                    "Symbol": s.symbol,
                    "Direction": s.direction,
                    "Score": f"{s.score}/100",
                    "Status": s.status,
                    "Zone": s.zone_type,
                    "Volume": (
                        f"{s.volume_ratio:.2f}x"
                    ),
                    "Entry": s.entry,
                    "SL": s.sl,
                    "TP1": s.tp1,
                    "TP2": s.tp2,
                    "RR": (
                        f"1:{s.rr:.2f}"
                    ),
                    "Sweep": (
                        f"{s.sweep_level:.2f}"
                    ),
                    "Candle": (
                        s.candle_time.strftime(
                            "%H:%M"
                        )
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

        # =============================================================
        # INSPECTOR
        # =============================================================

        st.subheader(
            "🔍 SMC Setup Inspector"
        )

        symbols = [
            s.symbol
            for s in active
        ]

        selected_symbol = st.selectbox(
            "Select setup",
            symbols,
        )

        selected_signal = next(
            s
            for s in active
            if s.symbol
            == selected_symbol
        )

        m1, m2, m3, m4, m5 = st.columns(
            5
        )

        m1.metric(
            "Direction",
            selected_signal.direction,
        )

        m2.metric(
            "SMC Score",
            f"{selected_signal.score}/100",
        )

        m3.metric(
            "Entry",
            f"₹{selected_signal.entry:.2f}",
        )

        m4.metric(
            "SL",
            f"₹{selected_signal.sl:.2f}",
        )

        m5.metric(
            "RR",
            f"1:{selected_signal.rr:.2f}",
        )

        # =============================================================
        # SEQUENCE
        # =============================================================

        st.markdown(
            "### SMC Sequence"
        )

        sequence = [
            "✅ 15M confirmed structure",
            "✅ HTF directional bias",
            "✅ 5M liquidity sweep",
            "✅ Displacement",
            f"✅ 5M {selected_signal.reason.split('5M ')[-1].split(' → ')[0] if '5M ' in selected_signal.reason else 'BOS/CHOCH'}",
            f"✅ {selected_signal.zone_type} zone",
            "✅ Price left zone",
            "✅ Genuine zone retest",
            "✅ Rejection candle",
            "✅ Volume gate",
            "✅ Structural SL",
            "✅ RR validation",
        ]

        for item in sequence:

            st.write(
                item
            )

        # =============================================================
        # CHART
        # =============================================================

        match = next(
            (
                i
                for i in instruments
                if i["symbol"]
                == selected_signal.symbol
            ),
            None,
        )

        if match:

            with st.spinner(
                "Loading chart..."
            ):

                chart_df, _ = (
                    client.get_candles(
                        match[
                            "instrument_key"
                        ],
                        5,
                        current
                        - dt.timedelta(
                            days=2
                        ),
                        current,
                    )
                )

                chart_df = (
                    filter_completed_candles(
                        chart_df,
                        5,
                        current,
                    )
                )

                if not chart_df.empty:

                    figure = build_chart(
                        chart_df,
                        selected_signal,
                    )

                    st.plotly_chart(
                        figure,
                        width="stretch",
                    )

        # =============================================================
        # REASON
        # =============================================================

        st.markdown(
            "### Why this setup passed"
        )

        st.write(
            selected_signal.reason
        )

    # =================================================================
    # DIAGNOSTICS
    # =================================================================

    failures = (
        st.session_state.get(
            "failed_diagnostics",
            [],
        )
    )

    if failures:

        with st.expander(
            f"⚠️ Diagnostics "
            f"({len(failures)})"
        ):

            rows = []

            for failure in failures:

                rows.append(
                    {
                        "Symbol": failure.symbol,
                        "Stage": failure.stage,
                        "HTTP": (
                            failure.http_status
                            or "N/A"
                        ),
                        "Error": failure.error_type,
                        "Reason": failure.reason,
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

    # =================================================================
    # FOOTER
    # =================================================================

    st.divider()

    st.caption(
        "Pure SMC analysis engine | "
        "Closed-candle only | "
        "No EMA/VWAP/MACD | "
        "No automatic order placement | "
        "Not investment advice."
    )


# =============================================================================
# 29. RUN
# =============================================================================

if __name__ == "__main__":
    main()
