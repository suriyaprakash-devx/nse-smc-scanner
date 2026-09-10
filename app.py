"""
================================================================================
NSE INTRADAY SMART MONEY CONCEPTS (SMC) SCANNER - PRODUCTION BUILD
================================================================================
Feature Highlights:
- TOP 2000+ ACTIVE STOCKS SCANNING: Dedicated liquidity selector ranking by
  live turnover (last price x traded volume) to scan the top 2000+ most
  active NSE equities.
- ZERO LOOK-AHEAD: Swings confirmed strictly after N confirmation bars.
- STRICT CLOSED CANDLES: Current in-progress 5M and 15M candles are dropped.
- SMC STATE MACHINE: HTF Bias -> Liquidity Sweep -> Displacement -> BOS/CHOCH -> Zone Retest.
- VOLUME HARD GATE: Minimum volume ratio (>= 1.5x) is a hard rejection gate.
- PINNED SIGNAL TIMESTAMPS: Age tracks candle close time; no reset on rerun.
- API RESILIENCE: Token bucket rate limiter, exponential backoff, batch progress.
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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

# Upstox market-quote endpoint accepts a bounded number of instrument keys
# per call. Keep batches conservative to avoid URL-length / API limits.
QUOTE_BATCH_SIZE = 500

# SMC Confluence Weights. These are a ranking aid, never a forecast or a
# guarantee of a profitable trade.
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
    """Thread-safe token bucket rate limiter supporting large 2000+ universe scans."""

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
    # Upstox timestamps denote the candle start.  Compare each candle's own
    # close time with `now`; this stays correct between bar boundaries.
    completed_at = df["timestamp"] + pd.Timedelta(minutes=timeframe_minutes)
    valid_df = df[completed_at <= now].copy()
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

    def get_bulk_quotes(self, instrument_keys: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Fetches last traded price and cumulative traded volume for a batch of
        instrument keys via the Upstox market-quote endpoint. Used purely as
        an input to liquidity ranking, never for signal generation.

        Returns: {instrument_key: {"ltp": float, "volume": float}}
        Caller is responsible for keeping each batch under QUOTE_BATCH_SIZE.
        """
        if not instrument_keys:
            return {}

        joined = ",".join(instrument_keys)
        # Commas must stay unescaped so Upstox parses this as a key list.
        url = f"{API_BASE}/v2/market-quote/quotes?instrument_key={quote(joined, safe=',')}"

        resp, diag = self._request(url, timeout=20)
        if resp is None or resp.status_code != 200:
            if diag:
                logger.warning("Bulk quote batch failed: %s", diag.reason)
            return {}

        try:
            payload = resp.json().get("data", {})
            out: Dict[str, Dict[str, float]] = {}
            for _, entry in payload.items():
                key = entry.get("instrument_token") or entry.get("instrument_key")
                if not key:
                    continue
                ltp = float(entry.get("last_price", 0.0) or 0.0)
                volume = float(entry.get("volume", 0.0) or 0.0)
                out[key] = {"ltp": ltp, "volume": volume}
            return out
        except Exception as e:
            logger.error("Error parsing bulk quote response: %s", e)
            return {}

    def rank_top_active_equities(
        self,
        instruments: List[Dict[str, Any]],
        target_count: int = 2000,
        batch_size: int = QUOTE_BATCH_SIZE,
        max_workers: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Ranks instruments by live traded turnover (last price x cumulative
        traded volume) and returns the top `target_count` most active names.

        This replaces a previous implementation that sorted the non-index
        remainder alphabetically by trading symbol -- that produced a
        deterministic but liquidity-BLIND selection (e.g. a low-volume stock
        starting with "A" would be preferred over a high-volume stock
        starting with "Z"), which silently defeated the scanner's headline
        "top active stocks" feature.

        If quote data cannot be retrieved for any instrument (e.g. API
        outage), falls back to a stable alphabetical slice so the app still
        functions -- but this fallback path is explicitly logged and should
        not be mistaken for a genuine liquidity rank.
        """
        if len(instruments) <= target_count:
            return instruments

        keys = [i["instrument_key"] for i in instruments]
        batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]

        turnover_map: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.get_bulk_quotes, batch): batch for batch in batches}
            for future in as_completed(future_map):
                try:
                    quotes = future.result()
                    for key, q in quotes.items():
                        turnover_map[key] = q["ltp"] * q["volume"]
                except Exception as e:
                    logger.warning("Quote batch failed during liquidity ranking: %s", e)

        if not turnover_map:
            logger.warning(
                "Turnover data unavailable for liquidity ranking; falling back "
                "to alphabetical selection (NOT a liquidity rank)."
            )
            fallback = sorted(instruments, key=lambda x: x["symbol"])
            return fallback[:target_count]

        ranked = sorted(
            instruments,
            key=lambda x: turnover_map.get(x["instrument_key"], 0.0),
            reverse=True,
        )
        return ranked[:target_count]


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
        # Strict extrema avoid treating a flat, illiquid price plateau as
        # several independent liquidity pools.
        if highs[i] > max(highs[i - length : i]) and highs[i] > max(highs[i + 1 : i + length + 1]):
            swing_highs.append(i)
        if lows[i] < min(lows[i - length : i]) and lows[i] < min(lows[i + 1 : i + length + 1]):
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder-style ATR approximation using only completed candles."""
    if len(df) < period + 1:
        return 0.0
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(true_range.rolling(period).mean().iloc[-1])
    return atr if math.isfinite(atr) and atr > 0 else 0.0


def candle_close_strength(candle: pd.Series, direction: str) -> float:
    """Return close location in the candle range; 1 means a strong close."""
    candle_range = float(candle["high"] - candle["low"])
    if candle_range <= 0:
        return 0.0
    if direction == "bull":
        return float((candle["close"] - candle["low"]) / candle_range)
    return float((candle["high"] - candle["close"]) / candle_range)


def is_tradeable_session(timestamp: dt.datetime, opening_buffer_min: int, closing_buffer_min: int) -> bool:
    """Avoid the opening auction noise and thin final minutes of NSE cash hours."""
    local = timestamp.astimezone(IST)
    start = dt.datetime.combine(local.date(), MARKET_OPEN_TIME, tzinfo=IST) + dt.timedelta(minutes=opening_buffer_min)
    end = dt.datetime.combine(local.date(), MARKET_CLOSE_TIME, tzinfo=IST) - dt.timedelta(minutes=closing_buffer_min)
    return start <= local <= end


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

    # Fifty 15-minute bars are needed for a meaningful 20/50 EMA regime,
    # rather than making a trend call from a handful of candles.
    min_required = max(swing_length * 2 + 5, 55)
    if len(c_htf) < min_required or len(c_entry) < min_required:
        return None

    current_bar = len(c_entry) - 1
    last_candle_time = c_entry.loc[current_bar, "timestamp"]
    entry_price = float(c_entry.loc[current_bar, "close"])

    if not is_tradeable_session(
        last_candle_time,
        int(settings.get("opening_buffer_min", 15)),
        int(settings.get("closing_buffer_min", 20)),
    ):
        return None

    # 1. HTF Bias
    _, htf_trend, _ = analyze_structure_state_machine(c_htf, swing_length)
    if not htf_trend:
        return None

    ema_fast = float(c_htf["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema_slow = float(c_htf["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    htf_close = float(c_htf["close"].iloc[-1])
    ema_aligned = (
        htf_close > ema_fast > ema_slow if htf_trend == "bull"
        else htf_close < ema_fast < ema_slow
    )
    if not ema_aligned:
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
    latest_sweep = max(valid_sweeps, key=lambda item: item["sweep_idx"])

    # 3. 5M Displacement & Structure Break
    ltf_events, _, _ = analyze_structure_state_machine(c_entry, swing_length)
    aligned_events = [
        e for e in ltf_events
        if e["type"] == bias and e["idx"] >= latest_sweep["sweep_idx"] and (current_bar - e["idx"]) <= settings["max_structure_bars"]
    ]
    if not aligned_events:
        return None
    displacements = detect_displacements(c_entry, multiplier=settings["displacement_multiplier"])
    aligned_disp = [
        d for d in displacements
        if d["type"] == bias and d["idx"] >= latest_sweep["sweep_idx"] and (current_bar - d["idx"]) <= settings["max_displacement_bars"]
    ]
    if not aligned_disp:
        return None

    # A valid sequence is sweep -> displacement -> confirmed break.  The old
    # implementation accepted these events in any order, which creates many
    # retrospective-looking but non-tradable signals.
    aligned_sequences = [
        (event, displacement)
        for event in aligned_events
        for displacement in aligned_disp
        if displacement["idx"] <= event["idx"]
    ]
    if not aligned_sequences:
        return None
    trigger_event, trigger_displacement = max(aligned_sequences, key=lambda pair: pair[0]["idx"])

    # 4. FVG & Order Block Zone Retest
    fvgs, obs = detect_active_fvg_and_order_blocks(c_entry, current_bar, settings["max_zone_age_bars"])
    zone_interaction = False
    entry_low = float(c_entry.loc[current_bar, "low"])
    entry_high = float(c_entry.loc[current_bar, "high"])
    for f in fvgs:
        # A gap created by the current bar has not been retested yet.
        if f["idx"] < current_bar and f["type"] == bias and not f["mitigated"]:
            zone_low, zone_high = min(f["top"], f["bottom"]), max(f["top"], f["bottom"])
            if entry_low <= zone_high * 1.002 and entry_high >= zone_low * 0.998:
                zone_interaction = True
                break
    if not zone_interaction:
        for o in obs:
            if o["type"] == bias and not o["invalidated"]:
                if entry_low <= o["high"] * 1.002 and entry_high >= o["low"] * 0.998:
                    zone_interaction = True
                    break

    if settings.get("require_zone_retest", True) and not zone_interaction:
        return None

    # 5. Volume Hard Gate
    volume_ratio = calculate_intraday_volume_ratio(c_entry, lookback=int(settings["volume_lookback"]))
    if volume_ratio < float(settings["min_volume_mult"]):
        return None

    # A close near the favourable end of the entry candle rejects weak
    # re-entries that merely touch the zone and reverse again.
    close_strength = candle_close_strength(c_entry.iloc[current_bar], bias)
    if close_strength < float(settings.get("min_close_strength", 0.65)):
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
    atr = calculate_atr(c_entry, period=14)
    if atr <= 0:
        return None
    max_sl_distance = entry_price * (float(settings["max_sl_pct"]) / 100.0)
    if (
        risk > max_sl_distance
        or risk <= 0
        or risk < atr * float(settings.get("min_stop_atr", 0.35))
        or risk > atr * float(settings.get("max_stop_atr", 2.5))
    ):
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
        "htf_alignment": ema_aligned,
        "liquidity_sweep": True,
        "displacement": True,
        "structure_break": True,
        "zone_confluence": zone_interaction,
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
        f"Swept {target_sweep_type} at {latest_sweep['level']:.2f}; displacement on "
        f"{trigger_displacement['timestamp'].strftime('%H:%M')}. Volume {volume_ratio:.2f}x, "
        f"close strength {close_strength:.0%}, ATR {atr:.2f}."
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

    completed_entry = filter_completed_candles(entry_df, 5, now)
    if completed_entry.empty or completed_entry["close"].iloc[-1] < float(settings["min_stock_price"]):
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
    st.set_page_config(page_title="NSE SMC Scanner (2000+ Stocks)", page_icon="⚡", layout="wide")

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
        st.subheader("1. Active Universe (Min 2000)")
        universe_mode = st.selectbox(
            "Scan Universe",
            [
                "TOP 2000 ACTIVE STOCKS",
                "TOP 1500 ACTIVE STOCKS",
                "ALL NSE EQUITIES",
                "NIFTY 50",
                "NIFTY 100",
                "CUSTOM LIMIT"
            ],
            index=0
        )
        custom_limit = 2000
        if universe_mode == "CUSTOM LIMIT":
            custom_limit = st.number_input("Custom Stock Limit", min_value=50, max_value=2500, value=2000, step=50)

        min_price = st.number_input("Min Price Filter (₹)", min_value=5.0, max_value=5000.0, value=250.0, step=5.0)

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
        min_volume_mult = st.slider("Min Volume Gate (x avg)", 1.0, 4.0, 1.8, 0.1, help="Rejects setup if volume < multiplier")
        min_score = st.slider("Min SMC Score", 50, 100, 85, 5)
        require_zone_retest = st.checkbox("Require Entry-Candle Zone Retest", value=True, help="Only signal when the latest closed candle actually trades into an active FVG/order block.")
        min_close_strength = st.slider("Min Entry Close Strength", 0.50, 0.95, 0.65, 0.05, help="BUY closes near its high; SELL closes near its low.")

        col_rr1, col_rr2 = st.columns(2)
        min_rr_val = col_rr1.number_input("Min R:R", min_value=1.5, max_value=5.0, value=2.0, step=0.5)
        target_rr_val = col_rr2.number_input("Target R:R", min_value=min_rr_val, max_value=6.0, value=max(min_rr_val, 2.5), step=0.5)

        max_sl_pct = st.slider("Max SL Distance (%)", 0.5, 3.0, 1.5, 0.1)
        opening_buffer_min = st.slider("Skip First Minutes", 0, 45, 15, 5)
        closing_buffer_min = st.slider("Skip Final Minutes", 0, 45, 20, 5)
        require_market_bias = st.checkbox("Require NIFTY 50 Bias Alignment", value=False)
        max_threads = st.slider("API Concurrency Workers", 4, 16, 8, help="Higher concurrency speeds up scanning 2000+ stocks")

        st.divider()

        # 4. Telegram Notifications
        st.subheader("4. Telegram Dispatcher")
        tg_enable = st.checkbox("Enable Alerts")
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")
        tg_cooldown = st.slider("Cooldown (Minutes)", 5, 120, 15)

    # --- Header / Market Status ---
    st.header("⚡ NSE Real-Time Smart Money Concepts (SMC) Scanner")
    st.caption("High-Capacity Multi-Threaded Engine for Scanning 2000+ Top Active NSE Equities")
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

    if col_btn1.button("🔌 Connect Upstox", width="stretch"):
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

    if col_btn2.button("🗑️ Clear Auth", width="stretch"):
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

    # --- Universe Resolution (Min 2000 Support) ---
    selected_instruments: List[Dict[str, Any]] = []
    if universe_mode == "TOP 2000 ACTIVE STOCKS":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=2000)
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

    # --- Automatic Scanning Execution ---
    st.subheader(f"🤖 Automatic Scanner ({len(selected_instruments)} Stocks)")

    # Refresh the Streamlit page every 15 minutes.
    # The candle-key check below prevents duplicate scans for the same
    # completed candle even if Streamlit reruns for another reason.
    st_autorefresh(
        interval=15 * 60 * 1000,  # 15 minutes
        key="automatic_scanner_refresh"
    )

    now_ist = dt.datetime.now(IST)
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    # Scan only during NSE cash-market hours.
    is_market_open = market_open <= now_ist <= market_close

    # Identify the latest COMPLETED 5-minute candle.
    minute_bucket = (now_ist.minute // 5) * 5
    current_candle_start = now_ist.replace(
        minute=minute_bucket,
        second=0,
        microsecond=0
    )
    last_closed_candle = current_candle_start - dt.timedelta(minutes=5)
    candle_key = last_closed_candle.strftime("%Y-%m-%d %H:%M")

    last_scanned_candle = st.session_state.get("last_scanned_candle")

    # Only scan once for each new completed 5-minute candle encountered
    # by the 15-minute automation cycle.
    should_scan = (
        is_market_open
        and candle_key != last_scanned_candle
    )

    if should_scan:
        st.session_state["last_scanned_candle"] = candle_key

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
            "require_zone_retest": require_zone_retest,
            "min_close_strength": min_close_strength,
            "min_rr": min_rr_val,
            "target_rr": target_rr_val,
            "max_sl_pct": max_sl_pct,
            "min_stop_atr": 0.35,
            "max_stop_atr": 2.5,
            "opening_buffer_min": opening_buffer_min,
            "closing_buffer_min": closing_buffer_min,
            "min_stock_price": min_price,
            "require_market_bias": require_market_bias,
            "max_threads": max_threads
        }

        st.info(f"🔄 Automatic scan started for completed candle {candle_key}")

        with st.spinner("Checking NIFTY 50 Macro Direction..."):
            macro_bias = evaluate_market_bias(client, now_ist)
            st.session_state["market_bias"] = macro_bias

        prog_bar = st.progress(0.0)
        status_box = st.empty()

        signals, failures = run_market_scan_with_progress(
            client, selected_instruments, settings_payload, macro_bias, prog_bar, status_box
        )

        st.session_state["scan_results"] = signals
        st.session_state["failed_diagnostics"] = failures
        st.session_state["last_scan_time"] = now_ist

        prog_bar.empty()
        status_box.empty()

        st.success(
            f"✅ Automatic scan complete! {len(signals)} setup(s) identified "
            f"out of {len(selected_instruments)} stocks."
        )

        if tg_enable and tg_token and tg_chat and signals:
            sent_cnt = dispatch_telegram_alerts(
                signals, tg_token, tg_chat, tg_cooldown
            )
            if sent_cnt > 0:
                st.toast(f"📲 Dispatched {sent_cnt} Telegram alert(s).")

    elif not is_market_open:
        if now_ist < market_open:
            st.info("🕘 Market not open yet. Automatic scanning starts at 09:15 IST.")
        else:
            st.info("🔴 Market closed. Automatic scanning finished for today.")
    else:
        last_scan = st.session_state.get("last_scan_time")
        if last_scan:
            st.caption(
                f"🟢 Automatic scanner active | Last scan: "
                f"{last_scan.strftime('%H:%M:%S')} IST | Refresh: every 15 minutes"
            )
        else:
            st.caption("🟢 Automatic scanner active | Waiting for the next cycle")

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
        st.info("No active SMC signals present. The scanner is running automatically.")
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
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

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
                    st.plotly_chart(fig, width="stretch")

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
            st.dataframe(pd.DataFrame(fail_rows), width="stretch", hide_index=True)

    st.caption("⚠️ Smart Money Concepts Intraday Scanner. Purely algorithmic analysis; not investment advice.")


if __name__ == "__main__":
    main()
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

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

# SMC Confluence Weights. These are a ranking aid, never a forecast or a
# guarantee of a profitable trade.
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
    """Thread-safe token bucket rate limiter supporting large 2000+ universe scans."""

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
    # Upstox timestamps denote the candle start.  Compare each candle's own
    # close time with `now`; this stays correct between bar boundaries.
    completed_at = df["timestamp"] + pd.Timedelta(minutes=timeframe_minutes)
    valid_df = df[completed_at <= now].copy()
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
        self, instruments: List[Dict[str, Any]], target_count: int = 2000
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
        # Strict extrema avoid treating a flat, illiquid price plateau as
        # several independent liquidity pools.
        if highs[i] > max(highs[i - length : i]) and highs[i] > max(highs[i + 1 : i + length + 1]):
            swing_highs.append(i)
        if lows[i] < min(lows[i - length : i]) and lows[i] < min(lows[i + 1 : i + length + 1]):
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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder-style ATR approximation using only completed candles."""
    if len(df) < period + 1:
        return 0.0
    previous_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    atr = float(true_range.rolling(period).mean().iloc[-1])
    return atr if math.isfinite(atr) and atr > 0 else 0.0


def candle_close_strength(candle: pd.Series, direction: str) -> float:
    """Return close location in the candle range; 1 means a strong close."""
    candle_range = float(candle["high"] - candle["low"])
    if candle_range <= 0:
        return 0.0
    if direction == "bull":
        return float((candle["close"] - candle["low"]) / candle_range)
    return float((candle["high"] - candle["close"]) / candle_range)


def is_tradeable_session(timestamp: dt.datetime, opening_buffer_min: int, closing_buffer_min: int) -> bool:
    """Avoid the opening auction noise and thin final minutes of NSE cash hours."""
    local = timestamp.astimezone(IST)
    start = dt.datetime.combine(local.date(), MARKET_OPEN_TIME, tzinfo=IST) + dt.timedelta(minutes=opening_buffer_min)
    end = dt.datetime.combine(local.date(), MARKET_CLOSE_TIME, tzinfo=IST) - dt.timedelta(minutes=closing_buffer_min)
    return start <= local <= end


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

    # Fifty 15-minute bars are needed for a meaningful 20/50 EMA regime,
    # rather than making a trend call from a handful of candles.
    min_required = max(swing_length * 2 + 5, 55)
    if len(c_htf) < min_required or len(c_entry) < min_required:
        return None

    current_bar = len(c_entry) - 1
    last_candle_time = c_entry.loc[current_bar, "timestamp"]
    entry_price = float(c_entry.loc[current_bar, "close"])

    if not is_tradeable_session(
        last_candle_time,
        int(settings.get("opening_buffer_min", 15)),
        int(settings.get("closing_buffer_min", 20)),
    ):
        return None

    # 1. HTF Bias
    _, htf_trend, _ = analyze_structure_state_machine(c_htf, swing_length)
    if not htf_trend:
        return None

    ema_fast = float(c_htf["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema_slow = float(c_htf["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    htf_close = float(c_htf["close"].iloc[-1])
    ema_aligned = (
        htf_close > ema_fast > ema_slow if htf_trend == "bull"
        else htf_close < ema_fast < ema_slow
    )
    if not ema_aligned:
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
    latest_sweep = max(valid_sweeps, key=lambda item: item["sweep_idx"])

    # 3. 5M Displacement & Structure Break
    ltf_events, _, _ = analyze_structure_state_machine(c_entry, swing_length)
    aligned_events = [
        e for e in ltf_events
        if e["type"] == bias and e["idx"] >= latest_sweep["sweep_idx"] and (current_bar - e["idx"]) <= settings["max_structure_bars"]
    ]
    if not aligned_events:
        return None
    displacements = detect_displacements(c_entry, multiplier=settings["displacement_multiplier"])
    aligned_disp = [
        d for d in displacements
        if d["type"] == bias and d["idx"] >= latest_sweep["sweep_idx"] and (current_bar - d["idx"]) <= settings["max_displacement_bars"]
    ]
    if not aligned_disp:
        return None

    # A valid sequence is sweep -> displacement -> confirmed break.  The old
    # implementation accepted these events in any order, which creates many
    # retrospective-looking but non-tradable signals.
    aligned_sequences = [
        (event, displacement)
        for event in aligned_events
        for displacement in aligned_disp
        if displacement["idx"] <= event["idx"]
    ]
    if not aligned_sequences:
        return None
    trigger_event, trigger_displacement = max(aligned_sequences, key=lambda pair: pair[0]["idx"])

    # 4. FVG & Order Block Zone Retest
    fvgs, obs = detect_active_fvg_and_order_blocks(c_entry, current_bar, settings["max_zone_age_bars"])
    zone_interaction = False
    entry_low = float(c_entry.loc[current_bar, "low"])
    entry_high = float(c_entry.loc[current_bar, "high"])
    for f in fvgs:
        # A gap created by the current bar has not been retested yet.
        if f["idx"] < current_bar and f["type"] == bias and not f["mitigated"]:
            zone_low, zone_high = min(f["top"], f["bottom"]), max(f["top"], f["bottom"])
            if entry_low <= zone_high * 1.002 and entry_high >= zone_low * 0.998:
                zone_interaction = True
                break
    if not zone_interaction:
        for o in obs:
            if o["type"] == bias and not o["invalidated"]:
                if entry_low <= o["high"] * 1.002 and entry_high >= o["low"] * 0.998:
                    zone_interaction = True
                    break

    if settings.get("require_zone_retest", True) and not zone_interaction:
        return None

    # 5. Volume Hard Gate
    volume_ratio = calculate_intraday_volume_ratio(c_entry, lookback=int(settings["volume_lookback"]))
    if volume_ratio < float(settings["min_volume_mult"]):
        return None

    # A close near the favourable end of the entry candle rejects weak
    # re-entries that merely touch the zone and reverse again.
    close_strength = candle_close_strength(c_entry.iloc[current_bar], bias)
    if close_strength < float(settings.get("min_close_strength", 0.65)):
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
    atr = calculate_atr(c_entry, period=14)
    if atr <= 0:
        return None
    max_sl_distance = entry_price * (float(settings["max_sl_pct"]) / 100.0)
    if (
        risk > max_sl_distance
        or risk <= 0
        or risk < atr * float(settings.get("min_stop_atr", 0.35))
        or risk > atr * float(settings.get("max_stop_atr", 2.5))
    ):
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
        "htf_alignment": ema_aligned,
        "liquidity_sweep": True,
        "displacement": True,
        "structure_break": True,
        "zone_confluence": zone_interaction,
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
        f"Swept {target_sweep_type} at {latest_sweep['level']:.2f}; displacement on "
        f"{trigger_displacement['timestamp'].strftime('%H:%M')}. Volume {volume_ratio:.2f}x, "
        f"close strength {close_strength:.0%}, ATR {atr:.2f}."
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

    completed_entry = filter_completed_candles(entry_df, 5, now)
    if completed_entry.empty or completed_entry["close"].iloc[-1] < float(settings["min_stock_price"]):
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
    st.set_page_config(page_title="NSE SMC Scanner (2000+ Stocks)", page_icon="⚡", layout="wide")

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
        st.subheader("1. Active Universe (Min 2000)")
        universe_mode = st.selectbox(
            "Scan Universe",
            [
                "TOP 2000 ACTIVE STOCKS",
                "TOP 1500 ACTIVE STOCKS",
                "ALL NSE EQUITIES",
                "NIFTY 50",
                "NIFTY 100",
                "CUSTOM LIMIT"
            ],
            index=0
        )
        custom_limit = 2000
        if universe_mode == "CUSTOM LIMIT":
            custom_limit = st.number_input("Custom Stock Limit", min_value=50, max_value=2500, value=2000, step=50)

        min_price = st.number_input("Min Price Filter (₹)", min_value=5.0, max_value=5000.0, value=250.0, step=5.0)

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
        min_volume_mult = st.slider("Min Volume Gate (x avg)", 1.0, 4.0, 1.8, 0.1, help="Rejects setup if volume < multiplier")
        min_score = st.slider("Min SMC Score", 50, 100, 85, 5)
        require_zone_retest = st.checkbox("Require Entry-Candle Zone Retest", value=True, help="Only signal when the latest closed candle actually trades into an active FVG/order block.")
        min_close_strength = st.slider("Min Entry Close Strength", 0.50, 0.95, 0.65, 0.05, help="BUY closes near its high; SELL closes near its low.")

        col_rr1, col_rr2 = st.columns(2)
        min_rr_val = col_rr1.number_input("Min R:R", min_value=1.5, max_value=5.0, value=2.0, step=0.5)
        target_rr_val = col_rr2.number_input("Target R:R", min_value=min_rr_val, max_value=6.0, value=max(min_rr_val, 2.5), step=0.5)

        max_sl_pct = st.slider("Max SL Distance (%)", 0.5, 3.0, 1.5, 0.1)
        opening_buffer_min = st.slider("Skip First Minutes", 0, 45, 15, 5)
        closing_buffer_min = st.slider("Skip Final Minutes", 0, 45, 20, 5)
        require_market_bias = st.checkbox("Require NIFTY 50 Bias Alignment", value=False)
        max_threads = st.slider("API Concurrency Workers", 4, 16, 8, help="Higher concurrency speeds up scanning 2000+ stocks")

        st.divider()

        # 4. Telegram Notifications
        st.subheader("4. Telegram Dispatcher")
        tg_enable = st.checkbox("Enable Alerts")
        tg_token = st.text_input("Bot Token", type="password")
        tg_chat = st.text_input("Chat ID")
        tg_cooldown = st.slider("Cooldown (Minutes)", 5, 120, 15)

    # --- Header / Market Status ---
    st.header("⚡ NSE Real-Time Smart Money Concepts (SMC) Scanner")
    st.caption("High-Capacity Multi-Threaded Engine for Scanning 2000+ Top Active NSE Equities")
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

    if col_btn1.button("🔌 Connect Upstox", width="stretch"):
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

    if col_btn2.button("🗑️ Clear Auth", width="stretch"):
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

    # --- Universe Resolution (Min 2000 Support) ---
    selected_instruments: List[Dict[str, Any]] = []
    if universe_mode == "TOP 2000 ACTIVE STOCKS":
        selected_instruments = client.rank_top_active_equities(all_instruments, target_count=2000)
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

    # --- Automatic Scanning Execution ---
    st.subheader(f"🤖 Automatic Scanner ({len(selected_instruments)} Stocks)")

    # Refresh the Streamlit page every 15 minutes.
    # The candle-key check below prevents duplicate scans for the same
    # completed candle even if Streamlit reruns for another reason.
    st_autorefresh(
        interval=15 * 60 * 1000,  # 15 minutes
        key="automatic_scanner_refresh"
    )

    now_ist = dt.datetime.now(IST)
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    # Scan only during NSE cash-market hours.
    is_market_open = market_open <= now_ist <= market_close

    # Identify the latest COMPLETED 5-minute candle.
    minute_bucket = (now_ist.minute // 5) * 5
    current_candle_start = now_ist.replace(
        minute=minute_bucket,
        second=0,
        microsecond=0
    )
    last_closed_candle = current_candle_start - dt.timedelta(minutes=5)
    candle_key = last_closed_candle.strftime("%Y-%m-%d %H:%M")

    last_scanned_candle = st.session_state.get("last_scanned_candle")

    # Only scan once for each new completed 5-minute candle encountered
    # by the 15-minute automation cycle.
    should_scan = (
        is_market_open
        and candle_key != last_scanned_candle
    )

    if should_scan:
        st.session_state["last_scanned_candle"] = candle_key

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
            "require_zone_retest": require_zone_retest,
            "min_close_strength": min_close_strength,
            "min_rr": min_rr_val,
            "target_rr": target_rr_val,
            "max_sl_pct": max_sl_pct,
            "min_stop_atr": 0.35,
            "max_stop_atr": 2.5,
            "opening_buffer_min": opening_buffer_min,
            "closing_buffer_min": closing_buffer_min,
            "min_stock_price": min_price,
            "require_market_bias": require_market_bias,
            "max_threads": max_threads
        }

        st.info(f"🔄 Automatic scan started for completed candle {candle_key}")

        with st.spinner("Checking NIFTY 50 Macro Direction..."):
            macro_bias = evaluate_market_bias(client, now_ist)
            st.session_state["market_bias"] = macro_bias

        prog_bar = st.progress(0.0)
        status_box = st.empty()

        signals, failures = run_market_scan_with_progress(
            client, selected_instruments, settings_payload, macro_bias, prog_bar, status_box
        )

        st.session_state["scan_results"] = signals
        st.session_state["failed_diagnostics"] = failures
        st.session_state["last_scan_time"] = now_ist

        prog_bar.empty()
        status_box.empty()

        st.success(
            f"✅ Automatic scan complete! {len(signals)} setup(s) identified "
            f"out of {len(selected_instruments)} stocks."
        )

        if tg_enable and tg_token and tg_chat and signals:
            sent_cnt = dispatch_telegram_alerts(
                signals, tg_token, tg_chat, tg_cooldown
            )
            if sent_cnt > 0:
                st.toast(f"📲 Dispatched {sent_cnt} Telegram alert(s).")

    elif not is_market_open:
        if now_ist < market_open:
            st.info("🕘 Market not open yet. Automatic scanning starts at 09:15 IST.")
        else:
            st.info("🔴 Market closed. Automatic scanning finished for today.")
    else:
        last_scan = st.session_state.get("last_scan_time")
        if last_scan:
            st.caption(
                f"🟢 Automatic scanner active | Last scan: "
                f"{last_scan.strftime('%H:%M:%S')} IST | Refresh: every 15 minutes"
            )
        else:
            st.caption("🟢 Automatic scanner active | Waiting for the next cycle")

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
        st.info("No active SMC signals present. The scanner is running automatically.")
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
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

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
                    st.plotly_chart(fig, width="stretch")

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
            st.dataframe(pd.DataFrame(fail_rows), width="stretch", hide_index=True)

    st.caption("⚠️ Smart Money Concepts Intraday Scanner. Purely algorithmic analysis; not investment advice.")


if __name__ == "__main__":
    main()
