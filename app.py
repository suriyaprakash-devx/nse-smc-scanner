"""
NSE EMA 6/30 Crossover Scanner
================================
A semi-algo Streamlit dashboard that scans all NSE equity stocks for
EMA 6 / EMA 30 crossover signals using the Upstox REST API (v2 + v3).

Usage
-----
    streamlit run app.py

• Detects BUY  (EMA 6 crosses above EMA 30)
• Detects SELL (EMA 6 crosses below EMA 30)
• Supports 5-minute and 10-minute candle timeframes
• Uses only fully completed candles — never in-progress ones
• Ranks signals by most recent crossover first
• Scans during NSE market hours (09:15 – 15:30 IST)
• Analysis and alerts only — NO trades are placed
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPORTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import streamlit as st
import requests
import pandas as pd
import time
import gzip
import io
import json
import threading
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IST = timezone(timedelta(hours=5, minutes=30))

MARKET_OPEN_H, MARKET_OPEN_M = 9, 15
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30

# Upstox instrument master (JSON, gzipped)
INSTRUMENTS_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)

# API base URLs — try v3 first, fall back to v2
UPSTOX_V3 = "https://api.upstox.com/v3"
UPSTOX_V2 = "https://api.upstox.com/v2"

# Interval mapping:  label → (v3_unit, v3_interval, v2_interval, minutes)
INTERVAL_MAP = {
    "5M":  ("minutes", "5",  "5minute",  5),
    "10M": ("minutes", "10", "10minute", 10),
}

# Rate-limiting & retry
API_DELAY_S   = 0.06       # ~16 req/s — well within Upstox limits
MAX_RETRIES   = 2
RETRY_DELAY_S = 1.0

# EMA parameters
EMA_SHORT   = 6
EMA_LONG    = 30
MIN_CANDLES = EMA_LONG + 2  # need at least this many for a meaningful cross

# UI auto-refresh while scanner is active (seconds)
REFRESH_S = 2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ema_scanner")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THREAD-SAFE SCANNER STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Module-level dict so it survives Streamlit reruns within the same
# server process.  Protected by a threading.Lock for safe cross-thread
# reads and writes.

_lock = threading.Lock()

_scanner: dict = {
    "running":        False,
    "stop_requested": False,
    "signals":        [],          # list[dict]
    "signal_keys":    set(),       # {(symbol, cross_time_str), …}
    "progress":       0,
    "total":          0,
    "current_stock":  "",
    "scan_number":    0,
    "last_scan_time": None,        # datetime | None
    "next_scan_time": None,        # datetime | None
    "error_count":    0,
    "skipped_count":  0,
    "status":         "Idle",
    "thread":         None,        # threading.Thread | None
}


# ── helpers for thread-safe access ──────────────────────────────

def _get(key=None):
    """Read one key or the whole state dict (minus non-serialisable bits)."""
    with _lock:
        if key is not None:
            return _scanner[key]
        return {k: v for k, v in _scanner.items() if k != "thread"}


def _set(**kw):
    """Update one or more keys."""
    with _lock:
        _scanner.update(kw)


def _add_signal(sig: dict) -> bool:
    """Append a signal if not a duplicate. Returns True if added."""
    with _lock:
        key = (sig["symbol"], sig["cross_time"])
        if key in _scanner["signal_keys"]:
            return False
        _scanner["signal_keys"].add(key)
        _scanner["signals"].append(sig)
        return True


def _signals_copy() -> list:
    with _lock:
        return list(_scanner["signals"])


def _clear_signals():
    with _lock:
        _scanner["signals"].clear()
        _scanner["signal_keys"].clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TIME UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _now() -> datetime:
    """Current IST datetime."""
    return datetime.now(IST)


def _is_market_open() -> bool:
    """True when inside NSE 09:15 – 15:30 on a weekday."""
    n = _now()
    if n.weekday() > 4:                    # Saturday / Sunday
        return False
    t_open  = n.replace(hour=MARKET_OPEN_H,  minute=MARKET_OPEN_M,
                        second=0, microsecond=0)
    t_close = n.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M,
                        second=0, microsecond=0)
    return t_open <= n <= t_close


def _next_candle_boundary(interval_min: int) -> datetime:
    """IST time when the next candle will *complete* (aligned to 09:15)."""
    n = _now()
    mkt_open = n.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                         second=0, microsecond=0)
    if n < mkt_open:
        return mkt_open + timedelta(minutes=interval_min)

    elapsed_s = (n - mkt_open).total_seconds()
    intervals = int(elapsed_s // (interval_min * 60))
    return mkt_open + timedelta(minutes=(intervals + 1) * interval_min)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NSE INSTRUMENT MASTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=86_400, show_spinner="📥 Downloading NSE instrument list …")
def load_instruments() -> pd.DataFrame:
    """
    Download the Upstox NSE BOD JSON file and return only equity rows.

    Returns a DataFrame with columns:
        instrument_key   – e.g. "NSE_EQ|INE002A01018"
        trading_symbol   – e.g. "RELIANCE"
        name             – e.g. "RELIANCE INDUSTRIES LIMITED"
    """
    resp = requests.get(INSTRUMENTS_URL, timeout=60)
    resp.raise_for_status()

    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as fh:
        raw = json.load(fh)

    df = pd.DataFrame(raw)

    # Normalise column names (lowercase, underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Filter: segment == "NSE_EQ"  AND  instrument_type == "EQ"
    mask = pd.Series([False] * len(df))
    if "segment" in df.columns:
        mask = mask | (df["segment"].str.upper() == "NSE_EQ")
    if "instrument_type" in df.columns:
        mask = mask & (df["instrument_type"].str.upper() == "EQ")
    if not mask.any() and "instrument_key" in df.columns:
        mask = df["instrument_key"].str.startswith("NSE_EQ|")

    keep_cols = []
    for c in ("instrument_key", "trading_symbol", "tradingsymbol", "name"):
        if c in df.columns:
            keep_cols.append(c)

    eq = df.loc[mask, keep_cols].copy()

    # Unify the symbol column name
    if "tradingsymbol" in eq.columns and "trading_symbol" not in eq.columns:
        eq.rename(columns={"tradingsymbol": "trading_symbol"}, inplace=True)
    if "trading_symbol" not in eq.columns:
        raise ValueError(
            f"Cannot find trading_symbol column. Available: {list(df.columns)}"
        )

    eq = eq.dropna(subset=["instrument_key", "trading_symbol"])
    eq = eq.drop_duplicates(subset=["instrument_key"])
    eq = eq.sort_values("trading_symbol").reset_index(drop=True)

    log.info("Loaded %d NSE equity instruments", len(eq))
    return eq


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UPSTOX API — CANDLE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _headers(token: str) -> dict:
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _fetch_intraday_candles(
    instrument_key: str,
    v3_unit: str,
    v3_interval: str,
    v2_interval: str,
    token: str,
) -> list:
    """
    Fetch today's intraday candles.  Tries v3 first, falls back to v2.

    Returns candles sorted ASCENDING by timestamp.
    Each candle = [timestamp_str, O, H, L, C, Volume, OI]
    """
    encoded = quote(instrument_key, safe="")
    urls = [
        f"{UPSTOX_V3}/historical-candle/intraday/{encoded}/{v3_unit}/{v3_interval}",
        f"{UPSTOX_V2}/historical-candle/intraday/{encoded}/{v2_interval}",
    ]
    hdrs = _headers(token)

    for url in urls:
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = requests.get(url, headers=hdrs, timeout=10)

                if r.status_code == 429:                     # rate-limited
                    wait = float(r.headers.get("Retry-After", RETRY_DELAY_S))
                    time.sleep(max(wait, RETRY_DELAY_S))
                    continue

                if r.status_code == 401:
                    raise PermissionError("Invalid or expired access token")

                if r.status_code >= 400:
                    break                                    # try next URL

                body = r.json()
                if body.get("status") != "success":
                    break

                candles = body.get("data", {}).get("candles", [])
                if candles:
                    candles.sort(key=lambda c: c[0])         # ascending
                    return candles

            except PermissionError:
                raise
            except requests.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
                continue
            except Exception:
                break
        # If v3 failed, try v2 next
        continue

    return []


def _fetch_historical_candles(
    instrument_key: str,
    v3_unit: str,
    v3_interval: str,
    v2_interval: str,
    token: str,
    lookback_days: int = 7,
) -> list:
    """
    Fetch historical candles for *past* days (up to but NOT including today).
    Used to bootstrap EMA calculations early in the session.
    """
    encoded = quote(instrument_key, safe="")
    today = _now().date()
    yesterday = today - timedelta(days=1)
    from_date = (today - timedelta(days=lookback_days)).isoformat()
    to_date = yesterday.isoformat()

    urls = [
        f"{UPSTOX_V3}/historical-candle/{encoded}/{v3_unit}/{v3_interval}/{to_date}/{from_date}",
        f"{UPSTOX_V2}/historical-candle/{encoded}/{v2_interval}/{to_date}/{from_date}",
    ]
    hdrs = _headers(token)

    for url in urls:
        for attempt in range(MAX_RETRIES + 1):
            try:
                r = requests.get(url, headers=hdrs, timeout=10)

                if r.status_code == 429:
                    wait = float(r.headers.get("Retry-After", RETRY_DELAY_S))
                    time.sleep(max(wait, RETRY_DELAY_S))
                    continue
                if r.status_code >= 400:
                    break

                body = r.json()
                if body.get("status") == "success":
                    candles = body.get("data", {}).get("candles", [])
                    if candles:
                        candles.sort(key=lambda c: c[0])
                        return candles
                break

            except requests.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_S)
                continue
            except Exception:
                break
        continue

    return []


def fetch_candles(
    instrument_key: str,
    v3_unit: str,
    v3_interval: str,
    v2_interval: str,
    token: str,
    hist_cache: dict,
) -> list:
    """
    Return a merged, ascending candle list combining:
      • cached historical candles (previous days)
      • fresh intraday candles (today)

    This ensures EMA 30 has enough look-back even early in the session.
    """
    # 1. Historical (cached per instrument_key per day)
    cache_key = instrument_key
    if cache_key not in hist_cache:
        hist = _fetch_historical_candles(
            instrument_key, v3_unit, v3_interval, v2_interval, token
        )
        hist_cache[cache_key] = hist
        time.sleep(API_DELAY_S)

    historical = hist_cache.get(cache_key, [])

    # 2. Intraday (always fresh)
    intraday = _fetch_intraday_candles(
        instrument_key, v3_unit, v3_interval, v2_interval, token
    )

    # 3. Merge & deduplicate by timestamp
    seen = set()
    merged = []
    for c in historical + intraday:
        ts = c[0]
        if ts not in seen:
            seen.add(ts)
            merged.append(c)

    merged.sort(key=lambda c: c[0])
    return merged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMPLETED-CANDLE FILTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _completed_candles(candles: list, interval_min: int, cutoff: datetime) -> list:
    """
    Return only candles whose interval has *fully elapsed* by *cutoff*.
    Candle timestamp = interval START, so a candle is complete when
    start + interval_minutes <= cutoff.
    """
    delta = timedelta(minutes=interval_min)
    out = []
    for c in candles:
        try:
            ts = pd.Timestamp(c[0])
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            else:
                ts = ts.tz_convert(IST)
            if (ts + delta) <= cutoff:
                out.append(c)
        except Exception:
            continue
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMA & CROSSOVER LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_emas(closes: list):
    """Return (ema_short, ema_long) as pandas Series."""
    s = pd.Series(closes, dtype=float)
    return (
        s.ewm(span=EMA_SHORT, adjust=False).mean(),
        s.ewm(span=EMA_LONG,  adjust=False).mean(),
    )


def detect_crossover(closes: list, timestamps: list):
    """
    Check the two most recent completed candles for an EMA 6/30 crossover.

    Returns a dict with signal details, or None.
    """
    if len(closes) < MIN_CANDLES:
        return None

    ema6, ema30 = _compute_emas(closes)

    cur6, cur30 = ema6.iloc[-1], ema30.iloc[-1]
    prv6, prv30 = ema6.iloc[-2], ema30.iloc[-2]

    sig = None
    if prv6 <= prv30 and cur6 > cur30:
        sig = "BUY"
    elif prv6 >= prv30 and cur6 < cur30:
        sig = "SELL"

    if sig is None:
        return None

    return {
        "signal":      sig,
        "ema6":        round(float(cur6), 2),
        "ema30":       round(float(cur30), 2),
        "cross_price": round(float(closes[-1]), 2),
        "cross_time":  timestamps[-1],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCAN ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_scan(
    instruments: pd.DataFrame,
    v3_unit: str,
    v3_interval: str,
    v2_interval: str,
    interval_min: int,
    token: str,
    hist_cache: dict,
):
    """Scan every instrument once.  Called from the background thread."""
    cutoff = _now()                                 # fix boundary for the pass
    total  = len(instruments)
    with _lock:
        _scanner["scan_number"] += 1
        _scanner["skipped_count"] = 0
    scan_no = _get("scan_number")
    _set(progress=0, total=total)
    found = 0

    for idx, row in instruments.iterrows():
        if _scanner["stop_requested"]:
            return

        ikey   = row["instrument_key"]
        symbol = row["trading_symbol"]
        _set(
            progress=idx + 1,
            current_stock=symbol,
            status=f"[Scan #{scan_no}]  {symbol}  ({idx + 1}/{total})",
        )

        try:
            raw = fetch_candles(
                ikey, v3_unit, v3_interval, v2_interval, token, hist_cache
            )
            if not raw:
                with _lock:
                    _scanner["skipped_count"] += 1
                time.sleep(API_DELAY_S)
                continue

            done = _completed_candles(raw, interval_min, cutoff)
            if len(done) < MIN_CANDLES:
                with _lock:
                    _scanner["skipped_count"] += 1
                time.sleep(API_DELAY_S)
                continue

            closes     = [c[4] for c in done]
            timestamps = [c[0] for c in done]
            result     = detect_crossover(closes, timestamps)

            if result:
                result.update(
                    symbol=symbol,
                    instrument_key=ikey,
                    timeframe=f"{interval_min}M",
                    scan_number=scan_no,
                )
                if _add_signal(result):
                    found += 1
                    log.info(
                        "SIGNAL  %s  %s  @ %.2f  [%s]",
                        result["signal"], symbol,
                        result["cross_price"], result["cross_time"],
                    )

        except PermissionError:
            _set(running=False, status="❌ Invalid / expired access token")
            return
        except Exception as exc:
            with _lock:
                _scanner["error_count"] += 1
            log.warning("Error scanning %s: %s", symbol, exc)

        time.sleep(API_DELAY_S)

    total_signals = len(_signals_copy())
    _set(
        last_scan_time=_now(),
        current_stock="",
        status=(
            f"Scan #{scan_no} done — {found} new signal(s), "
            f"{total_signals} total"
        ),
    )
    log.info(
        "Scan #%d complete: %d new signals, %d total, %d errors, %d skipped",
        scan_no, found, total_signals,
        _get("error_count"), _get("skipped_count"),
    )


def _scanner_loop(
    instruments: pd.DataFrame,
    v3_unit: str,
    v3_interval: str,
    v2_interval: str,
    interval_min: int,
    token: str,
):
    """Background-thread entry point.  Manages the scan schedule."""
    _set(running=True, stop_requested=False, error_count=0, skipped_count=0)
    hist_cache: dict = {}           # populated lazily, lives for the session

    try:
        # If outside market hours, wait (check every 10 s)
        if not _is_market_open():
            _set(status="⏳ Waiting for market to open …")
            while not _is_market_open() and not _scanner["stop_requested"]:
                n = _now()
                mkt = n.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M,
                                second=0, microsecond=0)
                if n > mkt:                          # already past open today
                    _set(status="Market closed for the day.")
                    return
                time.sleep(10)
            if _scanner["stop_requested"]:
                return

        # ── Immediate first scan ────────────────────────────────
        _run_scan(instruments, v3_unit, v3_interval, v2_interval,
                  interval_min, token, hist_cache)

        # ── Scheduled loop ──────────────────────────────────────
        while not _scanner["stop_requested"]:
            if not _is_market_open():
                _set(status="Market closed for the day.")
                break

            nxt = _next_candle_boundary(interval_min)
            _set(next_scan_time=nxt,
                 status=f"Next scan at {nxt.strftime('%H:%M:%S')} IST")

            # Wait until the candle completes
            while _now() < nxt and not _scanner["stop_requested"]:
                time.sleep(1)

            if _scanner["stop_requested"] or not _is_market_open():
                break

            # Small buffer so Upstox has flushed the completed candle
            time.sleep(3)

            _run_scan(instruments, v3_unit, v3_interval, v2_interval,
                      interval_min, token, hist_cache)

    finally:
        _set(running=False, next_scan_time=None,
             status="Scanner stopped")
        log.info("Scanner thread exited")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STREAMLIT UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _format_age(cross_time_str: str) -> str:
    """Human-readable age string for a crossover timestamp."""
    try:
        ts = pd.Timestamp(cross_time_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        secs = (_now() - ts).total_seconds()
        if secs < 0:
            return "just now"
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs // 60)}m ago"
        hrs = int(secs // 3600)
        mins = int((secs % 3600) // 60)
        return f"{hrs}h {mins}m ago"
    except Exception:
        return "–"


def _format_cross_time(cross_time_str: str) -> str:
    """Pretty-print the crossover candle timestamp."""
    try:
        ts = pd.Timestamp(cross_time_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        return ts.strftime("%d-%b %H:%M")
    except Exception:
        return cross_time_str


def main():
    # ── Page config ─────────────────────────────────────────────
    st.set_page_config(
        page_title="NSE EMA Scanner",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 NSE EMA 6 / 30 Crossover Scanner")
    st.caption(
        "Powered by Upstox API  ·  Analysis & alerts only — "
        "**no trades are placed**"
    )

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        token = st.text_input(
            "Upstox Access Token",
            type="password",
            help="Paste your Upstox API v2/v3 bearer token.",
        )

        timeframe = st.selectbox(
            "Candle Timeframe",
            options=list(INTERVAL_MAP.keys()),
            index=0,
        )

        st.divider()

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶️ Start", use_container_width=True)
        stop_btn  = c2.button("⏹ Stop",  use_container_width=True)

        if st.button("🗑️ Clear Signals", use_container_width=True):
            _clear_signals()
            st.rerun()

        st.divider()

        # Status panel
        st.subheader("📡 Status")
        snap = _get()

        now_str = _now().strftime("%H:%M:%S")
        mkt     = "🟢 Open" if _is_market_open() else "🔴 Closed"
        st.caption(f"🕐 IST {now_str}  ·  Market: {mkt}")

        colour = "🟢" if snap["running"] else "🔴"
        st.markdown(f"{colour} **{snap['status']}**")

        if snap["running"] and snap["total"] > 0:
            pct = snap["progress"] / snap["total"]
            st.progress(
                pct,
                text=(
                    f"{snap['current_stock']}  "
                    f"({snap['progress']}/{snap['total']})"
                ),
            )

        if snap["last_scan_time"]:
            st.caption(
                f"Last scan: {snap['last_scan_time'].strftime('%H:%M:%S')} IST"
            )
        if snap["next_scan_time"]:
            st.caption(
                f"Next scan: {snap['next_scan_time'].strftime('%H:%M:%S')} IST"
            )

        st.caption(f"Scans completed: {snap['scan_number']}")
        st.caption(f"API errors: {snap['error_count']}")
        st.caption(f"Stocks skipped (insufficient data): {snap['skipped_count']}")

    # ── Handle START ────────────────────────────────────────────
    if start_btn:
        if not token:
            st.error("⚠️ Please enter your Upstox access token in the sidebar.")
            st.stop()

        # Guard against duplicate threads
        if snap["running"]:
            th = _scanner.get("thread")
            if th and th.is_alive():
                st.warning("Scanner is already running.")
                st.stop()
            else:
                _set(running=False)

        # Load instruments
        try:
            instruments = load_instruments()
            st.sidebar.success(f"✅ Loaded {len(instruments)} NSE equities")
        except Exception as exc:
            st.error(f"Failed to load instruments: {exc}")
            st.stop()

        v3_unit, v3_interval, v2_interval, interval_min = INTERVAL_MAP[timeframe]

        th = threading.Thread(
            target=_scanner_loop,
            args=(instruments, v3_unit, v3_interval, v2_interval,
                  interval_min, token),
            daemon=True,
            name="ema_scanner",
        )
        with _lock:
            _scanner["thread"] = th
        th.start()

        log.info(
            "Scanner started: %s (%d stocks)",
            timeframe, len(instruments),
        )
        time.sleep(0.5)
        st.rerun()

    # ── Handle STOP ─────────────────────────────────────────────
    if stop_btn:
        _set(stop_requested=True)
        st.toast("⏹ Stop requested — scanner will halt after the current stock.")
        time.sleep(1)
        st.rerun()

    # ── About ───────────────────────────────────────────────────
    with st.expander("ℹ️  About this scanner", expanded=False):
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Short EMA | **{EMA_SHORT}** |
| Long EMA | **{EMA_LONG}** |
| Min candles needed | **{MIN_CANDLES}** |
| Scan interval | Aligned to candle close |
| Market hours | 09:15 – 15:30 IST (Mon – Fri) |

**BUY signal** — EMA {EMA_SHORT} crosses *above* EMA {EMA_LONG}
on the latest completed candle.

**SELL signal** — EMA {EMA_SHORT} crosses *below* EMA {EMA_LONG}
on the latest completed candle.

Historical data from the last 7 trading days is fetched once per
stock on the first scan so that EMA 30 is accurate even at market
open.  Subsequent scans reuse the cached historical data and
fetch only today's intraday candles.

⚠️ **No trades are ever placed.  This is analysis only.**
        """)

    # ── Signal Dashboard ────────────────────────────────────────
    st.header("📋 Signal Dashboard")

    signals = _signals_copy()

    if not signals:
        st.info(
            "No crossover signals detected yet.  "
            "Press **▶️ Start** to begin scanning."
        )
    else:
        # Sort by crossover time descending (most recent first)
        sigs = sorted(signals, key=lambda s: s["cross_time"], reverse=True)

        buy_n  = sum(1 for s in sigs if s["signal"] == "BUY")
        sell_n = len(sigs) - buy_n

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Signals", len(sigs))
        m2.metric("🟢 BUY",  buy_n)
        m3.metric("🔴 SELL", sell_n)

        rows = []
        for rank, s in enumerate(sigs, 1):
            rows.append({
                "Rank":        rank,
                "Symbol":      s["symbol"],
                "Signal":      s["signal"],
                "EMA 6":       s["ema6"],
                "EMA 30":      s["ema30"],
                "Cross Price": s["cross_price"],
                "Cross Time":  _format_cross_time(s["cross_time"]),
                "Timeframe":   s["timeframe"],
                "Age":         _format_age(s["cross_time"]),
            })

        df = pd.DataFrame(rows)

        # Conditional row colouring
        def _row_colour(row):
            if row["Signal"] == "BUY":
                bg = "background-color: rgba(0, 200, 83, 0.12)"
            else:
                bg = "background-color: rgba(255, 82, 82, 0.12)"
            return [bg] * len(row)

        styled = df.style.apply(_row_colour, axis=1)

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(len(rows) * 40 + 50, 700),
            column_config={
                "EMA 6":       st.column_config.NumberColumn(format="%.2f"),
                "EMA 30":      st.column_config.NumberColumn(format="%.2f"),
                "Cross Price": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    # ── Auto-refresh while running ──────────────────────────────
    if _get("running"):
        time.sleep(REFRESH_S)
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    main()
