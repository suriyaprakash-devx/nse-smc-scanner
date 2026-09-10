"""
NSE Intraday MA6/MA30 + Volume 1.5x + Structural SMC Scanner
Data source: Upstox only.
Orders: NEVER placed.

IMPROVED VERSION — changes from the original are marked with "# IMPROVED:"
comments so you can diff against your version easily.
"""

import os
import time
import gzip
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
IST = ZoneInfo("Asia/Kolkata")

UPSTOX_BASE = "https://api.upstox.com"
INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)
INTRADAY_V3 = UPSTOX_BASE + "/v3/historical-candle/intraday"

# Editable holiday list. Keep this aligned with the official NSE calendar.
NSE_HOLIDAYS = {
    "2026-01-26", "2026-03-03", "2026-03-26", "2026-03-31", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-05-27", "2026-06-26", "2026-08-15",
    "2026-09-14", "2026-10-02", "2026-10-20", "2026-11-10", "2026-11-24",
    "2026-12-25",
}

DEFAULTS = {
    "ma_fast": 6,
    "ma_slow": 30,
    "vol_multiplier": 1.5,
    "vol_lookback": 20,
    "swing_left": 3,
    "swing_right": 3,
    "sl_buffer_pct": 0.15,          # IMPROVED: 0.05% was too tight, near-zero buffer
    "atr_period": 14,               # IMPROVED: new — ATR-based dynamic SL buffer
    "min_rr": 1.5,
    "max_workers": 6,
    "request_interval": 0.12,
    "request_timeout": 12,
    "cross_confirm_bars": 3,        # IMPROVED: new — only fire near an actual MA cross
    "cooldown_bars": 6,             # IMPROVED: new — suppress repeat spam on same trend
    "min_avg_turnover": 500000,     # IMPROVED: new — liquidity filter (Rs per candle avg)
}


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "token": os.getenv("UPSTOX_ACCESS_TOKEN", "").strip(),
        "connected": False,
        "instruments": pd.DataFrame(),
        "signals": [],
        "last_processed_candle": None,
        "last_scan_time": "-",
        "diagnostics": [],
        "cleanup_date": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# TIME / MARKET
# ============================================================
def now_ist():
    return datetime.now(IST)


def is_trading_day(day):
    return day.weekday() < 5 and day.strftime("%Y-%m-%d") not in NSE_HOLIDAYS


def market_status():
    now = now_ist()
    if not is_trading_day(now.date()):
        return "CLOSED"
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return "OPEN" if market_start <= now <= market_end else "CLOSED"


def last_closed_5m_ist(now=None):
    """
    Returns the latest fully closed 5-minute candle start time.
    """
    now = now or now_ist()
    floored = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    return floored - timedelta(minutes=5)


# ============================================================
# INSTRUMENT MASTER (cached — no auth required for this endpoint)
# ============================================================
# IMPROVED: split out of UpstoxClient and cached via st.cache_data so the
# ~2000-row instrument file isn't re-downloaded on every reconnect/rerun.
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_instrument_master(timeout=15):
    response = requests.get(INSTRUMENT_URL, timeout=timeout)
    response.raise_for_status()

    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
        raw = gz.read()

    data = pd.read_json(io.BytesIO(raw))

    if data.empty:
        raise RuntimeError("Upstox NSE instrument file is empty.")

    required = {"segment", "instrument_type", "instrument_key", "trading_symbol"}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Instrument file is missing columns: {sorted(missing)}")

    df = data[
        (data["segment"] == "NSE_EQ") & (data["instrument_type"] == "EQ")
    ][["trading_symbol", "instrument_key", "name", "isin"]].copy()

    df = df.dropna(subset=["trading_symbol", "instrument_key"])
    df = df.drop_duplicates(subset=["instrument_key"])
    df = df.sort_values("trading_symbol").reset_index(drop=True)

    return df


# ============================================================
# UPSTOX CLIENT
# ============================================================
class UpstoxClient:
    def __init__(self, token, min_interval=0.12, timeout=12):
        self.token = token.strip()
        self.min_interval = float(min_interval)
        self.timeout = int(timeout)

        self._rate_lock = threading.Lock()
        self._last_request = 0.0

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        })

    def _wait_rate_limit(self):
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_request
            wait = self.min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, url, retries=3):
        last_error = None
        for attempt in range(retries + 1):
            try:
                self._wait_rate_limit()
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 429:
                    if attempt >= retries:
                        response.raise_for_status()
                    time.sleep(min(8, 2 * (2 ** attempt)))
                    continue

                if response.status_code in (500, 502, 503, 504):
                    if attempt >= retries:
                        response.raise_for_status()
                    time.sleep(min(8, 1.5 ** attempt))
                    continue

                response.raise_for_status()
                return response

            except requests.RequestException as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                time.sleep(min(8, 1.5 ** attempt))

        raise last_error

    def validate_token(self):
        """Small current-day request. No order is placed."""
        url = f"{INTRADAY_V3}/NSE_EQ%7CINE002A01018/minutes/5"
        response = self.get(url, retries=1)
        return response.ok

    def fetch_5m(self, instrument_key):
        """
        Fetch current-day 5-minute candles from Upstox V3.
        The currently forming candle is explicitly removed.
        """
        encoded_key = requests.utils.quote(instrument_key, safe="")
        url = f"{INTRADAY_V3}/{encoded_key}/minutes/5"

        response = self.get(url)
        payload = response.json()
        candles = payload.get("data", {}).get("candles", [])

        if not candles:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        rows = []
        for candle in candles:
            if len(candle) < 6:
                continue
            rows.append([
                pd.to_datetime(candle[0], utc=True).tz_convert(IST),
                float(candle[1]), float(candle[2]),
                float(candle[3]), float(candle[4]), float(candle[5]),
            ])

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if df.empty:
            return df

        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        # Closed-candle protection.
        cutoff = last_closed_5m_ist()
        df = df[df["timestamp"] <= cutoff].copy()
        return df


# ============================================================
# INDICATORS
# ============================================================
def add_indicators(df, fast=6, slow=30, atr_period=14):
    result = df.copy()
    result["ma6"] = result["close"].rolling(fast, min_periods=fast).mean()
    result["ma30"] = result["close"].rolling(slow, min_periods=slow).mean()

    # IMPROVED: ATR, used for a sane dynamic stop-loss buffer instead of a
    # fixed tiny percentage that gets whipsawed out on normal noise.
    prev_close = result["close"].shift(1)
    true_range = pd.concat([
        result["high"] - result["low"],
        (result["high"] - prev_close).abs(),
        (result["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    result["atr"] = true_range.rolling(atr_period, min_periods=atr_period).mean()

    return result


def volume_ratio(df, lookback=20):
    if len(df) < lookback + 1:
        return 0.0
    current_volume = float(df["volume"].iloc[-1])
    previous_average = float(df["volume"].iloc[-lookback - 1 : -1].mean())
    if previous_average <= 0:
        return 0.0
    return current_volume / previous_average


def avg_turnover(df, lookback=20):
    """IMPROVED: liquidity filter — avoids illiquid names with fake breakouts."""
    if len(df) < lookback:
        return 0.0
    window = df.iloc[-lookback:]
    return float((window["close"] * window["volume"]).mean())


def ma_cross_recent(work, direction, lookback):
    """
    IMPROVED: only treat MA6/MA30 alignment as a signal if a genuine
    crossover happened within the last `lookback` candles. Without this,
    the original logic fires on every candle for the entire duration a
    trend persists, producing repetitive, late, low-quality entries.
    """
    if len(work) < lookback + 1:
        return False

    diff = (work["ma6"] - work["ma30"]).iloc[-(lookback + 1):]
    diff = diff.dropna()
    if len(diff) < 2:
        return False

    for i in range(1, len(diff)):
        prev_val = diff.iloc[i - 1]
        cur_val = diff.iloc[i]
        if direction == "BUY" and prev_val <= 0 and cur_val > 0:
            return True
        if direction == "SELL" and prev_val >= 0 and cur_val < 0:
            return True
    return False


# ============================================================
# CONFIRMED SWINGS
# ============================================================
def confirmed_pivots(df, left=3, right=3):
    """
    A pivot is confirmed only after right-side candles exist.
    This prevents look-ahead from using future candles.
    """
    highs, lows = [], []
    if len(df) < left + right + 1:
        return highs, lows

    for i in range(left, len(df) - right):
        high_value = df["high"].iloc[i]
        low_value = df["low"].iloc[i]

        left_highs = df["high"].iloc[i - left : i]
        right_highs = df["high"].iloc[i + 1 : i + right + 1]
        left_lows = df["low"].iloc[i - left : i]
        right_lows = df["low"].iloc[i + 1 : i + right + 1]

        if high_value > left_highs.max() and high_value >= right_highs.max():
            highs.append((i, float(high_value), df["timestamp"].iloc[i]))

        if low_value < left_lows.min() and low_value <= right_lows.min():
            lows.append((i, float(low_value), df["timestamp"].iloc[i]))

    return highs, lows


# ============================================================
# STRUCTURAL SMC-STYLE SL / TP
# ============================================================
def structural_levels(df, direction, left=3, right=3, buffer_pct=0.15, atr_value=None):
    """
    Uses confirmed swing structure.

    BUY:  SL = latest confirmed swing low below entry. TP = confirmed swing highs above entry.
    SELL: SL = latest confirmed swing high above entry. TP = confirmed swing lows below entry.

    TP2 is never invented using risk multiplication.
    """
    highs, lows = confirmed_pivots(df, left, right)
    entry = float(df["close"].iloc[-1])

    highs = [item for item in highs if item[0] < len(df) - 1]
    lows = [item for item in lows if item[0] < len(df) - 1]

    # IMPROVED: dynamic buffer = max(pct-based, half an ATR), as an absolute
    # price delta rather than a percentage multiplier. A flat 0.05% buffer on
    # a stock trading at, say, Rs 2000 is a Rs 1 stop — smaller than typical
    # 5-minute noise, so it stops out on nothing.
    def buffer_distance(pivot_price):
        pct_buffer = pivot_price * buffer_pct / 100
        atr_buffer = (atr_value * 0.5) if atr_value and atr_value > 0 else 0.0
        return max(pct_buffer, atr_buffer)

    if direction == "BUY":
        prior_lows = [item for item in lows if item[1] < entry]
        prior_highs = [item for item in highs if item[1] > entry]
        if not prior_lows or not prior_highs:
            return None

        sl_pivot = max(prior_lows, key=lambda item: item[0])
        targets = sorted(prior_highs, key=lambda item: item[1])
        if not targets:
            return None

        sl = sl_pivot[1] - buffer_distance(sl_pivot[1])
        tp1 = targets[0][1]
        tp2 = targets[1][1] if len(targets) > 1 else None

    else:
        prior_highs = [item for item in highs if item[1] > entry]
        prior_lows = [item for item in lows if item[1] < entry]
        if not prior_highs or not prior_lows:
            return None

        sl_pivot = max(prior_highs, key=lambda item: item[0])
        targets = sorted(prior_lows, key=lambda item: item[1], reverse=True)
        if not targets:
            return None

        sl = sl_pivot[1] + buffer_distance(sl_pivot[1])
        tp1 = targets[0][1]
        tp2 = targets[1][1] if len(targets) > 1 else None

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    rr1 = abs(tp1 - entry) / risk
    rr2 = abs(tp2 - entry) / risk if tp2 is not None else None

    return {
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "rr1": rr1, "rr2": rr2,
        "sl_pivot_time": sl_pivot[2],
    }


# ============================================================
# SIGNAL ENGINE
# ============================================================
def compute_signal(df, symbol, cfg):
    minimum_candles = max(cfg["ma_slow"], cfg["vol_lookback"] + 1, cfg["atr_period"] + 1) + 10
    if len(df) < minimum_candles:
        return None

    work = add_indicators(df, cfg["ma_fast"], cfg["ma_slow"], cfg["atr_period"])
    latest = work.iloc[-1]

    if pd.isna(latest["ma6"]) or pd.isna(latest["ma30"]):
        return None

    if latest["ma6"] > latest["ma30"]:
        direction = "BUY"
    elif latest["ma6"] < latest["ma30"]:
        direction = "SELL"
    else:
        return None

    # IMPROVED: liquidity filter — skip thin names before doing anything else.
    turnover = avg_turnover(work, cfg["vol_lookback"])
    if turnover < cfg["min_avg_turnover"]:
        return None

    # IMPROVED: only act on a recent, genuine MA crossover — not a stale
    # trend that's already been running for hours.
    if not ma_cross_recent(work, direction, cfg["cross_confirm_bars"]):
        return None

    vr = volume_ratio(work, cfg["vol_lookback"])
    if vr < cfg["vol_multiplier"]:
        return None

    atr_value = float(latest["atr"]) if pd.notna(latest["atr"]) else None

    levels = structural_levels(
        work, direction, cfg["swing_left"], cfg["swing_right"],
        cfg["sl_buffer_pct"], atr_value,
    )
    if levels is None:
        return None

    entry = float(latest["close"])
    rr = levels["rr2"] if levels["rr2"] is not None else levels["rr1"]
    if rr < cfg["min_rr"]:
        return None

    # Setup/confluence score, NOT win probability.
    score = 25
    if vr >= cfg["vol_multiplier"]:
        score += 20
    if levels["rr1"] >= cfg["min_rr"]:
        score += 20
    if levels["rr2"] is not None and levels["rr2"] >= cfg["min_rr"]:
        score += 15
    else:
        score += 5

    # IMPROVED: trend-strength component — how separated MA6/MA30 are,
    # relative to ATR, rewards a decisive cross over a marginal one.
    if atr_value and atr_value > 0:
        separation = abs(latest["ma6"] - latest["ma30"]) / atr_value
        score += min(20, round(separation * 10))

    candle_time = latest["timestamp"]

    return {
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(levels["sl"], 2),
        "tp1": round(levels["tp1"], 2),
        "tp2": round(levels["tp2"], 2) if levels["tp2"] is not None else None,
        "rr": round(rr, 2),
        "score": int(min(score, 100)),
        "ma6": round(float(latest["ma6"]), 2),
        "ma30": round(float(latest["ma30"]), 2),
        "vol_ratio": round(vr, 2),
        "avg_turnover": round(turnover, 0),
        "candle_time": candle_time,
        "status": "LIVE",
        "smc_reason": (
            f"{direction}: fresh MA6/MA30 cross within "
            f"{cfg['cross_confirm_bars']} bars; confirmed swing structure; "
            f"ATR-aware structural SL; volume {vr:.2f}x avg"
        ),
    }


# ============================================================
# SCANNER
# ============================================================
def scan_one(client, row, cfg):
    try:
        candles = client.fetch_5m(row["instrument_key"])
        if candles.empty:
            return None, None
        signal = compute_signal(candles, row["trading_symbol"], cfg)
        return signal, None
    except Exception as exc:
        return None, {"symbol": row.get("trading_symbol", "?"), "error": str(exc)}


def run_full_scan(client, instruments, cfg):
    signals, diagnostics = [], []
    total = len(instruments)

    progress = st.progress(0.0)
    status_box = st.empty()

    workers = min(int(cfg["max_workers"]), max(1, total))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(scan_one, client, row, cfg)
            for _, row in instruments.iterrows()
        ]

        for index, future in enumerate(as_completed(futures), 1):
            signal, diagnostic = future.result()
            if signal:
                signals.append(signal)
            if diagnostic:
                diagnostics.append(diagnostic)

            if index == 1 or index % 10 == 0 or index == total:
                progress.progress(index / total)
                status_box.info(f"Scanned {index:,} / {total:,} equities")

    progress.empty()
    status_box.empty()
    return signals, diagnostics


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(page_title="NSE Intraday SMC Scanner", page_icon="⚡", layout="wide")
st.title("⚡ NSE Intraday MA6/MA30 + Volume + SMC Scanner")
st.caption(
    "Upstox-only data • 5-minute closed candles • fresh-cross + liquidity filtered • "
    "signal display only • no orders"
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🔐 Upstox")

token_input = st.sidebar.text_input(
    "UPSTOX_ACCESS_TOKEN",
    value=st.session_state.token,
    type="password",
    help="Your Upstox access token. No separate market-data API is used.",
)

connect_col, clear_col = st.sidebar.columns(2)

if connect_col.button("Connect", use_container_width=True):
    token = token_input.strip()
    if not token:
        st.session_state.connected = False
        st.sidebar.error("Enter an Upstox access token.")
    else:
        try:
            client = UpstoxClient(token)
            client.validate_token()
            st.session_state.token = token
            st.session_state.connected = True
            st.session_state.instruments = pd.DataFrame()
            st.sidebar.success("🟢 Upstox connected")
        except Exception as exc:
            st.session_state.connected = False
            st.sidebar.error(f"Connection failed: {exc}")

if clear_col.button("Clear", use_container_width=True):
    st.session_state.token = ""
    st.session_state.connected = False
    st.session_state.instruments = pd.DataFrame()
    st.session_state.signals = []
    st.session_state.last_processed_candle = None

st.sidebar.header("⚙ Strategy")

ma_fast = st.sidebar.number_input("MA Fast", 1, 100, DEFAULTS["ma_fast"])
ma_slow = st.sidebar.number_input("MA Slow", 2, 200, DEFAULTS["ma_slow"])
vol_multiplier = st.sidebar.number_input(
    "Volume multiplier", 1.0, 5.0, float(DEFAULTS["vol_multiplier"]), step=0.1
)
vol_lookback = st.sidebar.number_input("Volume lookback", 5, 100, DEFAULTS["vol_lookback"])
swing_left = st.sidebar.number_input("Swing left bars", 1, 10, DEFAULTS["swing_left"])
swing_right = st.sidebar.number_input("Swing right bars", 1, 10, DEFAULTS["swing_right"])
sl_buffer_pct = st.sidebar.number_input(
    "SL buffer % (floor; ATR-based buffer used if larger)",
    0.0, 2.0, float(DEFAULTS["sl_buffer_pct"]), step=0.01,
)
atr_period = st.sidebar.number_input("ATR period", 5, 50, DEFAULTS["atr_period"])
min_rr = st.sidebar.number_input("Minimum RR", 0.5, 10.0, float(DEFAULTS["min_rr"]), step=0.1)

st.sidebar.subheader("🎯 Signal quality")  # IMPROVED: new controls
cross_confirm_bars = st.sidebar.number_input(
    "Fresh-cross window (bars)", 1, 20, DEFAULTS["cross_confirm_bars"],
    help="Only signal if MA6/MA30 crossed within this many 5-min candles.",
)
cooldown_bars = st.sidebar.number_input(
    "Cooldown between repeat signals (bars)", 0, 50, DEFAULTS["cooldown_bars"],
    help="Suppress a new signal for the same symbol/direction within this many bars.",
)
min_avg_turnover = st.sidebar.number_input(
    "Min avg turnover per candle (₹)", 0, 50_000_000, DEFAULTS["min_avg_turnover"], step=50000,
    help="Filters out illiquid names likely to give false breakouts.",
)

st.sidebar.header("🚦 Scanner")
max_workers = st.sidebar.number_input("Parallel workers", 1, 12, DEFAULTS["max_workers"])
request_interval = st.sidebar.number_input(
    "API request spacing (sec)", 0.05, 2.0, float(DEFAULTS["request_interval"]), step=0.01
)
auto_scan = st.sidebar.checkbox("Auto scan on new 5-minute candle", value=True)

cfg = {
    "ma_fast": int(ma_fast),
    "ma_slow": int(ma_slow),
    "vol_multiplier": float(vol_multiplier),
    "vol_lookback": int(vol_lookback),
    "swing_left": int(swing_left),
    "swing_right": int(swing_right),
    "sl_buffer_pct": float(sl_buffer_pct),
    "atr_period": int(atr_period),
    "min_rr": float(min_rr),
    "max_workers": int(max_workers),
    "cross_confirm_bars": int(cross_confirm_bars),
    "cooldown_bars": int(cooldown_bars),
    "min_avg_turnover": float(min_avg_turnover),
}


# ============================================================
# DASHBOARD
# ============================================================
now = now_ist()
status = market_status()
closed_candle = last_closed_5m_ist(now)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Market", status)
c2.metric("NSE Equities", f"{len(st.session_state.instruments):,}")
c3.metric("Signals", len(st.session_state.signals))
c4.metric("Last closed 5M", closed_candle.strftime("%H:%M"))


# ============================================================
# NEW-DAY SESSION CLEANUP
# ============================================================
today_key = now.date().isoformat()
if st.session_state.cleanup_date is None:
    st.session_state.cleanup_date = today_key
elif st.session_state.cleanup_date != today_key:
    st.session_state.signals = []
    st.session_state.last_processed_candle = None
    st.session_state.diagnostics = []
    st.session_state.cleanup_date = today_key


# ============================================================
# LOAD INSTRUMENTS  (IMPROVED: cached loader, no token needed)
# ============================================================
if st.session_state.connected and st.session_state.instruments.empty:
    try:
        with st.spinner("Loading official Upstox NSE instrument master..."):
            st.session_state.instruments = get_instrument_master()
        st.success(f"Loaded {len(st.session_state.instruments):,} NSE equity instruments.")
    except Exception as exc:
        st.error(f"Instrument loading failed: {exc}")


# ============================================================
# SCAN CONTROLS
# ============================================================
manual_scan = st.button("🔎 Scan now", use_container_width=True)

if manual_scan and not st.session_state.connected:
    st.warning("Connect to Upstox first.")


# ============================================================
# AUTO / MANUAL SCAN
# ============================================================
should_scan = False

if st.session_state.connected and not st.session_state.instruments.empty:
    candle_key = closed_candle.isoformat()

    if manual_scan:
        should_scan = True
    elif (
        auto_scan
        and status == "OPEN"
        and st.session_state.last_processed_candle != candle_key
    ):
        should_scan = True


if should_scan:
    try:
        client = UpstoxClient(st.session_state.token, min_interval=float(request_interval))

        with st.spinner(
            f"Scanning {len(st.session_state.instruments):,} equities for "
            f"{closed_candle.strftime('%H:%M')}..."
        ):
            new_signals, diagnostics = run_full_scan(client, st.session_state.instruments, cfg)

        st.session_state.last_processed_candle = candle_key
        st.session_state.last_scan_time = now_ist().strftime("%H:%M:%S")
        st.session_state.diagnostics = diagnostics

        existing = {
            (item["symbol"], str(item["candle_time"])) for item in st.session_state.signals
        }

        # IMPROVED: cooldown — don't re-add a same-direction signal for a
        # symbol if one already fired recently. Prevents the signal table
        # filling up with the same name repeating every 5 minutes while a
        # trend simply continues.
        last_by_symbol = {}
        for item in st.session_state.signals:
            prev = last_by_symbol.get(item["symbol"])
            if prev is None or pd.to_datetime(item["candle_time"]) > pd.to_datetime(prev["candle_time"]):
                last_by_symbol[item["symbol"]] = item

        cooldown_delta = timedelta(minutes=5 * cfg["cooldown_bars"])
        added = 0

        for signal in new_signals:
            key = (signal["symbol"], str(signal["candle_time"]))
            if key in existing:
                continue

            prev = last_by_symbol.get(signal["symbol"])
            if prev and prev["direction"] == signal["direction"]:
                prev_time = pd.to_datetime(prev["candle_time"])
                cur_time = pd.to_datetime(signal["candle_time"])
                if (cur_time - prev_time) < cooldown_delta:
                    continue

            st.session_state.signals.append(signal)
            existing.add(key)
            last_by_symbol[signal["symbol"]] = signal
            added += 1

        st.success(f"Scan complete: {added} new signals.")

    except Exception as exc:
        st.error(f"Scan failed: {exc}")


# ============================================================
# SIGNAL TABLE
# ============================================================
st.subheader("📊 Signals")

if st.session_state.signals:
    signal_df = pd.DataFrame(st.session_state.signals)

    display_cols = [
        "symbol", "direction", "entry", "sl", "tp1", "tp2", "rr", "score",
        "ma6", "ma30", "vol_ratio", "avg_turnover", "candle_time", "status",
    ]
    display_cols = [c for c in display_cols if c in signal_df.columns]

    # IMPROVED: sort by most recent / highest score first for usability.
    signal_df = signal_df.sort_values(
        ["candle_time", "score"], ascending=[False, False]
    ).reset_index(drop=True)

    st.dataframe(signal_df[display_cols], use_container_width=True, hide_index=True)

    options = [
        f"{row.symbol} | {row.direction} | {row.candle_time}"
        for row in signal_df.itertuples()
    ]
    selected = st.selectbox("Signal details", options)
    symbol, direction, candle_time = selected.split(" | ", 2)

    matches = signal_df[
        (signal_df["symbol"] == symbol)
        & (signal_df["direction"] == direction)
        & (signal_df["candle_time"].astype(str) == candle_time)
    ]

    if not matches.empty:
        record = matches.iloc[0]

        st.subheader(f"{record['symbol']} — {record['direction']}")

        a, b, c, d, e = st.columns(5)
        a.metric("Entry", record["entry"])
        b.metric("SL", record["sl"])
        c.metric("TP1", record["tp1"])
        d.metric("TP2", record["tp2"] if pd.notna(record["tp2"]) else "N/A")
        e.metric("RR", f"1:{record['rr']}")

        st.write(
            f"**MA6:** {record['ma6']}  |  **MA30:** {record['ma30']}  |  "
            f"**Volume:** {record['vol_ratio']}x  |  **Setup score:** {record['score']}/100"
        )
        st.info(record["smc_reason"])

        # ------------------------------------
        # CHART
        # ------------------------------------
        instrument_match = st.session_state.instruments[
            st.session_state.instruments["trading_symbol"] == record["symbol"]
        ]

        if not instrument_match.empty:
            instrument_key = instrument_match.iloc[0]["instrument_key"]
            try:
                chart_client = UpstoxClient(
                    st.session_state.token, min_interval=float(request_interval)
                )
                chart_df = chart_client.fetch_5m(instrument_key)

                if not chart_df.empty:
                    chart_df = chart_df.tail(80)
                    chart_df = add_indicators(chart_df, cfg["ma_fast"], cfg["ma_slow"], cfg["atr_period"])

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=chart_df["timestamp"], open=chart_df["open"], high=chart_df["high"],
                        low=chart_df["low"], close=chart_df["close"], name="5M",
                    ))
                    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ma6"], mode="lines", name="MA6"))
                    fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["ma30"], mode="lines", name="MA30"))

                    fig.add_hline(y=float(record["entry"]), line_dash="solid", annotation_text="ENTRY")
                    fig.add_hline(y=float(record["sl"]), line_dash="dot", annotation_text="SL")
                    fig.add_hline(y=float(record["tp1"]), line_dash="dot", annotation_text="TP1")
                    if pd.notna(record["tp2"]):
                        fig.add_hline(y=float(record["tp2"]), line_dash="dot", annotation_text="TP2")

                    fig.update_layout(
                        height=650, xaxis_rangeslider_visible=False,
                        title=f"{record['symbol']} — 5-minute chart",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.warning(f"Chart data unavailable: {exc}")

else:
    st.info("No signals yet. Connect Upstox and run a scan during market hours.")


# ============================================================
# DIAGNOSTICS
# ============================================================
if st.session_state.diagnostics:
    with st.expander(f"Diagnostics ({len(st.session_state.diagnostics)} failures)"):
        st.dataframe(pd.DataFrame(st.session_state.diagnostics), use_container_width=True, hide_index=True)

st.caption(
    "Data source: Upstox only • "
    f"Last scan: {st.session_state.last_scan_time} IST • "
    "Orders are disabled"
)


# ============================================================
# AUTO REFRESH
# ============================================================
if auto_scan and status == "OPEN":
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=10_000, key="nse_auto_refresh")
    except ImportError:
        st.warning("Install streamlit-autorefresh for automatic scanning.")
