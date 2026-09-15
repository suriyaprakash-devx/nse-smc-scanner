"""
NSE Semi-Algo PDH/PDL Breakout Scanner — Upstox API
Analysis + alerts only. NEVER places orders.

Run:
    streamlit run app.py

Environment variables (recommended):
    UPSTOX_ACCESS_TOKEN=your_daily_access_token
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...

Optional:
    MAX_STOCKS=1000
    VOLUME_LOOKBACK=20
    VOLUME_MULTIPLIER=1.5
    MAX_WORKERS=8

Alerts are delivered via Telegram only. Configure TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID to receive them; without them, breakout signals still show
in the dashboard table but nothing is pushed anywhere.

Buy vs. sell volume note:
    Upstox's public historical-candle API returns OHLCV only — it does not
    tag individual trades as buyer- or seller-initiated, so there is no way
    to get a literal "number of buyers vs. number of sellers" from it. What
    this script computes instead is a standard proxy used by most retail
    scanners: the Close Location Value (CLV) method, also used inside
    Chaikin Money Flow. Volume for the candle is split according to where
    the close sits within the candle's high-low range:

        buy_fraction  = (close - low) / (high - low)
        buy_volume    = candle_volume * buy_fraction
        sell_volume   = candle_volume * (1 - buy_fraction)
        volume_delta  = buy_volume - sell_volume

    A close near the high implies most of the candle's volume was
    aggressive buying; a close near the low implies the opposite. This is
    an approximation, not tape/order-flow data — treat it as directional
    confirmation, not an exact trade count.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

API_BASE = "https://api.upstox.com"
NSE_INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

DATA_DIR = Path("scanner_data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "alert_state.json"
PD_FILE = DATA_DIR / "previous_day_levels.json"

VOLUME_LOOKBACK = int(os.getenv("VOLUME_LOOKBACK", "20"))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", "1.5"))
MAX_STOCKS = int(os.getenv("MAX_STOCKS", "1000"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))
HTTP_TIMEOUT = 12

# Upstox documents standard API limits of 50 req/s, 500/min and 2000/30 min
# for historical candles and other standard APIs. We deliberately use a
# conservative worker count and retry/backoff rather than hammering the API.
MAX_RETRIES = 4
ALERT_RETENTION_DAYS = 10

session = requests.Session()
session.headers.update({"Accept": "application/json"})
state_lock = Lock()


@dataclass
class Instrument:
    symbol: str
    instrument_key: str
    isin: str = ""


@dataclass
class Level:
    symbol: str
    instrument_key: str
    pdh: float
    pdl: float
    source_date: str


def now_ist() -> datetime:
    return datetime.now(IST)


def market_open(d: date) -> datetime:
    return datetime.combine(d, dtime(9, 15), IST)


def market_close(d: date) -> datetime:
    return datetime.combine(d, dtime(15, 30), IST)


def is_market_window(now: datetime) -> bool:
    return now.weekday() < 5 and market_open(now.date()) <= now <= market_close(now.date())


def scheduled_scan_times(d: date) -> list[datetime]:
    start = market_open(d)
    return [start + timedelta(minutes=10 * i) for i in range(39) if start + timedelta(minutes=10 * i) <= market_close(d)]


def latest_completed_schedule(now: datetime) -> datetime | None:
    if now.weekday() >= 5 or now < market_open(now.date()):
        return None
    times = scheduled_scan_times(now.date())
    eligible = [x for x in times if x <= now]
    return eligible[-1] if eligible else None


def token_from_env() -> str:
    return os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()


def api_get(path: str, token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    url = API_BASE + path
    last_error = ""

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
        except requests.RequestException as e:
            # Network-level failure (timeout, DNS, connection reset) - retry.
            last_error = str(e)
            time.sleep(min(8, 1.5 ** attempt))
            continue

        if r.status_code == 401:
            raise RuntimeError("UPSTOX_TOKEN_EXPIRED")
        if r.status_code == 429 or 500 <= r.status_code < 600:
            # Rate limited or a transient server error - back off and retry.
            last_error = f"HTTP {r.status_code}: {r.text[:300]}"
            time.sleep(min(8, 1.5 ** attempt))
            continue
        if r.status_code >= 400:
            # Any other 4xx (bad request, not found, etc.) won't be fixed by
            # retrying identically, so fail fast instead of burning through
            # MAX_RETRIES for something that will never succeed.
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

        return r.json()

    raise RuntimeError(last_error or "Upstox request failed")


def validate_token(token: str) -> dict[str, Any]:
    return api_get("/v2/user/profile", token)


@st.cache_data(ttl=3600, show_spinner=False)
def download_nse_instruments() -> list[dict[str, Any]]:
    r = requests.get(NSE_INSTRUMENT_URL, timeout=30)
    r.raise_for_status()
    import gzip
    raw = gzip.decompress(r.content)
    data = json.loads(raw.decode("utf-8"))
    if isinstance(data, dict):
        data = data.get("data", data.get("instruments", []))
    return data


def load_instruments(max_stocks: int) -> list[Instrument]:
    raw = download_nse_instruments()
    out = []
    seen = set()

    for x in raw:
        if x.get("segment") != "NSE_EQ":
            continue
        if x.get("instrument_type") != "EQ":
            continue
        key = x.get("instrument_key")
        symbol = x.get("trading_symbol") or x.get("symbol")
        if not key or not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Instrument(symbol=symbol, instrument_key=key, isin=x.get("isin", "")))

    # Full NSE_EQ universe is supported. MAX_STOCKS is an optional safety
    # throttle; set MAX_STOCKS=0 to scan the full list.
    if max_stocks > 0:
        out = out[:max_stocks]
    return out


def parse_candles(payload: dict[str, Any]) -> pd.DataFrame:
    candles = payload.get("data", {}).get("candles", [])
    rows = []
    for c in candles:
        if len(c) < 6:
            continue
        ts = pd.to_datetime(c[0], errors="coerce")
        if pd.isna(ts):
            continue
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        else:
            ts = ts.tz_convert(IST)
        rows.append({
            "timestamp": ts,
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def daily_candles(token: str, instrument_key: str, to_day: date, from_day: date) -> pd.DataFrame:
    path = (
        f"/v3/historical-candle/{quote(instrument_key, safe='')}/days/1/"
        f"{to_day.isoformat()}/{from_day.isoformat()}"
    )
    return parse_candles(api_get(path, token))


def intraday_candles(token: str, instrument_key: str, interval: int = 10) -> pd.DataFrame:
    path = f"/v3/historical-candle/intraday/{quote(instrument_key, safe='')}/minutes/{interval}"
    return parse_candles(api_get(path, token))


def previous_trading_day_levels(token: str, instrument: Instrument, today: date) -> Level | None:
    # Looks back 10 calendar days so weekends/holidays are naturally skipped
    # by taking the last row strictly before `today`.
    from_day = today - timedelta(days=10)
    df = daily_candles(token, instrument.instrument_key, today, from_day)
    df = df[df["timestamp"].dt.date < today]
    if df.empty:
        return None
    row = df.iloc[-1]
    return Level(
        symbol=instrument.symbol,
        instrument_key=instrument.instrument_key,
        pdh=float(row.high),
        pdl=float(row.low),
        source_date=str(row.timestamp.date()),
    )


def save_levels(levels: dict[str, Any], as_of: date) -> None:
    payload = {"as_of": as_of.isoformat(), "levels": levels}
    PD_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_levels() -> tuple[str | None, dict[str, Any]]:
    """Returns (as_of_date_str, levels_by_symbol). as_of is None if there's
    no file yet or it's in the legacy flat format."""
    if not PD_FILE.exists():
        return None, {}
    try:
        data = json.loads(PD_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None, {}
    if isinstance(data, dict) and "levels" in data:
        return data.get("as_of"), data.get("levels", {})
    return None, data if isinstance(data, dict) else {}


def levels_ready_for(today: date) -> dict[str, Any] | None:
    """Cached PDH/PDL levels, but only if they were already prepared for
    `today` - otherwise None so the caller knows to rebuild."""
    as_of, levels = load_levels()
    if levels and as_of == today.isoformat():
        return levels
    return None


def build_levels_parallel(token: str, instruments: list[Instrument], today: date) -> tuple[dict[str, Any], int]:
    levels = {}
    errors = 0

    def worker(ins: Instrument):
        try:
            lv = previous_trading_day_levels(token, ins, today)
            return ins.symbol, asdict(lv) if lv else None
        except Exception:
            return ins.symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, i) for i in instruments]
        for f in as_completed(futures):
            symbol, value = f.result()
            if value:
                levels[symbol] = value
            else:
                errors += 1

    save_levels(levels, today)
    return levels, errors


def load_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {"alerts": {}, "last_scan": None}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"alerts": {}, "last_scan": None}


def save_state(state: dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def alert_key(symbol: str, direction: str, level: float, source_date: str) -> str:
    return f"{source_date}|{symbol}|{direction}|{level:.8f}"


def prune_old_alerts(state: dict[str, Any], keep_days: int = ALERT_RETENTION_DAYS) -> None:
    """Drop alert-dedup entries older than keep_days so the state file
    doesn't grow forever."""
    cutoff = now_ist().date() - timedelta(days=keep_days)
    kept = {}
    for key, value in state.get("alerts", {}).items():
        date_part = key.split("|", 1)[0]
        try:
            key_date = date.fromisoformat(date_part)
        except ValueError:
            kept[key] = value  # unrecognized key format - keep to be safe
            continue
        if key_date >= cutoff:
            kept[key] = value
    state["alerts"] = kept


def buy_sell_volume_split(row: pd.Series) -> tuple[float, float, float]:
    """Approximate buy volume vs. sell volume for a single OHLCV candle
    using the Close Location Value (CLV) method (same idea as Chaikin
    Money Flow): the closer the close sits to the candle high, the more of
    the candle's volume is attributed to buyers, and vice versa.

    Returns (buy_volume, sell_volume, delta) where delta = buy - sell.
    This is a proxy for order flow, not actual buyer/seller trade counts -
    Upstox's historical-candle API doesn't expose that.
    """
    high, low, close, volume = float(row.high), float(row.low), float(row.close), float(row.volume)
    rng = high - low
    buy_fraction = (close - low) / rng if rng > 0 else 0.5
    buy_fraction = min(1.0, max(0.0, buy_fraction))
    buy_volume = volume * buy_fraction
    sell_volume = volume - buy_volume
    return buy_volume, sell_volume, buy_volume - sell_volume


def send_telegram(text: str) -> bool | None:
    """True if delivered, False if an attempt was made and failed, or None
    if Telegram isn't configured at all."""
    bot = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot or not chat:
        return None
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


def send_alert(symbol: str, direction: str, level: dict[str, Any], row: pd.Series,
               avg_volume: float, ratio: float, signal_time: datetime,
               buy_volume: float, sell_volume: float, delta: float) -> bool | None:
    candle_start = row.timestamp
    candle_end = candle_start + timedelta(minutes=10)
    text = (
        f"{signal_time:%H:%M} IST — {symbol} — {direction}\n"
        f"PDH: ₹{level['pdh']:.2f}\n"
        f"PDL: ₹{level['pdl']:.2f}\n"
        f"Breakout: ₹{row.close:.2f}\n"
        f"Volume Ratio: {ratio:.2f}×\n"
        f"Candle: {candle_start:%H:%M}–{candle_end:%H:%M}\n"
        f"Volume: {row.volume:,.0f}\n"
        f"Average Volume: {avg_volume:,.0f}\n"
        f"Est. Buy Volume: {buy_volume:,.0f}\n"
        f"Est. Sell Volume: {sell_volume:,.0f}\n"
        f"Volume Delta (Buy-Sell): {delta:+,.0f}\n"
        f"Reason: {'PDH' if direction == 'BUY' else 'PDL'} breakout + high volume "
        f"+ {'buyer' if direction == 'BUY' else 'seller'}-dominant volume delta"
    )
    return send_telegram(text)


def evaluate_stock(token: str, ins: Instrument, level: dict[str, Any],
                    scheduled_time: datetime) -> dict[str, Any] | None:
    try:
        df = intraday_candles(token, ins.instrument_key, 10)
        if df.empty:
            return None

        # At 09:15, no complete 10-min candle exists yet. From 09:25 onward,
        # the last candle whose end <= scheduled_time is the confirmation candle.
        df["end"] = df["timestamp"] + timedelta(minutes=10)
        completed = df[df["end"] <= scheduled_time].copy()
        if completed.empty:
            # No confirmation candle yet - nothing actionable to show for
            # this symbol at this scan. Skipped by the caller rather than
            # surfaced as a row, so the dashboard only lists real signals.
            return None

        row = completed.iloc[-1]
        prior = completed.iloc[:-1]

        # Average is calculated from prior completed 10-min candles only.
        avg_volume = float(prior.tail(VOLUME_LOOKBACK)["volume"].mean()) if not prior.empty else math.nan
        ratio = float(row.volume / avg_volume) if avg_volume and not math.isnan(avg_volume) else math.nan

        buy_volume, sell_volume, delta = buy_sell_volume_split(row)
        delta_pct = (delta / row.volume * 100.0) if row.volume else 0.0

        # row.close > pdh already implies row.high >= pdh (high is always >=
        # close), so the breakout test only needs the close + volume checks.
        # Volume delta must agree with direction (net buyers for a PDH
        # breakout, net sellers for a PDL breakdown) as an extra confirmation
        # on top of the raw volume-ratio spike.
        buy = row.close > level["pdh"] and ratio >= VOLUME_MULTIPLIER and delta > 0
        sell = row.close < level["pdl"] and ratio >= VOLUME_MULTIPLIER and delta < 0

        direction = "BUY" if buy else "SELL" if sell else None
        if direction is None:
            # Breakout/volume/delta conditions not all met - not a signal,
            # so skip it rather than returning a placeholder "WAIT" row.
            return None

        return {
            "Symbol": ins.symbol,
            "LTP": float(row.close),
            "PDH": float(level["pdh"]),
            "PDL": float(level["pdl"]),
            "Current 10m Candle": f"{row.open:.2f} / {row.high:.2f} / {row.low:.2f} / {row.close:.2f}",
            "Direction": direction,
            "Breakout": float(row.close),
            "Volume": float(row.volume),
            "Average Volume": avg_volume if not math.isnan(avg_volume) else None,
            "Volume Ratio": ratio if not math.isnan(ratio) else None,
            "Buy Volume": buy_volume,
            "Sell Volume": sell_volume,
            "Volume Delta": delta,
            "Delta %": delta_pct,
            "Candle Time": f"{row.timestamp:%H:%M}–{row.end:%H:%M}",
            "Signal Time": scheduled_time.strftime("%H:%M IST"),
            "Alert Status": "VALID",
            "_row": row.to_dict(),
        }
    except RuntimeError:
        raise
    except Exception:
        return None


def scan_all(token: str, instruments: list[Instrument], levels: dict[str, Any],
             scheduled_time: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (signals, signals) - kept as a pair for backward
    compatibility with callers that expect (results, signals); since
    evaluate_stock now only returns actionable rows, the two are the same
    list of BUY/SELL candidates."""
    signals = []

    def worker(ins: Instrument):
        lv = levels.get(ins.symbol)
        if not lv:
            return None
        return evaluate_stock(token, ins, lv, scheduled_time)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(worker, ins) for ins in instruments if ins.symbol in levels]
        for f in as_completed(futures):
            try:
                x = f.result()
                if x:
                    signals.append(x)
            except RuntimeError as e:
                if str(e) == "UPSTOX_TOKEN_EXPIRED":
                    raise
            except Exception:
                pass

    signals.sort(key=lambda x: (x["Direction"], x["Symbol"]))
    return signals, signals


def process_alerts(signals: list[dict[str, Any]], levels: dict[str, Any],
                    scheduled_time: datetime, state: dict[str, Any]) -> int:
    sent = 0
    with state_lock:
        for x in signals:
            lv = levels[x["Symbol"]]
            key_level = lv["pdh"] if x["Direction"] == "BUY" else lv["pdl"]
            key = alert_key(x["Symbol"], x["Direction"], key_level, lv["source_date"])
            if key in state["alerts"]:
                x["Alert Status"] = "DUPLICATE SUPPRESSED"
                continue

            row = pd.Series(x["_row"])
            avg = float(x["Average Volume"] or 0)
            ratio = float(x["Volume Ratio"] or 0)
            buy_vol = float(x["Buy Volume"] or 0)
            sell_vol = float(x["Sell Volume"] or 0)
            delta = float(x["Volume Delta"] or 0)
            delivered = send_alert(
                x["Symbol"], x["Direction"], lv, row, avg, ratio, scheduled_time,
                buy_vol, sell_vol, delta,
            )

            if delivered is None:
                x["Alert Status"] = "NOT SENT (Telegram not configured)"
            elif delivered:
                state["alerts"][key] = {
                    "symbol": x["Symbol"],
                    "direction": x["Direction"],
                    "level": key_level,
                    "candle_time": x["Candle Time"],
                    "signal_time": scheduled_time.isoformat(),
                }
                x["Alert Status"] = "ALERT SENT (Telegram)"
                sent += 1
            else:
                # Don't record a dedup key on failure, so the same breakout
                # is retried on the next scheduled scan instead of being
                # silently lost.
                x["Alert Status"] = "TELEGRAM FAILED (will retry next scan)"

        prune_old_alerts(state)
        state["last_scan"] = scheduled_time.isoformat()
        save_state(state)
    return sent


def render_dashboard(results: list[dict[str, Any]], scan_time: datetime, scanned_count: int) -> None:
    if not results:
        st.info(f"No breakout signals this scan (scanned {scanned_count} stocks with prepared levels).")
        return
    df = pd.DataFrame(results)
    if "_row" in df.columns:
        df = df.drop(columns=["_row"])
    cols = [
        "Symbol", "LTP", "PDH", "PDL", "Current 10m Candle",
        "Direction", "Breakout", "Volume", "Average Volume", "Volume Ratio",
        "Buy Volume", "Sell Volume", "Volume Delta", "Delta %",
        "Candle Time", "Signal Time", "Alert Status"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols].style.format({
            "LTP": "{:.2f}", "PDH": "{:.2f}", "PDL": "{:.2f}", "Breakout": "{:.2f}",
            "Volume": "{:,.0f}", "Average Volume": "{:,.0f}", "Volume Ratio": "{:.2f}",
            "Buy Volume": "{:,.0f}", "Sell Volume": "{:,.0f}", "Volume Delta": "{:+,.0f}",
            "Delta %": "{:+.1f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Last scheduled scan: {scan_time:%Y-%m-%d %H:%M:%S %Z} — "
        f"{scanned_count} stocks scanned, {len(results)} breakout signal(s) shown. "
        f"Buy/Sell Volume is an estimate (Close-Location-Value method), not tape data."
    )


def main():
    st.set_page_config(page_title="NSE PDH/PDL Semi-Algo Scanner", layout="wide")
    st.title("NSE PDH/PDL Breakout — Semi-Algo Scanner")
    st.caption("Analysis + alerts only. No order placement.")

    token = st.sidebar.text_input(
        "Upstox Access Token",
        value=token_from_env(),
        type="password",
        help="Paste today's Upstox access token or set UPSTOX_ACCESS_TOKEN.",
    )
    max_stocks = st.sidebar.number_input(
        "Max NSE stocks (0 = full NSE_EQ universe)",
        min_value=0, max_value=5000, value=MAX_STOCKS, step=100
    )
    st.sidebar.write(f"Volume rule: ≥ {VOLUME_MULTIPLIER:.1f}× average, delta-confirmed")
    st.sidebar.write("Schedule: 09:15, 09:25, …, 15:25 IST")
    st.sidebar.write("Alerts: Telegram only")
    st.sidebar.caption(
        "Buy/Sell Volume is estimated from each candle's close position "
        "(Close-Location-Value method) — Upstox candles don't include "
        "actual buyer/seller trade tags."
    )

    if not token:
        st.warning("Enter the Upstox access token in the sidebar.")
        st.stop()

    if "results" not in st.session_state:
        st.session_state.results = []
    if "scan_time" not in st.session_state:
        st.session_state.scan_time = None
    if "scanned_count" not in st.session_state:
        st.session_state.scanned_count = 0
    if "status" not in st.session_state:
        st.session_state.status = "Ready"

    today = now_ist().date()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("IST", now_ist().strftime("%H:%M:%S"))
    with c2:
        st.metric("Market", "OPEN" if is_market_window(now_ist()) else "CLOSED")
    with c3:
        st.metric("Status", st.session_state.status)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Validate Token"):
            try:
                p = validate_token(token)
                st.success(f"Token valid. Status: {p.get('status', 'success')}")
            except RuntimeError as e:
                st.error(str(e))
    with b2:
        if st.button("Load NSE Instruments"):
            try:
                ins = load_instruments(int(max_stocks))
                st.success(f"Loaded {len(ins)} NSE_EQ instruments.")
            except Exception as e:
                st.error(f"Instrument download failed: {e}")
    with b3:
        if st.button("Prepare PDH/PDL"):
            try:
                ins = load_instruments(int(max_stocks))
                levels, errors = build_levels_parallel(token, ins, today)
                st.success(f"Prepared {len(levels)} levels; {errors} symbols failed.")
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Preparation failed: {e}")

    levels = levels_ready_for(today)
    if levels:
        st.write(f"Stored PDH/PDL levels ready for today: **{len(levels)}**")
    else:
        _, stale = load_levels()
        note = f" ({len(stale)} cached from a previous day)" if stale else ""
        st.write(f"No PDH/PDL levels prepared for today yet.{note}")

    if st.button("Run Scheduled Scan Now"):
        target = latest_completed_schedule(now_ist()) or now_ist().replace(second=0, microsecond=0)
        try:
            ins = load_instruments(int(max_stocks))
            if not levels:
                levels, _ = build_levels_parallel(token, ins, today)
            signals, _ = scan_all(token, ins, levels, target)
            state = load_state()
            sent = process_alerts(signals, levels, target, state)
            st.session_state.results = signals
            st.session_state.scan_time = target
            st.session_state.scanned_count = len(levels)
            st.session_state.status = f"Scan complete — {len(signals)} signal(s), {sent} alert(s) delivered"
        except RuntimeError as e:
            st.session_state.status = str(e)
            st.error(str(e))
        except Exception as e:
            st.error(f"Scan failed: {e}")

    if st.session_state.scan_time:
        render_dashboard(st.session_state.results, st.session_state.scan_time, st.session_state.scanned_count)

    st.divider()
    st.subheader("Automatic scheduler")
    st.info(
        "For unattended operation, keep this browser tab open — the scan "
        "loop only runs while this Streamlit script is active. For real "
        "background/unattended operation, run the scan logic as a separate "
        "scheduled process (cron/systemd) instead of relying on an open tab."
    )

    if not st.session_state.get("scheduler_started"):
        if st.button("Start Automatic Scheduler"):
            st.session_state.scheduler_started = True
            st.rerun()
    else:
        if st.button("Stop Automatic Scheduler"):
            st.session_state.scheduler_started = False
            st.rerun()

    if st.session_state.get("scheduler_started"):
        st.success("Automatic scheduler is active.")
        now = now_ist()
        target = latest_completed_schedule(now)
        if target and st.session_state.get("auto_last_target") != target.isoformat():
            try:
                ins = load_instruments(int(max_stocks))
                current_levels = levels_ready_for(now.date())
                if not current_levels:
                    current_levels, _ = build_levels_parallel(token, ins, now.date())

                signals, _ = scan_all(token, ins, current_levels, target)
                state = load_state()
                sent = process_alerts(signals, current_levels, target, state)
                st.session_state.results = signals
                st.session_state.scan_time = target
                st.session_state.scanned_count = len(current_levels)
                st.session_state.auto_last_target = target.isoformat()
                st.session_state.status = f"Auto scan: {target:%H:%M} — {len(signals)} signal(s), {sent} delivered"
            except RuntimeError as e:
                st.session_state.status = str(e)
                st.error(str(e))
            except Exception as e:
                st.error(f"Automatic scan error: {e}")
            st.rerun()
        else:
            # Idle until close to the next scheduled boundary instead of
            # busy-polling every second. This only changes how often the
            # script reruns while idle - it does not change what gets
            # scanned or when.
            upcoming = [t for t in scheduled_scan_times(now.date()) if t > now] if now.weekday() < 5 else []
            wait_seconds = (upcoming[0] - now).total_seconds() if upcoming else 60
            time.sleep(max(1.0, min(wait_seconds, 30.0)))
            st.rerun()


if __name__ == "__main__":
    main()
