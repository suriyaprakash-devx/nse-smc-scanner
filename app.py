"""
NSE Semi-Algo PDH/PDL Breakout Scanner — Upstox API
Analysis + alerts only. NEVER places orders.

Run:
    streamlit run app.py

Environment variables (recommended):
    UPSTOX_ACCESS_TOKEN=your_daily_access_token
    TELEGRAM_BOT_TOKEN=...
    TELEGRAM_CHAT_ID=...
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=...
    SMTP_PASSWORD=...
    ALERT_EMAIL_TO=...

Optional:
    MAX_STOCKS=1000
    VOLUME_LOOKBACK=20
    VOLUME_MULTIPLIER=1.5
"""

from __future__ import annotations

import json
import math
import os
import smtplib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, date, time as dtime, timedelta
from email.message import EmailMessage
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

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


def sleep_until_next_scan() -> datetime:
    now = now_ist()
    candidates = []
    if now.date().weekday() < 5:
        candidates = [x for x in scheduled_scan_times(now.date()) if x > now]
    if candidates:
        target = candidates[0]
    else:
        d = now.date() + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        target = market_open(d)
    seconds = max(0.0, (target - now).total_seconds())
    if seconds > 0:
        time.sleep(seconds)
    return target


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
            if r.status_code == 401:
                raise RuntimeError("UPSTOX_TOKEN_EXPIRED")
            if r.status_code == 429 or 500 <= r.status_code < 600:
                last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                time.sleep(min(8, 1.5 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except RuntimeError:
            raise
        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(min(8, 1.5 ** attempt))

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
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    return pd.DataFrame(rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def historical_candles(
    token: str,
    instrument_key: str,
    interval: int,
    to_day: date,
    from_day: date,
) -> pd.DataFrame:
    path = (
        f"/v3/historical-candle/{quote(instrument_key, safe='')}/minutes/"
        f"{interval}/{to_day.isoformat()}/{from_day.isoformat()}"
    )
    return parse_candles(api_get(path, token))


def intraday_candles(token: str, instrument_key: str, interval: int = 10) -> pd.DataFrame:
    path = f"/v3/historical-candle/intraday/{quote(instrument_key, safe='')}/minutes/{interval}"
    return parse_candles(api_get(path, token))


def previous_trading_day_levels(token: str, instrument: Instrument, today: date) -> Level | None:
    # Daily candles are less request-efficient here than batching, but this
    # implementation uses the V3 daily endpoint per symbol and is simple.
    # It is called once per trading day, before market open.
    d1 = today - timedelta(days=10)
    df = historical_candles(token, instrument.instrument_key, 1440, today, d1)
    if df.empty:
        # V3 "days/1" is preferable; retry using the documented daily unit.
        path = (
            f"/v3/historical-candle/{quote(instrument.instrument_key, safe='')}/days/"
            f"1/{today.isoformat()}/{d1.isoformat()}"
        )
        df = parse_candles(api_get(path, token))
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


def save_levels(levels: dict[str, Any]) -> None:
    PD_FILE.write_text(json.dumps(levels, indent=2), encoding="utf-8")


def load_levels() -> dict[str, Any]:
    if not PD_FILE.exists():
        return {}
    try:
        return json.loads(PD_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


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

    save_levels(levels)
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


def send_telegram(text: str) -> bool:
    bot = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot or not chat:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot}/sendMessage",
            json={"chat_id": chat, "text": text},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False


def send_email(subject: str, body: str) -> bool:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    to = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not all([host, user, password, to]):
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception:
        return False


def send_alert(symbol: str, direction: str, level: dict[str, Any], row: pd.Series,
               avg_volume: float, ratio: float, signal_time: datetime) -> tuple[bool, bool]:
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
        f"Reason: {'PDH' if direction == 'BUY' else 'PDL'} breakout + high volume"
    )
    tg = send_telegram(text)
    em = send_email(f"NSE {direction} — {symbol}", text)
    return tg, em


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
            return {
                "Symbol": ins.symbol, "LTP": float(df.iloc[-1].close),
                "PDH": level["pdh"], "PDL": level["pdl"],
                "Current 10m Candle": "No completed 10m candle",
                "Direction": "WAIT", "Breakout": None,
                "Volume": None, "Average Volume": None, "Volume Ratio": None,
                "Candle Time": None, "Signal Time": scheduled_time,
                "Alert Status": "No confirmation candle"
            }

        row = completed.iloc[-1]
        prior = completed.iloc[:-1]

        # Average is calculated from prior completed 10-min candles only.
        avg_volume = float(prior.tail(VOLUME_LOOKBACK)["volume"].mean()) if not prior.empty else math.nan
        ratio = float(row.volume / avg_volume) if avg_volume and not math.isnan(avg_volume) else math.nan

        # "Cross since previous scan": use the candle's high/low for crossing,
        # and close for confirmation. This avoids missing an intraperiod cross.
        buy = row.close > level["pdh"] and row.high >= level["pdh"] and ratio >= VOLUME_MULTIPLIER
        sell = row.close < level["pdl"] and row.low <= level["pdl"] and ratio >= VOLUME_MULTIPLIER

        direction = "BUY" if buy else "SELL" if sell else "WAIT"
        breakout = float(row.close) if direction != "WAIT" else None

        return {
            "Symbol": ins.symbol,
            "LTP": float(row.close),
            "PDH": float(level["pdh"]),
            "PDL": float(level["pdl"]),
            "Current 10m Candle": f"{row.open:.2f} / {row.high:.2f} / {row.low:.2f} / {row.close:.2f}",
            "Direction": direction,
            "Breakout": breakout,
            "Volume": float(row.volume),
            "Average Volume": avg_volume if not math.isnan(avg_volume) else None,
            "Volume Ratio": ratio if not math.isnan(ratio) else None,
            "Candle Time": f"{row.timestamp:%H:%M}–{row.end:%H:%M}",
            "Signal Time": scheduled_time.strftime("%H:%M IST"),
            "Alert Status": "VALID" if direction != "WAIT" else "WAIT",
            "_row": row.to_dict(),
        }
    except RuntimeError:
        raise
    except Exception:
        return None


def scan_all(token: str, instruments: list[Instrument], levels: dict[str, Any],
             scheduled_time: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
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
                    results.append(x)
                    if x["Direction"] in ("BUY", "SELL"):
                        signals.append(x)
            except RuntimeError as e:
                if str(e) == "UPSTOX_TOKEN_EXPIRED":
                    raise
            except Exception:
                pass

    results.sort(key=lambda x: x["Symbol"])
    signals.sort(key=lambda x: (x["Direction"], x["Symbol"]))
    return results, signals


def process_alerts(signals: list[dict[str, Any]], levels: dict[str, Any],
                   scheduled_time: datetime, state: dict[str, Any]) -> int:
    sent = 0
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
        tg, em = send_alert(x["Symbol"], x["Direction"], lv, row, avg, ratio, scheduled_time)

        state["alerts"][key] = {
            "symbol": x["Symbol"],
            "direction": x["Direction"],
            "level": key_level,
            "candle_time": x["Candle Time"],
            "signal_time": scheduled_time.isoformat(),
            "telegram": tg,
            "email": em,
        }
        x["Alert Status"] = f"ALERT SENT (Telegram={tg}, Email={em})"
        sent += 1
    state["last_scan"] = scheduled_time.isoformat()
    save_state(state)
    return sent


def prepare_today(token: str, instruments: list[Instrument], force: bool = False) -> tuple[dict[str, Any], int]:
    today = now_ist().date()
    existing = load_levels()
    if not force and existing:
        source_dates = {v.get("source_date") for v in existing.values()}
        if len(source_dates) == 1 and next(iter(source_dates)) != str(today):
            return existing, 0
    return build_levels_parallel(token, instruments, today)


def render_dashboard(results: list[dict[str, Any]], scan_time: datetime) -> None:
    if not results:
        st.info("No stock data returned for this scan.")
        return
    df = pd.DataFrame(results)
    if "_row" in df.columns:
        df = df.drop(columns=["_row"])
    cols = [
        "Symbol", "LTP", "PDH", "PDL", "Current 10m Candle",
        "Direction", "Breakout", "Volume", "Average Volume",
        "Volume Ratio", "Candle Time", "Signal Time", "Alert Status"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)
    st.caption(f"Last scheduled scan: {scan_time:%Y-%m-%d %H:%M:%S %Z}")


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
    st.sidebar.write(f"Volume rule: ≥ {VOLUME_MULTIPLIER:.1f}× average")
    st.sidebar.write("Schedule: 09:15, 09:25, …, 15:25 IST")

    if not token:
        st.warning("Enter the Upstox access token in the sidebar.")
        st.stop()

    if "results" not in st.session_state:
        st.session_state.results = []
    if "scan_time" not in st.session_state:
        st.session_state.scan_time = None
    if "status" not in st.session_state:
        st.session_state.status = "Ready"

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
                st.session_state.instrument_count = len(ins)
            except Exception as e:
                st.error(f"Instrument download failed: {e}")
    with b3:
        if st.button("Prepare PDH/PDL"):
            try:
                ins = load_instruments(int(max_stocks))
                levels, errors = build_levels_parallel(token, ins, now_ist().date())
                st.success(f"Prepared {len(levels)} levels; {errors} symbols failed.")
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Preparation failed: {e}")

    levels = load_levels()
    st.write(f"Stored PDH/PDL levels: **{len(levels)}**")

    if st.button("Run Scheduled Scan Now"):
        target = latest_completed_schedule(now_ist()) or now_ist().replace(second=0, microsecond=0)
        try:
            ins = load_instruments(int(max_stocks))
            if not levels or any(v.get("source_date") == str(now_ist().date()) for v in levels.values()) is False:
                levels, _ = build_levels_parallel(token, ins, now_ist().date())
            results, signals = scan_all(token, ins, levels, target)
            state = load_state()
            sent = process_alerts(signals, levels, target, state)
            st.session_state.results = results
            st.session_state.scan_time = target
            st.session_state.status = f"Scan complete — {len(signals)} candidates, {sent} new alerts"
        except RuntimeError as e:
            st.session_state.status = str(e)
            st.error(str(e))
        except Exception as e:
            st.error(f"Scan failed: {e}")

    if st.session_state.results:
        render_dashboard(st.session_state.results, st.session_state.scan_time)

    st.divider()
    st.subheader("Automatic scheduler")
    st.info(
        "For unattended operation, keep this Streamlit process running. "
        "The scheduler wakes only at the exact 10-minute boundaries. "
        "The browser does not need to refresh every second."
    )

    if st.button("Start Automatic Scheduler"):
        st.session_state.scheduler_started = True

    if st.session_state.get("scheduler_started"):
        st.success("Automatic scheduler is active.")
        now = now_ist()
        target = latest_completed_schedule(now)
        if target and st.session_state.get("auto_last_target") != target.isoformat():
            try:
                ins = load_instruments(int(max_stocks))
                levels = load_levels()
                # Prepare levels once if today's levels are missing.
                if not levels or not any(v.get("source_date") == str(now.date()) for v in levels.values()):
                    levels, _ = build_levels_parallel(token, ins, now.date())

                results, signals = scan_all(token, ins, levels, target)
                state = load_state()
                sent = process_alerts(signals, levels, target, state)
                st.session_state.results = results
                st.session_state.scan_time = target
                st.session_state.auto_last_target = target.isoformat()
                st.session_state.status = f"Auto scan: {target:%H:%M} — {len(signals)} candidates, {sent} alerts"
                st.rerun()
            except RuntimeError as e:
                st.session_state.status = str(e)
                st.error(str(e))
            except Exception as e:
                st.error(f"Automatic scan error: {e}")

        # Streamlit reruns are used only to keep the dashboard alive. The
        # actual API scan is gated by the schedule above; there is no API
        # polling loop.
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
