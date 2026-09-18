"""
SMC Intraday Signal Assistant — Upstox Edition
================================================
A single-file Streamlit app that:
  - Lets you type in an NSE stock name/symbol
  - Pulls live + historical intraday candles from the Upstox API
  - Runs a rule-based Smart Money Concepts (SMC) engine on CLOSED candles only
    (Break of Structure, Change of Character, Liquidity Sweeps, Order Blocks,
    Fair Value Gaps, Displacement)
  - Shows a clear BUY / SELL / WAIT signal with Entry, Stop Loss and Target
  - Tracks an active "paper" setup and tells you when to EXIT (SL/Target hit)
  - NEVER places any order. This is a decision-support tool only.

--------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------
1) pip install streamlit pandas numpy requests plotly pytz streamlit-autorefresh
2) streamlit run smc_upstox_app.py
3) In the sidebar, paste a valid Upstox API v2 access token
   (generate it via Upstox's OAuth login flow — this app does not do the
   OAuth dance for you, since that requires a redirect/callback server).
4) Type a stock name (e.g. "RELIANCE", "TCS", "HDFC BANK"), pick it from the
   matches, choose a timeframe, and click "Start Monitoring".

--------------------------------------------------------------------------
IMPORTANT NOTES / LIMITATIONS (read before using with real money)
--------------------------------------------------------------------------
- This is EDUCATIONAL / DECISION-SUPPORT software. It does not place, modify
  or cancel any order. Every trade decision and execution is yours.
- SMC concepts (BOS, CHoCH, OB, FVG, liquidity sweeps, displacement) are
  discretionary in nature. This engine encodes one reasonable, rule-based
  interpretation of them — not "the" definitive definition. Validate the
  chart yourself before acting on any signal.
- The engine only acts on fully CLOSED candles. The most recent, still-forming
  candle of your chosen timeframe is always dropped before analysis, so
  nothing here "repaints" using an incomplete bar.
- Upstox's exact REST endpoint paths/params can change between API versions.
  This file targets the Upstox API v2 conventions. If your account is on a
  different API version, adjust `UPSTOX_BASE` and the two fetch functions.
- Historical intraday 1-minute data availability is limited by Upstox
  (typically the current + a few recent trading days). We fetch 1-minute
  data and resample it locally into your chosen timeframe (3/5/15 min) so
  we aren't dependent on Upstox supporting every timeframe natively.
"""

import time
from datetime import datetime, timedelta, time as dtime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ============================================================================
# CONSTANTS
# ============================================================================
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"

st.set_page_config(page_title="SMC Intraday Signal Assistant", layout="wide")


# ============================================================================
# TIME / MARKET HOURS HELPERS
# ============================================================================
def now_ist():
    return datetime.now(IST).replace(tzinfo=None)


def is_market_open():
    n = now_ist()
    if n.weekday() >= 5:
        return False
    return MARKET_OPEN <= n.time() <= MARKET_CLOSE


def market_status_label():
    if not is_market_open():
        n = now_ist()
        if n.weekday() >= 5:
            return "🔴 CLOSED (Weekend)"
        if n.time() < MARKET_OPEN:
            return "🟡 PRE-MARKET (opens 9:15 AM)"
        return "🔴 CLOSED (market ended 3:30 PM)"
    return "🟢 OPEN"


# ============================================================================
# INSTRUMENT MASTER (symbol -> Upstox instrument_key)
# ============================================================================
@st.cache_data(ttl=24 * 3600, show_spinner="Loading NSE instrument list...")
def load_instruments():
    df = pd.read_csv(INSTRUMENTS_URL)
    df = df[df["instrument_type"] == "EQ"]
    df = df[["instrument_key", "tradingsymbol", "name"]].dropna()
    return df.reset_index(drop=True)


def search_instrument(df, query):
    q = query.strip().upper()
    if not q:
        return pd.DataFrame()
    mask = df["tradingsymbol"].str.upper().str.contains(q, na=False) | df["name"].str.upper().str.contains(
        q, na=False
    )
    return df[mask].head(25)


# ============================================================================
# DATA FETCH (Upstox v2) — historical + intraday, merged and resampled
# ============================================================================
def api_headers(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def candles_to_df(candles):
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    if not candles:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles, columns=cols)
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
    df["timestamp"] = ts
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=20, show_spinner=False)
def fetch_intraday_1m(instrument_key, token):
    url = f"{UPSTOX_BASE}/historical-candle/intraday/{instrument_key}/1minute"
    r = requests.get(url, headers=api_headers(token), timeout=10)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return candles_to_df(candles)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_historical_1m(instrument_key, token, from_date, to_date):
    url = f"{UPSTOX_BASE}/historical-candle/{instrument_key}/1minute/{to_date}/{from_date}"
    r = requests.get(url, headers=api_headers(token), timeout=10)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return candles_to_df(candles)


def resample(df, tf_minutes):
    d = df.set_index("timestamp")
    o = d.resample(f"{tf_minutes}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    o = o.dropna(subset=["open", "high", "low", "close"])
    return o.reset_index()


def get_completed_candles(instrument_key, token, tf_minutes):
    """Fetch 1-min data, resample to tf_minutes, and DROP the still-forming
    last candle so the engine never looks at an incomplete bar."""
    today = now_ist().strftime("%Y-%m-%d")
    from_date = (now_ist() - timedelta(days=6)).strftime("%Y-%m-%d")

    hist = pd.DataFrame()
    intraday = pd.DataFrame()
    err = None
    try:
        hist = fetch_historical_1m(instrument_key, token, from_date, today)
    except Exception as e:
        err = str(e)
    try:
        intraday = fetch_intraday_1m(instrument_key, token)
    except Exception as e:
        err = str(e)

    df = pd.concat([hist, intraday], ignore_index=True)
    if df.empty:
        return df, err
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    df = resample(df, tf_minutes)
    if df.empty:
        return df, err

    n = now_ist()
    last_start = df.iloc[-1]["timestamp"]
    if last_start + timedelta(minutes=tf_minutes) > n:
        df = df.iloc[:-1]  # drop forming candle -> no look-ahead / no repaint

    return df.reset_index(drop=True), err


# ============================================================================
# SMC ANALYSIS ENGINE
# ============================================================================
def find_swings(df, left=2, right=2):
    """Fractal swing highs/lows using a symmetric lookback window."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    sh = [False] * n
    sl = [False] * n
    for i in range(left, n - right):
        wh = highs[i - left : i + right + 1]
        wl = lows[i - left : i + right + 1]
        if highs[i] == wh.max() and list(wh).count(highs[i]) == 1:
            sh[i] = True
        if lows[i] == wl.min() and list(wl).count(lows[i]) == 1:
            sl[i] = True
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"] = sl
    return out


def market_structure(df):
    """Walk candles chronologically; whenever a close breaks the most recent
    confirmed swing high/low, log a BOS (trend continuation) or CHoCH (trend
    reversal) event. Each swing level can only trigger one event (no dupes)."""
    events = []
    swing_highs = df[df["swing_high"]][["high"]].copy()
    swing_lows = df[df["swing_low"]][["low"]].copy()

    trend = None
    broken_sh, broken_sl = set(), set()

    for i in range(len(df)):
        close = df["close"].iloc[i]
        ts = df["timestamp"].iloc[i]

        sh_before = swing_highs[swing_highs.index < i]
        sl_before = swing_lows[swing_lows.index < i]
        cur_sh = sh_before["high"].iloc[-1] if len(sh_before) else None
        cur_sl = sl_before["low"].iloc[-1] if len(sl_before) else None

        if cur_sh is not None and close > cur_sh and cur_sh not in broken_sh:
            etype = "BOS" if trend in (None, "up") else "CHoCH"
            events.append(dict(idx=i, timestamp=ts, type=etype, direction="bull", price=cur_sh))
            trend = "up"
            broken_sh.add(cur_sh)
        if cur_sl is not None and close < cur_sl and cur_sl not in broken_sl:
            etype = "BOS" if trend in (None, "down") else "CHoCH"
            events.append(dict(idx=i, timestamp=ts, type=etype, direction="bear", price=cur_sl))
            trend = "down"
            broken_sl.add(cur_sl)

    return events, trend


def detect_liquidity_sweeps(df, lookback=20):
    """A sweep = price wicks beyond a recent swing extreme but CLOSES back
    inside it -> stop-hunt / liquidity grab, often precedes a reversal."""
    sweeps = []
    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback : i]
        recent_high = window.loc[window["swing_high"], "high"].max() if window["swing_high"].any() else None
        recent_low = window.loc[window["swing_low"], "low"].min() if window["swing_low"].any() else None
        row = df.iloc[i]
        if recent_high is not None and row["high"] > recent_high and row["close"] < recent_high:
            sweeps.append(
                dict(idx=i, timestamp=row["timestamp"], type="sell_side_sweep", level=float(recent_high), direction="bullish")
            )
        if recent_low is not None and row["low"] < recent_low and row["close"] > recent_low:
            sweeps.append(
                dict(idx=i, timestamp=row["timestamp"], type="buy_side_sweep", level=float(recent_low), direction="bearish")
            )
    return sweeps


def detect_fvg(df):
    """3-candle Fair Value Gap / imbalance."""
    fvgs = []
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if c1["high"] < c3["low"]:
            fvgs.append(dict(idx=i, timestamp=df.iloc[i - 1]["timestamp"], type="bullish_fvg", top=float(c3["low"]), bottom=float(c1["high"])))
        if c1["low"] > c3["high"]:
            fvgs.append(dict(idx=i, timestamp=df.iloc[i - 1]["timestamp"], type="bearish_fvg", top=float(c1["low"]), bottom=float(c3["high"])))
    return fvgs


def detect_displacement(df, atr_period=14, mult=1.5):
    """Momentum candle: body notably larger than recent average range."""
    body = (df["close"] - df["open"]).abs()
    rng = df["high"] - df["low"]
    atr = rng.rolling(atr_period, min_periods=5).mean()
    return body > (atr * mult)


def detect_order_blocks(df, events, disp_series):
    """For each structural break, find the last opposite-colour candle right
    before the displacement leg that caused it -> that candle is the OB."""
    obs = []
    for ev in events:
        i, direction = ev["idx"], ev["direction"]
        start = max(0, i - 10)
        found = None
        for j in range(i, start, -1):
            if j < len(disp_series) and bool(disp_series.iloc[j]):
                k = j - 1
                if k >= 0:
                    o, c = df["open"].iloc[k], df["close"].iloc[k]
                    if direction == "bull" and c < o:
                        found = k
                    elif direction == "bear" and c > o:
                        found = k
                break
        if found is not None:
            row = df.iloc[found]
            obs.append(
                dict(
                    idx=found,
                    timestamp=row["timestamp"],
                    type="bullish_ob" if direction == "bull" else "bearish_ob",
                    top=float(row["high"]),
                    bottom=float(row["low"]),
                    event_idx=i,
                )
            )
    return obs


def generate_signal(df):
    """Combine sweep -> CHoCH/BOS confirmation -> OB/FVG retracement zone
    -> confirmation candle, into a single actionable BUY/SELL/WAIT signal."""
    df = find_swings(df)
    events, trend = market_structure(df)
    sweeps = detect_liquidity_sweeps(df)
    fvgs = detect_fvg(df)
    disp = detect_displacement(df)
    obs = detect_order_blocks(df, events, disp)

    result = dict(
        signal="WAIT", reason="", entry=None, sl=None, target=None, trend=trend,
        events=events, sweeps=sweeps, fvgs=fvgs, obs=obs, disp=disp, df=df,
    )

    if len(df) < 30:
        result["reason"] = "Collecting data — need more completed candles before analysis is reliable."
        return result

    last_price = float(df["close"].iloc[-1])
    last_idx = len(df) - 1
    lookback_bars = 15

    recent_sweeps = [s for s in sweeps if s["idx"] >= last_idx - lookback_bars]

    bullish_setup = None
    bearish_setup = None
    for sw in recent_sweeps:
        if sw["direction"] == "bullish":
            confirm = [e for e in events if e["idx"] > sw["idx"] and e["direction"] == "bull"]
            if confirm:
                bullish_setup = (sw, confirm[0])
        if sw["direction"] == "bearish":
            confirm = [e for e in events if e["idx"] > sw["idx"] and e["direction"] == "bear"]
            if confirm:
                bearish_setup = (sw, confirm[0])

    def nearest_zone(after_idx, direction):
        pool = obs + fvgs
        cands = [
            z for z in pool
            if z["idx"] >= after_idx
            and (("bullish" in z["type"]) if direction == "bull" else ("bearish" in z["type"]))
        ]
        if not cands:
            return None
        cands.sort(key=lambda z: abs(last_price - (z["top"] + z["bottom"]) / 2))
        return cands[0]

    if bullish_setup:
        sw, choch = bullish_setup
        zone = nearest_zone(choch["idx"], "bull")
        if zone:
            top, bottom = zone["top"], zone["bottom"]
            in_zone = bottom <= last_price <= top * 1.002
            confirm_candle = df["close"].iloc[-1] > df["open"].iloc[-1]
            if in_zone and confirm_candle:
                entry = last_price
                sl = min(sw["level"], bottom) * 0.999
                risk = max(entry - sl, 0.01)
                future_highs = df.loc[df["swing_high"], "high"]
                targets = future_highs[future_highs > entry]
                target = float(targets.min()) if len(targets) else entry + risk * 2
                result.update(
                    signal="BUY",
                    reason=(
                        f"Buy-side liquidity swept at {sw['level']:.2f}, followed by a bullish "
                        f"{choch['type']}. Price reacted from a {zone['type'].replace('_', ' ')} zone "
                        f"({bottom:.2f}–{top:.2f}) with a bullish confirmation candle."
                    ),
                    entry=round(entry, 2), sl=round(sl, 2), target=round(target, 2),
                )
                return result
            result["reason"] = (
                f"Bullish sweep + {choch['type']} confirmed. Waiting for price to tap into "
                f"{zone['type'].replace('_', ' ')} zone ({bottom:.2f}–{top:.2f}) with a bullish close."
            )
        else:
            result["reason"] = "Bullish sweep + CHoCH detected, but no clear OB/FVG retracement zone yet."

    if bearish_setup and result["signal"] == "WAIT":
        sw, choch = bearish_setup
        zone = nearest_zone(choch["idx"], "bear")
        if zone:
            top, bottom = zone["top"], zone["bottom"]
            in_zone = bottom * 0.998 <= last_price <= top
            confirm_candle = df["close"].iloc[-1] < df["open"].iloc[-1]
            if in_zone and confirm_candle:
                entry = last_price
                sl = max(sw["level"], top) * 1.001
                risk = max(sl - entry, 0.01)
                future_lows = df.loc[df["swing_low"], "low"]
                targets = future_lows[future_lows < entry]
                target = float(targets.max()) if len(targets) else entry - risk * 2
                result.update(
                    signal="SELL",
                    reason=(
                        f"Sell-side liquidity swept at {sw['level']:.2f}, followed by a bearish "
                        f"{choch['type']}. Price reacted from a {zone['type'].replace('_', ' ')} zone "
                        f"({bottom:.2f}–{top:.2f}) with a bearish confirmation candle."
                    ),
                    entry=round(entry, 2), sl=round(sl, 2), target=round(target, 2),
                )
                return result
            result["reason"] = (
                f"Bearish sweep + {choch['type']} confirmed. Waiting for price to tap into "
                f"{zone['type'].replace('_', ' ')} zone ({bottom:.2f}–{top:.2f}) with a bearish close."
            )
        else:
            result["reason"] = "Bearish sweep + CHoCH detected, but no clear OB/FVG retracement zone yet."

    if result["signal"] == "WAIT" and not result["reason"]:
        result["reason"] = "No high-probability SMC setup right now (no recent sweep+structure-shift combo). Monitoring..."

    return result


# ============================================================================
# CHART
# ============================================================================
def build_chart(res, symbol, tf_minutes):
    df = res["df"]
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=symbol, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    )

    for ob in res["obs"]:
        color = "rgba(38,166,154,0.20)" if "bullish" in ob["type"] else "rgba(239,83,80,0.20)"
        fig.add_shape(
            type="rect", x0=ob["timestamp"], x1=df["timestamp"].iloc[-1],
            y0=ob["bottom"], y1=ob["top"], fillcolor=color, line=dict(width=0), layer="below",
        )

    for fv in res["fvgs"][-15:]:
        color = "rgba(41,98,255,0.15)" if "bullish" in fv["type"] else "rgba(255,152,0,0.15)"
        fig.add_shape(
            type="rect", x0=fv["timestamp"], x1=df["timestamp"].iloc[-1],
            y0=fv["bottom"], y1=fv["top"], fillcolor=color, line=dict(width=0), layer="below",
        )

    for sw in res["sweeps"][-10:]:
        fig.add_trace(
            go.Scatter(
                x=[sw["timestamp"]], y=[sw["level"]], mode="markers",
                marker=dict(symbol="x", size=10, color="#ffca28"),
                name="Liquidity Sweep", showlegend=False,
                hovertext=f"{sw['type']} @ {sw['level']:.2f}",
            )
        )

    for ev in res["events"][-12:]:
        color = "#26a69a" if ev["direction"] == "bull" else "#ef5350"
        fig.add_annotation(
            x=ev["timestamp"], y=ev["price"], text=ev["type"], showarrow=True, arrowhead=1,
            arrowcolor=color, font=dict(color=color, size=10), yshift=15 if ev["direction"] == "bull" else -15,
        )

    if res["signal"] in ("BUY", "SELL") and res["entry"]:
        fig.add_hline(y=res["entry"], line_dash="dot", line_color="#2962ff", annotation_text="Entry")
        fig.add_hline(y=res["sl"], line_dash="dot", line_color="#ef5350", annotation_text="Stop Loss")
        fig.add_hline(y=res["target"], line_dash="dot", line_color="#26a69a", annotation_text="Target")

    fig.update_layout(
        title=f"{symbol} — {tf_minutes}min (SMC view)", xaxis_rangeslider_visible=False,
        height=560, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark",
    )
    return fig


# ============================================================================
# SESSION STATE INIT
# ============================================================================
for key, default in [
    ("active_trade", None), ("trade_log", []), ("monitoring", False),
    ("selected_symbol", None), ("selected_key", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Setup")
access_token = st.sidebar.text_input("Upstox API v2 access token", type="password")
tf_minutes = st.sidebar.selectbox("Timeframe", [1, 3, 5, 15], index=2, format_func=lambda x: f"{x} min")
refresh_sec = st.sidebar.slider("Auto-refresh every (sec)", 15, 120, 30, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Find a stock")
query = st.sidebar.text_input("Stock name / symbol", placeholder="e.g. RELIANCE, TCS, HDFC BANK")

instrument_key, symbol_label = None, None
if access_token and query:
    try:
        inst_df = load_instruments()
        matches = search_instrument(inst_df, query)
        if len(matches):
            options = {f"{r.tradingsymbol} — {r.name}": r.instrument_key for r in matches.itertuples()}
            choice = st.sidebar.selectbox("Matches", list(options.keys()))
            instrument_key = options[choice]
            symbol_label = choice.split(" — ")[0]
        else:
            st.sidebar.warning("No matches found.")
    except Exception as e:
        st.sidebar.error(f"Could not load instrument list: {e}")
elif not access_token:
    st.sidebar.info("Paste your Upstox access token to search stocks.")

st.sidebar.markdown("---")
col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶ Start Monitoring", use_container_width=True, disabled=not instrument_key):
    st.session_state.monitoring = True
    st.session_state.selected_symbol = symbol_label
    st.session_state.selected_key = instrument_key
    st.session_state.active_trade = None
if col_b.button("⏹ Stop", use_container_width=True):
    st.session_state.monitoring = False

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This tool NEVER places, modifies or cancels orders. It only analyzes "
    "completed candles and shows you Entry / Stop Loss / Target for manual execution."
)

if not HAS_AUTOREFRESH:
    st.sidebar.caption("Tip: `pip install streamlit-autorefresh` for live auto-refresh during market hours.")


# ============================================================================
# MAIN AREA
# ============================================================================
st.title("📈 SMC Intraday Signal Assistant")
st.caption("Break of Structure • Change of Character • Liquidity Sweeps • Order Blocks • FVG • Displacement")

top1, top2, top3 = st.columns([2, 2, 3])
top1.metric("Market", market_status_label())
top2.metric("Now (IST)", now_ist().strftime("%H:%M:%S"))
top3.metric("Selected", st.session_state.selected_symbol or "—")

st.markdown(
    "> **Disclaimer:** Educational decision-support only. SMC signals are rule-based approximations "
    "of a discretionary methodology and can be wrong. No order is ever placed automatically — "
    "you decide whether, when and how to execute."
)

if not st.session_state.monitoring or not st.session_state.selected_key:
    st.info("Enter your access token, search a stock, and click **Start Monitoring** in the sidebar.")
    st.stop()

if HAS_AUTOREFRESH and is_market_open():
    st_autorefresh(interval=refresh_sec * 1000, key="live_refresh")
elif not is_market_open():
    st.warning("Market is currently closed (NSE hours: 9:15 AM – 3:30 PM, Mon–Fri). Showing last available data.")
    if st.button("🔄 Refresh now"):
        st.rerun()
else:
    if st.button("🔄 Refresh now"):
        st.rerun()

# ---- Fetch + analyze ----
df, err = get_completed_candles(st.session_state.selected_key, access_token, tf_minutes)

if df.empty:
    st.error(f"No candle data returned yet. {('Error: ' + err) if err else 'Try again in a moment, or check your access token.'}")
    st.stop()

res = generate_signal(df)

# ---- Manage active trade lifecycle (ENTER / EXIT) ----
last_row = df.iloc[-1]
exit_note = None
if st.session_state.active_trade is None:
    if res["signal"] in ("BUY", "SELL"):
        st.session_state.active_trade = dict(
            direction=res["signal"], entry=res["entry"], sl=res["sl"], target=res["target"],
            entry_time=str(last_row["timestamp"]),
        )
else:
    trade = st.session_state.active_trade
    if trade["direction"] == "BUY":
        if last_row["low"] <= trade["sl"]:
            exit_note = ("STOP LOSS HIT", trade["sl"])
        elif last_row["high"] >= trade["target"]:
            exit_note = ("TARGET HIT", trade["target"])
    else:
        if last_row["high"] >= trade["sl"]:
            exit_note = ("STOP LOSS HIT", trade["sl"])
        elif last_row["low"] <= trade["target"]:
            exit_note = ("TARGET HIT", trade["target"])

    if exit_note:
        st.session_state.trade_log.append(
            dict(
                symbol=st.session_state.selected_symbol, direction=trade["direction"],
                entry=trade["entry"], sl=trade["sl"], target=trade["target"],
                result=exit_note[0], exit_price=exit_note[1],
                entry_time=trade["entry_time"], exit_time=str(last_row["timestamp"]),
            )
        )
        st.session_state.active_trade = None

# ============================================================================
# SIGNAL PANEL
# ============================================================================
st.subheader("🎯 Current Signal")

if st.session_state.active_trade:
    t = st.session_state.active_trade
    badge = "🟢 IN TRADE — BUY" if t["direction"] == "BUY" else "🔴 IN TRADE — SELL"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", badge)
    c2.metric("Entry", f"{t['entry']:.2f}")
    c3.metric("Stop Loss", f"{t['sl']:.2f}")
    c4.metric("Target", f"{t['target']:.2f}")
    live_pl = (last_row["close"] - t["entry"]) if t["direction"] == "BUY" else (t["entry"] - last_row["close"])
    st.caption(f"Entered at {t['entry_time']} • Live unrealized: {live_pl:+.2f} pts (last close {last_row['close']:.2f})")
    st.info("Position is OPEN. This app will alert you here the moment SL or Target is hit on a completed candle. Manage/exit manually via your broker.")
elif exit_note:
    st.success(f"✅ {exit_note[0]} at {exit_note[1]:.2f} — trade closed. See log below. Watching for the next setup...")
else:
    if res["signal"] in ("BUY", "SELL"):
        color = "🟢 BUY" if res["signal"] == "BUY" else "🔴 SELL"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", color)
        c2.metric("Entry", f"{res['entry']:.2f}")
        c3.metric("Stop Loss", f"{res['sl']:.2f}")
        c4.metric("Target", f"{res['target']:.2f}")
        rr = abs(res["target"] - res["entry"]) / max(abs(res["entry"] - res["sl"]), 0.01)
        st.caption(f"Risk:Reward ≈ 1:{rr:.2f}")
        st.success(f"**ENTER now** — {res['reason']}")
    else:
        st.warning(f"⏳ **WAIT** — {res['reason']}")

st.caption(f"Market structure trend (from swings): **{res['trend'] or 'undetermined'}**  •  Last completed candle: {last_row['timestamp']}")

# ============================================================================
# CHART
# ============================================================================
st.plotly_chart(build_chart(res, st.session_state.selected_symbol, tf_minutes), use_container_width=True)

# ============================================================================
# DETECTED EVENTS
# ============================================================================
with st.expander("🔍 Detected structure events (recent)"):
    ev_df = pd.DataFrame(res["events"][-15:])
    if len(ev_df):
        st.dataframe(ev_df[["timestamp", "type", "direction", "price"]], use_container_width=True, hide_index=True)
    else:
        st.write("No BOS/CHoCH events yet.")

with st.expander("💧 Liquidity sweeps (recent)"):
    sw_df = pd.DataFrame(res["sweeps"][-10:])
    if len(sw_df):
        st.dataframe(sw_df[["timestamp", "type", "level", "direction"]], use_container_width=True, hide_index=True)
    else:
        st.write("No liquidity sweeps detected recently.")

with st.expander("📦 Order Blocks & Fair Value Gaps (recent)"):
    ob_df = pd.DataFrame(res["obs"][-10:])
    fv_df = pd.DataFrame(res["fvgs"][-10:])
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Order Blocks**")
        st.dataframe(ob_df[["timestamp", "type", "top", "bottom"]] if len(ob_df) else pd.DataFrame(), use_container_width=True, hide_index=True)
    with cc2:
        st.markdown("**Fair Value Gaps**")
        st.dataframe(fv_df[["timestamp", "type", "top", "bottom"]] if len(fv_df) else pd.DataFrame(), use_container_width=True, hide_index=True)

# ============================================================================
# TRADE LOG
# ============================================================================
st.subheader("📒 Signal / Trade Log")
if st.session_state.trade_log:
    st.dataframe(pd.DataFrame(st.session_state.trade_log), use_container_width=True, hide_index=True)
else:
    st.caption("No closed setups yet this session.")
