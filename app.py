import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import settings
from utils import (
    get_ist_now,
    format_ist_time,
    is_nse_market_open,
    mask_token,
    format_currency,
    format_pct
)
from upstox_client import upstox_client
from market_data import prepare_market_data, MarketDataUnavailableError
from indicators import compute_indicators
from smc_engine import smc_engine
from signal_engine import signal_engine
from risk_engine import risk_engine
from target_engine import target_engine
from telegram import telegram_notifier

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="NSE Semi-Algo — Trading Analyzer",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Container */
    .hero-header {
        text-align: center;
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Status Badges */
    .badge-open {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-closed {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Card Containers */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Main Directional Signal Card */
    .signal-card-buy {
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.25) 0%, rgba(15, 23, 42, 0.85) 75%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.25);
        margin: 1.2rem 0;
    }
    .signal-card-sell {
        background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.25) 0%, rgba(15, 23, 42, 0.85) 75%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.25);
        margin: 1.2rem 0;
    }
    
    .signal-text-buy {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        color: #10b981;
        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
        margin: 0.4rem 0;
    }
    .signal-text-sell {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        color: #ef4444;
        text-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
        margin: 0.4rem 0;
    }
    
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-val-highlight {
        font-size: 1.55rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Disclaimer Note */
    .disclaimer-text {
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
        margin-top: 0.8rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "upstox_token" not in st.session_state:
    st.session_state["upstox_token"] = ""
if "is_connected" not in st.session_state:
    st.session_state["is_connected"] = False
if "user_name" not in st.session_state:
    st.session_state["user_name"] = ""
if "current_symbol" not in st.session_state:
    st.session_state["current_symbol"] = "SAIL"
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# Header Banner
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚡ NSE SEMI-ALGO</div>
    <div class="hero-subtitle">Simple Streamlit Trading Analyzer • Upstox API v2</div>
</div>
""", unsafe_allow_html=True)

# Market Hours Live Badge
is_open, market_status_text = is_nse_market_open()
now_str = format_ist_time()
badge_class = "badge-open" if is_open else "badge-closed"

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; padding: 0 0.5rem;">
    <span class="{badge_class}">{market_status_text}</span>
    <span style="font-size: 0.85rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Time: {now_str}</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SECTION 1: UPSTOX API CONNECTION
# ==============================================================================
with st.container():
    st.markdown("### 🔑 UPSTOX API CONNECTION")

    if not st.session_state["is_connected"]:
        # Connection Form
        col1, col2 = st.columns([3, 1])
        with col1:
            token_input = st.text_input(
                "Upstox Access Token",
                type="password",
                placeholder="Paste Upstox Access Token here (e.g. eyJhbGci...)",
                help="Never displayed or logged. Kept securely in current session.",
                label_visibility="collapsed"
            )
        with col2:
            connect_btn = st.button("CONNECT", use_container_width=True, type="primary")

        if connect_btn:
            if not token_input.strip():
                st.error("🔴 Please enter your Upstox Access Token.")
            else:
                with st.spinner("Validating with Upstox API..."):
                    val_res = upstox_client.validate_token(token_input)
                    if val_res.get("valid"):
                        st.session_state["upstox_token"] = token_input.strip()
                        st.session_state["is_connected"] = True
                        st.session_state["user_name"] = val_res.get("user_name", "Trader")
                        upstox_client.set_token(token_input.strip())
                        st.rerun()
                    else:
                        st.error(f"🔴 **CONNECTION FAILED**: {val_res.get('message', 'Invalid Access Token')}")
    else:
        # Connected State
        c_status, c_disconnect = st.columns([3, 1])
        with c_status:
            masked = mask_token(st.session_state["upstox_token"])
            st.success(f"🟢 **UPSTOX CONNECTED** — Welcome, **{st.session_state['user_name']}** (`{masked}`)")
        with c_disconnect:
            if st.button("DISCONNECT", use_container_width=True):
                st.session_state["upstox_token"] = ""
                st.session_state["is_connected"] = False
                st.session_state["analysis_result"] = None
                upstox_client.set_token("")
                st.rerun()

# Halt execution until connected
if not st.session_state["is_connected"]:
    st.info("👆 Please enter your Upstox Access Token above and click **CONNECT** to proceed to stock analysis.")
    st.stop()

st.divider()

# ==============================================================================
# SECTION 2: STOCK INPUT & ANALYSIS TRIGGER
# ==============================================================================
st.markdown("### 🔍 STOCK ANALYSIS")

col_sym, col_tf, col_btn = st.columns([3, 1.5, 1.5])

with col_sym:
    sym_input = st.text_input(
        "Enter NSE Stock",
        value=st.session_state["current_symbol"],
        placeholder="e.g. SAIL, RELIANCE, TCS, INFY, SBIN",
        help="Type any NSE equity symbol. Instrument key is resolved automatically.",
        label_visibility="collapsed"
    ).upper().strip()

with col_tf:
    timeframe = st.selectbox(
        "Timeframe",
        options=["1m", "3m", "5m", "15m", "30m", "1h"],
        index=2,  # Default 5m
        label_visibility="collapsed"
    )

with col_btn:
    analyze_btn = st.button("ANALYZE STOCK", type="primary", use_container_width=True)

if analyze_btn and sym_input:
    st.session_state["current_symbol"] = sym_input
    with st.spinner(f"Fetching market data and running SMC + Technical analysis for {sym_input}..."):
        try:
            # 1. Fetch live market quote
            quote = upstox_client.get_market_quote(sym_input)
            
            # 2. Fetch completed candle data
            raw_candles = upstox_client.fetch_candles(sym_input, timeframe=timeframe)
            
            # 3. Clean & validate market data
            pkg = prepare_market_data(sym_input, quote, raw_candles, timeframe=timeframe)
            
            # 4. Compute Indicators & Price Action
            df_ind, ind_snap = compute_indicators(pkg.candles)
            
            # 5. Compute Smart Money Concepts (SMC)
            smc_snap = smc_engine.analyze(df_ind, pdh=quote.get("pdh"), pdl=quote.get("pdl"))
            
            # 6. Confluence Decision Engine (Strict BUY or SELL)
            signal_res = signal_engine.evaluate(
                symbol=sym_input,
                current_price=quote["last_price"],
                indicators=ind_snap,
                smc=smc_snap
            )
            
            # 7. Maximum Target Engine
            target_res = target_engine.calculate_target(
                direction=signal_res.direction,
                entry_price=signal_res.entry_price,
                indicators=ind_snap,
                smc=smc_snap
            )
            
            # 8. Dynamic Stop Loss & Risk Engine
            risk_res = risk_engine.calculate_stop_loss(
                direction=signal_res.direction,
                entry_price=signal_res.entry_price,
                indicators=ind_snap,
                smc=smc_snap,
                target_price=target_res.target_price
            )
            
            # Save into session state
            st.session_state["analysis_result"] = {
                "symbol": sym_input,
                "timeframe": timeframe,
                "quote": quote,
                "pkg": pkg,
                "indicators": ind_snap,
                "smc": smc_snap,
                "signal": signal_res,
                "target": target_res,
                "risk": risk_res,
                "df": df_ind
            }
            
            # Optional Telegram alert if configured
            if telegram_notifier.is_configured():
                telegram_notifier.send_signal_alert(
                    symbol=sym_input,
                    direction=signal_res.direction,
                    current_price=quote["last_price"],
                    entry=signal_res.entry_price,
                    target=target_res.target_price,
                    stop_loss=risk_res.stop_loss,
                    rr_ratio=risk_res.risk_reward_ratio
                )

        except MarketDataUnavailableError as mde:
            st.error(f"🔴 **Market data unavailable — please try again.** ({str(mde)})")
            st.session_state["analysis_result"] = None
        except Exception as e:
            st.error(f"🔴 **Market data unavailable — please try again.** (Error: {str(e)})")
            st.session_state["analysis_result"] = None

# ==============================================================================
# SECTION 3: DISPLAY ANALYSIS RESULTS
# ==============================================================================
res = st.session_state.get("analysis_result")

if res:
    sym = res["symbol"]
    q = res["quote"]
    sig = res["signal"]
    tgt = res["target"]
    rsk = res["risk"]
    df = res["df"]
    
    current_p = q["last_price"]
    prev_c = q["prev_close"]
    chg = q["change"]
    chg_p = q["change_pct"]
    
    # -------------------------------------------------------------
    # 3.1 STOCK CURRENT INFORMATION BAR
    # -------------------------------------------------------------
    st.markdown(f"## {sym}")
    
    # Current Price & Intraday Metrics Bar
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("CURRENT PRICE", format_currency(current_p), f"{chg_p:+.2f}%")
    with m2:
        st.metric("PREV CLOSE", format_currency(prev_c))
    with m3:
        st.metric("DAY HIGH", format_currency(q["high"]))
    with m4:
        st.metric("DAY LOW", format_currency(q["low"]))
    with m5:
        st.metric("VOLUME", f"{q['volume']:,}")
    with m6:
        st.metric("VWAP", format_currency(q["vwap"]))

    # -------------------------------------------------------------
    # 3.2 MAIN OUTPUT — ONLY BUY OR SELL (THE PRIMARY FOCUS)
    # -------------------------------------------------------------
    is_buy = sig.direction == "BUY"
    card_class = "signal-card-buy" if is_buy else "signal-card-sell"
    sig_text_class = "signal-text-buy" if is_buy else "signal-text-sell"
    emoji = "🟢" if is_buy else "🔴"
    target_color = "#10b981" if is_buy else "#f87171"
    sl_color = "#ef4444" if is_buy else "#34d399"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div style="font-size: 1.1rem; letter-spacing: 0.12em; color: #94a3b8; font-weight: 700; text-transform: uppercase;">
            {sym} • {res['timeframe'].upper()} TIMEFRAME
        </div>
        <div class="{sig_text_class}">
            {emoji} {sig.direction}
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.2rem;">
            <div>
                <div class="stat-label">Current Price</div>
                <div class="stat-val">{format_currency(current_p)}</div>
            </div>
            <div>
                <div class="stat-label">Entry Price</div>
                <div class="stat-val">{format_currency(sig.entry_price)}</div>
            </div>
            <div>
                <div class="stat-label">Maximum Target</div>
                <div class="stat-val-highlight" style="color: {target_color};">{format_currency(tgt.target_price)}</div>
                <div style="font-size: 0.8rem; color: {target_color}; font-weight: 600;">{tgt.target_distance_pct:+.1f}% move</div>
            </div>
            <div>
                <div class="stat-label">Stop Loss</div>
                <div class="stat-val-highlight" style="color: {sl_color};">{format_currency(rsk.stop_loss)}</div>
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">-{rsk.risk_pct:.1f}% risk</div>
            </div>
        </div>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 2.5rem; font-size: 0.95rem; color: #cbd5e1;">
            <span>⚖️ <b>Risk/Reward:</b> 1 : {rsk.risk_reward_ratio:.1f}</span>
            <span>🎯 <b>Target Reference:</b> {tgt.target_reference_name}</span>
            <span>🛡️ <b>SL Invalidation:</b> {rsk.invalidation_level_name}</span>
        </div>
        <div class="disclaimer-text">
            ⚠️ Technical analysis estimate — not guaranteed.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3.3 INTERACTIVE CANDLESTICK CHART
    # -------------------------------------------------------------
    st.markdown("### 📊 TECHNICAL & SMC CHART")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume")
    )
    
    # 1. Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlesticks",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        ),
        row=1, col=1
    )
    
    # 2. EMAs (9, 21, 50)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_9"], line=dict(color="#38bdf8", width=1.3), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_21"], line=dict(color="#f59e0b", width=1.3), name="EMA 21"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_50"], line=dict(color="#a855f7", width=1.5), name="EMA 50"), row=1, col=1)
    
    # 3. VWAP
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], line=dict(color="#ec4899", width=1.5, dash="dot"), name="VWAP"), row=1, col=1)

    # 4. Previous Day High & Low
    if res["smc"].pdh > 0:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=[res["smc"].pdh]*len(df), line=dict(color="#4ade80", width=1.2, dash="dash"), name="PDH"), row=1, col=1)
    if res["smc"].pdl > 0:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=[res["smc"].pdl]*len(df), line=dict(color="#f87171", width=1.2, dash="dash"), name="PDL"), row=1, col=1)

    # 5. Key Trade Levels: Entry, SL, Target
    fig.add_hline(y=sig.entry_price, line=dict(color="#e2e8f0", width=1.5, dash="dash"), annotation_text=f"Entry: ₹{sig.entry_price:.2f}", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=tgt.target_price, line=dict(color="#10b981", width=2.0), annotation_text=f"Target: ₹{tgt.target_price:.2f}", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=rsk.stop_loss, line=dict(color="#ef4444", width=2.0), annotation_text=f"SL: ₹{rsk.stop_loss:.2f}", annotation_position="bottom right", row=1, col=1)

    # 6. Volume Sub-plot
    colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors, name="Volume"),
        row=2, col=1
    )
    if "volume_ma" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["volume_ma"], line=dict(color="#fbbf24", width=1.2), name="Vol 20-MA"),
            row=2, col=1
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        height=540,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 3.4 SECONDARY SMC & CONFLUENCE BREAKDOWN (COLLAPSIBLE)
    # -------------------------------------------------------------
    with st.expander("🔬 Smart Money Concepts (SMC) & Indicator Confluence", expanded=False):
        c_smc1, c_smc2 = st.columns(2)
        with c_smc1:
            st.markdown("#### Market Structure & Zones")
            st.write(f"• **Trend:** `{res['smc'].trend}`")
            st.write(f"• **Structure:** `{res['smc'].market_structure}`")
            st.write(f"• **BOS / CHOCH:** `{res['smc'].bos_choch_event.get('details', 'None')}`")
            st.write(f"• **Liquidity Sweep:** `{res['smc'].sweep_event.get('details', 'None')}`")
            st.write(f"• **Dealing Range Zone:** `{res['smc'].dealing_range.get('zone', 'N/A')} ({res['smc'].dealing_range.get('pct', 0)}%)`")
            st.write(f"• **PDH / PDL:** `₹{res['smc'].pdh:.2f} / ₹{res['smc'].pdl:.2f} ({res['smc'].pdh_pdl_status})`")

        with c_smc2:
            st.markdown("#### Technical Confluence Score")
            st.write(f"• **BUY Score:** `{sig.buy_score:.1f}` | **SELL Score:** `{sig.sell_score:.1f}`")
            st.write(f"• **RSI (14):** `{res['indicators'].rsi:.1f}`")
            st.write(f"• **MACD Hist:** `{res['indicators'].macd_hist:+.2f}`")
            st.write(f"• **ADX (14):** `{res['indicators'].adx:.1f} (+DI {res['indicators'].plus_di:.1f}, -DI {res['indicators'].minus_di:.1f})`")
            st.write(f"• **Volume Surge Ratio:** `{res['indicators'].volume_ratio:.2f}x` ({'Surge' if sig.volume_surge else 'Normal'})")
            st.write(f"• **ATR Volatility:** `₹{res['indicators'].atr:.2f}`")

        st.markdown("#### Detected Confluences")
        for c in sig.confluence_factors:
            st.markdown(f"- {c}")

    # -------------------------------------------------------------
    # 3.5 TELEGRAM NOTIFICATION ACTION
    # -------------------------------------------------------------
    if telegram_notifier.is_configured():
        col_tg_left, col_tg_btn = st.columns([3, 1])
        with col_tg_left:
            st.caption("Telegram bot alerts configured.")
        with col_tg_btn:
            if st.button("✈️ Send Telegram Alert", use_container_width=True):
                sent = telegram_notifier.send_signal_alert(
                    symbol=sym,
                    direction=sig.direction,
                    current_price=current_p,
                    entry=sig.entry_price,
                    target=tgt.target_price,
                    stop_loss=rsk.stop_loss,
                    rr_ratio=rsk.risk_reward_ratio,
                    force=True
                )
                if sent:
                    st.toast("✅ Telegram alert sent successfully!")
                else:
                    st.error("Failed to send Telegram alert.")
