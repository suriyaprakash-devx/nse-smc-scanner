import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings
from utils import (
    get_ist_now,
    format_ist_time,
    is_nse_market_open,
    mask_token,
    format_currency,
    format_pct,
    MarketDataUnavailableError
)
from upstox_client import upstox_client
from market_data import prepare_market_data
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
    page_title="NSE Semi-Algo — Trading Analyzer & Scanner",
    page_icon="⚡",
    layout="wide",
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
        padding: 1.2rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.2rem;
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
    .badge-live {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-demo {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        font-size: 0.78rem;
        font-weight: 700;
    }
    
    /* Main Directional Signal Card */
    .signal-card-buy {
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.22) 0%, rgba(15, 23, 42, 0.9) 75%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.22);
        margin: 1.2rem 0;
    }
    .signal-card-sell {
        background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.22) 0%, rgba(15, 23, 42, 0.9) 75%);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 0 35px rgba(239, 68, 68, 0.22);
        margin: 1.2rem 0;
    }
    
    .signal-text-buy {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        color: #10b981;
        text-shadow: 0 0 22px rgba(16, 185, 129, 0.5);
        margin: 0.3rem 0;
    }
    .signal-text-sell {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        color: #ef4444;
        text-shadow: 0 0 22px rgba(239, 68, 68, 0.5);
        margin: 0.3rem 0;
    }
    
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
    }
    .stat-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-val-highlight {
        font-size: 1.45rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    
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
    st.session_state["current_symbol"] = "RELIANCE"
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "scanner_results" not in st.session_state:
    st.session_state["scanner_results"] = None

# Header Banner
st.markdown("""
<div class="hero-header">
    <div class="hero-title">⚡ NSE SEMI-ALGO</div>
    <div class="hero-subtitle">Intraday Trading Analyzer & Multi-Stock Scanner • SMC & EMA 6/30 • Upstox API v2/v3</div>
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
        col1, col2 = st.columns([3, 1])
        with col1:
            token_input = st.text_input(
                "Upstox Access Token",
                type="password",
                placeholder="Paste Upstox Access Token here (or enter 'demo' for sandbox testing)...",
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
        c_status, c_disconnect = st.columns([3, 1])
        with c_status:
            masked = mask_token(st.session_state["upstox_token"])
            mode_tag = "Sandbox / Demo Mode" if upstox_client.is_demo() else "LIVE PRODUCTION"
            st.success(f"🟢 **UPSTOX CONNECTED** — {st.session_state['user_name']} (`{masked}`) [{mode_tag}]")
        with c_disconnect:
            if st.button("DISCONNECT", use_container_width=True):
                st.session_state["upstox_token"] = ""
                st.session_state["is_connected"] = False
                st.session_state["analysis_result"] = None
                st.session_state["scanner_results"] = None
                upstox_client.set_token("")
                st.rerun()

if not st.session_state["is_connected"]:
    st.info("👆 Please enter your Upstox Access Token above and click **CONNECT** to proceed.")
    st.stop()

st.divider()

# ==============================================================================
# SECTION 2: TABS — MULTI-STOCK SCANNER vs SINGLE STOCK ANALYZER
# ==============================================================================
tab_scanner, tab_analyzer = st.tabs(["🚀 MULTI-STOCK SCANNER", "📈 SINGLE STOCK DEEP-DIVE"])

# Helper function to analyze a single stock
def run_stock_analysis(sym: str, tf: str) -> Dict[str, Any]:
    quote = upstox_client.get_market_quote(sym)
    raw_candles = upstox_client.fetch_candles(sym, timeframe=tf)
    pkg = prepare_market_data(sym, quote, raw_candles, timeframe=tf)
    df_ind, ind_snap = compute_indicators(pkg.candles)
    smc_snap = smc_engine.analyze(df_ind, pdh=quote.get("pdh"), pdl=quote.get("pdl"))
    signal_res = signal_engine.evaluate(
        symbol=sym,
        current_price=quote["last_price"],
        indicators=ind_snap,
        smc=smc_snap,
        timeframe=tf
    )
    target_res = target_engine.calculate_targets(
        direction=signal_res.direction,
        entry_price=signal_res.entry_price,
        indicators=ind_snap,
        smc=smc_snap
    )
    risk_res = risk_engine.calculate_stop_loss(
        direction=signal_res.direction,
        entry_price=signal_res.entry_price,
        indicators=ind_snap,
        smc=smc_snap,
        target_price=target_res.tp2
    )
    return {
        "symbol": sym,
        "timeframe": tf,
        "quote": quote,
        "pkg": pkg,
        "indicators": ind_snap,
        "smc": smc_snap,
        "signal": signal_res,
        "target": target_res,
        "risk": risk_res,
        "df": df_ind,
        "is_live": quote.get("is_live", False)
    }

# ------------------------------------------------------------------------------
# TAB 1: MULTI-STOCK SCANNER (ALL NSE STOCKS SUPPORTED)
# ------------------------------------------------------------------------------
with tab_scanner:
    st.markdown("### 📡 MULTI-STOCK MARKET SCANNER")
    st.caption("Scan official NSE index baskets, complete F&O universe (199 stocks), or custom symbols in parallel using SMC, EMA 6/30, and Volume filters.")

    # Load complete baskets from UpstoxClient
    presets_dict = upstox_client.get_watchlist_presets()
    if not presets_dict:
        presets_dict = {
            "NIFTY 50 (All 50 Stocks)": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE"],
            "NIFTY BANK (12 Stocks)": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "BANKBARODA"],
            "NSE F&O UNIVERSE (All 199 Stocks)": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "BHEL", "SAIL", "VEDL"]
        }

    all_syms = upstox_client.get_all_symbols()
    all_eq_batches = {
        "ALL NSE EQUITIES (Batch 1 - 50)": all_syms[:50],
        "ALL NSE EQUITIES (Batch 51 - 100)": all_syms[50:100],
        "ALL NSE EQUITIES (Batch 101 - 150)": all_syms[100:150],
        "CUSTOM WATCHLIST": []
    }
    available_presets = {**presets_dict, **all_eq_batches}

    col_w1, col_limit, col_w2, col_w3 = st.columns([2.5, 1.2, 1, 1.3])
    with col_w1:
        selected_preset = st.selectbox("Select Watchlist Basket", list(available_presets.keys()), index=0)

    basket_stocks = available_presets[selected_preset]

    with col_limit:
        limit_options = ["All in Basket", "Top 10", "Top 25", "Top 50", "Top 100"]
        default_lim_idx = 0 if len(basket_stocks) <= 50 else 2
        chosen_limit = st.selectbox("Batch Limit", limit_options, index=default_lim_idx)

    with col_w2:
        scanner_tf = st.selectbox("Timeframe", options=["5m", "10m", "15m", "30m"], index=0, key="scanner_tf")

    with col_w3:
        st.write("")
        st.write("")
        run_scan_btn = st.button("🔍 SCAN BASKET", type="primary", use_container_width=True)

    if selected_preset == "CUSTOM WATCHLIST":
        custom_symbols_input = st.text_area(
            "Enter NSE Symbols (comma-separated):",
            value="RELIANCE, TCS, HDFCBANK, INFY, SBIN, BHEL, IRCTC, ZOMATO, SAIL, TATAPOWER, TMCV, ETERNAL, TRENT, BEL, HAL",
            help="Type ANY valid NSE stock symbols separated by commas."
        )
        symbols_to_scan = [s.strip().upper() for s in custom_symbols_input.split(",") if s.strip()]
    else:
        symbols_to_scan = list(basket_stocks)
        if chosen_limit == "Top 10":
            symbols_to_scan = symbols_to_scan[:10]
        elif chosen_limit == "Top 25":
            symbols_to_scan = symbols_to_scan[:25]
        elif chosen_limit == "Top 50":
            symbols_to_scan = symbols_to_scan[:50]
        elif chosen_limit == "Top 100":
            symbols_to_scan = symbols_to_scan[:100]

    st.write(f"**Target Stocks ({len(symbols_to_scan)}):** `{'`, `'.join(symbols_to_scan[:30])}{' ...' if len(symbols_to_scan) > 30 else ''}`")

    if run_scan_btn and symbols_to_scan:
        progress_bar = st.progress(0)
        status_text = st.empty()
        scan_results = []
        total_stocks = len(symbols_to_scan)
        completed_count = 0

        def scan_worker(sym):
            try:
                res_obj = run_stock_analysis(sym, scanner_tf)
                sig = res_obj["signal"]
                tgt = res_obj["target"]
                rsk = res_obj["risk"]
                ind = res_obj["indicators"]
                q = res_obj["quote"]

                return {
                    "Stock": sym,
                    "Signal": f"🟢 BUY" if sig.direction == "BUY" else "🔴 SELL",
                    "Direction": sig.direction,
                    "LTP": round(q["last_price"], 2),
                    "Entry Zone": sig.entry_zone,
                    "Stop Loss": round(rsk.stop_loss, 2),
                    "TP1": round(tgt.tp1, 2),
                    "TP2": round(tgt.tp2, 2),
                    "TP3": round(tgt.tp3, 2),
                    "R:R": f"1:{rsk.risk_reward_ratio:.1f}",
                    "rr_val": rsk.risk_reward_ratio,
                    "Volume Ratio": f"{ind.volume_ratio:.2f}x",
                    "vol_val": ind.volume_ratio,
                    "EMA 6/30": f"₹{ind.ema_6:.1f} / ₹{ind.ema_30:.1f}",
                    "Structure": res_obj["smc"].market_structure,
                    "res_obj": res_obj
                }
            except Exception as e:
                return {
                    "Stock": sym,
                    "Signal": "⚠️ ERROR",
                    "Direction": "ERROR",
                    "LTP": 0.0,
                    "Entry Zone": "Data Unavailable",
                    "Stop Loss": 0.0,
                    "TP1": 0.0,
                    "TP2": 0.0,
                    "TP3": 0.0,
                    "R:R": "N/A",
                    "rr_val": 0.0,
                    "Volume Ratio": "N/A",
                    "vol_val": 0.0,
                    "EMA 6/30": "N/A",
                    "Structure": str(e),
                    "res_obj": None
                }

        # Multi-threaded scanning for lightning speed
        max_workers = min(8, max(1, total_stocks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sym = {executor.submit(scan_worker, s): s for s in symbols_to_scan}
            for future in as_completed(future_to_sym):
                completed_count += 1
                sym_done = future_to_sym[future]
                res = future.result()
                scan_results.append(res)
                progress_bar.progress(completed_count / total_stocks)
                status_text.text(f"⚡ Scanned {completed_count}/{total_stocks} stocks (latest: {sym_done})...")

        status_text.empty()
        progress_bar.empty()
        # Sort valid signals first
        scan_results.sort(key=lambda r: (0 if r["Direction"] in ["BUY", "SELL"] else 1, r["Stock"]))
        st.session_state["scanner_results"] = scan_results

    # Display Scanner Output Table
    if st.session_state["scanner_results"]:
        results = st.session_state["scanner_results"]
        total_valid = [r for r in results if r["Direction"] in ["BUY", "SELL"]]
        buys = [r for r in total_valid if r["Direction"] == "BUY"]
        sells = [r for r in total_valid if r["Direction"] == "SELL"]
        vol_surges = [r for r in total_valid if r.get("vol_val", 0) >= 1.5]

        # Metric summary row
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            st.metric("TOTAL SCANNED", len(results))
        with sm2:
            st.metric("🟢 BUY SIGNALS", len(buys))
        with sm3:
            st.metric("🔴 SELL SIGNALS", len(sells))
        with sm4:
            st.metric("⚡ VOLUME SURGES (>=1.5x)", len(vol_surges))

        # Dynamic Filters Row
        st.markdown("#### 🎯 Filter & Sort Results")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            filter_dir = st.selectbox("Signal Filter", ["ALL", "🟢 BUY ONLY", "🔴 SELL ONLY"], index=0)
        with f_col2:
            filter_vol = st.selectbox("Volume Surge", ["ALL", "⚡ Volume Surges (>=1.5x) Only"], index=0)
        with f_col3:
            filter_struct = st.selectbox("Market Structure", ["ALL", "BOS Only", "CHOCH Only"], index=0)
        with f_col4:
            sort_by = st.selectbox("Sort By", ["Volume Surge (High to Low)", "Risk:Reward (High to Low)", "Stock Symbol (A-Z)"], index=0)

        filtered = list(results)
        if filter_dir == "🟢 BUY ONLY":
            filtered = [r for r in filtered if r["Direction"] == "BUY"]
        elif filter_dir == "🔴 SELL ONLY":
            filtered = [r for r in filtered if r["Direction"] == "SELL"]

        if filter_vol == "⚡ Volume Surges (>=1.5x) Only":
            filtered = [r for r in filtered if r.get("vol_val", 0) >= 1.5]

        if filter_struct == "BOS Only":
            filtered = [r for r in filtered if "BOS" in str(r.get("Structure", ""))]
        elif filter_struct == "CHOCH Only":
            filtered = [r for r in filtered if "CHOCH" in str(r.get("Structure", ""))]

        if sort_by == "Volume Surge (High to Low)":
            filtered.sort(key=lambda r: r.get("vol_val", 0), reverse=True)
        elif sort_by == "Risk:Reward (High to Low)":
            filtered.sort(key=lambda r: r.get("rr_val", 0), reverse=True)
        elif sort_by == "Stock Symbol (A-Z)":
            filtered.sort(key=lambda r: r["Stock"])

        # Styled Table Display
        display_df = pd.DataFrame([{
            "Stock": r["Stock"],
            "Signal": r["Signal"],
            "LTP (₹)": f"₹{r['LTP']:,.2f}" if r["LTP"] > 0 else "N/A",
            "Entry Zone": r["Entry Zone"],
            "Stop Loss (₹)": f"₹{r['Stop Loss']:,.2f}" if r["Stop Loss"] > 0 else "N/A",
            "TP1 (₹)": f"₹{r['TP1']:,.2f}" if r["TP1"] > 0 else "N/A",
            "TP2 (₹)": f"₹{r['TP2']:,.2f}" if r["TP2"] > 0 else "N/A",
            "TP3 (₹)": f"₹{r.get('TP3', 0.0):,.2f}" if r.get("TP3", 0) > 0 else "N/A",
            "Risk/Reward": r["R:R"],
            "Volume Ratio": r["Volume Ratio"],
            "Structure": r["Structure"]
        } for r in filtered])

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Actions Row: CSV Export & Quick Deep Dive
        act_col1, act_col2, act_col3 = st.columns([1.5, 2, 1.2])
        with act_col1:
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Table as CSV",
                data=csv_data,
                file_name=f"nse_scanner_{scanner_tf}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with act_col2:
            inspect_sym = st.selectbox(
                "Select stock to view full interactive chart:",
                [r["Stock"] for r in total_valid],
                label_visibility="collapsed"
            )
        with act_col3:
            if st.button("📊 View Deep-Dive Chart", use_container_width=True):
                matched = next((r["res_obj"] for r in total_valid if r["Stock"] == inspect_sym), None)
                if matched:
                    st.session_state["analysis_result"] = matched
                    st.session_state["current_symbol"] = inspect_sym
                    st.toast(f"Loaded {inspect_sym} into Single Stock Deep-Dive tab!")

# ------------------------------------------------------------------------------
# TAB 2: SINGLE STOCK DEEP-DIVE ANALYZER
# ------------------------------------------------------------------------------
with tab_analyzer:
    st.markdown("### 📈 SINGLE STOCK DEEP-DIVE")
    st.caption("Deep-dive chart analysis, structural Order Blocks, FVGs, session VWAP, and Telegram alerts.")

    # Popular quick-click stock buttons
    st.markdown("**Popular NSE Stocks:**")
    quick_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "BHEL", "IRCTC", "ZOMATO", "TMCV", "TATAPOWER", "SAIL", "VEDL", "TRENT", "BEL"]
    quick_cols = st.columns(len(quick_stocks))
    for q_idx, q_sym in enumerate(quick_stocks):
        with quick_cols[q_idx]:
            if st.button(q_sym, key=f"quick_{q_sym}", use_container_width=True):
                st.session_state["current_symbol"] = q_sym
                st.rerun()

    # Build searchable dropdown with all 3,370+ NSE stocks
    stocks_meta = upstox_client.get_equity_stocks()
    stock_display_map = {}
    for s in stocks_meta:
        sym = s["symbol"]
        name = s.get("name", sym)
        disp = f"{sym} — {name}"
        stock_display_map[disp] = sym

    stock_display_map["TATAMOTORS — TATA MOTORS LIMITED (TMCV)"] = "TMCV"
    stock_display_map["ZOMATO — ETERNAL LIMITED (formerly Zomato)"] = "ETERNAL"

    # Category selector to narrow down dropdown list
    cat_col, search_col, tf_col, btn_col = st.columns([1.5, 2.5, 1, 1.2])
    with cat_col:
        cat_choices = ["All NSE Stocks (3,370+)", "NIFTY 50", "NSE F&O Universe (199)", "Nifty Bank", "Nifty IT", "Nifty Auto", "Nifty Metal", "Nifty Pharma"]
        sel_cat = st.selectbox("Category Filter", cat_choices, index=0)

    # Filter display options
    presets = upstox_client.get_watchlist_presets()
    cat_filter_set = None
    if sel_cat == "NIFTY 50":
        cat_filter_set = set(presets.get("NIFTY 50 (All 50 Stocks)", []))
    elif sel_cat == "NSE F&O Universe (199)":
        cat_filter_set = set(presets.get("NSE F&O UNIVERSE (All 199 Stocks)", []))
    elif sel_cat == "Nifty Bank":
        cat_filter_set = set(presets.get("NIFTY BANK (12 Stocks)", []))
    elif sel_cat == "Nifty IT":
        cat_filter_set = set(presets.get("NIFTY IT (10 Stocks)", []))
    elif sel_cat == "Nifty Auto":
        cat_filter_set = set(presets.get("NIFTY AUTO (15 Stocks)", []))
    elif sel_cat == "Nifty Metal":
        cat_filter_set = set(presets.get("NIFTY METAL (15 Stocks)", []))
    elif sel_cat == "Nifty Pharma":
        cat_filter_set = set(presets.get("NIFTY PHARMA (20 Stocks)", []))

    if cat_filter_set:
        filtered_disp_list = [d for d, s in stock_display_map.items() if s in cat_filter_set or any(c in d for c in cat_filter_set)]
    else:
        filtered_disp_list = list(stock_display_map.keys())

    # Match current symbol in dropdown
    curr_sym = st.session_state["current_symbol"]
    def_idx = 0
    for idx_d, d_name in enumerate(filtered_disp_list):
        if stock_display_map.get(d_name) == curr_sym or d_name.startswith(curr_sym + " "):
            def_idx = idx_d
            break

    with search_col:
        chosen_stock_disp = st.selectbox(
            "Search / Select Stock",
            options=filtered_disp_list if filtered_disp_list else [curr_sym],
            index=def_idx if def_idx < len(filtered_disp_list) else 0,
            help="Search by symbol or company name across all NSE stocks."
        )
        sym_input = stock_display_map.get(chosen_stock_disp, curr_sym)

    with tf_col:
        timeframe = st.selectbox(
            "Timeframe",
            options=["5m", "10m", "1m", "3m", "15m", "30m", "1h"],
            index=0,  # Default 5m
            key="single_tf"
        )

    with btn_col:
        st.write("")
        st.write("")
        analyze_btn = st.button("ANALYZE STOCK", type="primary", use_container_width=True)

    with st.expander("✏️ Or type/paste custom symbol manually"):
        manual_sym = st.text_input("Custom Symbol", value="", placeholder="e.g. TMCV, ETERNAL, MRF, SUZLON...").upper().strip()
        if manual_sym:
            sym_input = manual_sym

    if analyze_btn and sym_input:
        st.session_state["current_symbol"] = sym_input
        with st.spinner(f"Analyzing {sym_input} on {timeframe} timeframe..."):
            try:
                res_obj = run_stock_analysis(sym_input, timeframe)
                st.session_state["analysis_result"] = res_obj

                # Auto Telegram alert if configured
                if telegram_notifier.is_configured():
                    sig = res_obj["signal"]
                    rsk = res_obj["risk"]
                    tgt = res_obj["target"]
                    ind = res_obj["indicators"]
                    telegram_notifier.send_signal_alert(
                        symbol=sym_input,
                        direction=sig.direction,
                        current_price=res_obj["quote"]["last_price"],
                        entry=sig.entry_price,
                        entry_zone=sig.entry_zone,
                        stop_loss=rsk.stop_loss,
                        tp1=tgt.tp1,
                        tp2=tgt.tp2,
                        tp3=tgt.tp3,
                        rr_ratio=rsk.risk_reward_ratio,
                        volume_ratio=ind.volume_ratio,
                        timeframe=timeframe,
                        confluence_summary=sig.setup_name
                    )

            except MarketDataUnavailableError as mde:
                st.error(f"🔴 **DATA ERROR**: {str(mde)}")
                st.info("⚠️ In LIVE mode, signals are NEVER generated from simulated data when Upstox API calls fail.")
                st.session_state["analysis_result"] = None
            except Exception as e:
                st.error(f"🔴 **UNEXPECTED ERROR**: {str(e)}")
                st.session_state["analysis_result"] = None

    # Display Single Stock Analysis Results
    res = st.session_state.get("analysis_result")

    if res:
        sym = res["symbol"]
        q = res["quote"]
        sig = res["signal"]
        tgt = res["target"]
        rsk = res["risk"]
        df = res["df"]
        ind = res["indicators"]

        current_p = q["last_price"]
        prev_c = q["prev_close"]
        chg = q["change"]
        chg_p = q["change_pct"]

        # Stock Current Information Bar
        col_sym_title, col_data_badge = st.columns([3, 1])
        with col_sym_title:
            st.markdown(f"## {sym}")
        with col_data_badge:
            if res.get("is_live"):
                st.markdown("<div style='text-align: right;'><span class='badge-live'>🟢 LIVE UPSTOX DATA</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align: right;'><span class='badge-demo'>🟠 DEMO DATA</span></div>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        with m1:
            st.metric("LTP", format_currency(current_p), f"{chg_p:+.2f}%")
        with m2:
            st.metric("PREV CLOSE", format_currency(prev_c))
        with m3:
            st.metric("DAY HIGH", format_currency(q["high"]))
        with m4:
            st.metric("DAY LOW", format_currency(q["low"]))
        with m5:
            st.metric("PDH (PREV DAY)", format_currency(q["pdh"]))
        with m6:
            st.metric("PDL (PREV DAY)", format_currency(q["pdl"]))
        with m7:
            st.metric("VWAP", format_currency(ind.vwap))
        with m8:
            st.metric("VOL RATIO", f"{ind.volume_ratio:.2f}x")

        # Main Signal Card (BUY or SELL ONLY)
        is_buy = sig.direction == "BUY"
        card_class = "signal-card-buy" if is_buy else "signal-card-sell"
        sig_text_class = "signal-text-buy" if is_buy else "signal-text-sell"
        emoji = "🟢" if is_buy else "🔴"
        target_color = "#10b981" if is_buy else "#f87171"
        sl_color = "#ef4444" if is_buy else "#34d399"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size: 1.1rem; letter-spacing: 0.12em; color: #94a3b8; font-weight: 700; text-transform: uppercase;">
                {sym} • {res['timeframe'].upper()} TIMEFRAME • {sig.setup_name}
            </div>
            <div class="{sig_text_class}">
                {emoji} {sig.direction}
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 0.8rem;">
                <b>Entry Zone:</b> {sig.entry_zone} &nbsp;|&nbsp; <b>Signal Candle:</b> {sig.signal_candle_time}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.2rem; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.2rem;">
                <div>
                    <div class="stat-label">Current / Entry</div>
                    <div class="stat-val">{format_currency(sig.entry_price)}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Zone: {sig.entry_zone}</div>
                </div>
                <div>
                    <div class="stat-label">Stop Loss</div>
                    <div class="stat-val-highlight" style="color: {sl_color};">{format_currency(rsk.stop_loss)}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">-{rsk.risk_pct:.1f}% risk</div>
                </div>
                <div>
                    <div class="stat-label">Target 1 (TP1)</div>
                    <div class="stat-val-highlight" style="color: {target_color};">{format_currency(tgt.tp1)}</div>
                    <div style="font-size: 0.8rem; color: {target_color}; font-weight: 600;">{tgt.tp1_distance_pct:+.1f}% ({tgt.tp1_name})</div>
                </div>
                <div>
                    <div class="stat-label">Major Target (TP2)</div>
                    <div class="stat-val-highlight" style="color: {target_color};">{format_currency(tgt.tp2)}</div>
                    <div style="font-size: 0.8rem; color: {target_color}; font-weight: 600;">{tgt.tp2_distance_pct:+.1f}% ({tgt.tp2_name})</div>
                </div>
            </div>
            <div style="margin-top: 1.2rem; display: flex; justify-content: center; gap: 2rem; font-size: 0.95rem; color: #cbd5e1; flex-wrap: wrap;">
                <span>⚖️ <b>Risk/Reward:</b> 1 : {rsk.risk_reward_ratio:.1f}</span>
                <span>📊 <b>Volume:</b> {ind.volume:,} (Avg: {int(ind.volume_ma):,} | <b>{ind.volume_ratio:.2f}x</b>)</span>
                <span>⚡ <b>EMA 6 / 30:</b> ₹{ind.ema_6:.2f} / ₹{ind.ema_30:.2f}</span>
                <span>🛡️ <b>SL Invalidation:</b> {rsk.invalidation_level_name}</span>
            </div>
            <div class="disclaimer-text">
                ⚠️ Semi-algo technical analysis estimate for intraday — not guaranteed.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Candlestick Chart
        st.markdown("### 📊 TECHNICAL & SMC CHART")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=("", "Volume")
        )

        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candles",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444"
            ),
            row=1, col=1
        )

        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_6"], line=dict(color="#38bdf8", width=1.4), name="EMA 6"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_30"], line=dict(color="#f59e0b", width=1.5), name="EMA 30"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], line=dict(color="#ec4899", width=1.5, dash="dot"), name="VWAP (09:15 Reset)"), row=1, col=1)

        if res["smc"].pdh > 0:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=[res["smc"].pdh]*len(df), line=dict(color="#4ade80", width=1.2, dash="dash"), name="PDH (Prev Day High)"), row=1, col=1)
        if res["smc"].pdl > 0:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=[res["smc"].pdl]*len(df), line=dict(color="#f87171", width=1.2, dash="dash"), name="PDL (Prev Day Low)"), row=1, col=1)

        fig.add_hline(y=sig.entry_price, line=dict(color="#e2e8f0", width=1.4, dash="dash"), annotation_text=f"Entry: ₹{sig.entry_price:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=tgt.tp1, line=dict(color="#34d399", width=1.5, dash="dot"), annotation_text=f"TP1: ₹{tgt.tp1:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=tgt.tp2, line=dict(color="#10b981", width=2.0), annotation_text=f"TP2: ₹{tgt.tp2:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=rsk.stop_loss, line=dict(color="#ef4444", width=2.0), annotation_text=f"SL: ₹{rsk.stop_loss:.2f}", annotation_position="bottom right", row=1, col=1)

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
            height=550,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Breakdown drawer
        with st.expander("🔬 SMC Structure & 4-Factor Confluence Breakdown", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 1. Market Structure & SMC")
                st.write(f"• **Trend:** `{res['smc'].trend}`")
                st.write(f"• **Structure:** `{res['smc'].market_structure}`")
                st.write(f"• **BOS / CHOCH:** `{res['smc'].bos_choch_event.get('details', 'None')}`")
                st.write(f"• **Liquidity Sweep:** `{res['smc'].sweep_event.get('details', 'None')}`")
                st.write(f"• **Dealing Range:** `{res['smc'].dealing_range.get('zone', 'N/A')} ({res['smc'].dealing_range.get('pct', 0):.0f}% of range)`")
                st.write(f"• **PDH / PDL:** `₹{res['smc'].pdh:.2f} / ₹{res['smc'].pdl:.2f} ({res['smc'].pdh_pdl_status})`")
                st.write(f"• **Fresh Order Blocks:** `{len(res['smc'].fresh_order_blocks)} active`")
                st.write(f"• **Fresh FVGs:** `{len(res['smc'].fresh_fvgs)} active`")

            with c2:
                st.markdown("#### 2. Trend, Participation & Risk")
                st.write(f"• **EMA 6 vs 30:** `₹{ind.ema_6:.2f} vs ₹{ind.ema_30:.2f} ({'EMA 6 > 30' if ind.ema_6_above_30 else 'EMA 6 < 30'})`")
                st.write(f"• **Session VWAP:** `₹{ind.vwap:.2f} ({'Above VWAP' if current_p >= ind.vwap else 'Below VWAP'})`")
                st.write(f"• **Volume Surge:** `{ind.volume_ratio:.2f}x` ({'Surge (>=1.5x)' if sig.volume_surge else 'Normal'})")
                st.write(f"• **ATR (14):** `₹{ind.atr:.2f}`")
                st.write(f"• **RSI (14 Wilder):** `{ind.rsi:.1f}`")
                st.write(f"• **ADX (14 Wilder):** `{ind.adx:.1f} (+DI {ind.plus_di:.1f}, -DI {ind.minus_di:.1f})`")
                st.write(f"• **Risk/Reward:** `1 : {rsk.risk_reward_ratio:.1f}`")

            st.markdown("#### Calculated Signal Reasons")
            for r in sig.reasons:
                st.markdown(f"- {r}")

        # Telegram Action
        if telegram_notifier.is_configured():
            col_tg_left, col_tg_btn = st.columns([3, 1])
            with col_tg_left:
                st.caption("Telegram bot configured with persistent duplicate alert prevention.")
            with col_tg_btn:
                if st.button("✈️ Send Telegram Alert", use_container_width=True):
                    sent = telegram_notifier.send_signal_alert(
                        symbol=sym,
                        direction=sig.direction,
                        current_price=current_p,
                        entry=sig.entry_price,
                        entry_zone=sig.entry_zone,
                        stop_loss=rsk.stop_loss,
                        tp1=tgt.tp1,
                        tp2=tgt.tp2,
                        tp3=tgt.tp3,
                        rr_ratio=rsk.risk_reward_ratio,
                        volume_ratio=ind.volume_ratio,
                        timeframe=res["timeframe"],
                        confluence_summary=sig.setup_name,
                        force=True
                    )
                    if sent:
                        st.toast("✅ Telegram alert sent successfully!")
                    else:
                        st.error("Duplicate alert suppressed or delivery failed.")
