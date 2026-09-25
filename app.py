import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional
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
from smc_ml import analyze_smc_stock, run_smc_ml, Settings as SMCSettings, SMCSignalSnapshot

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="NSE Semi-Algo — Smart Money Concepts (SMC ML) Scanner & Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        color: #e2e8f0;
    }
    
    /* Header Container */
    .hero-header {
        text-align: center;
        padding: 1.4rem 0 1.2rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 1.2rem;
        background: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 45%, #c084fc 100%);
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
        padding: 0.28rem 0.85rem;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-closed {
        display: inline-block;
        padding: 0.28rem 0.85rem;
        border-radius: 9999px;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.35);
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
    
    /* Main Directional Signal Cards (BUY / SELL ONLY) */
    .signal-card-buy {
        background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.24) 0%, rgba(15, 23, 42, 0.95) 75%);
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 45px rgba(16, 185, 129, 0.25);
        margin: 1.2rem 0;
    }
    .signal-card-sell {
        background: radial-gradient(circle at 50% 0%, rgba(239, 68, 68, 0.24) 0%, rgba(15, 23, 42, 0.95) 75%);
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 45px rgba(239, 68, 68, 0.25);
        margin: 1.2rem 0;
    }
    
    .signal-text-buy {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        color: #10b981;
        text-shadow: 0 0 26px rgba(16, 185, 129, 0.55);
        margin: 0.3rem 0;
    }
    .signal-text-sell {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        color: #ef4444;
        text-shadow: 0 0 26px rgba(239, 68, 68, 0.55);
        margin: 0.3rem 0;
    }
    
    .stat-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .stat-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-val-highlight {
        font-size: 1.55rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Search Bar Wrapper */
    .search-container {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    
    .disclaimer-text {
        font-size: 0.75rem;
        color: #64748b;
        text-align: center;
        margin-top: 1rem;
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
    <div class="hero-title">⚡ NSE SEMI-ALGO — SMC ML SCANNER</div>
    <div class="hero-subtitle">Smart Money Concepts Machine Learning Indicator • NIFTY 200 Scanner • Full NSE Universe Search</div>
</div>
""", unsafe_allow_html=True)

# Market Hours Live Badge
is_open, market_status_text = is_nse_market_open()
now_str = format_ist_time()
badge_class = "badge-open" if is_open else "badge-closed"

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem; padding: 0 0.5rem;">
    <span class="{badge_class}">{market_status_text}</span>
    <span style="font-size: 0.85rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">Time (IST): {now_str}</span>
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
    st.info("👆 Please enter your Upstox Access Token above (or enter `demo` for sandbox testing) and click **CONNECT** to proceed.")
    st.stop()

st.divider()

# ==============================================================================
# STRATEGY EXECUTION HELPER (IN-MEMORY SMC ML)
# ==============================================================================
def run_stock_analysis(sym: str, tf: str) -> Dict[str, Any]:
    """Runs data pipeline and the Smart Money Concepts ML indicator in memory."""
    quote = upstox_client.get_market_quote(sym)
    raw_candles = upstox_client.fetch_candles(sym, timeframe=tf)
    pkg = prepare_market_data(sym, quote, raw_candles, timeframe=tf)
    df_ind, ind_snap = compute_indicators(pkg.candles)
    
    # Run the TradingView translated SMC ML indicator
    smc_snap = analyze_smc_stock(
        symbol=sym,
        df=df_ind,
        current_price=quote.get("last_price")
    )
    
    return {
        "symbol": sym,
        "timeframe": tf,
        "quote": quote,
        "pkg": pkg,
        "indicators": ind_snap,
        "smc": smc_snap,
        "df": df_ind,
        "is_live": quote.get("is_live", False)
    }

# ==============================================================================
# SECTION 2: TABS — MULTI-STOCK SCANNER vs SINGLE STOCK ANALYZER
# ==============================================================================
tab_scanner, tab_analyzer = st.tabs(["🚀 NIFTY 200 SCANNER", "🔍 SEARCH ANY NSE STOCK (DEEP-DIVE)"])

# ------------------------------------------------------------------------------
# TAB 1: NIFTY 200 MULTI-STOCK SCANNER
# ------------------------------------------------------------------------------
with tab_scanner:
    st.markdown("### 📡 NIFTY 200 & BASKET MARKET SCANNER")
    st.caption("Scan NIFTY 200, NIFTY 100, F&O Universe, or custom watchlist in parallel using TradingView's Smart Money Concept ML indicator. Tells BUY and SELL with Entry and Exit.")

    # Load complete baskets from UpstoxClient
    presets_dict = upstox_client.get_watchlist_presets()
    if not presets_dict:
        presets_dict = {
            "NIFTY 200 (Top 200 Stocks)": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC", "LT", "BAJFINANCE"],
            "NIFTY 50 (All 50 Stocks)": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY"],
            "NSE F&O UNIVERSE (All 199 Stocks)": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "BHEL", "SAIL", "VEDL"]
        }

    all_syms = upstox_client.get_all_symbols()
    all_eq_batches = {
        "ALL NSE EQUITIES (Batch 1 - 50)": all_syms[:50],
        "ALL NSE EQUITIES (Batch 51 - 100)": all_syms[50:100],
        "CUSTOM WATCHLIST": []
    }
    available_presets = {**presets_dict, **all_eq_batches}

    col_w1, col_limit, col_w2, col_w3 = st.columns([2.5, 1.2, 1, 1.3])
    with col_w1:
        # Default to NIFTY 200 basket
        preset_names = list(available_presets.keys())
        default_idx = 0
        for p_i, p_name in enumerate(preset_names):
            if "NIFTY 200" in p_name:
                default_idx = p_i
                break
        selected_preset = st.selectbox("Select Watchlist Basket", preset_names, index=default_idx)

    basket_stocks = available_presets[selected_preset]

    with col_limit:
        limit_options = ["All in Basket", "Top 25", "Top 50", "Top 100", "Top 10"]
        default_lim_idx = 0 if len(basket_stocks) <= 50 else 1
        chosen_limit = st.selectbox("Batch Limit", limit_options, index=default_lim_idx)

    with col_w2:
        scanner_tf = st.selectbox("Timeframe", options=["5m", "10m", "15m", "30m", "1h"], index=0, key="scanner_tf")

    with col_w3:
        st.write("")
        st.write("")
        run_scan_btn = st.button("🔍 SCAN BASKET", type="primary", use_container_width=True)

    if selected_preset == "CUSTOM WATCHLIST":
        custom_symbols_input = st.text_area(
            "Enter NSE Symbols (comma-separated):",
            value="RELIANCE, TCS, HDFCBANK, INFY, SBIN, BHEL, IRCTC, SUZLON, SAIL, TATAPOWER, TMCV, ETERNAL, TRENT, BEL, HAL",
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
                smc = res_obj["smc"]
                ind = res_obj["indicators"]
                q = res_obj["quote"]

                return {
                    "Stock": sym,
                    "Signal": f"🟢 BUY" if smc.direction == "BUY" else "🔴 SELL",
                    "Direction": smc.direction,
                    "LTP": round(q["last_price"], 2),
                    "Entry": round(smc.entry, 2),
                    "Stop Loss (Exit)": round(smc.stop, 2),
                    "TP1 (Exit)": round(smc.tp1, 2),
                    "TP2 (Exit)": round(smc.tp2, 2),
                    "Risk:Reward": f"1:{smc.risk_reward:.1f}",
                    "rr_val": smc.risk_reward,
                    "Volume Ratio": f"{ind.volume_ratio:.2f}x",
                    "vol_val": ind.volume_ratio,
                    "Method": smc.method,
                    "Trend": smc.trend,
                    "res_obj": res_obj
                }
            except Exception as e:
                return {
                    "Stock": sym,
                    "Signal": "⚠️ ERROR",
                    "Direction": "ERROR",
                    "LTP": 0.0,
                    "Entry": 0.0,
                    "Stop Loss (Exit)": 0.0,
                    "TP1 (Exit)": 0.0,
                    "TP2 (Exit)": 0.0,
                    "Risk:Reward": "N/A",
                    "rr_val": 0.0,
                    "Volume Ratio": "N/A",
                    "vol_val": 0.0,
                    "Method": "Data Error",
                    "Trend": str(e),
                    "res_obj": None
                }

        # Multi-threaded scanning for maximum performance
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
            filter_trend = st.selectbox("SMC Trend", ["ALL", "BULLISH Only", "BEARISH Only"], index=0)
        with f_col4:
            sort_by = st.selectbox("Sort By", ["Risk:Reward (High to Low)", "Volume Surge (High to Low)", "Stock Symbol (A-Z)"], index=0)

        filtered = list(results)
        if filter_dir == "🟢 BUY ONLY":
            filtered = [r for r in filtered if r["Direction"] == "BUY"]
        elif filter_dir == "🔴 SELL ONLY":
            filtered = [r for r in filtered if r["Direction"] == "SELL"]

        if filter_vol == "⚡ Volume Surges (>=1.5x) Only":
            filtered = [r for r in filtered if r.get("vol_val", 0) >= 1.5]

        if filter_trend == "BULLISH Only":
            filtered = [r for r in filtered if r.get("Trend") == "BULLISH"]
        elif filter_trend == "BEARISH Only":
            filtered = [r for r in filtered if r.get("Trend") == "BEARISH"]

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
            "Entry (₹)": f"₹{r['Entry']:,.2f}" if r["Entry"] > 0 else "N/A",
            "Stop Loss Exit (₹)": f"₹{r['Stop Loss (Exit)']:,.2f}" if r["Stop Loss (Exit)"] > 0 else "N/A",
            "TP1 Exit (₹)": f"₹{r['TP1 (Exit)']:,.2f}" if r["TP1 (Exit)"] > 0 else "N/A",
            "TP2 Exit (₹)": f"₹{r['TP2 (Exit)']:,.2f}" if r["TP2 (Exit)"] > 0 else "N/A",
            "Risk/Reward": r["Risk:Reward"],
            "Volume Ratio": r["Volume Ratio"],
            "Method": r["Method"],
            "Trend": r["Trend"]
        } for r in filtered])

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Quick Deep Dive selector
        act_col1, act_col2 = st.columns([3, 1])
        with act_col1:
            inspect_sym = st.selectbox(
                "Select stock from scan results to view full SMC ML interactive chart:",
                [r["Stock"] for r in total_valid],
                label_visibility="collapsed"
            )
        with act_col2:
            if st.button("📊 View Deep-Dive Chart", use_container_width=True):
                matched = next((r["res_obj"] for r in total_valid if r["Stock"] == inspect_sym), None)
                if matched:
                    st.session_state["analysis_result"] = matched
                    st.session_state["current_symbol"] = inspect_sym
                    st.toast(f"Loaded {inspect_sym} into Deep-Dive tab!")

# ------------------------------------------------------------------------------
# TAB 2: SEARCH ANY NSE STOCK IN WHOLE NSE MARKET (3,370+ STOCKS)
# ------------------------------------------------------------------------------
with tab_analyzer:
    st.markdown("### 🔍 SEARCH ANY STOCK IN WHOLE NSE MARKET")
    st.caption("Search across all 3,370+ NSE stocks in `nse_universe.json`. Evaluates the Smart Money Concept ML indicator and tells BUY or SELL only with Entry and Exit.")

    # Popular quick-click stock chips
    st.markdown("**Popular & Liquid NSE Stocks:**")
    quick_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN", "SUZLON", "BHEL", "IRCTC", "ETERNAL", "TATAPOWER", "SAIL", "VEDL", "TRENT", "BEL", "MRF"]
    quick_cols = st.columns(len(quick_stocks))
    for q_idx, q_sym in enumerate(quick_stocks):
        with quick_cols[q_idx]:
            if st.button(q_sym, key=f"quick_{q_sym}", use_container_width=True):
                st.session_state["current_symbol"] = q_sym
                st.rerun()

    # Load complete stock universe from nse_universe.json
    stocks_meta = upstox_client.get_equity_stocks()
    stock_display_map = {}
    for s in stocks_meta:
        sym = s["symbol"]
        name = s.get("name", sym)
        disp = f"{sym} — {name}"
        stock_display_map[disp] = sym

    stock_display_map["TATAMOTORS — TATA MOTORS LIMITED (TMCV)"] = "TMCV"
    stock_display_map["ZOMATO — ETERNAL LIMITED (formerly Zomato)"] = "ETERNAL"

    # Category selector and Direct Search Bar
    with st.container():
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        search_filter_col, search_input_col, tf_col, btn_col = st.columns([1.5, 2.5, 1, 1.2])

        with search_filter_col:
            cat_choices = ["NIFTY 200", "All NSE Stocks (3,370+)", "NIFTY 50", "NSE F&O Universe (199)", "Nifty Bank", "Nifty IT", "Nifty Auto", "Nifty Metal", "Nifty Pharma"]
            sel_cat = st.selectbox("Universe Basket Filter", cat_choices, index=0)

        # Filter display options
        presets = upstox_client.get_watchlist_presets()
        cat_filter_set = None
        if sel_cat == "NIFTY 200":
            cat_filter_set = set(presets.get("NIFTY 200 (Top 200 Stocks)", []))
        elif sel_cat == "NIFTY 50":
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

        with search_input_col:
            chosen_stock_disp = st.selectbox(
                "Search & Select Stock (Symbol or Name)",
                options=filtered_disp_list if filtered_disp_list else [curr_sym],
                index=def_idx if def_idx < len(filtered_disp_list) else 0,
                help="Type symbol or company name to search all 3,370+ stocks in nse_universe.json."
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

        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("✏️ Direct Symbol Input (Quick Type)"):
        manual_sym = st.text_input("Type Any NSE Stock Symbol directly:", value="", placeholder="e.g. SUZLON, RELIANCE, TMCV, 20MICRONS, MRF...").upper().strip()
        if manual_sym:
            sym_input = manual_sym

    if analyze_btn and sym_input:
        st.session_state["current_symbol"] = sym_input
        with st.spinner(f"Running Smart Money Concept ML indicator for {sym_input} on {timeframe}..."):
            try:
                res_obj = run_stock_analysis(sym_input, timeframe)
                st.session_state["analysis_result"] = res_obj
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
        smc = res["smc"]
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

        # ==============================================================================
        # MAIN SIGNAL CARD (BUY OR SELL ONLY WITH ENTRY AND EXIT)
        # ==============================================================================
        is_buy = smc.direction == "BUY"
        card_class = "signal-card-buy" if is_buy else "signal-card-sell"
        sig_text_class = "signal-text-buy" if is_buy else "signal-text-sell"
        emoji = "🟢" if is_buy else "🔴"
        target_color = "#10b981" if is_buy else "#f87171"
        sl_color = "#ef4444" if is_buy else "#34d399"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size: 1.1rem; letter-spacing: 0.12em; color: #94a3b8; font-weight: 700; text-transform: uppercase;">
                {sym} • {res['timeframe'].upper()} TIMEFRAME • SMC ML STRATEGY ({smc.method.upper()})
            </div>
            <div class="{sig_text_class}">
                {emoji} {smc.direction}
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 0.8rem;">
                <b>Signal Bar Time:</b> {smc.signal_time} &nbsp;|&nbsp; <b>SMC Structure Trend:</b> {smc.trend}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.2rem; text-align: center; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.4rem;">
                <div>
                    <div class="stat-label">🎯 ENTRY PRICE</div>
                    <div class="stat-val">{format_currency(smc.entry)}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Method: {smc.method}</div>
                </div>
                <div>
                    <div class="stat-label">🛑 EXIT (STOP LOSS)</div>
                    <div class="stat-val-highlight" style="color: {sl_color};">{format_currency(smc.stop)}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">-{smc.risk_pct:.1f}% risk (₹{smc.risk_amount:.2f})</div>
                </div>
                <div>
                    <div class="stat-label">🏁 EXIT (TARGET 1)</div>
                    <div class="stat-val-highlight" style="color: {target_color};">{format_currency(smc.tp1)}</div>
                    <div style="font-size: 0.8rem; color: {target_color}; font-weight: 600;">{smc.tp1_pct:+.1f}% (1:1 R:R)</div>
                </div>
                <div>
                    <div class="stat-label">🚀 EXIT (TARGET 2)</div>
                    <div class="stat-val-highlight" style="color: {target_color};">{format_currency(smc.tp2)}</div>
                    <div style="font-size: 0.8rem; color: {target_color}; font-weight: 600;">{smc.tp2_pct:+.1f}% (1:2 R:R)</div>
                </div>
            </div>
            <div style="margin-top: 1.2rem; display: flex; justify-content: center; gap: 2rem; font-size: 0.95rem; color: #cbd5e1; flex-wrap: wrap;">
                <span>⚖️ <b>Risk/Reward Ratio:</b> 1 : {smc.risk_reward:.1f}</span>
                <span>⚡ <b>ATR (14):</b> ₹{smc.atr:.2f}</span>
                <span>📊 <b>Volume:</b> {ind.volume:,} (<b>{ind.volume_ratio:.2f}x</b> avg)</span>
                <span>📦 <b>Order Blocks:</b> {smc.order_blocks_count}</span>
                <span>🌌 <b>FVG Zones:</b> {smc.fvgs_count}</span>
            </div>
            <div class="disclaimer-text">
                Smart Money Concepts quantitative ML translation. Intraday semi-algo recommendation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ==============================================================================
        # INTERACTIVE CANDLESTICK & SMC CHART
        # ==============================================================================
        st.markdown("### 📊 TECHNICAL & SMC ML CHART")

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

        # Indicators
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_6"], line=dict(color="#38bdf8", width=1.4), name="EMA 6"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema_30"], line=dict(color="#f59e0b", width=1.5), name="EMA 30"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["vwap"], line=dict(color="#ec4899", width=1.5, dash="dot"), name="VWAP"), row=1, col=1)

        # Entry, Stop Loss, TP1, TP2 Horizontal lines
        fig.add_hline(y=smc.entry, line=dict(color="#e2e8f0", width=1.5, dash="dash"), annotation_text=f"ENTRY: ₹{smc.entry:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=smc.tp1, line=dict(color="#34d399", width=1.5, dash="dot"), annotation_text=f"EXIT TP1: ₹{smc.tp1:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=smc.tp2, line=dict(color="#10b981", width=2.0), annotation_text=f"EXIT TP2: ₹{smc.tp2:.2f}", annotation_position="top right", row=1, col=1)
        fig.add_hline(y=smc.stop, line=dict(color="#ef4444", width=2.0), annotation_text=f"EXIT SL: ₹{smc.stop:.2f}", annotation_position="bottom right", row=1, col=1)

        # SMC Liquidity Pools
        if np.isfinite(smc.liquidity_above):
            fig.add_hline(y=smc.liquidity_above, line=dict(color="#f43f5e", width=1.2, dash="dashdot"), annotation_text=f"Pool Above: ₹{smc.liquidity_above:.2f}", annotation_position="top left", row=1, col=1)
        if np.isfinite(smc.liquidity_below):
            fig.add_hline(y=smc.liquidity_below, line=dict(color="#06b6d4", width=1.2, dash="dashdot"), annotation_text=f"Pool Below: ₹{smc.liquidity_below:.2f}", annotation_position="bottom left", row=1, col=1)

        # Volume Subplot
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
            paper_bgcolor="#080c14",
            plot_bgcolor="#0f172a",
            height=580,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # SMC Machine Learning Structure Breakdown Drawer
        with st.expander("🔬 SMC Machine Learning & Structural Details", expanded=True):
            raw = smc.raw_result
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 1. Smart Money Concepts ML")
                st.write(f"• **Direction Recommendation:** `{smc.direction}`")
                st.write(f"• **Entry Price:** `₹{smc.entry:.2f}`")
                st.write(f"• **Stop Loss (Exit):** `₹{smc.stop:.2f}` (-{smc.risk_pct:.1f}%)")
                st.write(f"• **Take Profit 1 (Exit):** `₹{smc.tp1:.2f}` ({smc.tp1_pct:+.1f}%)")
                st.write(f"• **Take Profit 2 (Exit):** `₹{smc.tp2:.2f}` ({smc.tp2_pct:+.1f}%)")
                st.write(f"• **Risk/Reward:** `1 : {smc.risk_reward:.1f}`")
                st.write(f"• **Method:** `{smc.method}`")
                st.write(f"• **Structure Trend:** `{smc.trend}`")

            with c2:
                st.markdown("#### 2. ML Probability & Liquidity Odds")
                odds_above = f"{smc.odds_pool_above*100:.1f}%" if np.isfinite(smc.odds_pool_above) else "Calculating..."
                st.write(f"• **Probability Pool Above reached first:** `{odds_above}`")
                st.write(f"• **Nearest Pool Above:** `₹{smc.liquidity_above:.2f}`" if np.isfinite(smc.liquidity_above) else "• **Nearest Pool Above:** `None`")
                st.write(f"• **Nearest Pool Below:** `₹{smc.liquidity_below:.2f}`" if np.isfinite(smc.liquidity_below) else "• **Nearest Pool Below:** `None`")
                st.write(f"• **Active Order Blocks:** `{smc.order_blocks_count}`")
                st.write(f"• **Active Fair Value Gaps (FVG):** `{smc.fvgs_count}`")
                st.write(f"• **ATR (Wilder 14):** `₹{smc.atr:.2f}`")
                st.write(f"• **Volume Surge:** `{ind.volume_ratio:.2f}x` ({'High Volume' if ind.volume_ratio >= 1.5 else 'Normal'})")

            # Show completed trades if any
            trades_df = raw.get("trades")
            if trades_df is not None and not trades_df.empty:
                st.markdown("#### 📋 Completed Trades Backtest Log (Current Series)")
                disp_trades = trades_df.tail(5)[["direction", "entry", "initial_stop", "tp1", "tp2", "exit_price", "result", "r_multiple"]].copy()
                disp_trades["direction"] = disp_trades["direction"].map({1: "BUY", -1: "SELL"})
                st.dataframe(disp_trades, use_container_width=True, hide_index=True)
