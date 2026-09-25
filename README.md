# NSE SEMI-ALGO — Smart Money Concepts (SMC ML) Scanner & Trading Analyzer

A high-performance, Streamlit-based **SMC Machine Learning trading analyzer and multi-stock scanner** for the National Stock Exchange (NSE) powered by Upstox API v2 / v3 and in-memory execution.

The application translates TradingView's Smart Money Concept ML indicator to deliver clear, decisive trade setups:
> **User connects Upstox → scans NIFTY 200 or searches ANY stock in the entire NSE universe (`nse_universe.json`) → system runs the SMC ML strategy → tells BUY or SELL only with precise Entry, Stop Loss (Exit), and Targets (Exit).**

---

## ⚡ Key Highlights

* **Smart Money Concepts (SMC ML) Engine**: Full in-memory implementation of the TradingView SMC ML indicator:
  - Confirmed swing pivots, Break of Structure (BOS), and Change of Character (CHoCH).
  - Prior-centred logistic MAP calibrated retest probabilities.
  - Reflection-principle liquidity draw odds (Odds pool above reached first).
  - Unmitigated Order Blocks and Fair Value Gaps (FVG).
  - Completed trades backtest evaluator with R-multiples.
* **Scan NIFTY 200 in Parallel**: Dedicated scanner preset with all **200 constituents of NIFTY 200** scanned concurrently via multi-threaded `ThreadPoolExecutor`.
* **Universal Search Across Whole NSE Market (3,370+ Stocks)**: Search bar and filter covering all 3,370+ NSE equities defined in [nse_universe.json](file:///c:/Users/suriya%20prakash/Downloads/nse-smc-scanner-main/nse_universe.json).
* **Decisive Output — BUY & SELL ONLY**:
  - Direction: **🟢 BUY** or **🔴 SELL**
  - **Entry Price**: Exact level / method (Break Close / Level Retest / Structure Flow)
  - **Exit 1 (Stop Loss)**: Structural invalidation level with -% risk
  - **Exit 2 (Target 1 / TP1)**: 1:1 Risk/Reward level with +% gain
  - **Exit 3 (Target 2 / TP2)**: 1:2 Risk/Reward major exit level with +% gain
* **Interactive Candlestick & SMC Chart**: Displays Candlesticks, Entry line, Stop Loss Exit line, TP1 & TP2 Exit lines, EMA 6/30, VWAP, Liquidity Pools, Order Blocks, and Volume with 20-MA.
* **Zero Disk CSV Storage**: Strategy executes strictly in memory for lightning fast performance.
* **Streamlined Architecture**: Unwanted files and Telegram alerts completely removed.

---

## 🚀 Quick Start (Local)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Step-by-Step Usage

1. **Step 1: Upstox Connection**
   - Enter your active Upstox Access Token.
   - Click **CONNECT**.
   *(Note: For offline or weekend testing without an active market session, enter `demo` or `test` to test in sandbox mode).*

2. **Step 2: Scan NIFTY 200 or Search Any Stock**
   - **Tab 1: NIFTY 200 Scanner**: Select batch limit (e.g., Top 25, Top 50, Top 100, All 200), choose timeframe, and click **SCAN BASKET**.
   - **Tab 2: Search Any Stock**: Type any symbol (e.g. `SUZLON`, `RELIANCE`, `TCS`, `MRF`, `20MICRONS`) or select from the dropdown containing all 3,370+ NSE stocks from `nse_universe.json`.

3. **Step 3: View Signals**
   - Instantly view **BUY or SELL**, **ENTRY PRICE**, **STOP LOSS (EXIT)**, and **TARGETS (EXIT)** with full interactive chart.

---

## ⚠️ Disclaimer

*Technical analysis estimate — not guaranteed.*
This application is designed for informational and analytical assistance. Trading in Indian equity markets involves substantial risk of loss. Always manage your position sizing and trade responsibly.
