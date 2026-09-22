# NSE SEMI-ALGO — Simple Streamlit Trading Analyzer

A clean, hyper-focused **Streamlit-based trading analyzer** for the National Stock Exchange (NSE) powered by the **Upstox API v2**.

The application is built around one singular, uncompromising goal:
> **User enters Upstox Access Token → connects → enters stock symbol (e.g. `SAIL`, `RELIANCE`, `TCS`) → system analyzes the stock → displays ONLY BUY or SELL with Current Price, Entry Price, Maximum Target, and Stop Loss.**

---

## ⚡ Key Highlights

* **All 3,370+ NSE Stocks Supported**: Complete catalog of all active NSE equities with company names, ISINs, and alias lookups (`TATAMOTORS`/`TMCV`, `ZOMATO`/`ETERNAL`).
* **Official NSE Watchlist Presets**: Instant one-click scanning of **NIFTY 50 (All 50 Stocks)**, **NIFTY NEXT 50**, **NIFTY 100**, **NSE F&O UNIVERSE (All 199 Derivative Stocks)**, **NIFTY BANK**, **NIFTY IT**, **NIFTY AUTO**, **NIFTY METAL**, **NIFTY PHARMA**, **NIFTY FMCG**, **NIFTY ENERGY**, **NIFTY PSU BANK**, **NIFTY REALTY**, and **HIGH MOMENTUM**.
* **Blazing Fast Multi-Stock Scanner**: Multi-threaded parallel scanning via `ThreadPoolExecutor` analyzing 50–200 stocks in seconds with live progress tracking.
* **Interactive Table & CSV Export**: Real-time filtering by Signal (BUY/SELL), Volume Surge ($\ge 1.5\times$), Market Structure (BOS/CHOCH), sorting, and 1-click CSV download.
* **Searchable Autocomplete in Deep-Dive**: Smart searchable dropdown containing all 3,370+ NSE companies with instant interactive Plotly charting.
* **Decisive Directional Output**: Strictly outputs **🟢 BUY** or **🔴 SELL** with complete structural trade management.
* **Accurate Previous Day High/Low (PDH / PDL)**: High and Low of the previous **completed** NSE trading session, accurately handling weekends, market closures, and official NSE holidays. Never uses today's in-progress OHLC.
* **Zero Fake Data in LIVE Mode**: If live Upstox API requests fail, the system reports `DATA ERROR` and **never** generates signals from synthetic data.
* **Core Semi-Algo Strategy**:
  - **EMA 6 & EMA 30**: Ultra-agile intraday momentum and trend alignment.
  - **Session-Reset VWAP**: Resets strictly at 09:15 at the start of every NSE trading session.
  - **Volume 20-MA & 1.5x Ratio**: Real participation filter displaying current volume, average volume, and volume ratio.
  - **Wilder's Smoothing**: Standard Wilder's smoothing for RSI (14), ATR (14), and ADX (14).
* **Proper 5m and 10m Timeframe Support**: 10-minute candles are strictly constructed with exact NSE 09:15 session alignment (`09:15–09:24`, `09:25–09:34`, etc.). Only completed candles are analyzed.
* **Enhanced Smart Money Concepts (SMC)**:
  - Confirmed non-repainting swings (HH, HL, LH, LL) and protected structural pivots.
  - Break of Structure (BOS) vs Change of Character (CHOCH).
  - Liquidity sweeps: SSL (Bullish), BSL (Bearish), PDH and PDL sweeps.
  - Fresh vs Mitigated Order Blocks and Fair Value Gaps (FVG).
  - Dealing range with 50% equilibrium and Premium/Discount zones.
* **Multi-Target & Structural Risk Engine**:
  - **TP1**: Nearest meaningful structural level / liquidity pool.
  - **TP2**: Major structural target (prioritizes PDH/PDL, major swings).
  - **TP3**: Extended structural / runner target.
  - **Structural Stop Loss**: Anchored below swing low/OB/sweep for BUY, above swing high/OB/sweep for SELL, with ATR buffer.
  - **Enforced Minimum R:R**: 1 : 2.0+.
* **Token Security & Persistent Alerts**: Upstox access tokens are masked and never logged. Telegram duplicate alert prevention is persistent across app restarts via `.sent_alerts.json`.

---

## 🧠 Confluence Decision Engine

Signals are grouped into 4 transparent categories:

1. **Structure (SMC)**:
   - Market structure sequence (HH/HL or LH/LL)
   - Bullish / Bearish BOS & CHOCH confirmation
   - Liquidity sweeps (SSL, BSL, PDH, PDL)
   - Fresh unmitigated Order Blocks & FVGs
   - Premium / Discount dealing range location
2. **Trend**:
   - EMA 6 vs EMA 30 crossover and alignment
   - Session-reset VWAP relationship (above VWAP for long, below for short)
3. **Participation**:
   - Volume 20-period moving average
   - Volume ratio (>= 1.5x volume surge detection)
4. **Risk & Execution**:
   - Structural invalidation stop loss
   - Multi-target planning (TP1, TP2, TP3) calibrated for at least 1:2.0 Risk/Reward
   - Suggested entry zone and confirmation price

---

## 🚀 Quick Start (Local)

### 1. Prerequisites
- Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🛠️ Step-by-Step Usage

1. **Step 1: Upstox Connection**
   - Enter your active Upstox Access Token (from [Upstox Developer Portal](https://developer.upstox.com)).
   - Click **CONNECT**.
   - You will see **🟢 UPSTOX CONNECTED** with your account details.
   *(Note: For offline or weekend testing without an active market session, enter `demo` or `test` to test in sandbox mode).*

2. **Step 2: Enter NSE Stock**
   - Type any NSE stock symbol: `SAIL`, `RELIANCE`, `TCS`, `INFY`, `SBIN`, etc.
   - Select timeframe (default: `5m`).
   - Click **ANALYZE STOCK**.

3. **Step 3: View Results**
   - Instantly view **CURRENT PRICE**, **BUY or SELL**, **ENTRY PRICE**, **ESTIMATED MAXIMUM TARGET**, and **STOP LOSS**.
   - Explore the candlestick chart and optional SMC drawer below.

---

## ☁️ Deployment

### Streamlit Community Cloud
1. Push repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Deploy an app**.
3. Select your repository and specify `app.py` as the main file path.
4. Click **Deploy**.

---

## ⚠️ Disclaimer

*Technical analysis estimate — not guaranteed.*
This application is designed for informational and analytical assistance. Trading in the Indian equity markets involves substantial risk of loss. Always manage your position sizing and trade responsibly.
