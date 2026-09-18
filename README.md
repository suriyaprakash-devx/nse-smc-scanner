# NSE SEMI-ALGO — Simple Streamlit Trading Analyzer

A clean, hyper-focused **Streamlit-based trading analyzer** for the National Stock Exchange (NSE) powered by the **Upstox API v2**.

The application is built around one singular, uncompromising goal:
> **User enters Upstox Access Token → connects → enters stock symbol (e.g. `SAIL`, `RELIANCE`, `TCS`) → system analyzes the stock → displays ONLY BUY or SELL with Current Price, Entry Price, Maximum Target, and Stop Loss.**

---

## ⚡ Key Highlights

* **Strict Binary Decision**: **NEVER** displays `WAIT`, `WATCH`, `NO TRADE`, `HOLD`, or `NEUTRAL`. Outputs strictly **🟢 BUY** or **🔴 SELL**.
* **Zero Instrument Key Hassle**: Automatically resolves official NSE instrument keys from the 1,100+ equity Upstox universe (e.g., `SAIL` ➔ `NSE_EQ|INE114A01011`).
* **5 Primary Outputs (Zero Distraction)**:
  1. **Current Price**: ₹XXX.XX (live Upstox tick data)
  2. **🟢 BUY** or **🔴 SELL**
  3. **Entry Price**: ₹XXX.XX (calculated valid entry / retest / breakout level)
  4. **Estimated Maximum Target**: ₹XXX.XX (structural upside/downside projection, with move %)
  5. **Stop Loss**: ₹XXX.XX (structural invalidation level, with risk %)
* **Risk / Reward**: Dynamically calculated (e.g., `1 : 2.5`).
* **Closed-Candle Analysis**: Eliminates look-ahead bias and repainting.
* **Token Security**: Tokens are held exclusively in Streamlit session state and are **never** logged, printed to console, or exposed.
* **Interactive Candlestick Chart**: Plotly chart with multi-timeframe support (`1m`, `3m`, `5m`, `15m`, `30m`, `1h`), EMAs (9, 21, 50), VWAP, Previous Day High/Low, and Volume sub-panel.
* **Optional Collapsible SMC Panel**: Deep-dive into Market Structure, BOS/CHOCH, Order Blocks, Liquidity Sweeps, FVGs, and Indicator scores without cluttering the main view.
* **Telegram Integration**: Instant alerts delivered to Telegram with duplicate suppression.

---

## 🧠 Confluence Decision Engine

Signals are generated using a multi-factor directional scoring model:

1. **Technical Indicators**:
   - EMA Alignment: EMA 9, EMA 21, EMA 50
   - RSI (14) momentum and crossing
   - MACD (12, 26, 9) histogram expansion
   - Intraday VWAP relative positioning
   - ADX (14) trend strength with +DI / -DI
   - Bollinger Bands (20, 2 std)
   - Volume surge filter (>= 1.5x 20-period Volume MA)
2. **Price Action**:
   - Candle structure & body size
   - Bullish & Bearish Engulfing
   - Range Breakout / Breakdown (20 bars)
   - Wick rejection analysis
3. **Smart Money Concepts (SMC)**:
   - Non-repainting Fractal Swing Highs and Swing Lows
   - Structural points: Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
   - Break of Structure (BOS) and Change of Character (CHOCH)
   - Liquidity Sweeps: Buy-Side Liquidity (BSL) and Sell-Side Liquidity (SSL)
   - Fair Value Gaps (FVG) with mitigation tracking
   - Bullish & Bearish Order Blocks
   - Dealing Range: Premium vs Discount zone equilibrium
4. **Previous Day Levels**:
   - Previous Day High (PDH) & Previous Day Low (PDL)

```python
if BUY_SCORE >= SELL_SCORE:
    DECISION = "BUY"
else:
    DECISION = "SELL"
```

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
