import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import requests
import pandas as pd
import pytz

from config import settings
from utils import get_ist_timezone, get_ist_now, mask_token, MarketDataUnavailableError
from calendar_utils import get_previous_trading_day, is_nse_holiday

logger = logging.getLogger("upstox_client")

WELL_KNOWN_NSE_INSTRUMENTS = {
    "SAIL": "NSE_EQ|INE114A01011",
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "ICICIBANK": "NSE_EQ|INE090A01021",
    "INFY": "NSE_EQ|INE009A01021",
    "BHARTIARTL": "NSE_EQ|INE397D01024",
    "ITC": "NSE_EQ|INE154A01025",
    "SBIN": "NSE_EQ|INE062A01020",
    "LICI": "NSE_EQ|INE115A01026",
    "LT": "NSE_EQ|INE018A01030",
    "HINDUNILVR": "NSE_EQ|INE030A01027",
    "BAJFINANCE": "NSE_EQ|INE296A01024",
    "HCLTECH": "NSE_EQ|INE860A01027",
    "MARUTI": "NSE_EQ|INE585B01010",
    "SUNPHARMA": "NSE_EQ|INE044A01036",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
    "ONGC": "NSE_EQ|INE213A01029",
    "KOTAKBANK": "NSE_EQ|INE237A01028",
    "NTPC": "NSE_EQ|INE733E01010",
    "AXISBANK": "NSE_EQ|INE238A01034",
    "TITAN": "NSE_EQ|INE280A01028",
    "ADANIENT": "NSE_EQ|INE423A01024",
    "ADANIPORTS": "NSE_EQ|INE742F01042",
    "ULTRACEMCO": "NSE_EQ|INE481G01011",
    "POWERGRID": "NSE_EQ|INE752E01010",
    "TATASTEEL": "NSE_EQ|INE081A01020",
    "COALINDIA": "NSE_EQ|INE522F01014",
    "BAJAJFINSV": "NSE_EQ|INE918I01026",
    "M&M": "NSE_EQ|INE101A01026",
    "JSWSTEEL": "NSE_EQ|INE019A01038",
    "GRASIM": "NSE_EQ|INE047A01021",
    "WIPRO": "NSE_EQ|INE075A01022",
    "NESTLEIND": "NSE_EQ|INE239A01024",
    "CIPLA": "NSE_EQ|INE059A01026",
    "TECHM": "NSE_EQ|INE669C01036",
    "HINDALCO": "NSE_EQ|INE038A01020",
    "DRREDDY": "NSE_EQ|INE089A01023",
    "EICHERMOT": "NSE_EQ|INE066A01021",
    "APOLLOHOSP": "NSE_EQ|INE437A01024",
    "BPCL": "NSE_EQ|INE029A01011",
    "DIVISLAB": "NSE_EQ|INE361B01024",
    "BRITANNIA": "NSE_EQ|INE216A01030",
    "ASIANPAINT": "NSE_EQ|INE021A01026",
    "SBILIFE": "NSE_EQ|INE123W01016",
    "HDFCLIFE": "NSE_EQ|INE795G01014",
    "TATACONSUM": "NSE_EQ|INE192A01025",
    "BAJAJ-AUTO": "NSE_EQ|INE917I01010",
    "HEROMOTOCO": "NSE_EQ|INE158A01026",
    "SHRIRAMFIN": "NSE_EQ|INE721A01013",
    "BEL": "NSE_EQ|INE263A01024",
    "ZOMATO": "NSE_EQ|INE758T01015",
    "PAYTM": "NSE_EQ|INE982J01020",
    "JIOFIN": "NSE_EQ|INE758E01017"
}

class UpstoxClient:
    """Production Upstox API v2 / v3 Client with rate limiting, secure session handling,
    and genuine previous completed NSE session (PDH/PDL) resolution.
    
    STRICT RULE: Never falls back to synthetic or fake data in LIVE mode.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token.strip() if token else None
        self.base_url = settings.UPSTOX_BASE_URL.rstrip("/")
        self.v3_base_url = settings.UPSTOX_V3_BASE_URL.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": f"{settings.APP_NAME}/{settings.APP_VERSION}"
        })
        self._universe_map: Dict[str, str] = dict(WELL_KNOWN_NSE_INSTRUMENTS)
        self._stocks_list: List[Dict[str, Any]] = []
        self._baskets: Dict[str, List[str]] = {}
        self._pd_cache: Dict[str, Dict[str, Any]] = {}
        self._load_local_universe()

    def set_token(self, token: str):
        self.token = token.strip() if token else None

    def is_demo(self) -> bool:
        """Determines if the client is operating in explicit demo/sandbox mode."""
        if not self.token:
            return True
        t = self.token.lower()
        return t in ["test", "demo", "sandbox"] or t.startswith("demo_")

    def _load_local_universe(self):
        """Loads cached instrument map, stock details, and watchlist baskets from nse_universe.json."""
        universe_path = os.path.join(os.path.dirname(__file__), "nse_universe.json")
        if os.path.exists(universe_path):
            try:
                with open(universe_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    keys = data.get("instrument_keys", {})
                    if keys:
                        self._universe_map.update(keys)
                    self._stocks_list = data.get("stocks", [])
                    self._baskets = data.get("baskets", {})
            except Exception as e:
                logger.debug(f"Could not load nse_universe.json: {e}")

    def _get_headers(self) -> dict:
        if not self.token:
            raise ValueError("Upstox Access Token is not set.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Executes HTTP request with exponential backoff on 429 rate limit."""
        max_retries = 3
        backoff = 0.8
        for attempt in range(max_retries):
            try:
                resp = self.session.request(method, url, timeout=12, **kwargs)
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return resp
            except (requests.ConnectionError, requests.Timeout):
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff *= 2
        return self.session.request(method, url, timeout=12, **kwargs)

    def validate_token(self, token: Optional[str] = None) -> Dict[str, Any]:
        """Validates Upstox access token against user profile endpoint.
        Does NOT log, display, or leak the access token.
        """
        chk_token = (token or self.token or "").strip()
        if not chk_token:
            return {
                "valid": False,
                "message": "Please enter your Upstox Access Token.",
                "user_name": None,
                "user_id": None
            }

        # Sandbox / demo mode check
        if chk_token.lower() in ["test", "demo", "sandbox"] or chk_token.startswith("demo_"):
            return {
                "valid": True,
                "message": "Connected in Sandbox / Demo Mode",
                "user_name": "NSE Demo Trader",
                "user_id": "DEMO_USER",
                "email": "demo@nse-semi-algo.demo",
                "is_sandbox": True
            }

        url = f"{self.base_url}/user/profile"
        headers = {
            "Authorization": f"Bearer {chk_token}",
            "Accept": "application/json"
        }

        try:
            resp = self._request_with_retry("GET", url, headers=headers)
            if resp.status_code == 200:
                res_data = resp.json().get("data", {})
                return {
                    "valid": True,
                    "message": "Connected successfully to Upstox API",
                    "user_name": res_data.get("user_name", "NSE Trader"),
                    "user_id": res_data.get("user_id", "N/A"),
                    "email": res_data.get("email", ""),
                    "is_sandbox": False
                }
            elif resp.status_code in [401, 403]:
                return {
                    "valid": False,
                    "message": "Invalid or expired Upstox Access Token. Please generate a fresh token.",
                    "user_name": None,
                    "user_id": None
                }
            else:
                err_text = ""
                try:
                    err_text = resp.json().get("errors", [{}])[0].get("message", "")
                except Exception:
                    pass
                return {
                    "valid": False,
                    "message": f"Upstox connection failed (HTTP {resp.status_code}): {err_text or 'Invalid response'}",
                    "user_name": None,
                    "user_id": None
                }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Network error connecting to Upstox: {str(e)}",
                "user_name": None,
                "user_id": None
            }

    def get_all_symbols(self) -> List[str]:
        """Returns sorted list of all supported NSE symbols."""
        if self._stocks_list:
            return [s["symbol"] for s in self._stocks_list]
        return sorted(list(self._universe_map.keys()))

    def get_watchlist_presets(self) -> Dict[str, List[str]]:
        """Returns pre-built categorized NSE watchlist baskets."""
        return dict(self._baskets)

    def get_equity_stocks(self) -> List[Dict[str, Any]]:
        """Returns list of all active NSE equity stocks with names and metadata."""
        return list(self._stocks_list)

    def resolve_instrument_key(self, symbol: str) -> str:
        """Automatically resolves ANY NSE equity stock symbol to Upstox instrument key."""
        clean_sym = symbol.strip().upper()
        if clean_sym in self._universe_map:
            return self._universe_map[clean_sym]

        # Common aliases & renames
        ALIASES = {
            "ZOMATO": "ETERNAL",
            "TATAMOTORS": "TMCV",
            "TATA MOTORS": "TMCV",
            "M&M": "M&M",
            "NIFTY": "Nifty 50",
            "BANKNIFTY": "Nifty Bank"
        }
        if clean_sym in ALIASES and ALIASES[clean_sym] in self._universe_map:
            return self._universe_map[ALIASES[clean_sym]]

        # Fallback check stripping -EQ, .NS, etc.
        base_sym = clean_sym.replace("-EQ", "").replace(".NS", "").replace(" ", "")
        if base_sym in self._universe_map:
            return self._universe_map[base_sym]

        return f"NSE_EQ|{clean_sym}"

    def _get_base_price_for_symbol(self, symbol: str) -> float:
        clean_sym = symbol.strip().upper()
        base_prices = {
            "SAIL": 142.50, "RELIANCE": 2980.0, "TCS": 4250.0, "INFY": 1890.0,
            "SBIN": 825.0, "HDFCBANK": 1650.0, "TATAMOTORS": 970.0, "TATASTEEL": 155.0,
            "BHEL": 265.0, "IRCTC": 890.0, "ZOMATO": 260.0, "ETERNAL": 260.0,
            "ITC": 495.0, "LT": 3650.0, "ICICIBANK": 1280.0, "BHARTIARTL": 1640.0,
            "TATAPOWER": 420.0, "VEDL": 480.0, "TITAN": 3450.0, "MRF": 135000.0,
            "AXISBANK": 1180.0, "KOTAKBANK": 1780.0, "MARUTI": 12500.0, "SUNPHARMA": 1750.0
        }
        if clean_sym in base_prices:
            return base_prices[clean_sym]
        h = sum(ord(c) * (i + 7) for i, c in enumerate(clean_sym))
        return round(float(45.0 + (h % 3200)), 2)

    def get_previous_day_ohlc(self, symbol: str) -> Dict[str, Any]:
        """Fetches High, Low, Close, Open of the previous COMPLETED NSE trading session.
        Never uses today's in-progress OHLC.
        Correctly accounts for weekends, NSE holidays, and caches per session.
        """
        clean_sym = symbol.strip().upper()
        inst_key = self.resolve_instrument_key(clean_sym)
        prev_trading_date = get_previous_trading_day()
        prev_date_str = prev_trading_date.strftime("%Y-%m-%d")
        cache_key = f"{clean_sym}_{prev_date_str}"

        # 1. Return from in-memory cache if already retrieved for this trading session
        if cache_key in self._pd_cache:
            return self._pd_cache[cache_key]

        # 2. Demo mode deterministic generation for any stock
        if self.is_demo():
            base = self._get_base_price_for_symbol(clean_sym)
            res = {
                "pdh": round(base * 1.018, 2),
                "pdl": round(base * 0.982, 2),
                "close": round(base, 2),
                "open": round(base * 0.995, 2),
                "session_date": prev_date_str,
                "is_live": False
            }
            self._pd_cache[cache_key] = res
            return res

        # 3. Live Upstox API: Fetch daily historical candles (last 15 days)
        ist_now = get_ist_now()
        to_date = ist_now.strftime("%Y-%m-%d")
        from_date = (ist_now - timedelta(days=20)).strftime("%Y-%m-%d")
        encoded_key = requests.utils.quote(inst_key)
        url = f"{self.base_url}/historical-candle/{encoded_key}/day/{to_date}/{from_date}"

        try:
            resp = self._request_with_retry("GET", url, headers=self._get_headers())
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                if candles:
                    # Upstox returns: [timestamp, open, high, low, close, volume, oi]
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
                    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                    df = df.sort_values("date").reset_index(drop=True)

                    # Strictly filter for completed sessions on or before previous trading date
                    completed_df = df[df["date"] <= prev_trading_date]
                    if not completed_df.empty:
                        last_session = completed_df.iloc[-1]
                        res = {
                            "pdh": round(float(last_session["high"]), 2),
                            "pdl": round(float(last_session["low"]), 2),
                            "close": round(float(last_session["close"]), 2),
                            "open": round(float(last_session["open"]), 2),
                            "session_date": str(last_session["date"]),
                            "is_live": True
                        }
                        self._pd_cache[cache_key] = res
                        return res
        except Exception as e:
            logger.warning(f"Daily candle fetch failed for {clean_sym}: {e}")

        # In LIVE mode, if data cannot be fetched, NEVER synthesize fake data!
        raise MarketDataUnavailableError(
            f"Failed to fetch previous session OHLC (PDH/PDL) for {clean_sym} from Upstox API. Data unavailable."
        )

    def get_market_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetches live market quote for a stock from Upstox API v2.
        PDH and PDL are strictly obtained from the previous completed NSE trading session.
        In LIVE mode, NEVER falls back to synthetic or fake data.
        """
        clean_sym = symbol.strip().upper()
        inst_key = self.resolve_instrument_key(clean_sym)

        # 1. Sandbox / Demo Mode
        if self.is_demo():
            return self._generate_simulated_quote(clean_sym)

        # 2. Get true PDH / PDL from previous completed session
        prev_ohlc = self.get_previous_day_ohlc(clean_sym)
        pdh = prev_ohlc["pdh"]
        pdl = prev_ohlc["pdl"]
        prev_close_ref = prev_ohlc["close"]

        # 3. Standard Upstox API v2 quote call
        encoded_key = requests.utils.quote(inst_key)
        url = f"{self.base_url}/market-quote/quotes?instrument_key={encoded_key}"

        try:
            resp = self._request_with_retry("GET", url, headers=self._get_headers())
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                quote_obj = None
                for k, v in data.items():
                    if clean_sym in k or inst_key.replace("|", ":") in k or inst_key in k:
                        quote_obj = v
                        break
                if not quote_obj and data:
                    quote_obj = list(data.values())[0]

                if quote_obj:
                    ltp = float(quote_obj.get("last_price", 0.0))
                    ohlc = quote_obj.get("ohlc", {})
                    # Prefer previous completed session close for accurate % change
                    prev_close = float(ohlc.get("close", prev_close_ref)) or prev_close_ref
                    day_high = float(ohlc.get("high", ltp))
                    day_low = float(ohlc.get("low", ltp))
                    day_open = float(ohlc.get("open", ltp))
                    volume = int(quote_obj.get("volume", 0))
                    vwap = float(quote_obj.get("average_price", 0.0) or ltp)
                    ts = quote_obj.get("timestamp") or get_ist_now().strftime("%Y-%m-%d %H:%M:%S")

                    if prev_close and prev_close > 0:
                        change = ltp - prev_close
                        change_pct = (change / prev_close) * 100.0
                    else:
                        change = 0.0
                        change_pct = 0.0

                    return {
                        "symbol": clean_sym,
                        "instrument_key": inst_key,
                        "last_price": round(ltp, 2),
                        "prev_close": round(prev_close, 2),
                        "open": round(day_open, 2),
                        "high": round(day_high, 2),
                        "low": round(day_low, 2),
                        "volume": volume,
                        "vwap": round(vwap, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "pdh": round(pdh, 2),
                        "pdl": round(pdl, 2),
                        "timestamp": ts,
                        "is_live": True,
                        "is_demo": False
                    }
                else:
                    raise MarketDataUnavailableError(f"No quote data returned for symbol {clean_sym}.")
            else:
                raise MarketDataUnavailableError(f"Upstox API quote call returned HTTP {resp.status_code}.")
        except MarketDataUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Live quote fetch failed for {clean_sym}: {e}")
            raise MarketDataUnavailableError(f"Failed to fetch live quote for {clean_sym} from Upstox: {e}")

    def fetch_candles(self, symbol: str, timeframe: str = "5m") -> pd.DataFrame:
        """Fetches candle data from Upstox API v2 / v3 and returns standard OHLCV DataFrame.
        In LIVE mode, NEVER falls back to synthetic or fake data.
        For 10m timeframe, requests 5m candles and relies on market_data.resample_to_10m.
        """
        clean_sym = symbol.strip().upper()
        inst_key = self.resolve_instrument_key(clean_sym)

        # 1. Demo Mode
        if self.is_demo():
            minutes = 10 if timeframe == "10m" else (5 if timeframe == "5m" else 5)
            return self._generate_simulated_candles(clean_sym, minutes)

        # 2. Determine base fetch interval
        fetch_minutes = 5
        if timeframe == "1m":
            fetch_minutes = 1
        elif timeframe == "3m":
            fetch_minutes = 3
        elif timeframe == "5m":
            fetch_minutes = 5
        elif timeframe == "10m":
            fetch_minutes = 5  # Fetch 5m candles to resample into aligned 10m candles
        elif timeframe == "15m":
            fetch_minutes = 15
        elif timeframe == "30m":
            fetch_minutes = 30
        elif timeframe == "1h":
            fetch_minutes = 60

        df_result: Optional[pd.DataFrame] = None

        # 3. Try Upstox v3 intraday minutes endpoint
        try:
            url = f"{self.v3_base_url}/historical-candle/intraday/{requests.utils.quote(inst_key)}/minutes/{fetch_minutes}"
            resp = self._request_with_retry("GET", url, headers=self._get_headers())
            if resp.status_code == 200:
                candles = resp.json().get("data", {}).get("candles", [])
                if candles and len(candles) >= 15:
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    df_result = df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"Upstox V3 candle request skipped/failed: {e}")

        # 4. If v3 failed or had insufficient candles, try Upstox v2 historical endpoint
        if df_result is None or len(df_result) < 30:
            try:
                ist_now = get_ist_now()
                to_date = ist_now.strftime("%Y-%m-%d")
                from_date = (ist_now - timedelta(days=7)).strftime("%Y-%m-%d")
                v2_interval = "1minute" if fetch_minutes == 1 else ("30minute" if fetch_minutes == 30 else ("day" if fetch_minutes >= 60 else "1minute"))
                url = f"{self.base_url}/historical-candle/{requests.utils.quote(inst_key)}/{v2_interval}/{to_date}/{from_date}"
                resp = self._request_with_retry("GET", url, headers=self._get_headers())
                if resp.status_code == 200:
                    candles = resp.json().get("data", {}).get("candles", [])
                    if candles:
                        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"][:len(candles[0])])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp").reset_index(drop=True)
                        if fetch_minutes > 1 and v2_interval == "1minute":
                            # Resample 1m to requested fetch_minutes
                            df.set_index("timestamp", inplace=True)
                            resampled = df.resample(f"{fetch_minutes}min").agg({
                                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
                            }).dropna().reset_index()
                            df_result = resampled[["timestamp", "open", "high", "low", "close", "volume"]]
                        else:
                            df_result = df[["timestamp", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.debug(f"Upstox V2 candle request skipped/failed: {e}")

        if df_result is not None and not df_result.empty:
            return df_result

        # STRICT: In LIVE mode, NEVER generate synthetic candles if API failed!
        raise MarketDataUnavailableError(
            f"Failed to fetch live {timeframe} candle data for {clean_sym} from Upstox API. Data unavailable."
        )

    def _generate_simulated_quote(self, symbol: str) -> Dict[str, Any]:
        """Generates realistic market quote when testing offline in explicit DEMO mode."""
        import numpy as np
        seed = int(sum(ord(c) for c in symbol)) % 10000
        np.random.seed(seed)

        base = self._get_base_price_for_symbol(symbol)
        prev_close = round(base, 2)
        pdh = round(base * 1.018, 2)
        pdl = round(base * 0.982, 2)

        drift = round(float(np.random.normal(0, base * 0.008)), 2)
        ltp = round(base + drift, 2)
        change = round(ltp - prev_close, 2)
        change_pct = round((change / prev_close) * 100.0, 2)
        high = round(max(ltp, prev_close) + abs(drift * 0.4), 2)
        low = round(min(ltp, prev_close) - abs(drift * 0.4), 2)
        vwap = round((high + low + ltp) / 3.0, 2)
        vol = int(np.random.randint(1500000, 8000000))

        return {
            "symbol": symbol.upper(),
            "instrument_key": self.resolve_instrument_key(symbol),
            "last_price": ltp,
            "prev_close": prev_close,
            "open": round(prev_close + (drift * 0.2), 2),
            "high": high,
            "low": low,
            "volume": vol,
            "vwap": vwap,
            "change": change,
            "change_pct": change_pct,
            "pdh": pdh,
            "pdl": pdl,
            "timestamp": get_ist_now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_live": False,
            "is_demo": True
        }

    def _generate_simulated_candles(self, symbol: str, minutes: int = 5) -> pd.DataFrame:
        """Generates realistic closed intraday OHLCV candles for offline DEMO mode."""
        import numpy as np
        seed = int(sum(ord(c) for c in symbol)) % 10000
        np.random.seed(seed)

        base = self._get_base_price_for_symbol(symbol)
        n_bars = 75

        now = get_ist_now()
        start_time = now - timedelta(minutes=n_bars * minutes)

        candles = []
        curr = base
        for i in range(n_bars):
            wave = np.sin(i / 6.0) * (base * 0.012)
            noise = np.random.normal(0, base * 0.002)
            c_open = curr
            curr = base + wave + noise
            c_close = curr
            c_high = max(c_open, c_close) + abs(np.random.normal(0, base * 0.0015))
            c_low = min(c_open, c_close) - abs(np.random.normal(0, base * 0.0015))
            c_vol = int(np.random.randint(25000, 350000))
            ts = start_time + timedelta(minutes=i * minutes)

            candles.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(float(c_open), 2),
                "high": round(float(c_high), 2),
                "low": round(float(c_low), 2),
                "close": round(float(c_close), 2),
                "volume": c_vol
            })

        df = pd.DataFrame(candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

upstox_client = UpstoxClient()
