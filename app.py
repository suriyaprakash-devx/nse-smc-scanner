"""
NSE INTRADAY SMC SCANNER
========================

Features:
- Streamlit UI
- Upstox access-token validation
- Current NSE equity instrument file
- Upstox instrument_key support
- 15-minute HTF SMC
- 5-minute entry SMC
- BOS
- CHOCH
- Liquidity Sweep
- FVG
- Order Block
- Displacement
- Volume >= 1.5x average
- SMC scoring
- BUY / SELL
- Entry / SL / TP1 / TP2
- Candlestick chart
- Optional Telegram alerts
- No automatic order placement

Install:
    pip install streamlit pandas numpy requests plotly

Run:
    streamlit run app.py
"""

import datetime as dt
import gzip
import json
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from urllib.parse import quote


# ============================================================
# CONFIG
# ============================================================

API_BASE = "https://api.upstox.com"

NSE_INSTRUMENT_URL = (
    "https://assets.upstox.com/"
    "market-quote/instruments/exchange/NSE.json.gz"
)

IST = dt.timezone(
    dt.timedelta(hours=5, minutes=30)
)


# ============================================================
# SCORE
# TOTAL = 100
# ============================================================

SCORE_WEIGHTS = {

    "htf_structure": 15,

    "liquidity_sweep": 15,

    "choch": 15,

    "bos": 15,

    "displacement": 10,

    "fvg": 10,

    "order_block": 10,

    "volume": 10

}


# ============================================================
# UPSTOX CLIENT
# ============================================================

class UpstoxClient:

    def __init__(self, token: str):

        self.token = token.strip()

        self.session = requests.Session()

        self.session.headers.update({

            "Accept": "application/json",

            "Content-Type": "application/json",

            "Authorization":
                f"Bearer {self.token}"

        })


    # ========================================================
    # TOKEN VALIDATION
    # ========================================================

    def validate_token(self) -> bool:

        try:

            url = (
                f"{API_BASE}/v2/user/profile"
            )

            response = self.session.get(
                url,
                timeout=15
            )

            if response.status_code != 200:

                st.error(
                    f"Upstox authentication failed "
                    f"({response.status_code})\n\n"
                    f"{response.text[:500]}"
                )

                return False

            data = response.json()

            if data.get("status") == "success":

                return True

            st.error(
                f"Unexpected Upstox response:\n"
                f"{data}"
            )

            return False

        except requests.RequestException as e:

            st.error(
                f"Connection error:\n{e}"
            )

            return False

        except Exception as e:

            st.error(
                f"Token validation error:\n{e}"
            )

            return False


    # ========================================================
    # PROFILE
    # ========================================================

    def get_profile(self):

        try:

            url = (
                f"{API_BASE}/v2/user/profile"
            )

            response = self.session.get(
                url,
                timeout=15
            )

            response.raise_for_status()

            return response.json()

        except Exception as e:

            st.warning(
                f"Profile error: {e}"
            )

            return None


    # ========================================================
    # NSE EQUITIES
    #
    # IMPORTANT:
    # DO NOT USE /v2/instruments
    #
    # We download the official NSE JSON instrument file.
    # ========================================================

    def get_nse_equities(
        self
    ) -> List[Dict[str, Any]]:

        try:

            st.info(
                "Downloading current NSE instrument list..."
            )

            response = requests.get(
                NSE_INSTRUMENT_URL,
                timeout=60
            )

            if response.status_code != 200:

                raise Exception(
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            # Decompress GZIP
            raw_data = gzip.decompress(
                response.content
            )

            data = json.loads(
                raw_data.decode("utf-8")
            )

            if not isinstance(data, list):

                raise Exception(
                    "Invalid NSE instrument file format."
                )

            equities = []

            for instrument in data:

                if not isinstance(
                    instrument,
                    dict
                ):

                    continue

                segment = instrument.get(
                    "segment"
                )

                instrument_type = instrument.get(
                    "instrument_type"
                )

                exchange = instrument.get(
                    "exchange"
                )

                # Current Upstox NSE EQ filtering
                if (
                    segment == "NSE_EQ"
                    and instrument_type == "EQ"
                    and exchange == "NSE"
                ):

                    instrument_key = (
                        instrument.get(
                            "instrument_key"
                        )
                    )

                    trading_symbol = (
                        instrument.get(
                            "trading_symbol"
                        )
                    )

                    if not instrument_key:
                        continue

                    if not trading_symbol:
                        continue

                    equities.append({

                        "instrument_key":
                            instrument_key,

                        "instrument_token":
                            instrument_key,

                        "symbol":
                            trading_symbol,

                        "trading_symbol":
                            trading_symbol,

                        "name":
                            instrument.get(
                                "name",
                                ""
                            ),

                        "exchange":
                            exchange,

                        "segment":
                            segment,

                        "instrument_type":
                            instrument_type,

                        "isin":
                            instrument.get(
                                "isin",
                                ""
                            )

                    })

            # Remove duplicates
            unique = {}

            for item in equities:

                unique[
                    item["instrument_key"]
                ] = item

            equities = list(
                unique.values()
            )

            return equities

        except Exception as e:

            st.error(
                "Failed to fetch NSE instruments:\n"
                f"{e}"
            )

            return []


    # ========================================================
    # HISTORICAL CANDLES - V3
    #
    # Example:
    #
    # /v3/historical-candle/
    # NSE_EQ%7CINE002A01018/
    # minutes/5/
    # 2026-09-05/
    # 2026-09-02
    #
    # ========================================================

    def get_candles(
        self,
        instrument_key: str,
        minutes: int,
        start: dt.datetime,
        end: dt.datetime
    ) -> pd.DataFrame:

        try:

            # Upstox requires the instrument key
            # in the URL.
            encoded_key = quote(
                instrument_key,
                safe=""
            )

            from_date = (
                start.strftime("%Y-%m-%d")
            )

            to_date = (
                end.strftime("%Y-%m-%d")
            )

            url = (
                f"{API_BASE}/v3/"
                f"historical-candle/"
                f"{encoded_key}/"
                f"minutes/"
                f"{minutes}/"
                f"{to_date}/"
                f"{from_date}"
            )

            response = self.session.get(
                url,
                timeout=30
            )

            if response.status_code != 200:

                return pd.DataFrame()

            raw = response.json()

            if raw.get("status") != "success":

                return pd.DataFrame()

            data = raw.get(
                "data",
                {}
            )

            candles = data.get(
                "candles",
                []
            )

            if not candles:

                return pd.DataFrame()

            rows = []

            for candle in candles:

                if len(candle) < 6:

                    continue

                rows.append({

                    "timestamp":
                        candle[0],

                    "open":
                        candle[1],

                    "high":
                        candle[2],

                    "low":
                        candle[3],

                    "close":
                        candle[4],

                    "volume":
                        candle[5]

                })

            if not rows:

                return pd.DataFrame()

            df = pd.DataFrame(
                rows
            )

            df["timestamp"] = (
                pd.to_datetime(
                    df["timestamp"],
                    utc=True
                )
                .dt
                .tz_convert(IST)
            )

            numeric_columns = [

                "open",
                "high",
                "low",
                "close",
                "volume"

            ]

            for column in numeric_columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

            df = df.dropna()

            df = df.sort_values(
                "timestamp"
            )

            df = df.drop_duplicates(
                subset=["timestamp"]
            )

            df = df.reset_index(
                drop=True
            )

            return df

        except Exception:

            return pd.DataFrame()


# ============================================================
# SWING DETECTION
# ============================================================

def find_swings(
    df: pd.DataFrame,
    length: int = 5
) -> Dict[str, List[int]]:

    if len(df) < (
        length * 2 + 1
    ):

        return {
            "high": [],
            "low": []
        }

    highs = df["high"].values

    lows = df["low"].values

    swing_highs = []

    swing_lows = []

    for i in range(
        length,
        len(df) - length
    ):

        left_highs = highs[
            i - length:i
        ]

        right_highs = highs[
            i + 1:i + length + 1
        ]

        left_lows = lows[
            i - length:i
        ]

        right_lows = lows[
            i + 1:i + length + 1
        ]

        if (
            highs[i] >
            left_highs.max()
            and
            highs[i] >
            right_highs.max()
        ):

            swing_highs.append(i)

        if (
            lows[i] <
            left_lows.min()
            and
            lows[i] <
            right_lows.min()
        ):

            swing_lows.append(i)

    return {

        "high":
            swing_highs,

        "low":
            swing_lows

    }


# ============================================================
# BOS
# ============================================================

def detect_bos(
    df: pd.DataFrame,
    swings: Dict[str, List[int]]
) -> List[Dict[str, Any]]:

    events = []

    last_high = None

    last_low = None

    broken_high = None

    broken_low = None

    for idx, row in df.iterrows():

        if idx in swings["high"]:

            last_high = float(
                row["high"]
            )

            broken_high = False

        if idx in swings["low"]:

            last_low = float(
                row["low"]
            )

            broken_low = False

        # Bullish BOS
        if (
            last_high is not None
            and
            row["close"] > last_high
            and
            broken_high is not True
        ):

            events.append({

                "type": "bull",

                "price":
                    float(row["close"]),

                "idx":
                    idx

            })

            broken_high = True

        # Bearish BOS
        if (
            last_low is not None
            and
            row["close"] < last_low
            and
            broken_low is not True
        ):

            events.append({

                "type": "bear",

                "price":
                    float(row["close"]),

                "idx":
                    idx

            })

            broken_low = True

    return events


# ============================================================
# CHOCH
# ============================================================

def detect_choch(
    df: pd.DataFrame,
    swings: Dict[str, List[int]],
    previous_structure: Optional[str]
) -> List[Dict[str, Any]]:

    events = []

    if previous_structure is None:

        return events

    if previous_structure == "bear":

        if swings["high"]:

            latest_high_idx = (
                swings["high"][-1]
            )

            latest_high = float(
                df.loc[
                    latest_high_idx,
                    "high"
                ]
            )

            if (
                df["close"].iloc[-1]
                >
                latest_high
            ):

                events.append({

                    "type": "bull",

                    "idx":
                        len(df) - 1

                })

    elif previous_structure == "bull":

        if swings["low"]:

            latest_low_idx = (
                swings["low"][-1]
            )

            latest_low = float(
                df.loc[
                    latest_low_idx,
                    "low"
                ]
            )

            if (
                df["close"].iloc[-1]
                <
                latest_low
            ):

                events.append({

                    "type": "bear",

                    "idx":
                        len(df) - 1

                })

    return events


# ============================================================
# LIQUIDITY SWEEP
# ============================================================

def detect_liquidity_sweep(
    df: pd.DataFrame,
    swings: Dict[str, List[int]]
) -> List[Dict[str, Any]]:

    sweeps = []

    # Sell-side liquidity sweep
    # Low taken + close back above

    for low_idx in swings["low"]:

        low_price = float(
            df.loc[
                low_idx,
                "low"
            ]
        )

        later = df.loc[
            low_idx + 1:
        ]

        if later.empty:

            continue

        swept = later[
            (
                later["low"]
                <
                low_price
            )
            &
            (
                later["close"]
                >
                low_price
            )
        ]

        if not swept.empty:

            idx = swept.index[-1]

            sweeps.append({

                "direction":
                    "sell",

                "level":
                    low_price,

                "idx":
                    idx

            })

    # Buy-side liquidity sweep
    # High taken + close back below

    for high_idx in swings["high"]:

        high_price = float(
            df.loc[
                high_idx,
                "high"
            ]
        )

        later = df.loc[
            high_idx + 1:
        ]

        if later.empty:

            continue

        swept = later[
            (
                later["high"]
                >
                high_price
            )
            &
            (
                later["close"]
                <
                high_price
            )
        ]

        if not swept.empty:

            idx = swept.index[-1]

            sweeps.append({

                "direction":
                    "buy",

                "level":
                    high_price,

                "idx":
                    idx

            })

    return sweeps


# ============================================================
# FVG
# ============================================================

def detect_fvg(
    df: pd.DataFrame
) -> List[Dict[str, Any]]:

    fvgs = []

    for i in range(
        2,
        len(df)
    ):

        current = df.iloc[i]

        two_ago = df.iloc[i - 2]

        # Bullish FVG
        if (
            current["low"]
            >
            two_ago["high"]
        ):

            fvgs.append({

                "type":
                    "bull",

                "top":
                    float(current["low"]),

                "bottom":
                    float(two_ago["high"]),

                "idx":
                    i

            })

        # Bearish FVG
        if (
            current["high"]
            <
            two_ago["low"]
        ):

            fvgs.append({

                "type":
                    "bear",

                "top":
                    float(two_ago["low"]),

                "bottom":
                    float(current["high"]),

                "idx":
                    i

            })

    return fvgs


# ============================================================
# ORDER BLOCK
# ============================================================

def detect_order_blocks(
    df: pd.DataFrame
) -> List[Dict[str, Any]]:

    blocks = []

    for i in range(
        1,
        len(df)
    ):

        previous = df.iloc[
            i - 1
        ]

        current = df.iloc[
            i
        ]

        # Bullish OB
        if (
            previous["close"]
            <
            previous["open"]
            and
            current["close"]
            >
            current["open"]
        ):

            blocks.append({

                "type":
                    "bull",

                "high":
                    float(previous["high"]),

                "low":
                    float(previous["low"]),

                "idx":
                    i - 1

            })

        # Bearish OB
        if (
            previous["close"]
            >
            previous["open"]
            and
            current["close"]
            <
            current["open"]
        ):

            blocks.append({

                "type":
                    "bear",

                "high":
                    float(previous["high"]),

                "low":
                    float(previous["low"]),

                "idx":
                    i - 1

            })

    return blocks


# ============================================================
# DISPLACEMENT
# ============================================================

def detect_displacement(
    df: pd.DataFrame,
    multiplier: float = 1.5
) -> List[Dict[str, Any]]:

    displacement = []

    lookback = 5

    if len(df) <= lookback:

        return displacement

    bodies = abs(
        df["close"] -
        df["open"]
    )

    for i in range(
        lookback,
        len(df)
    ):

        body = float(
            bodies.iloc[i]
        )

        previous_bodies = bodies.iloc[
            i - lookback:i
        ]

        average_body = float(
            previous_bodies.mean()
        )

        if average_body <= 0:

            continue

        if (
            body
            >
            average_body * multiplier
        ):

            direction = (
                "bull"
                if
                df.iloc[i]["close"]
                >
                df.iloc[i]["open"]
                else
                "bear"
            )

            displacement.append({

                "type":
                    direction,

                "idx":
                    i,

                "price":
                    float(
                        df.iloc[i]["close"]
                    )

            })

    return displacement


# ============================================================
# VOLUME
# ============================================================

def calculate_volume_ratio(
    df: pd.DataFrame,
    lookback: int = 20
) -> float:

    if len(df) < (
        lookback + 1
    ):

        return 0.0

    previous_volume = df[
        "volume"
    ].iloc[
        -lookback - 1:-1
    ]

    average_volume = float(
        previous_volume.mean()
    )

    current_volume = float(
        df["volume"].iloc[-1]
    )

    if average_volume <= 0:

        return 0.0

    return (
        current_volume
        /
        average_volume
    )


# ============================================================
# SCORE
# ============================================================

def calculate_smc_score(
    components: Dict[str, bool]
) -> int:

    score = 0

    for key, value in components.items():

        if value:

            score += SCORE_WEIGHTS.get(
                key,
                0
            )

    return score


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    htf_df: pd.DataFrame,
    entry_df: pd.DataFrame,
    settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:

    if (
        htf_df.empty
        or
        entry_df.empty
    ):

        return None

    swing_length = int(
        settings["swing_length"]
    )

    # ========================================================
    # 15M HTF
    # ========================================================

    htf_swings = find_swings(
        htf_df,
        swing_length
    )

    htf_bos = detect_bos(
        htf_df,
        htf_swings
    )

    if not htf_bos:

        return None

    bias = htf_bos[-1]["type"]

    # ========================================================
    # 5M ENTRY
    # ========================================================

    entry_swings = find_swings(
        entry_df,
        swing_length
    )

    entry_bos = detect_bos(
        entry_df,
        entry_swings
    )

    entry_choch = detect_choch(
        entry_df,
        entry_swings,
        bias
    )

    sweeps = detect_liquidity_sweep(
        entry_df,
        entry_swings
    )

    fvgs = detect_fvg(
        entry_df
    )

    order_blocks = detect_order_blocks(
        entry_df
    )

    displacement = detect_displacement(
        entry_df
    )

    volume_ratio = calculate_volume_ratio(
        entry_df,
        int(settings["volume_lookback"])
    )

    volume_ok = (
        volume_ratio
        >=
        float(
            settings["volume_multiplier"]
        )
    )

    # ========================================================
    # CONFLUENCE
    # ========================================================

    required_sweep = (
        "sell"
        if bias == "bull"
        else
        "buy"
    )

    components = {

        "htf_structure":
            True,

        "liquidity_sweep":
            any(
                s["direction"]
                ==
                required_sweep
                for s in sweeps
            ),

        "choch":
            any(
                c["type"]
                ==
                bias
                for c in entry_choch
            ),

        "bos":
            any(
                b["type"]
                ==
                bias
                for b in entry_bos
            ),

        "displacement":
            any(
                d["type"]
                ==
                bias
                for d in displacement
            ),

        "fvg":
            any(
                f["type"]
                ==
                bias
                for f in fvgs
            ),

        "order_block":
            any(
                o["type"]
                ==
                bias
                for o in order_blocks
            ),

        "volume":
            volume_ok

    }

    score = calculate_smc_score(
        components
    )

    # ========================================================
    # REQUIRE VOLUME
    # ========================================================

    if not volume_ok:

        return None

    # ========================================================
    # MIN SCORE
    # ========================================================

    if (
        score
        <
        int(settings["min_score"])
    ):

        return None

    signal = (
        "BUY"
        if bias == "bull"
        else
        "SELL"
    )

    # ========================================================
    # ENTRY
    # ========================================================

    entry_price = float(
        entry_df.iloc[-1]["close"]
    )

    # ========================================================
    # STOP LOSS
    # ========================================================

    lookback = int(
        settings["sl_lookback"]
    )

    if signal == "BUY":

        recent_low = float(
            entry_df["low"]
            .iloc[-lookback:]
            .min()
        )

        sl = recent_low

        if sl >= entry_price:

            sl = (
                entry_price
                *
                0.995
            )

    else:

        recent_high = float(
            entry_df["high"]
            .iloc[-lookback:]
            .max()
        )

        sl = recent_high

        if sl <= entry_price:

            sl = (
                entry_price
                *
                1.005
            )

    # ========================================================
    # RISK
    # ========================================================

    risk = abs(
        entry_price - sl
    )

    if risk <= 0:

        return None

    rr_factor = float(
        settings["risk_reward"]
    )

    # ========================================================
    # TARGETS
    # ========================================================

    if signal == "BUY":

        tp1 = (
            entry_price
            +
            risk * 2
        )

        tp2 = (
            entry_price
            +
            risk * rr_factor
        )

    else:

        tp1 = (
            entry_price
            -
            risk * 2
        )

        tp2 = (
            entry_price
            -
            risk * rr_factor
        )

    return {

        "signal":
            signal,

        "score":
            int(score),

        "price":
            round(
                entry_price,
                2
            ),

        "entry":
            round(
                entry_price,
                2
            ),

        "sl":
            round(
                sl,
                2
            ),

        "tp1":
            round(
                tp1,
                2
            ),

        "tp2":
            round(
                tp2,
                2
            ),

        "rr":
            f"1:{rr_factor:g}",

        "volume_ratio":
            round(
                volume_ratio,
                2
            ),

        "htf_bias":
            "Bullish"
            if bias == "bull"
            else
            "Bearish",

        "components":
            components

    }


# ============================================================
# SCAN ONE STOCK
# ============================================================

def scan_stock(
    client: UpstoxClient,
    instrument: Dict[str, Any],
    settings: Dict[str, Any]
) -> Optional[Dict[str, Any]]:

    instrument_key = instrument.get(
        "instrument_key"
    )

    if not instrument_key:

        return None

    now = dt.datetime.now(
        IST
    )

    # ========================================================
    # 15 MIN DATA
    # ========================================================

    start_15 = (
        now
        -
        dt.timedelta(
            days=5
        )
    )

    # ========================================================
    # 5 MIN DATA
    # ========================================================

    start_5 = (
        now
        -
        dt.timedelta(
            days=3
        )
    )

    htf_df = client.get_candles(

        instrument_key,

        15,

        start_15,

        now

    )

    entry_df = client.get_candles(

        instrument_key,

        5,

        start_5,

        now

    )

    if (
        htf_df.empty
        or
        entry_df.empty
    ):

        return None

    return generate_signal(
        htf_df,
        entry_df,
        settings
    )


# ============================================================
# MARKET SCANNER
# ============================================================

def scan_market(
    client: UpstoxClient,
    settings: Dict[str, Any],
    max_stocks: Optional[int] = None
):

    stocks = client.get_nse_equities()

    if not stocks:

        return [], [], 0

    # Optional limit
    if max_stocks:

        stocks = stocks[
            :max_stocks
        ]

    results = []

    failed = []

    lock = threading.Lock()

    def worker(
        instrument
    ):

        symbol = instrument.get(
            "symbol",
            "UNKNOWN"
        )

        try:

            result = scan_stock(
                client,
                instrument,
                settings
            )

            if result:

                with lock:

                    results.append({

                        "symbol":
                            symbol,

                        **result

                    })

        except Exception:

            with lock:

                failed.append(
                    symbol
                )

    # ========================================================
    # CONCURRENCY
    # ========================================================

    max_threads = int(
        settings["max_threads"]
    )

    threads = []

    for instrument in stocks:

        thread = threading.Thread(
            target=worker,
            args=(instrument,)
        )

        thread.start()

        threads.append(
            thread
        )

        while sum(
            t.is_alive()
            for t in threads
        ) >= max_threads:

            time.sleep(
                0.1
            )

    for thread in threads:

        thread.join()

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda x:
        x["score"],
        reverse=True
    )

    return (
        results,
        failed,
        len(stocks)
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_alert(
    token: str,
    chat_id: str,
    message: str
):

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    payload = {

        "chat_id":
            chat_id,

        "text":
            message

    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=15
        )

        if response.status_code != 200:

            st.warning(
                "Telegram error: "
                +
                response.text[:500]
            )

    except Exception as e:

        st.warning(
            f"Telegram failed: {e}"
        )


# ============================================================
# CHART
# ============================================================

def render_chart(
    df: pd.DataFrame,
    symbol: str
):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(

            x=df["timestamp"],

            open=df["open"],

            high=df["high"],

            low=df["low"],

            close=df["close"],

            name="5M Candles"

        )
    )

    fig.update_layout(

        title=
            f"{symbol} - 5 Minute Chart",

        xaxis_title=
            "Time",

        yaxis_title=
            "Price",

        height=600,

        xaxis_rangeslider_visible=False

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# COMPONENT FORMAT
# ============================================================

def format_components(
    components
):

    lines = []

    for key, value in components.items():

        name = key.replace(
            "_",
            " "
        ).title()

        if value:

            lines.append(
                f"✓ {name}"
            )

        else:

            lines.append(
                f"✗ {name}"
            )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN APP
# ============================================================

def main():

    st.set_page_config(

        page_title=
            "NSE SMC Scanner",

        page_icon=
            "📈",

        layout=
            "wide"

    )

    st.title(
        "📈 NSE Intraday SMC Scanner"
    )

    st.caption(
        "15M HTF + 5M Entry | "
        "SMC + Volume ≥ 1.5x"
    )

    # ========================================================
    # SESSION STATE
    # ========================================================

    if "connected" not in st.session_state:

        st.session_state[
            "connected"
        ] = False

    if "upstox_token" not in st.session_state:

        st.session_state[
            "upstox_token"
        ] = ""

    if "scan_results" not in st.session_state:

        st.session_state[
            "scan_results"
        ] = []

    if "failed_symbols" not in st.session_state:

        st.session_state[
            "failed_symbols"
        ] = []

    if "total_scanned" not in st.session_state:

        st.session_state[
            "total_scanned"
        ] = 0

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header(
            "⚙️ Strategy Settings"
        )

        swing_length = st.number_input(

            "Swing Length",

            min_value=2,

            max_value=10,

            value=5,

            step=1

        )

        volume_lookback = st.number_input(

            "Volume Lookback",

            min_value=5,

            max_value=50,

            value=20,

            step=1

        )

        volume_multiplier = st.slider(

            "Minimum Volume",

            min_value=1.0,

            max_value=3.0,

            value=1.5,

            step=0.1

        )

        min_score = st.slider(

            "Minimum SMC Score",

            min_value=0,

            max_value=100,

            value=65,

            step=5

        )

        rr = st.selectbox(

            "Risk Reward",

            ["2", "3", "4"],

            index=0

        )

        sl_lookback = st.number_input(

            "SL Lookback",

            min_value=5,

            max_value=50,

            value=20,

            step=1

        )

        max_threads = st.slider(

            "Concurrent Requests",

            min_value=1,

            max_value=10,

            value=5

        )

        st.divider()

        st.header(
            "🔔 Telegram"
        )

        telegram_token = st.text_input(

            "Telegram Bot Token",

            type="password"

        )

        telegram_chat = st.text_input(

            "Telegram Chat ID"

        )

        telegram_enable = st.checkbox(

            "Enable Telegram Alerts"

        )

        st.divider()

        st.caption(
            "⚠️ Analysis only. "
            "No automatic order placement."
        )

    # ========================================================
    # UPSTOX CONNECTION
    # ========================================================

    st.subheader(
        "🔐 Upstox Connection"
    )

    token_input = st.text_input(

        "Enter Today's Upstox Access Token",

        type="password",

        placeholder=
            "Paste your Upstox access token"

    )

    col1, col2 = st.columns(2)

    with col1:

        connect = st.button(

            "🔌 CONNECT",

            type="primary",

            use_container_width=True

        )

    with col2:

        clear = st.button(

            "🗑️ CLEAR TOKEN",

            use_container_width=True

        )

    # ========================================================
    # CLEAR
    # ========================================================

    if clear:

        st.session_state[
            "upstox_token"
        ] = ""

        st.session_state[
            "connected"
        ] = False

        st.session_state[
            "scan_results"
        ] = []

        st.rerun()

    # ========================================================
    # CONNECT
    # ========================================================

    if connect:

        if not token_input.strip():

            st.error(
                "Please enter your Upstox access token."
            )

        else:

            with st.spinner(
                "Checking Upstox token..."
            ):

                client = UpstoxClient(
                    token_input
                )

                valid = (
                    client.validate_token()
                )

            if valid:

                st.session_state[
                    "upstox_token"
                ] = token_input.strip()

                st.session_state[
                    "connected"
                ] = True

                profile = (
                    client.get_profile()
                )

                st.success(
                    "🟢 Upstox Connected Successfully"
                )

                if profile:

                    user_data = profile.get(
                        "data",
                        {}
                    )

                    user_name = (
                        user_data.get(
                            "user_name",
                            "N/A"
                        )
                    )

                    broker = (
                        user_data.get(
                            "broker",
                            "N/A"
                        )
                    )

                    st.info(
                        f"User: {user_name}\n\n"
                        f"Broker: {broker}"
                    )

            else:

                st.session_state[
                    "connected"
                ] = False

                st.error(
                    "🔴 Invalid / Expired "
                    "Upstox Access Token"
                )

    # ========================================================
    # CONNECTION CHECK
    # ========================================================

    if not st.session_state.get(
        "connected",
        False
    ):

        st.info(
            "Enter your Upstox access token "
            "and click CONNECT."
        )

        st.stop()

    token = st.session_state[
        "upstox_token"
    ]

    client = UpstoxClient(
        token
    )

    st.success(
        "🟢 Upstox connection active"
    )

    # ========================================================
    # MARKET SCANNER
    # ========================================================

    st.subheader(
        "🔎 NSE Market Scanner"
    )

    st.write(
        "Scans NSE equities using "
        "15-minute HTF and 5-minute entry SMC."
    )

    scan_button = st.button(

        "🚀 SCAN NSE MARKET NOW",

        type="primary",

        use_container_width=True

    )

    if scan_button:

        settings = {

            "swing_length":
                int(swing_length),

            "volume_lookback":
                int(volume_lookback),

            "volume_multiplier":
                float(volume_multiplier),

            "min_score":
                int(min_score),

            "risk_reward":
                rr,

            "sl_lookback":
                int(sl_lookback),

            "max_threads":
                int(max_threads)

        }

        with st.spinner(
            "Downloading NSE instruments "
            "and scanning market..."
        ):

            results, failed, total = (
                scan_market(
                    client,
                    settings
                )
            )

        st.session_state[
            "scan_results"
        ] = results

        st.session_state[
            "failed_symbols"
        ] = failed

        st.session_state[
            "total_scanned"
        ] = total

        st.success(

            f"Scan complete. "
            f"{len(results)} strong setup(s) "
            f"found from {total} NSE equities."

        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = st.session_state[
        "scan_results"
    ]

    if not results:

        st.info(
            "Click SCAN NSE MARKET NOW "
            "to start scanning."
        )

        st.stop()

    # ========================================================
    # SIGNAL TABLE
    # ========================================================

    st.subheader(
        "📊 Strong Signals"
    )

    result_df = pd.DataFrame(
        results
    )

    columns = [

        "symbol",

        "signal",

        "score",

        "htf_bias",

        "entry",

        "sl",

        "tp1",

        "tp2",

        "rr",

        "volume_ratio"

    ]

    st.dataframe(

        result_df[
            columns
        ],

        use_container_width=True,

        hide_index=True

    )

    # ========================================================
    # STRONGEST SETUP
    # ========================================================

    top = results[0]

    st.subheader(

        f"🔥 Strongest "
        f"{top['signal']} - "
        f"{top['symbol']}"

    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "SMC Score",
            f"{top['score']}/100"
        )

    with c2:

        st.metric(
            "Entry",
            f"₹{top['entry']}"
        )

    with c3:

        st.metric(
            "Stop Loss",
            f"₹{top['sl']}"
        )

    with c4:

        st.metric(
            "Volume",
            f"{top['volume_ratio']}x"
        )

    c5, c6, c7 = st.columns(3)

    with c5:

        st.metric(
            "TP1",
            f"₹{top['tp1']}"
        )

    with c6:

        st.metric(
            "TP2",
            f"₹{top['tp2']}"
        )

    with c7:

        st.metric(
            "Risk Reward",
            top["rr"]
        )

    # ========================================================
    # SMC COMPONENTS
    # ========================================================

    st.subheader(
        "🧠 SMC Confluence"
    )

    st.text(
        format_components(
            top["components"]
        )
    )

    # ========================================================
    # CHART
    # ========================================================

    st.subheader(
        "📈 Stock Chart"
    )

    symbols = [

        result["symbol"]

        for result in results

    ]

    selected_symbol = st.selectbox(

        "Select Stock",

        symbols

    )

    # Find selected instrument
    instruments = (
        client.get_nse_equities()
    )

    selected_instrument = next(

        (
            instrument

            for instrument
            in instruments

            if instrument.get(
                "symbol"
            )
            ==
            selected_symbol

        ),

        None

    )

    if selected_instrument:

        now = dt.datetime.now(
            IST
        )

        chart_start = (
            now
            -
            dt.timedelta(
                days=2
            )
        )

        chart_df = client.get_candles(

            selected_instrument[
                "instrument_key"
            ],

            5,

            chart_start,

            now

        )

        if not chart_df.empty:

            render_chart(

                chart_df,

                selected_symbol

            )

        else:

            st.warning(
                "Unable to load chart candles."
            )

    # ========================================================
    # TELEGRAM
    # ========================================================

    if (
        telegram_enable
        and
        telegram_token
        and
        telegram_chat
    ):

        st.subheader(
            "📨 Telegram Alerts"
        )

        sent_count = 0

        for signal in results:

            message = (

                "*SMC ALERT*\n\n"

                f"Stock: "
                f"{signal['symbol']}\n"

                f"Signal: "
                f"STRONG "
                f"{signal['signal']}\n"

                f"Score: "
                f"{signal['score']}/100\n"

                f"HTF Bias: "
                f"{signal['htf_bias']}\n"

                f"Entry: ₹"
                f"{signal['entry']}\n"

                f"SL: ₹"
                f"{signal['sl']}\n"

                f"TP1: ₹"
                f"{signal['tp1']}\n"

                f"TP2: ₹"
                f"{signal['tp2']}\n"

                f"RR: "
                f"{signal['rr']}\n"

                f"Volume: "
                f"{signal['volume_ratio']}x"

            )

            send_telegram_alert(

                telegram_token,

                telegram_chat,

                message

            )

            sent_count += 1

        st.success(
            f"{sent_count} Telegram alert(s) sent."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    failed_symbols = (
        st.session_state.get(
            "failed_symbols",
            []
        )
    )

    total_scanned = (
        st.session_state.get(
            "total_scanned",
            0
        )
    )

    st.divider()

    st.caption(

        f"Stocks scanned: "
        f"{total_scanned} | "
        f"Failed requests: "
        f"{len(failed_symbols)}"

    )

    if failed_symbols:

        with st.expander(
            "Show failed symbols"
        ):

            st.write(
                failed_symbols
            )

    st.caption(
        "⚠️ Analysis only. "
        "This application does NOT place "
        "orders automatically."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()