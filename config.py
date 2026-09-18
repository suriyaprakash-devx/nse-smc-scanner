import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "NSE Semi-Algo Trading Analyzer"
    APP_VERSION: str = "2.0.0"
    
    # Timezone & Trading Hours (IST)
    TIMEZONE: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30
    
    # Upstox API v2 / v3
    UPSTOX_BASE_URL: str = os.getenv("UPSTOX_BASE_URL", "https://api.upstox.com/v2")
    UPSTOX_V3_BASE_URL: str = os.getenv("UPSTOX_V3_BASE_URL", "https://api.upstox.com/v3")
    UPSTOX_NSE_INSTRUMENTS_URL: str = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    
    # Analysis & Engine Settings
    DEFAULT_TIMEFRAME: str = "5m"
    ALLOWED_TIMEFRAMES: tuple = ("1m", "3m", "5m", "15m", "30m", "1h")
    DEFAULT_VOLUME_MULTIPLIER: float = float(os.getenv("VOLUME_MULTIPLIER", "1.5"))
    SWING_WINDOW: int = int(os.getenv("SWING_WINDOW", "3"))
    MIN_RR_RATIO: float = float(os.getenv("MIN_RR_RATIO", "1.5"))
    EQ_TOLERANCE_PCT: float = 0.0015  # 0.15% for Equal Highs / Equal Lows
    
    # Telegram Integration (Optional)
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

settings = Settings()
