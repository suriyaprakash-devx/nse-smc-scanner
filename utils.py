from datetime import datetime
from typing import Optional, Tuple
import pytz
from config import settings

class MarketDataUnavailableError(Exception):
    """Raised when market data is missing, corrupted, stale, or synthetic in live mode."""
    pass

def get_ist_timezone() -> pytz.BaseTzInfo:
    return pytz.timezone(settings.TIMEZONE)

def get_ist_now() -> datetime:
    return datetime.now(get_ist_timezone())

def format_ist_time(dt: Optional[datetime] = None, include_date: bool = False) -> str:
    if dt is None:
        dt = get_ist_now()
    elif dt.tzinfo is None:
        dt = get_ist_timezone().localize(dt)
    else:
        dt = dt.astimezone(get_ist_timezone())
        
    if include_date:
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")
    return dt.strftime("%H:%M:%S IST")

from calendar_utils import (
    get_market_session_status,
    is_nse_holiday,
    is_nse_trading_day,
    get_previous_trading_day
)

def is_nse_market_open() -> Tuple[bool, str]:
    """Returns True/False and human-readable status string for NSE market hours, weekends, and holidays."""
    return get_market_session_status()

def mask_token(token: Optional[str]) -> str:
    """Masks Upstox access token for display. Never reveals the full secret."""
    if not token:
        return "Not Set"
    token = str(token).strip()
    if len(token) <= 8:
        return "••••••••"
    return f"{token[:4]}••••••••{token[-4:]}"

def format_currency(val: float) -> str:
    """Formats numeric value in Indian Rupee format."""
    try:
        return f"₹{float(val):,.2f}"
    except (ValueError, TypeError):
        return "₹0.00"

def format_pct(val: float) -> str:
    """Formats percentage change with sign."""
    try:
        f = float(val)
        prefix = "+" if f > 0 else ""
        return f"{prefix}{f:.2f}%"
    except (ValueError, TypeError):
        return "0.00%"
