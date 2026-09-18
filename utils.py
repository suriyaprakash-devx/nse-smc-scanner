from datetime import datetime
from typing import Optional, Tuple
import pytz
from config import settings

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

def is_nse_market_open() -> Tuple[bool, str]:
    """Returns True/False and human-readable status string for NSE regular market hours (09:15 - 15:30 IST)."""
    now = get_ist_now()
    weekday = now.weekday()  # 0 = Monday, 6 = Sunday
    
    if weekday >= 5:
        return False, "🔴 Market Closed (Weekend)"
        
    open_time = now.replace(hour=settings.MARKET_OPEN_HOUR, minute=settings.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    close_time = now.replace(hour=settings.MARKET_CLOSE_HOUR, minute=settings.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    
    if open_time <= now <= close_time:
        return True, "🟢 Market Open"
    elif now < open_time:
        return False, "🔴 Market Closed (Pre-Market)"
    else:
        return False, "🔴 Market Closed (Post-Market)"

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
