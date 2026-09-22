"""NSE Trading Calendar & Holiday Management.
Handles NSE equity market holidays, weekends, and trading session determinations.
"""

from datetime import date, datetime, timedelta
from typing import Optional, Tuple
import pytz

from config import settings

# NSE Official Trading Holidays (format: YYYY-MM-DD -> Holiday Name)
NSE_HOLIDAYS = {
    # 2024
    "2024-01-22": "Special Holiday (Ram Mandir Pran Pratishtha)",
    "2024-01-26": "Republic Day",
    "2024-03-08": "Mahashivratri",
    "2024-03-25": "Holi",
    "2024-03-29": "Good Friday",
    "2024-04-11": "Id-Ul-Fitr (Ramzan Id)",
    "2024-04-17": "Shri Ram Navami",
    "2024-05-01": "Maharashtra Day",
    "2024-05-20": "General Parliamentary Elections (Mumbai)",
    "2024-06-17": "Bakri Id",
    "2024-07-17": "Moharram",
    "2024-08-15": "Independence Day",
    "2024-10-02": "Mahatma Gandhi Jayanti",
    "2024-11-01": "Diwali Laxmi Pujan (Evening Muhurat Only)",
    "2024-11-15": "Gurunanak Jayanti",
    "2024-11-20": "Maharashtra Assembly Elections",
    "2024-12-25": "Christmas",

    # 2025
    "2025-02-26": "Mahashivratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-Ul-Fitr (Ramzan Id)",
    "2025-04-10": "Shri Mahavir Jayanti",
    "2025-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Maharashtra Day",
    "2025-06-07": "Bakri Id",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Mahatma Gandhi Jayanti / Dussehra",
    "2025-10-21": "Diwali Laxmi Pujan",
    "2025-10-22": "Diwali Balipratipada",
    "2025-11-05": "Prakash Gurpurb Sri Guru Nanak Dev",
    "2025-12-25": "Christmas",

    # 2026
    "2026-01-26": "Republic Day",
    "2026-02-16": "Mahashivratri",
    "2026-03-03": "Holi",
    "2026-03-20": "Id-Ul-Fitr",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-27": "Bakri Id",
    "2026-06-25": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-09-15": "Milad-un-Nabi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-08": "Diwali Laxmi Pujan",
    "2026-11-10": "Diwali Balipratipada",
    "2026-11-24": "Gurunanak Jayanti",
    "2026-12-25": "Christmas",
}


def get_ist_timezone() -> pytz.BaseTzInfo:
    return pytz.timezone(settings.TIMEZONE)


def get_ist_now() -> datetime:
    return datetime.now(get_ist_timezone())


def is_nse_holiday(check_date: date) -> Tuple[bool, str]:
    """Returns (True, HolidayName) if check_date is an official NSE trading holiday."""
    date_str = check_date.strftime("%Y-%m-%d")
    if date_str in NSE_HOLIDAYS:
        return True, NSE_HOLIDAYS[date_str]
    return False, ""


def is_nse_trading_day(check_date: date) -> bool:
    """Returns True if check_date is a weekday and not an NSE official holiday."""
    if check_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    is_hol, _ = is_nse_holiday(check_date)
    return not is_hol


def get_previous_trading_day(ref_dt: Optional[datetime] = None) -> date:
    """Finds the date of the previous completed NSE trading session.
    
    If ref_dt is not provided, current IST time is used.
    If ref_dt is before or during today's market session, today is NOT completed,
    so we return the trading day strictly BEFORE today.
    If today is a weekend or holiday, we roll back to the last completed trading day.
    """
    if ref_dt is None:
        ref_dt = get_ist_now()
    elif ref_dt.tzinfo is None:
        ref_dt = get_ist_timezone().localize(ref_dt)
    else:
        ref_dt = ref_dt.astimezone(get_ist_timezone())

    cur_date = ref_dt.date()
    
    # Always step back at least one day because today's session is either:
    # 1) In-progress, 2) Not started, or 3) Current day.
    # The requirement strictly states:
    # "PDH = High of the previous completed NSE trading session."
    # "Never use today's high/low as PDH/PDL."
    candidate = cur_date - timedelta(days=1)
    
    while True:
        if is_nse_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)


def get_market_session_status(now: Optional[datetime] = None) -> Tuple[bool, str]:
    """Evaluates NSE market status considering hours, weekends, and holidays."""
    if now is None:
        now = get_ist_now()
    elif now.tzinfo is None:
        now = get_ist_timezone().localize(now)
    else:
        now = now.astimezone(get_ist_timezone())

    cur_date = now.date()
    
    # Check weekend
    if cur_date.weekday() >= 5:
        day_name = "Saturday" if cur_date.weekday() == 5 else "Sunday"
        return False, f"🔴 Market Closed (Weekend - {day_name})"
        
    # Check NSE holiday
    is_hol, hol_name = is_nse_holiday(cur_date)
    if is_hol:
        return False, f"🔴 Market Closed (NSE Holiday: {hol_name})"

    open_time = now.replace(
        hour=settings.MARKET_OPEN_HOUR,
        minute=settings.MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    close_time = now.replace(
        hour=settings.MARKET_CLOSE_HOUR,
        minute=settings.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )

    if open_time <= now <= close_time:
        return True, "🟢 Market Open"
    elif now < open_time:
        return False, "🔴 Market Closed (Pre-Market)"
    else:
        return False, "🔴 Market Closed (Post-Market)"
