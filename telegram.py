import os
import json
import logging
import hashlib
import time
from datetime import datetime
from typing import Optional, Set, Dict, Any
import requests
from config import settings
from utils import format_ist_time, get_ist_now

logger = logging.getLogger("telegram_service")

CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), ".sent_alerts.json")

class TelegramNotifier:
    """Dispatches real-time signal alerts to Telegram with persistent duplicate suppression.
    Never sends NO TRADE alerts. Only alerts for BUY or SELL signals.
    Strictly protects sensitive bot tokens and credentials from logs and error traces.
    """

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = (bot_token or settings.TELEGRAM_BOT_TOKEN).strip()
        self.chat_id = (chat_id or settings.TELEGRAM_CHAT_ID).strip()
        self._sent_cache: Dict[str, str] = self._load_persistent_cache()

    def _load_persistent_cache(self) -> Dict[str, str]:
        """Loads sent alert hashes from .sent_alerts.json."""
        if os.path.exists(CACHE_FILE_PATH):
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.debug(f"Could not load .sent_alerts.json: {e}")
        return {}

    def _save_persistent_cache(self):
        """Saves sent alert hashes to .sent_alerts.json."""
        try:
            with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._sent_cache, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not write .sent_alerts.json: {e}")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _generate_hash(
        self, symbol: str, direction: str, entry: float, sl: float, tp1: float, tp2: float, date_str: str
    ) -> str:
        # Prevents duplicate alert across application restarts
        raw = f"{symbol.upper()}_{direction.upper()}_{entry:.2f}_{sl:.2f}_{tp1:.2f}_{tp2:.2f}_{date_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def send_signal_alert(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        entry: float,
        entry_zone: str,
        stop_loss: float,
        tp1: float,
        tp2: float,
        rr_ratio: float,
        volume_ratio: float,
        timeframe: str = "5m",
        tp3: Optional[float] = None,
        confluence_summary: Optional[str] = None,
        time_str: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Sends formatted signal alert to configured Telegram chat with persistent duplicate protection.
        Strictly BUY or SELL only.
        """
        if not self.is_configured():
            return False

        direction_clean = direction.upper().strip()
        if direction_clean not in ["BUY", "SELL"]:
            return False

        today_str = get_ist_now().strftime("%Y-%m-%d")
        sig_hash = self._generate_hash(symbol, direction_clean, entry, stop_loss, tp1, tp2, today_str)

        if sig_hash in self._sent_cache and not force:
            logger.info(f"Duplicate Telegram alert persistently suppressed for {symbol} ({direction_clean})")
            return False

        time_val = time_str or format_ist_time(include_date=True)
        dir_emoji = "🟢 BUY" if direction_clean == "BUY" else "🔴 SELL"

        tp3_line = f"🎯 TP3 (Extended): ₹{tp3:,.2f}\n" if (tp3 and tp3 > 0) else ""
        conf_line = f"🔬 Confluence: {confluence_summary}\n" if confluence_summary else ""

        message = (
            f"⚡ NSE SEMI-ALGO SIGNAL ALERT\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 Stock: {symbol.upper()} • {timeframe.upper()}\n"
            f"SIGNAL: {dir_emoji}\n"
            f"Current Price: ₹{current_price:,.2f}\n"
            f"Entry: ₹{entry:,.2f} (Zone: {entry_zone})\n"
            f"Stop Loss: ₹{stop_loss:,.2f}\n"
            f"🎯 TP1 (Structural): ₹{tp1:,.2f}\n"
            f"🎯 TP2 (Major): ₹{tp2:,.2f}\n"
            f"{tp3_line}"
            f"⚖️ Risk/Reward: 1 : {rr_ratio:.1f}\n"
            f"📊 Volume Ratio: {volume_ratio:.2f}x\n"
            f"{conf_line}"
            f"⏰ Time: {time_val}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Semi-algo technical analysis — not financial advice."
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }

        try:
            resp = requests.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                self._sent_cache[sig_hash] = time_val
                self._save_persistent_cache()
                logger.info(f"Telegram alert sent for {symbol} ({direction_clean})")
                return True
            else:
                # Sanitize error to never reveal bot token
                safe_err = resp.text.replace(self.bot_token, "REDACTED_TOKEN")
                logger.warning(f"Telegram API responded HTTP {resp.status_code}: {safe_err}")
                return False
        except Exception as e:
            safe_e = str(e).replace(self.bot_token, "REDACTED_TOKEN")
            logger.error(f"Failed to deliver Telegram alert: {safe_e}")
            return False

telegram_notifier = TelegramNotifier()
