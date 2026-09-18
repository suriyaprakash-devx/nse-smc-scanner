import logging
import hashlib
import time
from typing import Optional, Set
import requests
from config import settings
from utils import format_ist_time

logger = logging.getLogger("telegram_service")

class TelegramNotifier:
    """Dispatches real-time signal alerts to Telegram with duplicate suppression."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = (bot_token or settings.TELEGRAM_BOT_TOKEN).strip()
        self.chat_id = (chat_id or settings.TELEGRAM_CHAT_ID).strip()
        self._sent_hashes: Set[str] = set()

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _generate_hash(self, symbol: str, direction: str, entry: float, sl: float, target: float) -> str:
        # Prevents duplicate alert within current session if signal parameters are identical
        raw = f"{symbol.upper()}_{direction}_{entry:.1f}_{sl:.1f}_{target:.1f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def send_signal_alert(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        entry: float,
        target: float,
        stop_loss: float,
        rr_ratio: float,
        time_str: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """Sends formatted signal alert to configured Telegram chat."""
        if not self.is_configured():
            return False

        sig_hash = self._generate_hash(symbol, direction, entry, stop_loss, target)
        if sig_hash in self._sent_hashes and not force:
            logger.info(f"Duplicate Telegram alert suppressed for {symbol} ({direction})")
            return False

        time_val = time_str or format_ist_time()
        dir_emoji = "🟢 BUY" if direction.upper() == "BUY" else "🔴 SELL"

        message = (
            f"⚡ NSE SEMI-ALGO\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 {symbol.upper()}\n"
            f"SIGNAL: {dir_emoji}\n"
            f"Current Price: ₹{current_price:,.2f}\n"
            f"Entry: ₹{entry:,.2f}\n"
            f"Maximum Target: ₹{target:,.2f}\n"
            f"Stop Loss: ₹{stop_loss:,.2f}\n"
            f"R:R: 1:{rr_ratio:.1f}\n"
            f"Time: {time_val}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ Technical analysis estimate — not guaranteed."
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message
        }

        try:
            resp = requests.post(url, json=payload, timeout=6)
            if resp.status_code == 200:
                self._sent_hashes.add(sig_hash)
                logger.info(f"Telegram alert sent for {symbol} ({direction})")
                return True
            else:
                logger.warning(f"Telegram API responded HTTP {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to deliver Telegram alert: {e}")
            return False

telegram_notifier = TelegramNotifier()
