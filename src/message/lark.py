# -*- coding: utf-8 -*-
"""
Moon Dev's Feishu (Lark) Bot Implementation
Built with love by Moon Dev

Feishu Robot Message Push Module
Supports sending text messages and urgent card messages

Configuration (.env):
    FEISHU_APP_ID: Feishu App ID
    FEISHU_APP_SECRET: Feishu App Secret
    FEISHU_USER_OPEN_ID: User Open ID to receive messages
"""

import os
import time
import json
import requests
from typing import Optional, Dict, Any
from termcolor import cprint
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)


class FeishuBot:
    """Feishu Bot class for sending message notifications"""

    # API endpoints
    TOKEN_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    MESSAGE_URL = "https://open.feishu.cn/open-apis/im/v1/messages"

    def __init__(
        self,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        user_open_id: Optional[str] = None
    ):
        """
        Initialize Feishu Bot

        Args:
            app_id: Feishu App ID, defaults to env variable
            app_secret: Feishu App Secret, defaults to env variable
            user_open_id: User Open ID to receive messages, defaults to env variable
        """
        self.app_id = app_id or os.getenv("FEISHU_APP_ID")
        self.app_secret = app_secret or os.getenv("FEISHU_APP_SECRET")
        self.user_open_id = user_open_id or os.getenv("FEISHU_USER_OPEN_ID")

        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        if not all([self.app_id, self.app_secret]):
            cprint("Warning: Feishu credentials not configured. Set FEISHU_APP_ID and FEISHU_APP_SECRET in .env", "yellow")

    def _get_access_token(self) -> Optional[str]:
        """
        Get tenant_access_token with auto caching and refresh

        Returns:
            str: access_token, None on failure
        """
        # Check if cached token is still valid (refresh 5 minutes early)
        if self._access_token and time.time() < self._token_expires_at - 300:
            return self._access_token

        try:
            response = requests.post(
                self.TOKEN_URL,
                json={
                    "app_id": self.app_id,
                    "app_secret": self.app_secret
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                self._access_token = data.get("tenant_access_token")
                # Token usually valid for 2 hours
                expire_seconds = data.get("expire", 7200)
                self._token_expires_at = time.time() + expire_seconds
                cprint("Feishu access token refreshed", "green")
                return self._access_token
            else:
                cprint(f"Failed to get Feishu token: {data.get('msg')}", "red")
                return None

        except Exception as e:
            cprint(f"Feishu token request error: {repr(e)}", "red")
            return None

    def _send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: Dict[str, Any],
        receive_id_type: str = "open_id"
    ) -> bool:
        """
        Generic method to send messages

        Args:
            receive_id: Receiver ID
            msg_type: Message type (text, interactive, etc.)
            content: Message content
            receive_id_type: ID type (open_id, user_id, union_id, email, chat_id)

        Returns:
            bool: Whether sending was successful
        """
        token = self._get_access_token()
        if not token:
            return False

        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8"
            }

            payload = {
                "receive_id": receive_id,
                "msg_type": msg_type,
                "content": json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else content
            }

            response = requests.post(
                f"{self.MESSAGE_URL}?receive_id_type={receive_id_type}",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get("code") == 0:
                cprint(f"Feishu message sent successfully ({msg_type})", "green")
                return True
            else:
                cprint(f"Feishu message failed: {data.get('msg')}", "red")
                return False

        except Exception as e:
            cprint(f"Feishu message error: {repr(e)}", "red")
            return False

    def send_text(
        self,
        text: str,
        user_open_id: Optional[str] = None
    ) -> bool:
        """
        Send text message

        Args:
            text: Message text content
            user_open_id: Receiver Open ID, defaults to configured ID

        Returns:
            bool: Whether sending was successful

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_text("Hello from Moon Dev!")
        """
        receive_id = user_open_id or self.user_open_id
        if not receive_id:
            cprint("No user_open_id configured for Feishu", "red")
            return False

        content = {"text": text}
        return self._send_message(receive_id, "text", content)

    def send_urgent_card(
        self,
        title: str,
        content: str,
        user_open_id: Optional[str] = None,
        color: str = "red",
        button_text: str = "View Details",
        button_url: Optional[str] = None
    ) -> bool:
        """
        Send urgent card message (Interactive Card)

        Args:
            title: Card title
            content: Card content (supports Markdown)
            user_open_id: Receiver Open ID, defaults to configured ID
            color: Card color (red, orange, yellow, green, blue, purple, grey)
            button_text: Button text (ignored, kept for compatibility)
            button_url: Button URL (ignored, kept for compatibility)

        Returns:
            bool: Whether sending was successful

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_urgent_card(
            ...     title="Trading Alert",
            ...     content="BTC price broke $100,000!",
            ...     color="red"
            ... )
        """
        receive_id = user_open_id or self.user_open_id
        if not receive_id:
            cprint("No user_open_id configured for Feishu", "red")
            return False

        # Build card elements - content only, no buttons or footer
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": content
                }
            }
        ]

        # Build card message
        card = {
            "config": {
                "wide_screen_mode": True,
                "enable_forward": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": color
            },
            "elements": elements
        }

        return self._send_message(receive_id, "interactive", card)

    def send_table_card(
        self,
        title: str,
        headers: list,
        rows: list,
        user_open_id: Optional[str] = None,
        color: str = "blue"
    ) -> bool:
        """
        Send table card using multi-column layout

        Args:
            title: Card title
            headers: List of column headers (e.g., ["#", "Symbol", "Price", "Vol 4H", "24H%"])
            rows: List of row data, each row is a list matching headers
            user_open_id: Receiver Open ID, defaults to configured ID
            color: Card color

        Returns:
            bool: Whether sending was successful

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_table_card(
            ...     title="Top 15 Altcoins",
            ...     headers=["#", "Symbol", "Price", "Volume"],
            ...     rows=[
            ...         ["1", "HYPE", "$25.30", "$1.2B"],
            ...         ["2", "DOGE", "$0.42", "$800M"],
            ...     ]
            ... )
        """
        receive_id = user_open_id or self.user_open_id
        if not receive_id:
            cprint("No user_open_id configured for Feishu", "red")
            return False

        elements = []

        # Build header row using column_set
        header_columns = []
        for h in headers:
            header_columns.append({
                "tag": "column",
                "width": "weighted",
                "weight": 1,
                "vertical_align": "top",
                "elements": [
                    {
                        "tag": "div",
                        "text": {
                            "tag": "lark_md",
                            "content": f"**{h}**"
                        }
                    }
                ]
            })

        elements.append({
            "tag": "column_set",
            "flex_mode": "none",
            "background_style": "grey",
            "columns": header_columns
        })

        # Build data rows
        for row in rows:
            row_columns = []
            for cell in row:
                row_columns.append({
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "vertical_align": "top",
                    "elements": [
                        {
                            "tag": "div",
                            "text": {
                                "tag": "plain_text",
                                "content": str(cell)
                            }
                        }
                    ]
                })

            elements.append({
                "tag": "column_set",
                "flex_mode": "none",
                "background_style": "default",
                "columns": row_columns
            })

        # Build card message
        card = {
            "config": {
                "wide_screen_mode": True,
                "enable_forward": True
            },
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": title
                },
                "template": color
            },
            "elements": elements
        }

        return self._send_message(receive_id, "interactive", card)

    def send_trading_alert(
        self,
        action: str,
        symbol: str,
        price: float,
        reason: str,
        user_open_id: Optional[str] = None
    ) -> bool:
        """
        Send trading alert card (convenience method)

        Args:
            action: Trading action (BUY, SELL, ALERT)
            symbol: Trading pair symbol
            price: Current price
            reason: Trigger reason
            user_open_id: Receiver Open ID

        Returns:
            bool: Whether sending was successful

        Example:
            >>> bot = FeishuBot()
            >>> bot.send_trading_alert(
            ...     action="BUY",
            ...     symbol="BTC/USDT",
            ...     price=99500.50,
            ...     reason="RSI oversold signal triggered"
            ... )
        """
        # Choose color and icon based on action
        action_config = {
            "BUY": {"color": "green", "icon": "[BUY]"},
            "SELL": {"color": "red", "icon": "[SELL]"},
            "ALERT": {"color": "orange", "icon": "[ALERT]"},
            "STOP_LOSS": {"color": "red", "icon": "[STOP]"},
            "TAKE_PROFIT": {"color": "green", "icon": "[PROFIT]"}
        }

        config = action_config.get(action.upper(), {"color": "blue", "icon": "[INFO]"})

        title = f"{config['icon']} {action.upper()} - {symbol}"
        content = f"""**Symbol:** {symbol}
**Action:** {action.upper()}
**Price:** ${price:,.2f}
**Reason:** {reason}"""

        return self.send_urgent_card(
            title=title,
            content=content,
            user_open_id=user_open_id,
            color=config["color"]
        )


# Create default instance
feishu_bot = FeishuBot()


# Convenience functions
def send_text(text: str, user_open_id: Optional[str] = None) -> bool:
    """
    Convenience function to send text message

    Args:
        text: Message text
        user_open_id: Receiver Open ID (optional)

    Returns:
        bool: Whether sending was successful
    """
    return feishu_bot.send_text(text, user_open_id)


def send_urgent_card(
    title: str,
    content: str,
    user_open_id: Optional[str] = None,
    color: str = "red",
    button_text: str = "View Details",
    button_url: Optional[str] = None
) -> bool:
    """
    Convenience function to send urgent card message

    Args:
        title: Card title
        content: Card content
        user_open_id: Receiver Open ID (optional)
        color: Card color
        button_text: Button text
        button_url: Button URL

    Returns:
        bool: Whether sending was successful
    """
    return feishu_bot.send_urgent_card(
        title=title,
        content=content,
        user_open_id=user_open_id,
        color=color,
        button_text=button_text,
        button_url=button_url
    )


def send_table_card(
    title: str,
    headers: list,
    rows: list,
    user_open_id: Optional[str] = None,
    color: str = "blue"
) -> bool:
    """
    Convenience function to send table card with multi-column layout

    Args:
        title: Card title
        headers: List of column headers
        rows: List of row data
        user_open_id: Receiver Open ID (optional)
        color: Card color

    Returns:
        bool: Whether sending was successful
    """
    return feishu_bot.send_table_card(
        title=title,
        headers=headers,
        rows=rows,
        user_open_id=user_open_id,
        color=color
    )


if __name__ == "__main__":
    # Test code
    cprint("Moon Dev's Feishu Bot Test", "cyan")

    # Test sending text
    success = send_text("Hello from Moon Dev! This is a test message.")
    print(f"Text message: {'Success' if success else 'Failed'}")

    # Test sending urgent card
    success = send_urgent_card(
        title="Test Alert",
        content="**This is a test message**\n\nSystem running normally, all modules ready.",
        color="blue",
        button_text="View Docs",
        button_url="https://github.com"
    )
    print(f"Card message: {'Success' if success else 'Failed'}")

    # Test trading alert
    success = feishu_bot.send_trading_alert(
        action="BUY",
        symbol="BTC/USDT",
        price=99500.50,
        reason="RSI oversold signal triggered, recommend buying"
    )
    print(f"Trading alert: {'Success' if success else 'Failed'}")
