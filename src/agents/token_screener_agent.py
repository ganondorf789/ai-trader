"""
Moon Dev's Token Screener Agent (V2 - Optimized with Token List V3 API)

Screens Solana tokens using Birdeye Token List V3 API for maximum efficiency.
Most data comes from a single API call, only OHLCV needs separate requests.

Screening Criteria:
1. liquidity_usd >= 800,000          - High liquidity, harder to rug
2. volume_24h_usd >= 10,000,000      - Strong trading volume
3. number_holders >= 4,000           - Good holder distribution
4. price_change_24h >= -30%          - Not in free fall
5. current_price <= 3d_low * 1.12    - Near 3-day low (12% tolerance)
6. current_price >= 3d_low * 0.75    - Not totally dead
7. pool_age >= 12 hours              - Not brand new (uses recent_listing_time)
8. (optional) 1h_volume >= 8% of 24h - Volume surge signal

Built with love by Moon Dev
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Load environment variables
load_dotenv()

# Get API key
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
if not BIRDEYE_API_KEY:
    raise ValueError("BIRDEYE_API_KEY not found in environment variables!")

BASE_URL = "https://public-api.birdeye.so"

# ============================================================================
# SCREENING CRITERIA - Adjust these thresholds as needed
# ============================================================================

# Minimum liquidity in USD (higher = harder to rug)
MIN_LIQUIDITY_USD = 800_000

# Minimum 24h trading volume in USD
MIN_VOLUME_24H_USD = 10_000_000

# Minimum number of holders
MIN_HOLDERS = 4_000

# Maximum 24h price drop percentage (e.g., -30 means max 30% drop allowed)
MAX_PRICE_DROP_24H = -30

# Price must be within this multiple of 3-day low (1.12 = within 12% above low)
MAX_PRICE_VS_3D_LOW = 1.12

# Price must be at least this multiple of 3-day low (0.75 = not more than 25% below)
MIN_PRICE_VS_3D_LOW = 0.75

# Minimum pool age in hours
MIN_POOL_AGE_HOURS = 12

# Optional: Minimum 1h volume as percentage of 24h volume (volume surge signal)
# Set to 0 to disable this filter
# 8-10% is a good range for volume surge when enabled
MIN_1H_VOLUME_PERCENT = 4  # Set to 0 to disable (default), 8-10 to enable

# How many tokens to fetch (API returns up to 100 per call)
TOKEN_FETCH_LIMIT = 100

# Sleep between API calls to avoid rate limiting (seconds)
API_SLEEP = 0.3

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_FOLDER = Path(__file__).parent.parent / "data" / "screener_agent"
RESULTS_FILE = DATA_FOLDER / "screened_tokens.csv"
HISTORY_FILE = DATA_FOLDER / "screening_history.csv"


class TokenScreenerAgent:
    """
    Moon Dev's Token Screener Agent V2
    Uses Token List V3 API for efficient bulk data fetching
    Only needs OHLCV calls for 3-day low calculation
    """

    def __init__(self):
        self.headers = {
            "X-API-KEY": BIRDEYE_API_KEY,
            "x-chain": "solana",
            "accept": "application/json"
        }
        self.session = requests.Session()
        self.screened_tokens = []

        # Create data directory
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)

        cprint("\n" + "="*60, "cyan")
        cprint(" Moon Dev's Token Screener Agent V2 ", "white", "on_magenta")
        cprint(" Using Token List V3 API for efficiency ", "white", "on_blue")
        cprint("="*60, "cyan")

    def get_tokens_v3(self, limit=TOKEN_FETCH_LIMIT):
        """
        Fetch tokens using Token List V3 API
        This returns comprehensive data in a single call:
        - liquidity, volume (1h/2h/4h/8h/24h), holder count
        - price changes, trade counts, buy/sell ratios
        - recent_listing_time for pool age
        """
        cprint(f"\n Fetching tokens via V3 API (sorted by 24h volume)...", "cyan")

        url = f"{BASE_URL}/defi/v3/token/list"

        # V3 API parameters - filter by minimum liquidity and volume at API level
        params = {
            "sort_by": "volume_24h_usd",
            "sort_type": "desc",
            "offset": 0,
            "limit": min(limit, 100),  # Max 100 per call
            "min_liquidity": MIN_LIQUIDITY_USD,
            "min_volume_24h_usd": MIN_VOLUME_24H_USD
        }

        all_tokens = []

        try:
            response = self.session.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if data.get("success"):
                    items = data.get("data", {}).get("items", [])
                    all_tokens = items
                    cprint(f" Fetched {len(items)} tokens from V3 API", "green")

                    # Check if there are more pages
                    has_next = data.get("data", {}).get("hasNext", False)
                    if has_next and len(all_tokens) < limit:
                        cprint(f" More tokens available (hasNext=True)", "yellow")
                else:
                    cprint(f" API returned success=false", "red")
            else:
                cprint(f" V3 API error: {response.status_code}", "red")
                cprint(f" Response: {response.text[:200]}", "red")

        except Exception as e:
            cprint(f" Error fetching V3 tokens: {e}", "red")

        return all_tokens

    def get_ohlcv_data(self, address, days_back=3, timeframe="1H"):
        """
        Get OHLCV data to find 3-day low
        This is the only additional API call needed per token
        """
        now = datetime.now()
        time_to = int(now.timestamp())
        time_from = int((now - timedelta(days=days_back)).timestamp())

        url = f"{BASE_URL}/defi/ohlcv"
        params = {
            "address": address,
            "type": timeframe,
            "time_from": time_from,
            "time_to": time_to
        }

        try:
            response = self.session.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", {})
                items = data.get("items", [])

                if items:
                    df = pd.DataFrame(items)
                    df['datetime'] = pd.to_datetime(df['unixTime'], unit='s')
                    return df
            return None

        except Exception as e:
            return None

    def calculate_pool_age_hours(self, recent_listing_time, last_trade_unix_time):
        """
        Calculate pool age in hours
        Uses recent_listing_time if available, otherwise estimates from last_trade_unix_time
        """
        now = datetime.now().timestamp()

        if recent_listing_time:
            # recent_listing_time is Unix timestamp
            age_hours = (now - recent_listing_time) / 3600
            return age_hours

        # If no listing time, token is likely old (established token)
        # Return a large value to pass the filter
        return 99999

    def screen_token_v3(self, token_data):
        """
        Apply screening criteria to token data from V3 API
        Most filters can be checked directly without additional API calls
        Only OHLCV is needed for 3-day low calculation
        """
        address = token_data.get("address", "")
        symbol = token_data.get("symbol", "???")
        name = token_data.get("name", "")

        # Extract all metrics from V3 response
        liquidity = float(token_data.get("liquidity", 0) or 0)
        volume_24h = float(token_data.get("volume_24h_usd", 0) or 0)
        volume_1h = float(token_data.get("volume_1h_usd", 0) or 0)
        holder_count = int(token_data.get("holder", 0) or 0)
        price_change_24h = float(token_data.get("price_change_24h_percent", 0) or 0)
        current_price = float(token_data.get("price", 0) or 0)
        recent_listing_time = token_data.get("recent_listing_time")
        last_trade_time = token_data.get("last_trade_unix_time")

        # ===== PRE-FILTER: Check for obvious failures =====

        # Check for extreme price drops (potential rug)
        price_change_8h = float(token_data.get("price_change_8h_percent", 0) or 0)
        price_change_4h = float(token_data.get("price_change_4h_percent", 0) or 0)

        if price_change_24h < -80 or price_change_8h < -80 or price_change_4h < -80:
            cprint(f" {symbol}: Potential rug pull (extreme price drop)", "red")
            return None

        # ===== FILTER 1: Liquidity (already filtered at API level, but double check) =====
        if liquidity < MIN_LIQUIDITY_USD:
            cprint(f" {symbol}: Low liquidity ${liquidity:,.0f} < ${MIN_LIQUIDITY_USD:,}", "yellow")
            return None

        # ===== FILTER 2: 24h Volume (already filtered at API level) =====
        if volume_24h < MIN_VOLUME_24H_USD:
            cprint(f" {symbol}: Low volume ${volume_24h:,.0f} < ${MIN_VOLUME_24H_USD:,}", "yellow")
            return None

        # ===== FILTER 3: Holder Count =====
        if holder_count < MIN_HOLDERS:
            cprint(f" {symbol}: Low holders {holder_count:,} < {MIN_HOLDERS:,}", "yellow")
            return None

        # ===== FILTER 4: 24h Price Change =====
        if price_change_24h < MAX_PRICE_DROP_24H:
            cprint(f" {symbol}: Price dropped too much {price_change_24h:.1f}% < {MAX_PRICE_DROP_24H}%", "yellow")
            return None

        # ===== FILTER 7: Pool Age (check before OHLCV to save API calls) =====
        pool_age_hours = self.calculate_pool_age_hours(recent_listing_time, last_trade_time)

        if pool_age_hours < MIN_POOL_AGE_HOURS:
            cprint(f" {symbol}: Pool too new {pool_age_hours:.1f}h < {MIN_POOL_AGE_HOURS}h", "yellow")
            return None

        # ===== FILTER 8: Volume Surge (check before OHLCV) =====
        volume_1h_percent = (volume_1h / volume_24h * 100) if volume_24h > 0 else 0
        volume_surge = volume_1h_percent >= MIN_1H_VOLUME_PERCENT

        if MIN_1H_VOLUME_PERCENT > 0 and not volume_surge:
            cprint(f" {symbol}: No volume surge {volume_1h_percent:.1f}% < {MIN_1H_VOLUME_PERCENT}%", "yellow")
            return None

        # ===== FILTER 5 & 6: Price vs 3-day Low (requires OHLCV API call) =====
        cprint(f" {symbol}: Passed initial filters, checking 3-day low...", "white")

        ohlcv = self.get_ohlcv_data(address, days_back=3)
        time.sleep(API_SLEEP)

        if ohlcv is not None and len(ohlcv) > 0:
            three_day_low = ohlcv['l'].min()  # 'l' is low price

            if three_day_low > 0 and current_price > 0:
                price_vs_low_ratio = current_price / three_day_low

                # Filter 5: Price should be within 12% of 3-day low
                if price_vs_low_ratio > MAX_PRICE_VS_3D_LOW:
                    cprint(f" {symbol}: Price {price_vs_low_ratio:.2f}x above 3d low (max {MAX_PRICE_VS_3D_LOW}x)", "yellow")
                    return None

                # Filter 6: Price should not be more than 25% below 3-day low
                if price_vs_low_ratio < MIN_PRICE_VS_3D_LOW:
                    cprint(f" {symbol}: Price {price_vs_low_ratio:.2f}x of 3d low (min {MIN_PRICE_VS_3D_LOW}x)", "yellow")
                    return None
            else:
                cprint(f" {symbol}: Invalid price data", "yellow")
                return None
        else:
            cprint(f" {symbol}: Could not get OHLCV data", "yellow")
            return None

        # ===== TOKEN PASSED ALL FILTERS =====
        cprint(f" {symbol}: PASSED all filters!", "green", attrs=["bold"])

        return {
            "address": address,
            "symbol": symbol,
            "name": name,
            "liquidity_usd": liquidity,
            "volume_24h_usd": volume_24h,
            "volume_1h_usd": volume_1h,
            "volume_1h_percent": round(volume_1h_percent, 2),
            "holder_count": holder_count,
            "price_change_24h": round(price_change_24h, 2),
            "current_price": current_price,
            "three_day_low": three_day_low,
            "price_vs_3d_low": round(price_vs_low_ratio, 3),
            "pool_age_hours": round(pool_age_hours, 1) if pool_age_hours < 99999 else "established",
            "volume_surge": volume_surge,
            "market_cap": token_data.get("market_cap", 0),
            "fdv": token_data.get("fdv", 0),
            "buy_24h": token_data.get("buy_24h", 0),
            "sell_24h": token_data.get("sell_24h", 0),
            "unique_wallet_24h": token_data.get("unique_wallet_24h", 0),
            "screened_at": datetime.now().isoformat(),
            "birdeye_link": f"https://birdeye.so/token/{address}?chain=solana",
            "dexscreener_link": f"https://dexscreener.com/solana/{address}"
        }

    def run(self):
        """
        Main screening loop using V3 API
        """
        cprint("\n Starting token screening (V3 API)...", "cyan")
        cprint(f" Criteria:", "white")
        cprint(f"   Liquidity >= ${MIN_LIQUIDITY_USD:,}", "white")
        cprint(f"   24h Volume >= ${MIN_VOLUME_24H_USD:,}", "white")
        cprint(f"   Holders >= {MIN_HOLDERS:,}", "white")
        cprint(f"   24h Price Change >= {MAX_PRICE_DROP_24H}%", "white")
        cprint(f"   Price <= {MAX_PRICE_VS_3D_LOW}x of 3-day low", "white")
        cprint(f"   Price >= {MIN_PRICE_VS_3D_LOW}x of 3-day low", "white")
        cprint(f"   Pool Age >= {MIN_POOL_AGE_HOURS} hours", "white")
        if MIN_1H_VOLUME_PERCENT > 0:
            cprint(f"   1h Volume >= {MIN_1H_VOLUME_PERCENT}% of 24h (volume surge)", "white")
        cprint("")

        # Get tokens via V3 API (pre-filtered by volume and liquidity)
        tokens = self.get_tokens_v3()

        if not tokens:
            cprint(" No tokens found matching initial criteria", "red")
            return []

        cprint(f"\n Screening {len(tokens)} tokens...", "cyan")

        # Screen each token
        passed_tokens = []

        for i, token in enumerate(tokens, 1):
            symbol = token.get("symbol", "???")
            address = token.get("address", "")[:8]

            cprint(f"\n[{i}/{len(tokens)}] {symbol} ({address}...)", "cyan")

            result = self.screen_token_v3(token)

            if result:
                passed_tokens.append(result)
                self.display_result(result)

        # Save results
        self.save_results(passed_tokens)

        # Summary
        cprint("\n" + "="*60, "cyan")
        cprint(f" SCREENING COMPLETE", "white", "on_green")
        cprint(f" Fetched: {len(tokens)} tokens (pre-filtered by V3 API)", "white")
        cprint(f" Passed all criteria: {len(passed_tokens)} tokens", "green")
        cprint("="*60, "cyan")

        return passed_tokens

    def display_result(self, token):
        """
        Display a passed token with Moon Dev style
        """
        cprint("\n" + "-"*50, "green")
        cprint(f" {token['symbol']} - {token['name']}", "white", "on_green")
        cprint("-"*50, "green")
        cprint(f"   Address: {token['address']}", "white")
        cprint(f"   Liquidity: ${token['liquidity_usd']:,.0f}", "cyan")
        cprint(f"   24h Volume: ${token['volume_24h_usd']:,.0f}", "cyan")
        cprint(f"   1h Volume: ${token['volume_1h_usd']:,.0f} ({token['volume_1h_percent']}%)", "cyan")
        cprint(f"   Holders: {token['holder_count']:,}", "cyan")
        cprint(f"   24h Change: {token['price_change_24h']}%", "cyan")
        cprint(f"   Price: ${token['current_price']:.8f}", "cyan")
        cprint(f"   3-Day Low: ${token['three_day_low']:.8f}", "cyan")
        cprint(f"   Price vs 3d Low: {token['price_vs_3d_low']}x", "cyan")
        cprint(f"   Pool Age: {token['pool_age_hours']} hours", "cyan")
        cprint(f"   Market Cap: ${token.get('market_cap', 0):,.0f}", "cyan")
        cprint(f"   Volume Surge: {'YES' if token['volume_surge'] else 'No'}",
               "green" if token['volume_surge'] else "white")
        cprint(f"   Birdeye: {token['birdeye_link']}", "blue")
        cprint(f"   DexScreener: {token['dexscreener_link']}", "blue")

    def save_results(self, tokens):
        """
        Save screening results to CSV
        """
        if not tokens:
            cprint(" No tokens to save", "yellow")
            return

        df = pd.DataFrame(tokens)

        # Save current results
        df.to_csv(RESULTS_FILE, index=False)
        cprint(f"\n Saved {len(tokens)} tokens to {RESULTS_FILE}", "green")

        # Append to history
        if HISTORY_FILE.exists():
            history_df = pd.read_csv(HISTORY_FILE)
            combined = pd.concat([history_df, df], ignore_index=True)
            combined.to_csv(HISTORY_FILE, index=False)
        else:
            df.to_csv(HISTORY_FILE, index=False)

        cprint(f" Updated screening history at {HISTORY_FILE}", "green")


def main():
    """
    Main entry point for standalone execution
    """
    agent = TokenScreenerAgent()
    results = agent.run()

    if results:
        cprint(f"\n Found {len(results)} tokens matching all criteria!", "green", attrs=["bold"])
        for token in results:
            cprint(f"  - {token['symbol']}: {token['dexscreener_link']}", "cyan")
    else:
        cprint("\n No tokens matched all screening criteria", "yellow")


if __name__ == "__main__":
    main()
