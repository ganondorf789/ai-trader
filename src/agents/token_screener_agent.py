"""
Moon Dev's Token Screener Agent (V3 - With Swarm Analysis & Feishu)

Screens multi-chain tokens using Birdeye Token List V3 API for maximum efficiency.
Most data comes from a single API call, only OHLCV needs separate requests.
Found tokens are analyzed by AI Swarm and results sent to Feishu.

Screening Criteria (all configurable):
1. liquidity_usd >= MIN_LIQUIDITY_USD       - High liquidity, harder to rug
2. volume_24h_usd >= MIN_VOLUME_24H_USD     - Strong trading volume
3. number_holders >= MIN_HOLDERS            - Good holder distribution
4. price_change_24h >= MAX_PRICE_DROP_24H   - Not in free fall
5. current_price <= N-day_low * MAX_PRICE_VS_LOW  - Near N-day low
6. current_price >= N-day_low * MIN_PRICE_VS_LOW  - Not totally dead
7. pool_age >= MIN_POOL_AGE_HOURS           - Not brand new
8. (optional) 1h_volume >= MIN_1H_VOLUME_PERCENT of 24h - Volume surge signal

N = PRICE_LOW_DAYS (configurable, default 3)

Built with love by Moon Dev
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Get API key
BIRDEYE_API_KEY = os.getenv("BIRDEYE_API_KEY")
if not BIRDEYE_API_KEY:
    raise ValueError("BIRDEYE_API_KEY not found in environment variables!")

BASE_URL = "https://public-api.birdeye.so"

# Import Swarm Agent and Feishu Bot
from src.agents.swarm_agent import SwarmAgent
from src.message.lark import send_text, send_urgent_card, send_table_card

# ============================================================================
# SCREENING CRITERIA - Adjust these thresholds as needed
# ============================================================================

# Minimum liquidity in USD (higher = harder to rug)
MIN_LIQUIDITY_USD = 0

# Minimum 24h trading volume in USD
MIN_VOLUME_24H_USD = 5_000_000

# Minimum number of holders
MIN_HOLDERS = 0

# Maximum 24h price drop percentage (e.g., -25 means max 25% drop allowed)
MAX_PRICE_DROP_24H = -25

# Number of days to look back for price low calculation
PRICE_LOW_DAYS = 14

# Price must be within this multiple of N-day low (1.15 = within 15% above low)
MAX_PRICE_VS_LOW = 1.15

# Price must be at least this multiple of N-day low (0.80 = not more than 20% below)
MIN_PRICE_VS_LOW = 0.80

# Minimum pool age in hours (168 = 7 days)
MIN_POOL_AGE_HOURS = 168

# Optional: Minimum 1h volume as percentage of 24h volume (volume surge signal)
# Set to 0 to disable this filter
# 8-10% is a good range for volume surge when enabled
MIN_1H_VOLUME_PERCENT = 0  # Set to 0 to disable (default), 8-10 to enable

# Sleep between API calls to avoid rate limiting (seconds)
API_SLEEP = 0

# ============================================================================
# STABLECOIN EXCLUSION LIST - Skip these tokens
# ============================================================================
EXCLUDED_SYMBOLS = [
    # Major stablecoins
    "USDC",  "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX",
    "PYUSD", "USD1", "USDG", "USDS", "USX", "USDD", "EURC", "EURS",
    # Wrapped stablecoins
    "sUSD", "aUSDC", "aUSDT", "cUSDC", "cUSDT",
    # Solana wrapped stables
    "UXD", "CASH", "USDH", "PAI",
    "USDE","USDON","SUSDE","USDT"
]

# ============================================================================
# MULTI-CHAIN SETTINGS - Run screening on multiple chains sequentially
# ============================================================================
ENABLED_CHAINS = ["solana"]  # Chains to screen (in order)
# Supported chains: solana, ethereum, arbitrum, avalanche, bsc, optimism, polygon, base, zksync, sui

# ============================================================================
# FEISHU & SWARM SETTINGS
# ============================================================================
FEISHU_ENABLED = True  # Set to False to disable Feishu notifications
SWARM_ENABLED = True   # Set to False to skip AI analysis

# ============================================================================
# CONTINUOUS RUN SETTINGS
# ============================================================================
RUN_INTERVAL_MINUTES = 4 * 60  # Run every 60 minutes (1 hour)
CONTINUOUS_MODE = True     # Set to False for single run

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_FOLDER = Path(__file__).parent.parent / "data" / "screener_agent"
RESULTS_FILE = DATA_FOLDER / "screened_tokens.csv"
HISTORY_FILE = DATA_FOLDER / "screening_history.csv"


class TokenScreenerAgent:
    """
    Moon Dev's Token Screener Agent V3
    Uses Token List V3 API for efficient bulk data fetching
    Supports multiple chains: solana, ethereum, arbitrum, etc.
    Only needs OHLCV calls for 3-day low calculation
    """

    def __init__(self, chain="solana"):
        """
        Initialize the screener for a specific chain

        Args:
            chain: Chain to screen (solana, ethereum, arbitrum, avalanche, bsc, optimism, polygon, base, zksync, sui)
        """
        self.chain = chain.lower()
        self.headers = {
            "X-API-KEY": BIRDEYE_API_KEY,
            "x-chain": self.chain,
            "accept": "application/json"
        }
        self.session = requests.Session()
        self.screened_tokens = []

        # Create data directory
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)

        cprint("\n" + "="*60, "cyan")
        cprint(" Moon Dev's Token Screener Agent V3 ", "white", "on_magenta")
        cprint(f" Chain: {self.chain.upper()} ", "white", "on_blue")
        cprint("="*60, "cyan")

    def get_tokens_v3(self):
        """
        Fetch all tokens using Token List V3 API with pagination
        This returns comprehensive data:
        - liquidity, volume (1h/2h/4h/8h/24h), holder count
        - price changes, trade counts, buy/sell ratios
        - recent_listing_time for pool age

        Uses while loop to fetch all pages (API max 100 per call)
        """
        cprint(f"\n Fetching tokens via V3 API (sorted by 24h volume)...", "cyan")

        url = f"{BASE_URL}/defi/v3/token/list"
        all_tokens = []
        offset = 0
        page_size = 100  # API maximum per call

        try:
            while True:
                # V3 API parameters - filter by minimum volume at API level
                params = {
                    "sort_by": "volume_24h_usd",
                    "sort_type": "desc",
                    "offset": offset,
                    "limit": page_size,
                    "min_volume_24h_usd": MIN_VOLUME_24H_USD
                }

                response = self.session.get(url, headers=self.headers, params=params)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("success"):
                        items = data.get("data", {}).get("items", [])

                        if not items:
                            # No more items
                            break

                        all_tokens.extend(items)
                        cprint(f" Fetched {len(items)} tokens (total: {len(all_tokens)})", "green")

                        # Check if there are more pages
                        has_next = data.get("data", {}).get("has_next", False)
                        if not has_next:
                            break

                        # Move to next page
                        offset += page_size

                        # Small delay to avoid rate limiting
                        if API_SLEEP > 0:
                            time.sleep(API_SLEEP)
                    else:
                        cprint(f" API returned success=false", "red")
                        break
                else:
                    cprint(f" V3 API error: {response.status_code}", "red")
                    cprint(f" Response: {response.text[:200]}", "red")
                    break

            cprint(f" Total fetched: {len(all_tokens)} tokens from V3 API", "green")

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

        # Skip stablecoins
        if symbol.upper() in EXCLUDED_SYMBOLS:
            cprint(f" {symbol}: Skipping stablecoin", "yellow")
            return None

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

        # ===== FILTER 5 & 6: Price vs N-day Low (requires OHLCV API call) =====
        cprint(f" {symbol}: Passed initial filters, checking {PRICE_LOW_DAYS}d low...", "white")

        ohlcv = self.get_ohlcv_data(address, days_back=PRICE_LOW_DAYS)
        time.sleep(API_SLEEP)

        if ohlcv is not None and len(ohlcv) > 0:
            period_low = ohlcv['l'].min()  # 'l' is low price

            if period_low > 0 and current_price > 0:
                price_vs_low_ratio = current_price / period_low

                # Filter 5: Price should be within configured % of N-day low
                if price_vs_low_ratio > MAX_PRICE_VS_LOW:
                    cprint(f" {symbol}: Price {price_vs_low_ratio:.2f}x above {PRICE_LOW_DAYS}d low (max {MAX_PRICE_VS_LOW}x)", "yellow")
                    return None

                # Filter 6: Price should not be too far below N-day low
                if price_vs_low_ratio < MIN_PRICE_VS_LOW:
                    cprint(f" {symbol}: Price {price_vs_low_ratio:.2f}x of {PRICE_LOW_DAYS}d low (min {MIN_PRICE_VS_LOW}x)", "yellow")
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
            "chain": self.chain,
            "liquidity_usd": liquidity,
            "volume_24h_usd": volume_24h,
            "volume_1h_usd": volume_1h,
            "volume_1h_percent": round(volume_1h_percent, 2),
            "holder_count": holder_count,
            "price_change_24h": round(price_change_24h, 2),
            "current_price": current_price,
            "period_low": period_low,
            "price_vs_low": round(price_vs_low_ratio, 3),
            "pool_age_hours": round(pool_age_hours, 1) if pool_age_hours < 99999 else "established",
            "volume_surge": volume_surge,
            "market_cap": token_data.get("market_cap", 0),
            "fdv": token_data.get("fdv", 0),
            "buy_24h": token_data.get("buy_24h", 0),
            "sell_24h": token_data.get("sell_24h", 0),
            "unique_wallet_24h": token_data.get("unique_wallet_24h", 0),
            "screened_at": datetime.now().isoformat(),
            "birdeye_link": f"https://birdeye.so/token/{address}?chain={self.chain}",
            "dexscreener_link": f"https://dexscreener.com/{self.chain}/{address}"
        }

    def run(self):
        """
        Main screening loop using V3 API
        """
        cprint(f"\n Starting token screening on {self.chain.upper()} (V3 API)...", "cyan")
        cprint(f" Criteria:", "white")
        cprint(f"   Liquidity >= ${MIN_LIQUIDITY_USD:,}", "white")
        cprint(f"   24h Volume >= ${MIN_VOLUME_24H_USD:,}", "white")
        cprint(f"   Holders >= {MIN_HOLDERS:,}", "white")
        cprint(f"   24h Price Change >= {MAX_PRICE_DROP_24H}%", "white")
        cprint(f"   Price <= {MAX_PRICE_VS_LOW}x of {PRICE_LOW_DAYS}-day low", "white")
        cprint(f"   Price >= {MIN_PRICE_VS_LOW}x of {PRICE_LOW_DAYS}-day low", "white")
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
        cprint(f"   {PRICE_LOW_DAYS}-Day Low: ${token['period_low']:.8f}", "cyan")
        cprint(f"   Price vs {PRICE_LOW_DAYS}d Low: {token['price_vs_low']}x", "cyan")
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


# ============================================================================
# SWARM ANALYSIS
# ============================================================================

def create_analysis_prompt(tokens):
    """Create prompt for swarm agents to analyze screened tokens"""

    prompt = f"""You are analyzing tokens that passed a strict screening filter for potential buying opportunities.

All tokens below have already passed these criteria:
- Liquidity >= ${MIN_LIQUIDITY_USD:,} (hard to rug)
- 24h Volume >= ${MIN_VOLUME_24H_USD:,} (active trading)
- Holders >= {MIN_HOLDERS:,} (good distribution)
- Price within {int((MAX_PRICE_VS_LOW - 1) * 100)}% of {PRICE_LOW_DAYS}-day low (near bottom)
- Pool age >= {MIN_POOL_AGE_HOURS} hours (not brand new)
- No extreme price drops (not rugging)

Here are the screened tokens:

"""

    for i, token in enumerate(tokens, 1):
        prompt += f"\n{i}. **{token['symbol']}** ({token['name']}) [{token.get('chain', 'unknown').upper()}]\n"
        prompt += f"   - Price: ${token['current_price']:.8f}\n"
        prompt += f"   - Market Cap: ${token.get('market_cap', 0):,.0f}\n"
        prompt += f"   - 24h Volume: ${token['volume_24h_usd']:,.0f}\n"
        prompt += f"   - 1h Volume: ${token['volume_1h_usd']:,.0f} ({token['volume_1h_percent']}% of 24h)\n"
        prompt += f"   - Liquidity: ${token['liquidity_usd']:,.0f}\n"
        prompt += f"   - Holders: {token['holder_count']:,}\n"
        prompt += f"   - 24h Price Change: {token['price_change_24h']}%\n"
        prompt += f"   - Price vs {PRICE_LOW_DAYS}-Day Low: {token['price_vs_low']}x\n"
        prompt += f"   - Volume Surge: {'YES' if token['volume_surge'] else 'No'}\n"
        prompt += f"   - Buy/Sell 24h: {token.get('buy_24h', 0)}/{token.get('sell_24h', 0)}\n"
        prompt += f"   - Unique Wallets 24h: {token.get('unique_wallet_24h', 0)}\n"

    prompt += f"""

Based on the data above, analyze each token and recommend:
1. Which token(s) would you BUY now and why?
2. Which token(s) should be AVOIDED and why?
3. What's your confidence level (1-10) for your top pick?

Focus on: volume momentum, price position vs {PRICE_LOW_DAYS}d low, holder distribution, and market cap.
Be concise and direct. Give actionable recommendations."""

    return prompt


def format_volume(volume):
    """Format volume for display"""
    if volume >= 1_000_000_000:
        return f"${volume/1_000_000_000:.2f}B"
    elif volume >= 1_000_000:
        return f"${volume/1_000_000:.2f}M"
    else:
        return f"${volume/1_000:.2f}K"


def run_swarm_analysis(tokens):
    """Run swarm analysis on screened tokens"""

    if not SWARM_ENABLED:
        cprint(" Swarm analysis disabled", "yellow")
        return None

    if not tokens:
        cprint(" No tokens to analyze", "yellow")
        return None

    cprint("\n" + "="*60, "cyan")
    cprint(" Running AI Swarm Analysis...", "cyan", attrs=["bold"])
    cprint("="*60, "cyan")

    swarm = SwarmAgent()
    prompt = create_analysis_prompt(tokens)
    result = swarm.query(prompt)

    return result


def display_swarm_results(result):
    """Display swarm analysis results"""

    if not result:
        return

    cprint("\n" + "="*60, "green")
    cprint(" AI SWARM ANALYSIS RESULTS", "green", attrs=["bold"])
    cprint("="*60, "green")

    # Show consensus first
    if "consensus_summary" in result:
        cprint("\n CONSENSUS:", "magenta", attrs=["bold"])
        cprint(f"{result['consensus_summary']}\n", "white")

    # Show individual responses
    cprint(" INDIVIDUAL AI RECOMMENDATIONS:", "yellow", attrs=["bold"])

    reverse_mapping = {}
    if "model_mapping" in result:
        for ai_num, provider in result["model_mapping"].items():
            reverse_mapping[provider.lower()] = ai_num

    for provider, data in result.get("responses", {}).items():
        if data.get("success"):
            ai_label = reverse_mapping.get(provider, provider.upper())
            cprint(f"\n {ai_label} ({provider.upper()}):", "cyan")
            response_text = data.get("response", "")
            # Truncate long responses
            if len(response_text) > 600:
                response_text = response_text[:600] + "..."
            cprint(f"{response_text}", "white")

    # Metadata
    if "metadata" in result:
        meta = result["metadata"]
        cprint(f"\n Models: {meta.get('successful_responses', 0)}/{meta.get('total_models', 0)} | Time: {meta.get('total_time', 0):.1f}s", "blue")


def send_feishu_report(tokens, swarm_result):
    """Send screening results and analysis to Feishu"""

    if not FEISHU_ENABLED:
        cprint(" Feishu notifications disabled", "yellow")
        return

    if not tokens:
        cprint(" No tokens to report", "yellow")
        return

    cprint("\n Sending Feishu notifications...", "cyan")

    try:
        # ============================================
        # Message 1: Screened Tokens Table
        # ============================================
        headers = ["#", "Symbol", "Price", "MCap", "Vol24h", "24h%", f"vs{PRICE_LOW_DAYS}dLow"]
        rows = []

        for i, t in enumerate(tokens, 1):
            price = f"${t['current_price']:.6f}" if t['current_price'] < 1 else f"${t['current_price']:.2f}"
            mcap = format_volume(t.get('market_cap', 0))
            vol = format_volume(t['volume_24h_usd'])
            chg = f"{t['price_change_24h']:+.1f}%"
            vs_low = f"{t['price_vs_low']:.3f}x"
            rows.append([str(i), t['symbol'][:8], price, mcap, vol, chg, vs_low])

        send_table_card(
            title=f"[1/3] Token Screener - {len(tokens)} Tokens Found",
            headers=headers,
            rows=rows,
            color="blue"
        )
        cprint(" Feishu Message 1/3 sent (Token Table)", "green")
        time.sleep(1)

        # ============================================
        # Message 2: AI Consensus
        # ============================================
        if swarm_result and "consensus_summary" in swarm_result:
            content_parts = []
            content_parts.append("**Screening Criteria:**")
            content_parts.append(f"- Liquidity >= ${MIN_LIQUIDITY_USD:,}")
            content_parts.append(f"- Volume 24h >= ${MIN_VOLUME_24H_USD:,}")
            content_parts.append(f"- Holders >= {MIN_HOLDERS:,}")
            content_parts.append(f"- Price vs {PRICE_LOW_DAYS}d Low <= {MAX_PRICE_VS_LOW}x")
            content_parts.append("")
            content_parts.append("**AI Consensus:**")
            content_parts.append(swarm_result["consensus_summary"])

            if "metadata" in swarm_result:
                meta = swarm_result["metadata"]
                content_parts.append("")
                content_parts.append(f"Models: {meta.get('successful_responses', 0)}/{meta.get('total_models', 0)}")

            send_urgent_card(
                title="[2/3] AI Consensus Analysis",
                content="\n".join(content_parts),
                color="green"
            )
            cprint(" Feishu Message 2/3 sent (AI Consensus)", "green")
            time.sleep(1)

        # ============================================
        # Message 3: Individual AI Recommendations
        # ============================================
        if swarm_result and "responses" in swarm_result:
            content_parts = []

            model_mapping = swarm_result.get("model_mapping", {})
            reverse_mapping = {v.lower(): k for k, v in model_mapping.items()}

            for provider, data in swarm_result["responses"].items():
                if data.get("success") and data.get("response"):
                    ai_label = reverse_mapping.get(provider, provider.upper())
                    response_text = data["response"]
                    # Truncate for Feishu
                    if len(response_text) > 400:
                        response_text = response_text[:400] + "..."
                    content_parts.append(f"**{ai_label} ({provider.upper()}):**")
                    content_parts.append(response_text)
                    content_parts.append("")

            if content_parts:
                send_urgent_card(
                    title="[3/3] AI Individual Analysis",
                    content="\n".join(content_parts),
                    color="blue"
                )
                cprint(" Feishu Message 3/3 sent (AI Analysis)", "green")

    except Exception as e:
        cprint(f" Feishu notification error: {e}", "red")


def run_single_chain(chain):
    """
    Run screening for a single chain (screening only, no AI analysis)

    Args:
        chain: Chain name (solana, ethereum, etc.)

    Returns:
        List of screened tokens for this chain
    """
    cprint(f"\n{'='*60}", "magenta")
    cprint(f" SCREENING CHAIN: {chain.upper()}", "white", "on_magenta")
    cprint(f"{'='*60}", "magenta")

    agent = TokenScreenerAgent(chain=chain)
    results = agent.run()

    if results:
        cprint(f"\n Found {len(results)} tokens on {chain.upper()} matching all criteria!", "green", attrs=["bold"])
        for token in results:
            cprint(f"  - [{chain.upper()}] {token['symbol']}: {token['dexscreener_link']}", "cyan")
        return results
    else:
        cprint(f"\n No tokens matched screening criteria on {chain.upper()}", "yellow")
        return []


def run_screening_cycle():
    """
    Run one complete screening cycle across all enabled chains
    Collects all results first, then runs AI analysis on combined results
    """
    cprint("\n" + "="*60, "cyan")
    cprint(" Moon Dev's Multi-Chain Token Screener", "white", "on_cyan")
    cprint(f" Chains: {', '.join(c.upper() for c in ENABLED_CHAINS)}", "cyan")
    cprint(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
    cprint("="*60, "cyan")

    all_results = {}
    total_tokens = 0

    # ============================================
    # Phase 1: Screen all chains and collect results
    # ============================================
    cprint("\n" + "="*60, "yellow")
    cprint(" PHASE 1: Screening All Chains", "white", "on_yellow")
    cprint("="*60, "yellow")

    for chain in ENABLED_CHAINS:
        results = run_single_chain(chain)
        all_results[chain] = results
        total_tokens += len(results)

        # Small delay between chains to avoid rate limiting
        if chain != ENABLED_CHAINS[-1]:
            cprint(f"\n Waiting 2 seconds before next chain...", "yellow")
            time.sleep(2)

    # Screening summary
    cprint("\n" + "="*60, "green")
    cprint(" SCREENING COMPLETE", "white", "on_green")
    cprint("="*60, "green")

    for chain, results in all_results.items():
        cprint(f" {chain.upper()}: {len(results)} tokens found", "cyan")

    cprint(f"\n Total: {total_tokens} tokens across {len(ENABLED_CHAINS)} chains", "green", attrs=["bold"])

    if total_tokens == 0:
        if FEISHU_ENABLED:
            chains_str = ", ".join(c.upper() for c in ENABLED_CHAINS)
            send_text(f"Token Screener: No tokens matched on {chains_str} at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return total_tokens

    # ============================================
    # Phase 2: AI Swarm Analysis
    # ============================================
    cprint("\n" + "="*60, "yellow")
    cprint(" PHASE 2: AI Swarm Analysis", "white", "on_yellow")
    cprint("="*60, "yellow")

    # Combine all tokens from all chains
    all_tokens_combined = []
    for chain, results in all_results.items():
        all_tokens_combined.extend(results)

    cprint(f"\n {len(all_tokens_combined)} tokens to analyze with AI Swarm", "cyan", attrs=["bold"])

    # Show breakdown by chain
    chain_counts = {}
    for token in all_tokens_combined:
        chain = token.get('chain', 'unknown')
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
    for chain, count in chain_counts.items():
        cprint(f"   - {chain.upper()}: {count} tokens", "white")

    # Run swarm analysis on all tokens
    swarm_result = run_swarm_analysis(all_tokens_combined)

    # Display results
    if swarm_result:
        display_swarm_results(swarm_result)

    # Send to Feishu
    send_feishu_report(all_tokens_combined, swarm_result)

    return total_tokens


def main():
    """
    Main entry point - supports both single run and continuous mode
    """
    import sys

    # Check for --once flag to force single run
    single_run = "--once" in sys.argv or not CONTINUOUS_MODE

    if single_run:
        cprint("\n" + "="*60, "yellow")
        cprint(" SINGLE RUN MODE", "white", "on_yellow")
        cprint("="*60, "yellow")
        run_screening_cycle()
    else:
        # Continuous mode
        cprint("\n" + "="*60, "green")
        cprint(" CONTINUOUS MODE - Token Screener", "white", "on_green")
        cprint(f" Running every {RUN_INTERVAL_MINUTES} minutes", "green")
        cprint(f" Chains: {', '.join(c.upper() for c in ENABLED_CHAINS)}", "green")
        cprint(" Press Ctrl+C to stop", "yellow")
        cprint("="*60, "green")

        run_count = 0

        try:
            while True:
                run_count += 1
                cprint(f"\n{'#'*60}", "magenta")
                cprint(f" RUN #{run_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "white", "on_magenta")
                cprint(f"{'#'*60}", "magenta")

                # Run screening cycle
                total_tokens = run_screening_cycle()

                # Calculate next run time
                next_run = datetime.now() + timedelta(minutes=RUN_INTERVAL_MINUTES)
                cprint(f"\n Run #{run_count} complete. Found {total_tokens} tokens.", "green")
                cprint(f" Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
                cprint(f" Sleeping for {RUN_INTERVAL_MINUTES} minutes...", "yellow")

                # Sleep until next run
                time.sleep(RUN_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n\n" + "="*60, "yellow")
            cprint(" Token Screener stopped by user", "white", "on_yellow")
            cprint(f" Total runs completed: {run_count}", "cyan")
            cprint("="*60, "yellow")


if __name__ == "__main__":
    main()
