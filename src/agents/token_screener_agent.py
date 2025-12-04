"""
Moon Dev's Token Screener Agent (V3 - With Swarm Analysis & Feishu)

Screens Solana tokens using Birdeye Token List V3 API for maximum efficiency.
Most data comes from a single API call, only OHLCV needs separate requests.
Found tokens are analyzed by AI Swarm and results sent to Feishu.

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
MIN_1H_VOLUME_PERCENT = 0  # Set to 0 to disable (default), 8-10 to enable

# How many tokens to fetch (API returns up to 100 per call)
TOKEN_FETCH_LIMIT = 100

# Sleep between API calls to avoid rate limiting (seconds)
API_SLEEP = 0.3

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
]

# ============================================================================
# MULTI-CHAIN SETTINGS - Run screening on multiple chains sequentially
# ============================================================================
ENABLED_CHAINS = ["solana", "ethereum"]  # Chains to screen (in order)
# Supported chains: solana, ethereum, arbitrum, avalanche, bsc, optimism, polygon, base, zksync, sui

# ============================================================================
# FEISHU & SWARM SETTINGS
# ============================================================================
FEISHU_ENABLED = True  # Set to False to disable Feishu notifications
SWARM_ENABLED = True   # Set to False to skip AI analysis

# ============================================================================
# CACHE SETTINGS - Skip tokens analyzed within this time window
# ============================================================================
CACHE_EXPIRY_HOURS = 1  # Don't re-analyze tokens within this time window

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_FOLDER = Path(__file__).parent.parent / "data" / "screener_agent"
RESULTS_FILE = DATA_FOLDER / "screened_tokens.csv"
HISTORY_FILE = DATA_FOLDER / "screening_history.csv"
CACHE_FILE = DATA_FOLDER / "analysis_cache.json"


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
            "chain": self.chain,
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


# ============================================================================
# CACHE MANAGEMENT - Skip recently analyzed tokens
# ============================================================================

def load_analysis_cache():
    """Load the analysis cache from file"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
            return cache
        except Exception as e:
            cprint(f" Error loading cache: {e}", "yellow")
    return {}


def save_analysis_cache(cache):
    """Save the analysis cache to file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        cprint(f" Error saving cache: {e}", "yellow")


def is_token_cached(address, cache):
    """Check if token was analyzed within CACHE_EXPIRY_HOURS"""
    if address not in cache:
        return False

    cached_time = cache[address].get("analyzed_at")
    if not cached_time:
        return False

    try:
        cached_datetime = datetime.fromisoformat(cached_time)
        expiry_time = cached_datetime + timedelta(hours=CACHE_EXPIRY_HOURS)
        if datetime.now() < expiry_time:
            return True
    except Exception:
        pass

    return False


def update_cache_for_tokens(tokens, cache):
    """Update cache with newly analyzed tokens"""
    now = datetime.now().isoformat()
    for token in tokens:
        address = token.get("address")
        if address:
            cache[address] = {
                "symbol": token.get("symbol"),
                "analyzed_at": now,
                "price": token.get("current_price"),
                "market_cap": token.get("market_cap")
            }
    return cache


def filter_uncached_tokens(tokens):
    """Filter out tokens that were analyzed within CACHE_EXPIRY_HOURS"""
    cache = load_analysis_cache()

    uncached = []
    cached_count = 0

    for token in tokens:
        address = token.get("address")
        if is_token_cached(address, cache):
            cached_count += 1
            cprint(f" {token.get('symbol')}: Skipping (cached within {CACHE_EXPIRY_HOURS}h)", "yellow")
        else:
            uncached.append(token)

    if cached_count > 0:
        cprint(f"\n Skipped {cached_count} cached tokens, {len(uncached)} new tokens to analyze", "cyan")

    return uncached, cache


# ============================================================================
# SWARM ANALYSIS
# ============================================================================

def create_analysis_prompt(tokens):
    """Create prompt for swarm agents to analyze screened tokens"""

    prompt = """You are analyzing tokens that passed a strict screening filter for potential buying opportunities.

All tokens below have already passed these criteria:
- Liquidity >= $800K (hard to rug)
- 24h Volume >= $10M (active trading)
- Holders >= 4,000 (good distribution)
- Price within 12% of 3-day low (near bottom)
- Pool age >= 12 hours (not brand new)
- No extreme price drops (not rugging)

Here are the screened tokens:

"""

    for i, token in enumerate(tokens, 1):
        prompt += f"\n{i}. **{token['symbol']}** ({token['name']})\n"
        prompt += f"   - Price: ${token['current_price']:.8f}\n"
        prompt += f"   - Market Cap: ${token.get('market_cap', 0):,.0f}\n"
        prompt += f"   - 24h Volume: ${token['volume_24h_usd']:,.0f}\n"
        prompt += f"   - 1h Volume: ${token['volume_1h_usd']:,.0f} ({token['volume_1h_percent']}% of 24h)\n"
        prompt += f"   - Liquidity: ${token['liquidity_usd']:,.0f}\n"
        prompt += f"   - Holders: {token['holder_count']:,}\n"
        prompt += f"   - 24h Price Change: {token['price_change_24h']}%\n"
        prompt += f"   - Price vs 3-Day Low: {token['price_vs_3d_low']}x\n"
        prompt += f"   - Volume Surge: {'YES' if token['volume_surge'] else 'No'}\n"
        prompt += f"   - Buy/Sell 24h: {token.get('buy_24h', 0)}/{token.get('sell_24h', 0)}\n"
        prompt += f"   - Unique Wallets 24h: {token.get('unique_wallet_24h', 0)}\n"

    prompt += """

Based on the data above, analyze each token and recommend:
1. Which token(s) would you BUY now and why?
2. Which token(s) should be AVOIDED and why?
3. What's your confidence level (1-10) for your top pick?

Focus on: volume momentum, price position vs 3d low, holder distribution, and market cap.
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
        headers = ["#", "Symbol", "Price", "MCap", "Vol24h", "24h%", "vs3dLow"]
        rows = []

        for i, t in enumerate(tokens, 1):
            price = f"${t['current_price']:.6f}" if t['current_price'] < 1 else f"${t['current_price']:.2f}"
            mcap = format_volume(t.get('market_cap', 0))
            vol = format_volume(t['volume_24h_usd'])
            chg = f"{t['price_change_24h']:+.1f}%"
            vs_low = f"{t['price_vs_3d_low']:.3f}x"
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
            content_parts.append(f"- Price vs 3d Low <= {MAX_PRICE_VS_3D_LOW}x")
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
    Run screening for a single chain

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

        # Filter out cached tokens (analyzed within CACHE_EXPIRY_HOURS)
        uncached_tokens, cache = filter_uncached_tokens(results)

        if uncached_tokens:
            cprint(f"\n {len(uncached_tokens)} new {chain.upper()} tokens to analyze with AI Swarm", "cyan", attrs=["bold"])

            # Run swarm analysis only on uncached tokens
            swarm_result = run_swarm_analysis(uncached_tokens)

            # Display results
            if swarm_result:
                display_swarm_results(swarm_result)

            # Send to Feishu
            send_feishu_report(uncached_tokens, swarm_result)

            # Update cache with analyzed tokens
            cache = update_cache_for_tokens(uncached_tokens, cache)
            save_analysis_cache(cache)
            cprint(f" Cache updated with {len(uncached_tokens)} {chain.upper()} tokens", "green")

        else:
            cprint(f"\n All {len(results)} {chain.upper()} tokens were recently analyzed (within {CACHE_EXPIRY_HOURS}h)", "yellow")
            cprint(" Skipping swarm analysis and notifications", "yellow")

        return results

    else:
        cprint(f"\n No tokens matched screening criteria on {chain.upper()}", "yellow")
        return []


def main():
    """
    Main entry point - runs screening on all enabled chains sequentially
    """
    cprint("\n" + "="*60, "cyan")
    cprint(" Moon Dev's Multi-Chain Token Screener", "white", "on_cyan")
    cprint(f" Chains: {', '.join(c.upper() for c in ENABLED_CHAINS)}", "cyan")
    cprint("="*60, "cyan")

    all_results = {}
    total_tokens = 0

    # Run screening for each enabled chain sequentially
    for chain in ENABLED_CHAINS:
        results = run_single_chain(chain)
        all_results[chain] = results
        total_tokens += len(results)

        # Small delay between chains to avoid rate limiting
        if chain != ENABLED_CHAINS[-1]:
            cprint(f"\n Waiting 2 seconds before next chain...", "yellow")
            time.sleep(2)

    # Final summary
    cprint("\n" + "="*60, "green")
    cprint(" MULTI-CHAIN SCREENING COMPLETE", "white", "on_green")
    cprint("="*60, "green")

    for chain, results in all_results.items():
        cprint(f" {chain.upper()}: {len(results)} tokens found", "cyan")

    cprint(f"\n Total: {total_tokens} tokens across {len(ENABLED_CHAINS)} chains", "green", attrs=["bold"])

    if total_tokens == 0 and FEISHU_ENABLED:
        chains_str = ", ".join(c.upper() for c in ENABLED_CHAINS)
        send_text(f"Token Screener: No tokens matched on {chains_str} at {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
