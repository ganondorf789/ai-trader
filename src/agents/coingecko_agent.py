from pathlib import Path

"""
🌙 Moon Dev's Birdeye Agent 🦅
Provides comprehensive access to Birdeye API data and market intelligence for Solana tokens

=================================
📚 FILE OVERVIEW & DOCUMENTATION
=================================

This file implements a multi-agent AI trading system that analyzes crypto markets using CoinGecko data.
The system consists of three specialized agents working together:

1. Agent One (Technical Analysis) 📊
   - Focuses on charts, patterns, and technical indicators
   - Uses shorter-term analysis for trading opportunities
   - Configured with AGENT_ONE_MODEL and AGENT_ONE_MAX_TOKENS

2. Agent Two (Fundamental Analysis) 🌍
   - Analyzes macro trends and fundamental data
   - Provides longer-term market perspective
   - Configured with AGENT_TWO_MODEL and AGENT_TWO_MAX_TOKENS

3. Token Extractor Agent 🔍
   - Monitors agent conversations
   - Extracts mentioned tokens/symbols
   - Maintains historical token discussion data
   - Uses minimal tokens/temperature for precise extraction

Key Components:
--------------
1. Configuration Section
   - Model selection for each agent
   - Response length control (max_tokens)
   - Creativity control (temperature)
   - Round timing configuration

2. Memory System
   - Stores agent conversations in JSON files
   - Maintains token discussion history in CSV
   - Keeps track of last 50 rounds
   - Auto-cleans old memory files

3. Birdeye API Integration
   - Comprehensive Solana token market data access
   - Rate limiting and error handling
   - Multiple endpoints (prices, trends, history, OHLCV)

4. Game Loop Structure
   - Runs in continuous rounds
   - Each round:
     a. Fetch fresh market data
     b. Agent One analyzes
     c. Agent Two responds
     d. Extract mentioned tokens
     e. Generate round synopsis
     f. Wait for next round

5. Output Formatting
   - Colorful terminal output
   - Clear section headers
   - Structured agent responses
   - Easy-to-read summaries

File Structure:
--------------
1. Configuration & Constants
2. Helper Functions (print_banner, print_section)
3. Core Classes:
   - AIAgent: Base agent functionality
   - BirdeyeAPI: API wrapper for Solana token data
   - TokenExtractorAgent: Symbol extraction
   - MultiAgentSystem: Orchestrates everything

Usage:
------
1. Ensure environment variables are set:
   - ANTHROPIC_KEY
   - BIRDEYE_API_KEY

2. Run the file directly:
   python src/agents/birdeye_agent.py

3. Or import the classes:
   from agents.birdeye_agent import MultiAgentSystem

Configuration:
-------------
Adjust the constants at the top of the file to:
- Change agent models
- Modify response lengths
- Control creativity levels
- Adjust round timing

Memory Files:
------------
- src/data/agent_memory/agent_one.json
- src/data/agent_memory/agent_two.json
- src/data/agent_discussed_tokens.csv

Author: Moon Dev 🌙
"""

# API Base URLs for different providers
DEEPSEEK_BASE_URL = "https://api.deepseek.com"  # DeepSeek API
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen3 Max API
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # GLM-4.6 API

# 🤖 Agent Prompts & Personalities
AGENT_ONE_PROMPT = """
You are Agent One - The Technical Analysis Expert 📊
Your role is to analyze charts, patterns, and market indicators to identify trading opportunities.

Focus on:
- Price action and chart patterns
- Technical indicators (RSI, MACD, etc.)
- Volume analysis
- Support/resistance levels
- Short to medium-term opportunities

Remember to be specific about entry/exit points and always consider Moon Dev's risk management rules! 🎯
"""

AGENT_TWO_PROMPT = """
You are Agent Two - The Fundamental Analysis Expert 🌍
Your role is to analyze macro trends, project fundamentals, and long-term potential.

Focus on:
- Project fundamentals and technology
- Team and development activity
- Market trends and sentiment
- Competitor analysis
- Long-term growth potential

Always consider the bigger picture and help guide Moon Dev's long-term strategy! 🚀
"""

TOKEN_EXTRACTOR_PROMPT = """
You are the Token Extraction Agent 🔍
Your role is to identify and extract all cryptocurrency symbols and tokens mentioned in conversations.

Rules:
- Extract both well-known (BTC, ETH) and newer tokens
- Include tokens mentioned by name or symbol
- Format as a clean list of symbols
- Be thorough but avoid duplicates
- When only a name is given, provide the symbol

Keep Moon Dev's token tracking clean and organized! 📝
"""

SYNOPSIS_AGENT_PROMPT = """
You are the Round Synopsis Agent 📊
Your role is to create clear, concise summaries of trading discussions.

Guidelines:
- Summarize key points in 1-2 sentences
- Focus on actionable decisions
- Highlight agreement between agents
- Note significant market observations
- Track progress toward the $10M goal

Help Moon Dev keep track of the trading journey! 🎯
"""

# 🤖 Agent Model Selection (Fixed assignments)
AGENT_ONE_MODEL = "deepseek-chat"  # Agent One: DeepSeek only
AGENT_TWO_MODEL = "qwen-max"  # Agent Two: Qwen3 Max only
TOKEN_EXTRACTOR_MODEL = "glm-4"  # Token Extractor: GLM-4.6 only

# 🎮 Game Configuration
MINUTES_BETWEEN_ROUNDS = 30  # Time to wait between trading rounds (in minutes)

# 🔧 Agent Response Configuration
# Max Tokens (Controls response length):
AGENT_ONE_MAX_TOKENS = 1000    # Technical analysis needs decent space (500-1000 words)
AGENT_TWO_MAX_TOKENS = 1000    # Fundamental analysis might need more detail (600-1200 words)
EXTRACTOR_MAX_TOKENS = 100     # Keep it brief, just token lists (50-100 words)
SYNOPSIS_MAX_TOKENS = 100      # Brief round summaries (50-100 words)

# Temperature (Controls response creativity/randomness):
AGENT_ONE_TEMP = 0.7    # Balanced creativity for technical analysis (0.5-0.8)
AGENT_TWO_TEMP = 0.7    # Balanced creativity for fundamental analysis (0.5-0.8)
EXTRACTOR_TEMP = 0      # Zero creativity, just extract tokens (always 0)
SYNOPSIS_TEMP = 0.3     # Low creativity for consistent summaries (0.2-0.4)

# Token Log File
TOKEN_LOG_FILE = Path("src/data/agent_discussed_tokens.csv")

# Available Models:
# - claude-3-opus-20240229    (Most powerful, longest responses)
# - claude-3-sonnet-20240229  (Balanced performance)
# - claude-3-haiku-20240307   (Fastest, shorter responses)
# - claude-2.1                (Previous generation)
# - claude-2.0                (Previous generation)

"""
Response Length Guide (max_tokens):
50-100:   Ultra concise, bullet points
100-200:  Short paragraphs
500-800:  Detailed explanation
1000+:    In-depth analysis

Temperature Guide:
0.0:  Deterministic, same response every time
0.3:  Very focused, minimal variation
0.7:  Creative but stays on topic
1.0:  Maximum creativity/variation
"""

"""
SYSTEM GOAL:
Two AI agents collaborate to grow a $10,000 portfolio to $10,000,000 using Birdeye's
comprehensive Solana token data. They analyze market trends, identify opportunities, and make
strategic decisions together while maintaining a conversation log in the data folder.

Agent One: Technical Analysis Expert 📊
Agent Two: Fundamental/Macro Analysis Expert 🌍
"""


import os
import requests
import pandas as pd
import json
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from termcolor import colored, cprint
from pathlib import Path
import openai

# Local imports
from src.config import *

# Load environment variables
load_dotenv()

def print_banner():
    """Print a fun colorful banner"""
    cprint("\n" + "="*70, "white", "on_blue")
    cprint("🌙 🎮 Moon Dev's Solana Trading Game (Birdeye) 🎮 🌙", "white", "on_magenta", attrs=["bold"])
    cprint("="*70 + "\n", "white", "on_blue")

def print_section(title: str, color: str = "on_blue"):
    """Print a section header"""
    cprint(f"\n{'='*35}", "white", color)
    cprint(f" {title} ", "white", color, attrs=["bold"])
    cprint(f"{'='*35}\n", "white", color)

# Create data directory for agent memory in the correct project structure
AGENT_MEMORY_DIR = Path("src/data/agent_memory")
AGENT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_old_memory_files():
    """Clean up old memory files from previous naming conventions"""
    old_files = ['haiku_memory.json', 'sonnet_memory.json']
    for file in old_files:
        try:
            old_file = AGENT_MEMORY_DIR / file
            if old_file.exists():
                old_file.unlink()
                cprint(f"🧹 Cleaned up old memory file: {file}", "white", "on_blue")
        except Exception as e:
            cprint(f"⚠️ Error cleaning up {file}: {e}", "white", "on_yellow")

print(f"📁 Agent memory directory: {AGENT_MEMORY_DIR}")
cleanup_old_memory_files()  # Clean up old files on startup

class AIAgent:
    """Individual AI Agent for collaborative decision making"""

    def __init__(self, name: str, model: str = None):
        self.name = name
        self.model = model or AI_MODEL

        # Initialize client based on model (Agent One=DeepSeek, Agent Two=Qwen)
        if "deepseek" in self.model.lower():
            api_key = os.getenv("DEEPSEEK_KEY")
            if not api_key:
                raise ValueError("🚨 DEEPSEEK_KEY not found in environment variables!")
            self.client = openai.OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
            print(f"🚀 {name} using DeepSeek model: {model}")
        elif "qwen" in self.model.lower():
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("🚨 DASHSCOPE_API_KEY not found in environment variables!")
            self.client = openai.OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
            print(f"🌟 {name} using Qwen3 Max model: {model}")
            
        # Use a simpler memory file name
        self.memory_file = AGENT_MEMORY_DIR / f"{name.lower().replace(' ', '_')}.json"
        self.load_memory()
        
    def load_memory(self):
        """Load agent's memory from file"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {
                'conversations': [],
                'decisions': [],
                'portfolio_history': []
            }
            self.save_memory()
            
    def save_memory(self):
        """Save agent's memory to file"""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=2)
            
    def think(self, market_data: Dict, other_agent_message: str = None) -> str:
        """Process market data and other agent's message to make decisions"""
        try:
            print_section(f"🤔 {self.name} is thinking...", "on_magenta")
            
            # Get the right configuration based on agent name
            max_tokens = AGENT_ONE_MAX_TOKENS if self.name == "Agent One" else AGENT_TWO_MAX_TOKENS
            temperature = AGENT_ONE_TEMP if self.name == "Agent One" else AGENT_TWO_TEMP
            prompt = AGENT_ONE_PROMPT if self.name == "Agent One" else AGENT_TWO_PROMPT
            
            # Add market data context
            market_context = f"""
Current Market Data:
{json.dumps(market_data, indent=2)}

Previous Agent Message:
{other_agent_message if other_agent_message else 'No previous message'}

Remember to format your response like this:

🤖 Hey Moon Dev! {self.name} here!
=================================

📊 Market Vibes:
[Your main market thoughts in simple terms]

💡 Opportunities I See:
- [Opportunity 1]
- [Opportunity 2]
- [Opportunity 3]

🎯 My Recommendations:
1. [Clear action item]
2. [Clear action item]
3. [Clear action item]

💰 Portfolio Impact:
[How this helps reach our $10M goal]

🌙 Moon Dev Wisdom:
[Fun reference to Moon Dev's trading style]
"""
            
            # Get AI response (OpenAI-compatible API for DeepSeek/Qwen)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": market_context}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            response_text = response.choices[0].message.content
            
            # Clean up the response
            response = (response_text
                .replace("TextBlock(text='", "")
                .replace("')", "")
                .replace("\\n", "\n")
                .replace("*", "")
                .replace("```", "")
                .strip())
            
            # Add extra newlines between sections for readability
            sections = ["Market Vibes:", "Opportunities I See:", "My Recommendations:", "Portfolio Impact:", "Moon Dev Wisdom:"]
            for section in sections:
                response = response.replace(section, f"\n{section}\n")
            
            # Save to memory
            self.memory['conversations'].append({
                'timestamp': datetime.now().isoformat(),
                'market_data': market_data,
                'other_message': other_agent_message,
                'response': response
            })
            self.save_memory()
            
            return response
            
        except Exception as e:
            cprint(f"❌ Error in agent thinking: {str(e)}", "white", "on_red")
            return f"Error processing market data: {str(e)}"

class BirdeyeAPI:
    """Utility class for Birdeye API calls 🦅"""

    # Default Solana token addresses for major coins
    DEFAULT_TOKENS = {
        'SOL': 'So11111111111111111111111111111111111111112',
        'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
        'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
        'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
        'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
        'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
    }

    def __init__(self):
        self.api_key = os.getenv("BIRDEYE_API_KEY")
        if not self.api_key:
            print("⚠️ Warning: BIRDEYE_API_KEY not found in environment variables!")
        self.base_url = "https://public-api.birdeye.so"
        self.headers = {
            "X-API-KEY": self.api_key,
            "x-chain": "solana"
        }
        print("🦅 Moon Dev's Birdeye API initialized!")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 429:
                print("⚠️ Rate limit hit! Waiting before retry...")
                time.sleep(60)  # Wait 60 seconds before retry
                return self._make_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {str(e)}")
            return {}

    def get_ping(self) -> bool:
        """Check API server status by making a simple request"""
        try:
            response = self._make_request("defi/price", {'address': self.DEFAULT_TOKENS['SOL']})
            return response.get('success', False)
        except:
            return False

    def get_price(self, addresses: Union[str, List[str]]) -> Dict:
        """Get current price data for tokens

        Args:
            addresses: Token address(es) (Solana mint addresses)
        """
        if isinstance(addresses, str):
            addresses = [addresses]

        results = {}
        for address in addresses:
            print(f"🔍 Getting price for: {address[:8]}...")
            response = self._make_request("defi/price", {'address': address})
            if response.get('success') and response.get('data'):
                results[address] = response['data']

        return results

    def get_multi_price(self, addresses: List[str]) -> Dict:
        """Get prices for multiple tokens at once

        Args:
            addresses: List of token addresses
        """
        print(f"🔍 Getting prices for {len(addresses)} tokens...")
        params = {'list_address': ','.join(addresses)}
        return self._make_request("defi/multi_price", params)

    def get_token_overview(self, address: str) -> Dict:
        """Get comprehensive token overview data

        Args:
            address: Token address (Solana mint address)
        """
        print(f"📊 Getting token overview for {address[:8]}...")
        return self._make_request("defi/token_overview", {'address': address})

    def get_trending(self) -> List[Dict]:
        """Get trending tokens on Solana"""
        print("🔥 Getting trending tokens...")
        response = self._make_request("defi/token_trending")
        if response.get('success') and response.get('data'):
            return response['data'].get('items', [])
        return []

    def get_token_list(self, sort_by: str = 'v24hUSD', sort_type: str = 'desc', limit: int = 50) -> List[Dict]:
        """Get list of tokens sorted by various metrics

        Args:
            sort_by: Sort field (v24hUSD, mc, etc.)
            sort_type: 'asc' or 'desc'
            limit: Number of tokens to return
        """
        print(f"📋 Getting top {limit} tokens by {sort_by}...")
        params = {
            'sort_by': sort_by,
            'sort_type': sort_type,
            'offset': 0,
            'limit': limit
        }
        response = self._make_request("defi/tokenlist", params)
        if response.get('success') and response.get('data'):
            return response['data'].get('tokens', [])
        return []

    def get_token_security(self, address: str) -> Dict:
        """Get token security information

        Args:
            address: Token address
        """
        print(f"🔒 Getting security info for {address[:8]}...")
        return self._make_request("defi/token_security", {'address': address})

    def get_history_price(self, address: str, address_type: str = 'token',
                          time_type: str = '24h') -> Dict:
        """Get historical price data

        Args:
            address: Token address
            address_type: 'token' or 'pair'
            time_type: '24h', '7d', '30d', etc.
        """
        print(f"📈 Getting {time_type} price history for {address[:8]}...")
        params = {
            'address': address,
            'address_type': address_type,
            'type': time_type
        }
        return self._make_request("defi/history_price", params)

    def get_ohlcv(self, address: str, timeframe: str = '15m',
                  time_from: Optional[int] = None, time_to: Optional[int] = None) -> Dict:
        """Get OHLCV candlestick data

        Args:
            address: Token address
            timeframe: Candle interval (1m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M)
            time_from: Unix timestamp start
            time_to: Unix timestamp end
        """
        if time_to is None:
            time_to = int(time.time())
        if time_from is None:
            time_from = time_to - (7 * 24 * 60 * 60)  # Default 7 days

        print(f"📊 Getting OHLCV data for {address[:8]}...")
        params = {
            'address': address,
            'type': timeframe,
            'time_from': time_from,
            'time_to': time_to
        }
        return self._make_request("defi/ohlcv", params)

    def get_trades(self, address: str, limit: int = 50) -> List[Dict]:
        """Get recent trades for a token

        Args:
            address: Token address
            limit: Number of trades to return
        """
        print(f"💱 Getting recent trades for {address[:8]}...")
        params = {
            'address': address,
            'limit': limit
        }
        response = self._make_request("defi/txs/token", params)
        if response.get('success') and response.get('data'):
            return response['data'].get('items', [])
        return []

    def get_price_volume(self, address: str) -> Dict:
        """Get price and volume data for a token

        Args:
            address: Token address
        """
        print(f"📈 Getting price/volume for {address[:8]}...")
        return self._make_request("defi/price_volume/single", {'address': address})

class TokenExtractorAgent:
    """Agent that extracts token/crypto symbols from conversations (GLM-4.6 only)"""

    def __init__(self):
        self.model = TOKEN_EXTRACTOR_MODEL
        # Initialize GLM client
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("🚨 ZHIPU_API_KEY not found in environment variables!")
        self.client = openai.OpenAI(api_key=api_key, base_url=GLM_BASE_URL)
        print(f"🔮 Token Extractor using GLM-4.6 model: {self.model}")
        self.token_history = self._load_token_history()
        cprint("🔍 Token Extractor Agent initialized!", "white", "on_cyan")
        
    def _load_token_history(self) -> pd.DataFrame:
        """Load or create token history DataFrame"""
        if TOKEN_LOG_FILE.exists():
            return pd.read_csv(TOKEN_LOG_FILE)
        else:
            df = pd.DataFrame(columns=['timestamp', 'round', 'token', 'context'])
            df.to_csv(TOKEN_LOG_FILE, index=False)
            return df
            
    def extract_tokens(self, round_num: int, agent_one_msg: str, agent_two_msg: str) -> List[Dict]:
        """Extract tokens/symbols from agent messages"""
        try:
            print_section("🔍 Extracting Mentioned Tokens", "on_cyan")

            user_content = f"""
Agent One said:
{agent_one_msg}

Agent Two said:
{agent_two_msg}

Extract all token symbols and return as a simple list.
"""
            # Use GLM API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TOKEN_EXTRACTOR_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=EXTRACTOR_MAX_TOKENS,
                temperature=EXTRACTOR_TEMP
            )
            response_text = response.choices[0].message.content

            # Clean up response and split into list
            tokens = response_text.strip().split('\n')
            tokens = [t.strip().upper() for t in tokens if t.strip()]
            
            # Create records for each token
            timestamp = datetime.now().isoformat()
            records = []
            for token in tokens:
                records.append({
                    'timestamp': timestamp,
                    'round': round_num,
                    'token': token,
                    'context': f"Round {round_num} discussion"
                })
                
            # Log to DataFrame
            new_records = pd.DataFrame(records)
            self.token_history = pd.concat([self.token_history, new_records], ignore_index=True)
            self.token_history.to_csv(TOKEN_LOG_FILE, index=False)
            
            # Print extracted tokens
            cprint("\n📝 Tokens Mentioned This Round:", "white", "on_cyan")
            for token in tokens:
                cprint(f"• {token}", "white", "on_cyan")
            
            return records
            
        except Exception as e:
            cprint(f"❌ Error extracting tokens: {str(e)}", "white", "on_red")
            return []

class MultiAgentSystem:
    """System managing multiple AI agents analyzing Birdeye Solana data"""

    def __init__(self):
        print_banner()
        self.api = BirdeyeAPI()
        self.agent_one = AIAgent("Agent One", AGENT_ONE_MODEL)
        self.agent_two = AIAgent("Agent Two", AGENT_TWO_MODEL)
        self.token_extractor = TokenExtractorAgent()
        self.round_history = []  # Store round synopses
        self.max_history_rounds = 50  # Keep last 50 rounds of context
        cprint("🎮 Moon Dev's Trading Game System Ready! 🎮", "white", "on_green", attrs=["bold"])
        
    def generate_round_synopsis(self, agent_one_response: str, agent_two_response: str) -> str:
        """Generate a brief synopsis of the round's key points using Synopsis Agent"""
        try:
            user_content = f"""
Agent One said:
{agent_one_response}

Agent Two said:
{agent_two_response}

Create a brief synopsis of this trading round.
"""
            # Use Agent One's DeepSeek client for synopsis
            response = self.agent_one.client.chat.completions.create(
                model=self.agent_one.model,
                messages=[
                    {"role": "system", "content": SYNOPSIS_AGENT_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=SYNOPSIS_MAX_TOKENS,
                temperature=SYNOPSIS_TEMP
            )
            synopsis = response.choices[0].message.content.strip()

            return synopsis

        except Exception as e:
            cprint(f"⚠️ Error generating synopsis: {e}", "white", "on_yellow")
            return "Synopsis generation failed"
    
    def get_recent_history(self) -> str:
        """Get formatted string of recent round synopses"""
        if not self.round_history:
            return "No previous rounds yet."
            
        history = "\n".join([
            f"Round {i+1}: {synopsis}"
            for i, synopsis in enumerate(self.round_history[-10:])  # Show last 10 rounds
        ])
        return f"\n📜 Recent Trading History:\n{history}\n"
        
    def run_conversation_cycle(self):
        """Run one cycle of agent conversation"""
        try:
            print_section("🔄 Starting New Trading Round!", "on_blue")

            # Get fresh market data from Birdeye
            cprint("📊 Gathering Solana Market Intelligence...", "white", "on_magenta")
            market_data = {
                'trending': self.api.get_trending(),
                'top_tokens': self.api.get_token_list(sort_by='v24hUSD', limit=20),
                'sol': self.api.get_token_overview(BirdeyeAPI.DEFAULT_TOKENS['SOL']),
                'bonk': self.api.get_token_overview(BirdeyeAPI.DEFAULT_TOKENS['BONK']),
                'jup': self.api.get_token_overview(BirdeyeAPI.DEFAULT_TOKENS['JUP']),
            }
            
            # Add round history to market context
            market_data['recent_history'] = self.get_recent_history()
            
            # Agent One starts the conversation
            print_section("🤖 Agent One's Analysis", "on_blue")
            agent_one_response = self.agent_one.think(market_data)
            print(agent_one_response)
            
            # Agent Two responds
            print_section("🤖 Agent Two's Response", "on_magenta")
            agent_two_response = self.agent_two.think(market_data, agent_one_response)
            print(agent_two_response)
            
            # Extract tokens from conversation
            self.token_extractor.extract_tokens(
                len(self.round_history) + 1,
                agent_one_response,
                agent_two_response
            )
            
            # Generate and store round synopsis
            synopsis = self.generate_round_synopsis(agent_one_response, agent_two_response)
            self.round_history.append(synopsis)
            
            # Keep only last N rounds
            if len(self.round_history) > self.max_history_rounds:
                self.round_history = self.round_history[-self.max_history_rounds:]
            
            # Print round synopsis
            print_section("📝 Round Synopsis", "on_green")
            cprint(synopsis, "white", "on_green")
            
            cprint("\n🎯 Trading Round Complete! 🎯", "white", "on_green", attrs=["bold"])
            
        except Exception as e:
            cprint(f"\n❌ Error in trading round: {str(e)}", "white", "on_red")

def main():
    """Main function to run the multi-agent system"""
    print_banner()
    cprint("🎮 Welcome to Moon Dev's Solana Trading Game! 🎮", "white", "on_magenta", attrs=["bold"])
    cprint("Two AI agents analyze Solana tokens via Birdeye to turn $10,000 into $10,000,000!", "white", "on_blue")
    cprint("Let the trading begin! 🚀\n", "white", "on_green", attrs=["bold"])
    
    system = MultiAgentSystem()
    
    try:
        round_number = 1
        while True:
            print_section(f"🎮 Round {round_number} 🎮", "on_blue")
            system.run_conversation_cycle()
            next_round_time = datetime.now() + timedelta(minutes=MINUTES_BETWEEN_ROUNDS)
            cprint(f"\n⏳ Next round starts in {MINUTES_BETWEEN_ROUNDS} minutes (at {next_round_time.strftime('%H:%M:%S')})...", 
                  "white", "on_magenta")
            time.sleep(MINUTES_BETWEEN_ROUNDS * 60)  # Convert minutes to seconds
            round_number += 1
            
    except KeyboardInterrupt:
        cprint("\n👋 Thanks for playing Moon Dev's Trading Game! 🌙", "white", "on_magenta", attrs=["bold"])
    except Exception as e:
        cprint(f"\n❌ Game Error: {str(e)}", "white", "on_red")

if __name__ == "__main__":
    main()