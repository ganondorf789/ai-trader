# 🌙 Moon Dev's Trading Agents Guide

**Complete Onboarding to Moon Dev's AI Trading Agent Ecosystem**

Built with love by Moon Dev 🚀

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Core Trading Agents](#core-trading-agents)
   - [Trading Agent](#1-trading-agent-tradingagentpy)
   - [Strategy Agent](#2-strategy-agent-strategyagentpy)
   - [Risk Agent](#3-risk-agent-riskagentpy)
3. [Market Intelligence Agents](#market-intelligence-agents)
   - [Sentiment Agent](#4-sentiment-agent-sentimentagentpy)
   - [Whale Agent](#5-whale-agent-whaleagentpy)
   - [Funding Agent](#6-funding-agent-fundingagentpy)
   - [Liquidation Agent](#7-liquidation-agent-liquidationagentpy)
   - [Chart Analysis Agent](#8-chart-analysis-agent-chartanalysisagentpy)
4. [Token Discovery Agents](#token-discovery-agents)
   - [Sniper Agent](#9-sniper-agent-sniperagentpy)
   - [Solana Agent](#10-solana-agent-solanaagentpy)
   - [CopyBot Agent](#11-copybot-agent-copybotag entpy)
5. [How Agents Work Together](#how-agents-work-together)
6. [Quick Start Guide](#quick-start-guide)
7. [Configuration](#configuration)

---

## Overview

Moon Dev's trading system uses a **multi-agent architecture** where specialized AI agents work independently or together to analyze markets, generate signals, manage risk, and execute trades.

### 🎯 Agent Categories

| Category | Agents | Purpose |
|----------|--------|---------|
| **Core Trading** | Trading, Strategy, Risk | Execute trades and manage portfolio |
| **Market Intelligence** | Sentiment, Whale, Funding, Liquidation, Chart | Provide market context |
| **Token Discovery** | Sniper, Solana, CopyBot | Find opportunities |

### 🔄 Agent Communication Flow

```
Market Intelligence Agents
    ↓ (Signals/Context)
Core Trading Agents
    ↓ (Decisions)
Risk Agent
    ↓ (Validation)
Execution (via nice_funcs.py)
```

---

## Core Trading Agents

These are the main agents that make trading decisions and execute them.

### 1. Trading Agent (`trading_agent.py`)

**The LLM Decision Maker 🤖**

#### What It Does:
- **Pure AI analysis** of market data using Claude
- No hard-coded rules - LLM decides everything
- Manages portfolio allocation across all tokens
- Analyzes technical indicators (MA20, MA40, RSI, volume)
- Can incorporate strategy signals from other agents

#### How It Works:
```python
# 1. Collect market data for all tokens
market_data = collect_all_tokens()

# 2. LLM analyzes each token
for token in MONITORED_TOKENS:
    analysis = llm.analyze(token, market_data)
    # Returns: BUY/SELL/NOTHING + confidence %

# 3. Handle exits first (close SELL/NOTHING positions)
handle_exits()

# 4. LLM decides portfolio allocation
allocation = llm.allocate_portfolio()
# Returns: {token: usd_amount}

# 5. Execute entries
execute_allocations(allocation)
```

#### Key Methods:
- `analyze_market_data(token, market_data)` - LLM analysis
- `allocate_portfolio()` - LLM decides capital distribution
- `execute_allocations(allocation_dict)` - Executes via `n.ai_entry()`
- `handle_exits()` - Closes positions via `n.chunk_kill()`
- `run_trading_cycle(strategy_signals=None)` - Main loop

#### Configuration:
```python
# src/config.py
AI_MODEL = "claude-3-haiku-20240307"  # LLM model
usd_size = 25  # Portfolio size
MAX_POSITION_PERCENTAGE = 30  # Max per position
CASH_PERCENTAGE = 20  # Minimum cash buffer
MONITORED_TOKENS = [...]  # Tokens to trade
```

#### When to Use:
- ✅ You want AI to make all decisions
- ✅ Market conditions change frequently
- ✅ You prefer adaptive, LLM-driven behavior

#### Standalone Usage:
```bash
python src/agents/trading_agent.py
# Runs every 15 minutes (configurable)
```

---

### 2. Strategy Agent (`strategy_agent.py`)

**The Rule-Based Strategist 🧠**

#### What It Does:
- Loads custom strategies from `src/strategies/custom/`
- Each strategy generates signals: BUY, SELL, NOTHING
- **AI validates** all signals before execution
- Can work independently OR feed signals to Trading Agent

#### How It Works:
```python
# 1. Load custom strategies
strategies = [ExampleStrategy(), MyStrategy()]

# 2. Generate signals for each token
for strategy in strategies:
    signal = strategy.generate_signals(token)
    # Returns: {token, direction, signal_strength, metadata}

# 3. AI validates all signals together
evaluation = llm.evaluate_signals(signals, market_data)
# AI checks for confirmation/contradiction, risk factors

# 4. Execute ONLY approved signals
for signal in approved_signals:
    if signal.direction == 'BUY':
        n.ai_entry(token, target_size)
    elif signal.direction == 'SELL':
        n.chunk_kill(token, max_order_size, slippage)
```

#### Key Methods:
- `get_signals(token)` - Collect + validate + execute signals
- `evaluate_signals(signals, market_data)` - LLM validation
- `execute_strategy_signals(approved_signals)` - Trade execution
- `combine_with_portfolio(signals, portfolio)` - Portfolio integration

#### Creating Custom Strategies:
```python
# src/strategies/custom/my_strategy.py
class MyStrategy(BaseStrategy):
    name = "MyStrategy"

    def generate_signals(self):
        # Your logic here (RSI, MA crossovers, etc.)
        return {
            "token": "address",
            "signal": 0.8,  # 0-1 confidence
            "direction": "BUY",
            "metadata": {}
        }
```

Then register in `strategy_agent.py`:
```python
from src.strategies.custom.my_strategy import MyStrategy

self.enabled_strategies.extend([
    ExampleStrategy(),
    MyStrategy()  # Add yours!
])
```

#### Configuration:
```python
# src/config.py
ENABLE_STRATEGIES = True
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum signal strength
```

#### When to Use:
- ✅ You have specific trading rules to code
- ✅ You want AI to validate (not decide)
- ✅ You need deterministic behavior
- ✅ Example: "Buy when RSI < 30 AND volume spike"

#### Standalone Usage:
```bash
# Modify strategy_agent.py to run standalone
python src/agents/strategy_agent.py
```

---

### 3. Risk Agent (`risk_agent.py`)

**The Safety Guardian 🛡️**

#### What It Does:
- **Circuit breaker** - Stops trading when limits hit
- Monitors max loss/gain over configurable timeframe
- AI-powered override decisions for edge cases
- Minimum balance protection
- Runs FIRST before any trading decisions

#### How It Works:
```python
# 1. Check PnL over last N hours
pnl = calculate_pnl_over_hours(MAX_LOSS_GAIN_CHECK_HOURS)

# 2. Compare to limits
if USE_PERCENTAGE:
    limit_breach = pnl_pct < -MAX_LOSS_PERCENT or pnl_pct > MAX_GAIN_PERCENT
else:
    limit_breach = pnl < -MAX_LOSS_USD or pnl > MAX_GAIN_USD

# 3. Check minimum balance
balance = get_usdc_balance()
if balance < MINIMUM_BALANCE_USD:
    if USE_AI_CONFIRMATION:
        decision = llm.should_close_positions(balance, positions, market_data)
    else:
        close_all_positions()

# 4. If limit hit, ask AI for override
if limit_breach:
    decision = llm.should_override_limit(positions, market_data, limit_type)
    if decision == "RESPECT_LIMIT":
        block_trading()
        close_positions_if_needed()
```

#### AI Override Logic:

**For Max Loss:**
- AI is **EXTREMELY conservative**
- Only overrides if strong reversal signals (90%+ confidence)
- All positions must show reversal potential

**For Max Gain:**
- AI can be **more lenient**
- Looks for continued momentum (60%+ confidence)
- Most positions should show upward momentum

#### Key Methods:
- `check_risk_limits()` - Main risk check
- `should_override_limit(limit_type, position_data)` - AI decision
- `close_positions()` - Emergency position closing
- `run()` - Runs risk checks every cycle

#### Configuration:
```python
# src/config.py
# Choose percentage or USD-based limits
USE_PERCENTAGE = False

# USD-based (if USE_PERCENTAGE = False)
MAX_LOSS_USD = 25
MAX_GAIN_USD = 25
MINIMUM_BALANCE_USD = 50

# Percentage-based (if USE_PERCENTAGE = True)
MAX_LOSS_PERCENT = 5  # 5% loss
MAX_GAIN_PERCENT = 5  # 5% gain

# How far back to check
MAX_LOSS_GAIN_CHECK_HOURS = 12

# AI confirmation for balance breach
USE_AI_CONFIRMATION = True  # False = close immediately
```

#### When to Use:
- ✅ Always! Risk Agent should run FIRST in main loop
- ✅ You want automated risk management
- ✅ You need circuit breakers
- ✅ You want AI to handle edge cases

#### Standalone Usage:
```bash
python src/agents/risk_agent.py
```

---

## How Core Agents Work Together

### 🔄 Integration Modes

#### Mode 1: Trading Agent Only (Pure LLM)
```python
from src.agents.trading_agent import TradingAgent

agent = TradingAgent()
agent.run_trading_cycle()  # LLM makes all decisions
```

**Flow:**
```
Market Data → LLM Analysis → Portfolio Allocation → Execute
```

---

#### Mode 2: Strategy Agent Only (Rules + AI Validation)
```python
from src.agents.strategy_agent import StrategyAgent

agent = StrategyAgent()
for token in MONITORED_TOKENS:
    signals = agent.get_signals(token)
    # Signals auto-execute if validated!
```

**Flow:**
```
Your Strategy → Signal → AI Validates → Execute Trade
```

---

#### Mode 3: Combined (Best of Both Worlds! 🚀)
```python
from src.agents.strategy_agent import StrategyAgent
from src.agents.trading_agent import TradingAgent

strategy_agent = StrategyAgent()
trading_agent = TradingAgent()

# 1. Get validated strategy signals (but don't auto-execute)
all_signals = {}
for token in MONITORED_TOKENS:
    signals = strategy_agent.get_signals(token)
    if signals:
        all_signals[token] = signals

# 2. Pass signals to trading agent as additional context
trading_agent.run_trading_cycle(strategy_signals=all_signals)
```

**Flow:**
```
Your Strategy → Signal → AI Validates
                           ↓
                    (Pass to LLM)
                           ↓
Market Data + Strategy Signals → LLM Analysis → Portfolio Allocation → Execute
```

**Why This Works:**
- Strategy signals act as **additional context** for LLM
- LLM can agree/disagree with strategies
- LLM considers both technical analysis AND strategy recommendations
- Most robust decision-making!

---

#### Mode 4: Full System with Risk Agent (Recommended! 🛡️)
```python
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.trading_agent import TradingAgent

risk_agent = RiskAgent()
strategy_agent = StrategyAgent()
trading_agent = TradingAgent()

while True:
    # ALWAYS run risk agent FIRST!
    risk_status = risk_agent.check_risk_limits()

    if risk_status == "BLOCKED":
        print("🛑 Trading blocked by risk agent")
        continue

    # Collect strategy signals
    all_signals = {}
    for token in MONITORED_TOKENS:
        signals = strategy_agent.get_signals(token)
        if signals:
            all_signals[token] = signals

    # Run trading agent with signals as context
    trading_agent.run_trading_cycle(strategy_signals=all_signals)

    time.sleep(900)  # 15 minutes
```

**Flow:**
```
Risk Agent Check
    ↓ (PASS/BLOCKED)
Strategy Signals → AI Validates
    ↓
Market Data + Signals → LLM Analysis → Portfolio Allocation
    ↓
Risk Agent Final Check → Execute Trades
```

---

## Market Intelligence Agents

These agents provide market context and signals to inform trading decisions.

### 4. Sentiment Agent (`sentiment_agent.py`)

**The Twitter Whisperer 🐦**

#### What It Does:
- Monitors Twitter sentiment for specified tokens
- Uses HuggingFace models for sentiment scoring
- Tracks sentiment over time
- Announces via voice when sentiment shifts dramatically

#### How It Works:
```python
# 1. Fetch tweets for tracked tokens
tweets = fetch_tweets(TOKENS_TO_TRACK, TWEETS_PER_RUN)

# 2. Analyze sentiment using HuggingFace
for tweet in tweets:
    sentiment_score = model.predict(tweet.text)
    # Returns: -1 (bearish) to +1 (bullish)

# 3. Track sentiment history
save_sentiment_to_history(token, sentiment_score, timestamp)

# 4. Alert if significant change
if abs(sentiment_score) > SENTIMENT_ANNOUNCE_THRESHOLD:
    announce_via_voice(f"{token} sentiment: {sentiment_score}")
```

#### Configuration:
```python
# sentiment_agent.py
TOKENS_TO_TRACK = ["solana", "bitcoin", "ethereum"]
TWEETS_PER_RUN = 30
CHECK_INTERVAL_MINUTES = 15
SENTIMENT_ANNOUNCE_THRESHOLD = 0.4  # -1 to 1 scale
```

#### Setup Required:
1. Run `src/scripts/twitter_login.py` to generate cookies
2. Add Twitter credentials to `.env`

#### When to Use:
- ✅ Gauge market sentiment before trades
- ✅ Detect narrative shifts
- ✅ Identify sentiment extremes (contrarian signals)

#### Standalone Usage:
```bash
python src/agents/sentiment_agent.py
```

---

### 5. Whale Agent (`whale_agent.py`)

**Dez the Whale Watcher 🐋**

#### What It Does:
- Monitors BTC open interest changes
- Detects whale activity (large OI moves)
- AI analyzes if moves are significant
- Announces via voice when whales move

#### How It Works:
```python
# 1. Fetch current BTC open interest
current_oi = api.get_oi_data('BTC')

# 2. Compare to historical average
oi_history = deque(maxlen=LOOKBACK_PERIODS)
avg_oi = np.mean(oi_history)
pct_change = ((current_oi - avg_oi) / avg_oi) * 100

# 3. Detect whale activity
if abs(pct_change) > WHALE_THRESHOLD_MULTIPLIER * avg_change:
    # 4. AI analysis
    market_data = get_btc_ohlcv(TIMEFRAME, LOOKBACK_BARS)
    decision = llm.analyze_whale_move(
        pct_change, current_oi, previous_oi, market_data
    )
    # Returns: BUY/SELL/NOTHING + reasoning + confidence

    # 5. Announce
    announce_via_voice(decision)
```

#### AI Analysis Considers:
- Large OI increases + price up = strong momentum
- Large OI decreases + price down = capitulation (buy signal or trend confirmation)
- Ratio of long vs short OI

#### Configuration:
```python
# whale_agent.py
CHECK_INTERVAL_MINUTES = 5
LOOKBACK_PERIODS = {'15min': 15}
WHALE_THRESHOLD_MULTIPLIER = 1.31  # 31% above average
MODEL_OVERRIDE = "deepseek-chat"  # Fast & cheap
```

#### When to Use:
- ✅ Detect whale accumulation/distribution
- ✅ Identify potential trend reversals
- ✅ Front-run major moves

#### Standalone Usage:
```bash
python src/agents/whale_agent.py
```

---

### 6. Funding Agent (`funding_agent.py`)

**Fran the Funding Rate Monitor 💰**

#### What It Does:
- Tracks funding rates across different tokens
- Detects extreme funding rates (positive or negative)
- AI analyzes if rates signal opportunity
- Announces via voice when thresholds hit

#### How It Works:
```python
# 1. Fetch funding rates for tracked symbols
for symbol in SYMBOL_NAMES:
    funding_rate = api.get_funding_rate(symbol)
    annual_rate = funding_rate * 365  # Annualized

    # 2. Check thresholds
    if annual_rate < NEGATIVE_THRESHOLD or annual_rate > POSITIVE_THRESHOLD:
        # 3. Get market context (BTC for overall direction)
        btc_data = get_btc_ohlcv(TIMEFRAME, LOOKBACK_BARS)

        # 4. AI analysis
        decision = llm.analyze_funding(
            symbol, funding_rate, btc_data
        )
        # Returns: BUY/SELL/NOTHING + reasoning + confidence

        # 5. Announce
        announce_via_voice(decision)
```

#### AI Analysis Considers:
- **Negative funding + uptrend** = shorts getting squeezed (BUY signal)
- **High funding + downtrend** = longs getting liquidated (SELL signal)
- BTC trend for overall market direction

#### Configuration:
```python
# funding_agent.py
CHECK_INTERVAL_MINUTES = 15
NEGATIVE_THRESHOLD = -5  # -5% annual
POSITIVE_THRESHOLD = 20  # 20% annual
SYMBOL_NAMES = {
    'BTC': 'Bitcoin',
    'FARTCOIN': 'Fart Coin'
}
MODEL_OVERRIDE = "deepseek-chat"
```

#### When to Use:
- ✅ Catch funding rate extremes
- ✅ Identify crowded trades
- ✅ Find contrarian opportunities

#### Standalone Usage:
```bash
python src/agents/funding_agent.py
```

---

### 7. Liquidation Agent (`liquidation_agent.py`)

**Luna the Liquidation Monitor 🌊**

#### What It Does:
- Tracks liquidation volume (longs vs shorts)
- Detects sudden liquidation spikes
- AI analyzes if liquidations signal reversals
- Announces via voice when significant

#### How It Works:
```python
# 1. Fetch recent liquidations
liquidations = api.get_liquidations(LIQUIDATION_ROWS)

# 2. Separate longs vs shorts
current_longs = sum(liquidations[liquidations.side == 'long'].usd)
current_shorts = sum(liquidations[liquidations.side == 'short'].usd)

# 3. Compare to historical average
avg_longs = np.mean(long_history)
avg_shorts = np.mean(short_history)
pct_change_longs = ((current_longs - avg_longs) / avg_longs) * 100
pct_change_shorts = ((current_shorts - avg_shorts) / avg_shorts) * 100

# 4. Detect significant events
total_pct_change = ((current_longs + current_shorts) - (avg_longs + avg_shorts)) / (avg_longs + avg_shorts) * 100

if total_pct_change > LIQUIDATION_THRESHOLD * 100:
    # 5. AI analysis
    market_data = get_btc_ohlcv(TIMEFRAME, LOOKBACK_BARS)
    decision = llm.analyze_liquidations(
        pct_change_longs, pct_change_shorts, market_data
    )

    # 6. Announce
    announce_via_voice(decision)
```

#### AI Analysis Considers:
- **Large long liquidations** = potential bottoms (shorts taking profit)
- **Large short liquidations** = potential tops (longs taking profit)
- Ratio of long vs short liquidations

#### Configuration:
```python
# liquidation_agent.py
CHECK_INTERVAL_MINUTES = 10
LIQUIDATION_ROWS = 10000  # Recent liquidations to fetch
LIQUIDATION_THRESHOLD = 0.5  # 50% above average
COMPARISON_WINDOW = 15  # 15 minutes for comparison
MODEL_OVERRIDE = "deepseek-chat"
```

#### When to Use:
- ✅ Detect capitulation bottoms
- ✅ Identify blow-off tops
- ✅ Gauge market panic/euphoria

#### Standalone Usage:
```bash
python src/agents/liquidation_agent.py
```

---

### 8. Chart Analysis Agent (`chartanalysis_agent.py`)

**Chuck the Chart Analyzer 📊**

#### What It Does:
- Generates candlestick charts with indicators
- Uses AI vision to analyze chart patterns
- Supports multiple timeframes (15m, 1h, 4h, 1d)
- Saves charts for review

#### How It Works:
```python
# 1. Fetch OHLCV data
for symbol in SYMBOLS:
    for timeframe in TIMEFRAMES:
        ohlcv = get_ohlcv(symbol, timeframe, LOOKBACK_BARS)

        # 2. Calculate indicators
        ohlcv['SMA20'] = talib.SMA(ohlcv['close'], 20)
        ohlcv['SMA50'] = talib.SMA(ohlcv['close'], 50)
        ohlcv['RSI'] = talib.RSI(ohlcv['close'], 14)

        # 3. Generate chart image
        chart_image = mpf.plot(ohlcv, type='candle', style=CHART_STYLE, volume=True)
        save_chart(f"{symbol}_{timeframe}.png")

        # 4. AI vision analysis
        decision = llm.analyze_chart(
            image=chart_image,
            symbol=symbol,
            timeframe=timeframe
        )
        # Returns: BUY/SELL/NOTHING + reasoning + confidence

        # 5. Announce
        announce_via_voice(decision)
```

#### AI Vision Analysis Considers:
- Chart patterns (head & shoulders, triangles, etc.)
- Support/resistance levels
- Indicator confluence
- Volume confirmation

#### Configuration:
```python
# chartanalysis_agent.py
CHECK_INTERVAL_MINUTES = 10
TIMEFRAMES = ['15m', '1h', '4h', '1d']
LOOKBACK_BARS = 100
SYMBOLS = ["BTC", "FARTCOIN"]
INDICATORS = ['SMA20', 'SMA50', 'SMA200', 'RSI', 'MACD']
```

#### When to Use:
- ✅ Visual pattern recognition
- ✅ Multiple timeframe analysis
- ✅ Chart review and education

#### Standalone Usage:
```bash
python src/agents/chartanalysis_agent.py
```

---

## Token Discovery Agents

These agents find new trading opportunities.

### 9. Sniper Agent (`sniper_agent.py`)

**The New Token Scanner 🎯**

#### What It Does:
- Watches for NEW Solana token launches in real-time
- Displays fun animations when new tokens appear
- Auto-opens tokens in browser (DexScreener or Birdeye)
- Plays sound effects for new launches
- Saves token data for analysis

#### How It Works:
```python
# 1. Continuously poll Moon Dev API for new tokens
while True:
    new_tokens = api.get_recent_tokens()

    # 2. Check for unseen tokens
    for token in new_tokens:
        if token.address not in seen_tokens:
            # 3. Show attention animation
            attention_animation()

            # 4. Display token info
            print_token_info(token)

            # 5. Play sound effect
            play_sound(random.choice(SOUND_EFFECTS))

            # 6. Auto-open in browser
            if AUTO_OPEN_BROWSER:
                url = get_dexscreener_url(token) if USE_DEXSCREENER else get_birdeye_url(token)
                webbrowser.open(url)

            # 7. Save to CSV
            save_token_data(token)

            seen_tokens.add(token.address)

    time.sleep(CHECK_INTERVAL)
```

#### Configuration:
```python
# sniper_agent.py
CHECK_INTERVAL = 10  # Seconds between checks
PAST_TOKENS_TO_SHOW = 40  # Display history
AUTO_OPEN_BROWSER = True
USE_DEXSCREENER = True  # vs Birdeye
SOUND_ENABLED = True
EXCLUDE_PATTERNS = ['So111...']  # Filter out SOL
```

#### When to Use:
- ✅ Catch tokens at launch
- ✅ Monitor new listings
- ✅ Find early opportunities

#### Standalone Usage:
```bash
python src/agents/sniper_agent.py
```

---

### 10. Solana Agent (`solana_agent.py`)

**The Solana Token Analyzer 🔍**

#### What It Does:
- Analyzes recent token launches using Moon Dev's criteria
- Filters tokens by market cap, liquidity, volume, holder distribution
- Scores tokens and saves "top picks"
- More sophisticated than sniper agent

#### How It Works:
```python
# 1. Load recent tokens from sniper agent data
tokens = pd.read_csv(SNIPER_DATA)

# 2. Analyze each token
for token in tokens:
    # Get comprehensive token data
    overview = token_overview(token.address)
    security = token_security_info(token.address)
    creation = token_creation_info(token.address)

    # 3. Apply filters
    passes = (
        MIN_MARKET_CAP < overview.mcap < MAX_MARKET_CAP and
        MIN_LIQUIDITY < overview.liquidity < MAX_LIQUIDITY and
        overview.volume_24h > MIN_VOLUME_24H and
        overview.top_holders_pct < MAX_TOP_HOLDERS_PCT and
        overview.unique_holders > MIN_UNIQUE_HOLDERS and
        MIN_AGE_HOURS < token_age_hours < MAX_AGE_HOURS and
        overview.buy_tx_pct > MIN_BUY_TX_PCT
    )

    # 4. If passes, save to top picks
    if passes:
        save_to_top_picks(token)

        # 5. Auto-open in browser
        if AUTO_OPEN_BROWSER:
            webbrowser.open(get_dexscreener_url(token))
```

#### Analysis Criteria:
- Market cap: $1k - $1M
- Liquidity: $1k - $500k
- 24h volume: > $5k
- Top 10 holders: < 60%
- Unique holders: > 100
- Token age: 1-48 hours
- Buy transactions: > 60%
- Trades last hour: > 10

#### Configuration:
```python
# solana_agent.py
CHECK_INTERVAL = 600  # 10 minutes
MIN_MARKET_CAP = 1000
MAX_MARKET_CAP = 1000000
MIN_LIQUIDITY = 1000
MAX_LIQUIDITY = 500000
MIN_VOLUME_24H = 5000
MAX_TOP_HOLDERS_PCT = 60
MIN_UNIQUE_HOLDERS = 100
MIN_AGE_HOURS = 1
MAX_AGE_HOURS = 48
```

#### When to Use:
- ✅ Find quality new tokens
- ✅ Avoid rug pulls
- ✅ Systematic token discovery

#### Standalone Usage:
```bash
python src/agents/solana_agent.py
```

---

### 11. CopyBot Agent (`copybot_agent.py`)

**The Copy Trading Analyzer 📋**

#### What It Does:
- Analyzes current copybot portfolio positions
- Identifies which positions deserve larger allocations
- AI evaluates relative performance
- Works with external copy trading bot

#### How It Works:
```python
# 1. Load copybot portfolio
portfolio = pd.read_csv(COPYBOT_PORTFOLIO_PATH)

# 2. For each position
for token in portfolio.tokens:
    # Get market data
    ohlcv = collect_token_data(token)

    # Get position performance
    position_data = portfolio[portfolio['Mint Address'] == token]

    # 3. AI analysis
    decision = llm.analyze_copybot_position(
        position_data,
        ohlcv,
        portfolio  # Compare to other positions
    )
    # Returns: BUY/SELL/NOTHING + reasoning + confidence

    # 4. If BUY signal, increase position
    if decision == "BUY":
        target_size = calculate_target_size(decision.confidence)
        n.ai_entry(token, target_size)
```

#### AI Analysis Considers:
- Position performance vs others in portfolio
- Technical indicators
- Volume profile
- Risk/reward
- Market conditions

#### Configuration:
```python
# copybot_agent.py
COPYBOT_PORTFOLIO_PATH = '/path/to/current_portfolio.csv'
```

#### When to Use:
- ✅ You run a copy trading bot
- ✅ Want to scale winning positions
- ✅ Automate copybot follow-up

#### Standalone Usage:
```bash
python src/agents/copybot_agent.py
```

---

## How Agents Work Together

### 🎯 Main Loop Integration

Here's how you can orchestrate ALL agents together:

```python
# main.py (example)

from src.agents.risk_agent import RiskAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.trading_agent import TradingAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.whale_agent import WhaleAgent
from src.agents.funding_agent import FundingAgent
from src.agents.liquidation_agent import LiquidationAgent

# Initialize all agents
risk_agent = RiskAgent()
strategy_agent = StrategyAgent()
trading_agent = TradingAgent()
sentiment_agent = SentimentAgent()
whale_agent = WhaleAgent()
funding_agent = FundingAgent()
liquidation_agent = LiquidationAgent()

# Track signals from intelligence agents
market_signals = {
    'sentiment': None,
    'whale': None,
    'funding': None,
    'liquidation': None
}

def main_loop():
    while True:
        print("\n🌙 Moon Dev Trading System - Starting Cycle")

        # 1. RISK CHECK (ALWAYS FIRST!)
        risk_status = risk_agent.check_risk_limits()
        if risk_status == "BLOCKED":
            print("🛑 Trading blocked by risk agent")
            time.sleep(900)
            continue

        # 2. COLLECT MARKET INTELLIGENCE (Run in background/parallel)
        market_signals['sentiment'] = sentiment_agent.get_current_sentiment()
        market_signals['whale'] = whale_agent.check_whale_activity()
        market_signals['funding'] = funding_agent.check_funding_rates()
        market_signals['liquidation'] = liquidation_agent.check_liquidations()

        # 3. STRATEGY SIGNALS
        strategy_signals = {}
        for token in MONITORED_TOKENS:
            signals = strategy_agent.get_signals(token)
            if signals:
                strategy_signals[token] = signals

        # 4. TRADING AGENT (with all context)
        # Pass strategy signals AND market intelligence
        trading_agent.run_trading_cycle(
            strategy_signals=strategy_signals,
            market_context=market_signals
        )

        print("✅ Cycle complete!")
        time.sleep(900)  # 15 minutes

if __name__ == "__main__":
    main_loop()
```

### 🔄 Information Flow

```
Token Discovery Agents (Sniper, Solana, CopyBot)
    ↓ (New opportunities)
MONITORED_TOKENS list
    ↓
Market Intelligence Agents (Sentiment, Whale, Funding, Liquidation, Chart)
    ↓ (Market context)
Strategy Agent (Custom rules + AI validation)
    ↓ (Validated signals)
Trading Agent (LLM analysis + allocation)
    ↓ (Proposed trades)
Risk Agent (Circuit breakers + AI overrides)
    ↓ (Approved trades)
Execution (n.ai_entry, n.chunk_kill)
```

---

## Quick Start Guide

### 1. Configure Your System

Edit `src/config.py`:

```python
# Tokens to trade
MONITORED_TOKENS = [
    '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',  # FART
    'HeLp6NuQkmYB4pYWo2zYs22mESHXPQYzXbB8n4V98jwC',  # AI16Z
]

# Position sizing
usd_size = 25
max_usd_order_size = 3
MAX_POSITION_PERCENTAGE = 30
CASH_PERCENTAGE = 20

# Risk management
USE_PERCENTAGE = False
MAX_LOSS_USD = 25
MAX_GAIN_USD = 25
MINIMUM_BALANCE_USD = 50

# AI settings
AI_MODEL = "claude-3-haiku-20240307"
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 1024

# Agent intervals
SLEEP_BETWEEN_RUNS_MINUTES = 15
```

### 2. Set Up Environment Variables

Create `.env` file:

```bash
# AI Models
ANTHROPIC_KEY=sk-ant-...
OPENAI_KEY=sk-...
DEEPSEEK_KEY=...
GROQ_API_KEY=...

# Blockchain
SOLANA_PRIVATE_KEY=...
RPC_ENDPOINT=...

# APIs
BIRDEYE_API_KEY=...
MOONDEV_API_KEY=...
COINGECKO_API_KEY=...
```

### 3. Run Individual Agents

```bash
# Core trading
python src/agents/trading_agent.py       # LLM trader
python src/agents/strategy_agent.py      # Strategy executor
python src/agents/risk_agent.py          # Risk manager

# Market intelligence
python src/agents/sentiment_agent.py     # Twitter sentiment
python src/agents/whale_agent.py         # Whale watching
python src/agents/funding_agent.py       # Funding rates
python src/agents/liquidation_agent.py   # Liquidations
python src/agents/chartanalysis_agent.py # Chart analysis

# Token discovery
python src/agents/sniper_agent.py        # New tokens
python src/agents/solana_agent.py        # Token analyzer
python src/agents/copybot_agent.py       # Copy trading
```

### 4. Run Main Orchestrator

```bash
python src/main.py  # Runs all active agents in loop
```

Configure which agents are active in `main.py`:

```python
ACTIVE_AGENTS = {
    'risk': True,        # Always recommended!
    'trading': True,
    'strategy': False,   # Enable if you have custom strategies
    'sentiment': False,
    'whale': False,
    'funding': False,
    'liquidation': False,
}
```

---

## Configuration

### Global Settings (`src/config.py`)

```python
# Trading
MONITORED_TOKENS = [...]
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]
usd_size = 25
max_usd_order_size = 3
slippage = 199

# Risk Management
CASH_PERCENTAGE = 20
MAX_POSITION_PERCENTAGE = 30
MAX_LOSS_USD = 25
MAX_GAIN_USD = 25
MINIMUM_BALANCE_USD = 50
MAX_LOSS_GAIN_CHECK_HOURS = 12

# AI
AI_MODEL = "claude-3-haiku-20240307"
AI_TEMPERATURE = 0.7
AI_MAX_TOKENS = 1024

# Timing
SLEEP_BETWEEN_RUNS_MINUTES = 15
```

### Per-Agent Overrides

Most agents support model overrides:

```python
# In agent file (e.g., whale_agent.py)
MODEL_OVERRIDE = "deepseek-chat"  # Override AI_MODEL from config
AI_TEMPERATURE = 0.5              # Override temperature
AI_MAX_TOKENS = 50                # Override max tokens
```

Available model overrides:
- `"deepseek-chat"` - Fast & cheap
- `"deepseek-reasoner"` - Reasoning model
- `"0"` - Use config.py default

---

## Best Practices

### 🎯 Strategy Selection

| Goal | Recommended Agents |
|------|-------------------|
| **Automated Trading** | Risk + Trading Agent |
| **Rule-Based Trading** | Risk + Strategy Agent |
| **Hybrid Trading** | Risk + Strategy + Trading |
| **Market Monitoring** | Intelligence agents only |
| **Token Discovery** | Sniper + Solana agents |

### 🛡️ Risk Management

**ALWAYS run Risk Agent first!**

```python
# Correct order:
1. Risk Agent
2. Other agents
3. Execution

# Wrong order:
1. Trading Agent
2. Risk Agent  # Too late!
```

### 💰 Cost Optimization

**Model Selection by Use Case:**

| Agent | Recommended Model | Why |
|-------|------------------|-----|
| **Trading Agent** | Claude Haiku | Quality decisions |
| **Strategy Agent** | Claude Haiku | Strategy validation |
| **Risk Agent** | DeepSeek Chat | Fast & cheap checks |
| **Whale Agent** | DeepSeek Chat | Frequent checks |
| **Funding Agent** | DeepSeek Chat | Frequent checks |
| **Liquidation Agent** | DeepSeek Chat | Frequent checks |
| **Chart Agent** | Claude Sonnet | Vision quality |

**Cost Comparison (per 1M tokens):**
- DeepSeek Chat: ~$0.14
- Claude Haiku: ~$0.25
- Claude Sonnet: ~$3.00
- Claude Opus: ~$15.00

### 📊 Monitoring

Watch these metrics:
- API costs (track LLM usage)
- Win rate (via logs)
- Max drawdown
- Position sizes
- Agent decisions (logs)

---

## Troubleshooting

### "No trades executing"
- Check `MONITORED_TOKENS` has valid addresses
- Verify API keys in `.env`
- Check Risk Agent isn't blocking
- Review agent logs for errors

### "Risk Agent blocking everything"
- Check PnL over `MAX_LOSS_GAIN_CHECK_HOURS`
- Adjust `MAX_LOSS_USD` or `MAX_GAIN_USD`
- Review risk agent logs
- Set `USE_AI_CONFIRMATION = True` for overrides

### "Strategies not loading"
- Verify strategies in `src/strategies/custom/`
- Check `ENABLE_STRATEGIES = True` in config
- Import strategies in `strategy_agent.py`
- Review strategy agent logs

### "Agent crashes"
- Check API rate limits
- Verify all environment variables
- Review error logs in console
- Check account balances

---

## Advanced Topics

### Creating Custom Agents

```python
from src.agents.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__('mycustom')
        # Your initialization

    def run(self):
        # Your agent logic
        pass
```

### Agent Communication

Agents can share data via:
1. Return values (synchronous)
2. CSV files (`src/data/agent_name/`)
3. Database (future)

### Parallel Execution

Run intelligence agents in parallel:

```python
import threading

threads = [
    threading.Thread(target=sentiment_agent.run),
    threading.Thread(target=whale_agent.run),
    threading.Thread(target=funding_agent.run),
]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

---

## Resources

- **Main README**: `/README.md`
- **Model Factory**: `src/models/README.md`
- **Nice Functions**: `src/nice_funcs.py`
- **Strategies**: `src/strategies/`
- **Data**: `src/data/`

---

## Support

Questions? Issues? Join the Moon Dev community:
- YouTube: [@moondevonyt](https://www.youtube.com/@moondevonyt)
- Bootcamp: [algotradecamp.com](https://algotradecamp.com)
- Discord: [Join here](https://algotradecamp.com)

---

**Built with love by Moon Dev 🌙**

*Remember: Trading involves risk. Always test strategies thoroughly before deploying with real capital. These agents are educational tools - use at your own risk!*
