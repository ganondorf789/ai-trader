# Trading Agent

Pure LLM-based trader that uses Claude/GPT/Grok to analyze markets and execute trades.

## What It Does
- Analyzes token data using AI
- Makes buy/sell decisions
- Manages portfolio allocation
- No hardcoded rules, pure AI reasoning

## Usage
```bash
python src/agents/trading_agent.py
```

## Configuration
Edit top of `trading_agent.py`:
```python
AI_MODEL_TYPE = 'claude'  # or 'openai', 'xai', 'groq'
AI_MODEL_NAME = None      # or specific model
```

## Key Functions
- `analyze_market_data()` - AI analyzes token and decides action
- `allocate_portfolio()` - AI allocates USD across opportunities
- `execute_trades()` - Executes the AI decisions

## Output
Saves decisions to `src/data/trading_agent/[date]/`