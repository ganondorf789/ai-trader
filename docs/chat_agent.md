# Chat Agent

Interactive AI assistant for market analysis via chat.

## What It Does
- Answers market questions
- Analyzes tokens on demand
- Provides trade recommendations
- Explains market conditions

## Usage
```bash
python src/agents/chat_agent.py
```

## Commands
- `analyze [token]` - Full token analysis
- `price [token]` - Quick price check
- `compare [token1] [token2]` - Compare tokens
- `market` - Overall market status
- `/quit` - Exit

## Features
- Remembers conversation context
- Can execute trades (with confirmation)
- Pulls live data from APIs
- Supports multiple AI models

## Configuration
```python
CHAT_MODEL_TYPE = 'claude'
CHAT_ALLOW_TRADING = False  # Set True to enable trades
```

## Output
Chat history saved to `src/data/chat_agent/[date]/`