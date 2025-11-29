# Strategy Agent

Runs pre-built strategies from `src/strategies/` folder with AI validation.

## What It Does
- Loads strategy files (moving averages, RSI, etc.)
- Generates signals from strategies
- Uses AI to validate/filter signals
- Manages multiple strategies in parallel

## Usage
```bash
python src/agents/strategy_agent.py
```

## Add New Strategy
Create file in `src/strategies/`:
```python
class YourStrategy(BaseStrategy):
    def generate_signals(self, token_address, market_data):
        return {"action": "BUY", "confidence": 80}
```

## Configuration
In `config.py`:
```python
ACTIVE_STRATEGIES = ['moving_average', 'rsi_divergence']
STRATEGY_USE_AI_CONFIRMATION = True
```

## Output
Saves to `src/data/strategy_agent/[strategy_name]/[date]/`