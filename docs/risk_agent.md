# Risk Agent

First line of defense. Runs before any trades to check risk limits.

## What It Does
- Checks portfolio against risk limits
- Monitors drawdowns and exposure
- Force closes positions hitting stop losses
- Prevents overleveraging

## Usage
```bash
python src/agents/risk_agent.py
```

## Configuration
In `config.py`:
```python
MAX_POSITION_PERCENTAGE = 10  # Max 10% per token
MAX_LOSS_USD = 1000          # Daily loss limit
MAX_GAIN_USD = 5000          # Daily gain limit
MINIMUM_BALANCE_USD = 100    # Stop trading if below
STOP_LOSS_PERCENTAGE = 20    # Per position stop
```

## Risk Checks
1. Position size limits
2. Daily P&L limits
3. Minimum balance
4. Concentration risk
5. Correlation risk

## Output
Logs to console, creates risk reports in `src/data/risk_agent/`