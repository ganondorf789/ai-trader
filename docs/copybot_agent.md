# CopyBot Agent

Copies trades from profitable wallets.

## What It Does
- Monitors high-performance wallets
- Copies their trades in real-time
- Filters by wallet win rate
- Implements position sizing rules

## Usage
```bash
python src/agents/copybot_agent.py
```

## Wallet Selection
- Min 60% win rate over 30 days
- Min 50 trades history
- Profit >$10k
- Active in last 24h

## Copy Settings
```python
COPYBOT_MAX_WALLETS = 5        # Follow up to 5
COPYBOT_POSITION_MULTIPLIER = 0.1  # Use 10% of their size
COPYBOT_DELAY_SECONDS = 5      # Wait 5s after their trade
```

## Risk Management
- Max $500 per copied trade
- Skip if token in excluded list
- Stop copying after 3 losses

## Data Source
Moon Dev API: `/api/copybot-follow-list`

## Output
`src/data/copybot_agent/[date]/copied_trades.csv`