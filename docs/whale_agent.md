# Whale Agent

Tracks large wallet movements and whale behavior.

## What It Does
- Monitors wallets holding >$100k of tokens
- Detects whale accumulation/distribution
- Tracks smart money flow
- Alerts on unusual large transactions

## Usage
```bash
python src/agents/whale_agent.py
```

## Tracking Features
- New whale entries/exits
- Whale wallet clustering
- Average buy/sell size changes
- Wallet age and success rate

## Configuration
```python
WHALE_THRESHOLD_USD = 100000  # Min to be considered whale
WHALE_ALERT_SIZE = 50000      # Alert on trades above
```

## Data Sources
- BirdEye holder API
- Helius transaction history
- On-chain analysis

## Output
`src/data/whale_agent/[date]/whale_movements.json`