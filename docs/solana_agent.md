# Solana Agent

Discovers trending tokens on Solana using BirdEye data.

## What It Does
- Scans top gainers/volume leaders
- Filters by market cap and liquidity
- Checks token age and holder count
- Ranks by opportunity score

## Usage
```bash
python src/agents/solana_agent.py
```

## Discovery Criteria
- Volume >$100k/24h
- Liquidity >$50k
- Holders >100
- Not in excluded list
- Market cap <$10M (configurable)

## Data Points
- Price change % (1h, 24h, 7d)
- Volume/market cap ratio
- Holder growth rate
- Liquidity depth

## Configuration
```python
SOLANA_MIN_LIQUIDITY = 50000
SOLANA_MIN_HOLDERS = 100
SOLANA_MAX_MCAP = 10000000
```

## Output
`src/data/solana_agent/[date]/trending_tokens.json`