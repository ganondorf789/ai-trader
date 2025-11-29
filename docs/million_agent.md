# Million Agent

Focused on finding tokens with 100x-1000x potential.

## What It Does
- Scans for micro-cap gems
- Analyzes tokenomics for moon potential
- Checks early holder distribution
- Identifies narrative catalysts

## Usage
```bash
python src/agents/million_agent.py
```

## Selection Criteria
- Market cap <$100k
- Liquidity >$10k
- Clean contract (no honeypot)
- Strong community forming
- Unique narrative/utility

## Risk Management
- Max $50 per moonshot
- Max 10 positions
- Auto-sell 50% at 10x
- Let rest ride to 100x+

## Configuration
```python
MILLION_MAX_MCAP = 100000
MILLION_MAX_POSITION = 50
MILLION_TAKE_PROFIT_1 = 10  # 10x
```

## Output
`src/data/million_agent/[date]/moonshot_picks.json`