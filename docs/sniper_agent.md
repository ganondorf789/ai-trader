# Sniper Agent

Finds and snipes new token launches on Solana.

## What It Does
- Monitors Raydium/Orca pool creation
- Analyzes token safety in <100ms
- Auto-buys if criteria met
- Implements anti-rug checks

## Usage
```bash
python src/agents/sniper_agent.py
```

## Safety Checks
- Liquidity locked check
- Mint authority revoked
- Top holder distribution
- Contract verification
- Bundled transaction detection

## Configuration
```python
SNIPER_MIN_LIQUIDITY = 10000    # $10k minimum
SNIPER_MAX_BUY = 100            # Max $100 per snipe
SNIPER_AUTO_SELL_MULTIPLIER = 2 # Sell at 2x
```

## Speed Requirements
- RPC: Helius/QuickNode paid tier
- Decision time: <100ms
- Use Jito for MEV protection

## Output
`src/data/sniper_agent/[date]/sniped_tokens.csv`