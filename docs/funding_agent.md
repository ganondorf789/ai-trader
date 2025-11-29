# Funding Agent

Monitors perpetual funding rates across exchanges.

## What It Does
- Tracks funding rates on Hyperliquid, Drift, etc.
- Identifies funding arbitrage opportunities
- Detects extreme funding as reversal signals
- Monitors open interest changes

## Usage
```bash
python src/agents/funding_agent.py
```

## Key Signals
- Funding >0.1% = Overleveraged longs
- Funding <-0.05% = Overleveraged shorts
- Funding flips = Potential trend change
- OI spike + funding = Squeeze setup

## Configuration
```python
FUNDING_EXTREME_THRESHOLD = 0.1  # % per 8 hours
FUNDING_ARB_MINIMUM = 0.05      # Min spread to trade
```

## Data Source
Moon Dev API: `/api/funding-data`

## Output
`src/data/funding_agent/[date]/funding_rates.csv`