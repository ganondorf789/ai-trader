# Liquidation Agent

Tracks leveraged position liquidations in real-time.

## What It Does
- Monitors liquidation levels across perp exchanges
- Identifies cascade liquidation zones
- Predicts stop-loss hunting moves
- Alerts on unusual liquidation spikes

## Usage
```bash
python src/agents/liquidation_agent.py
```

## Key Metrics
- 24h liquidation volume
- Long vs short liquidation ratio
- Liquidation heatmap by price level
- Largest single liquidations

## Trading Signals
- Mass liquidations = Local bottom/top
- One-sided liquidations = Continuation
- No liquidations = Low volatility ahead

## Configuration
```python
LIQUIDATION_SPIKE_THRESHOLD = 10000000  # $10M in 1hr
```

## Data Source
Moon Dev API: `/api/liquidation-data`

## Output
`src/data/liquidation_agent/[date]/liquidations.json`