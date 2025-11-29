# Chart Analysis Agent

Technical analysis using indicators and patterns.

## What It Does
- Calculates 20+ technical indicators
- Identifies chart patterns (triangles, flags, etc.)
- Detects support/resistance levels
- Generates confluence scores

## Usage
```bash
python src/agents/chartanalysis_agent.py
```

## Indicators Used
- Moving Averages (SMA, EMA, VWAP)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume (OBV, Volume Profile)

## Pattern Detection
- Head and shoulders
- Cup and handle
- Ascending/descending triangles
- Bull/bear flags

## Configuration
```python
CHART_TIMEFRAMES = ['5m', '1H', '4H', '1D']
MIN_CONFLUENCE_SCORE = 3  # Min indicators agreeing
```

## Output
`src/data/chartanalysis_agent/[date]/technical_analysis.json`