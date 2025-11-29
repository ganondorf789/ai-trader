# Sentiment Agent

Analyzes social media sentiment for trading signals.

## What It Does
- Monitors Twitter/Discord/Telegram mentions
- Calculates sentiment scores (-100 to +100)
- Detects FOMO/FUD extremes
- Tracks sentiment velocity changes

## Usage
```bash
python src/agents/sentiment_agent.py
```

## Data Sources
- CoinGecko sentiment API
- BirdEye social metrics
- Custom scrapers (if configured)

## Key Metrics
- Mention count & velocity
- Positive/negative ratio
- Influencer vs retail sentiment
- Sentiment divergence from price

## Configuration
```python
SENTIMENT_THRESHOLD_BUY = 70   # Bullish above
SENTIMENT_THRESHOLD_SELL = 30  # Bearish below
```

## Output
`src/data/sentiment_agent/[date]/sentiment_scores.csv`