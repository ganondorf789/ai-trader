# Tweet Agent

Generates trading-related tweets automatically.

## What It Does
- Creates market commentary tweets
- Generates token analysis threads
- Writes educational content
- Schedules tweet timing

## Usage
```bash
python src/agents/tweet_agent.py
```

## Tweet Types
- Market updates
- Token discoveries
- Technical analysis
- Risk warnings
- Educational threads

## Configuration
```python
TWEET_FREQUENCY = 'hourly'  # or 'daily', 'manual'
TWEET_MAX_LENGTH = 280
TWEET_INCLUDE_CHARTS = True
TWEET_AUTO_POST = False  # Set True to auto-post
```

## Safety
- No financial advice disclaimers
- No token shilling
- Fact-based content only

## Output
`src/data/tweets/[date]/generated_tweets.txt`