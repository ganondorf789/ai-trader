# TX Agent

Transaction analysis and monitoring.

## What It Does
- Monitors pending transactions
- Analyzes transaction patterns
- Detects MEV attacks
- Tracks gas optimization

## Usage
```bash
python src/agents/tx_agent.py
```

## Features
- Transaction simulation
- Priority fee optimization
- Jito bundle creation
- Failed transaction analysis

## Configuration
```python
TX_PRIORITY_FEE = 10000  # Lamports
TX_USE_JITO = True      # Use Jito for MEV protection
TX_MAX_RETRIES = 3
```

## Monitoring
- Success/failure rates
- Average confirmation time
- Slippage analysis
- MEV extraction amounts

## Output
`src/data/tx_agent/[date]/transaction_log.csv`