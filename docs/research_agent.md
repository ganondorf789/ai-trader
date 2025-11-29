# Research Agent

Deep-dive token research and due diligence.

## What It Does
- Comprehensive token analysis
- Team background checks
- Whitepaper analysis
- Competitor comparison
- Risk assessment

## Usage
```bash
python src/agents/research_agent.py
```

## Research Areas
- Tokenomics (supply, vesting, burns)
- Team (doxxed, experience, past projects)
- Technology (innovation, code quality)
- Community (size, engagement, growth)
- Partnerships (real vs fake)

## Data Sources
- GitHub activity
- Social media presence
- On-chain metrics
- News mentions
- Forum discussions

## Configuration
```python
RESEARCH_DEPTH = 'comprehensive'  # or 'quick', 'standard'
RESEARCH_INCLUDE_COMPETITORS = True
```

## Output
`src/data/research_agent/[date]/[token]_research.md`