# Moon Dev AI Trading Strategy Orchestration Plan 🌙

## Mission: Complete Backtesting of All Trading Strategies

### Objective
Each AI agent (AI2-AI8) will backtest assigned trading strategies using the backtesting.py library and produce comprehensive performance statistics.

### AI Agent Task Distribution

**AI2**: BandDivergence, BandMACDDivergence, BandReversalEdge
**AI3**: BandSqueezeTrend, BandStochFusion, BandStrengthMomentum  
**AI4**: BandSyncMomentum, BandedMACD, ChikouVolatility
**AI5**: CoTrendalNeutral, ConfluencePattern, DeltaSentiment
**AI6**: DivergenceAnchor, DivergenceBand, DivergenceVolatility
**AI7**: DivergentBandReversion, DivergentBands, DivergentCrossover
**AI8**: DivergentMomentum, DivergentPulse, DivergentVolume

### Working Directory Structure
- **Base Path**: `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/03_14_2025/`
- **Strategy Files**: Located in `backtests/` folder
- **Output**: Create new files in `AI_BACKTEST_RESULTS/` folder (create this folder)

### Backtest Requirements

1. **Data Source**: `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv`
2. **Portfolio Size**: $1,000,000
3. **Library**: Use backtesting.py framework
4. **Indicators**: TA-Lib or pandas_ta

### Key Implementation Rules

#### Data Handling:
```python
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
```

#### Indicator Calculations:
- Always use `self.I()` wrapper
- Use talib functions: `self.I(talib.SMA, self.data.Close, timeperiod=20)`

#### Position Sizing:
```python
position_size = int(round(position_size))  # Always convert to integer
```

#### Output Requirements:
```python
stats = bt.run()
print(stats)
print(stats._strategy)
```

### Completion Signal
When backtest is complete and stats are printed, send exactly 50 emojis: 🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀✨🌙🚀

### Action Items for Each AI:
1. Read assigned strategy files from `backtests/` folder
2. Create working backtest implementation 
3. Execute backtest with BTC-USD-15m data
4. Print comprehensive statistics
5. Save final code in `AI_BACKTEST_RESULTS/` folder
6. Send completion signal (50 emojis)

### Start Command: "START BACKTESTING NOW! 🌙🚀"

---
*Orchestrated by AI1 - Moon Dev Trading Command Center* 🌙