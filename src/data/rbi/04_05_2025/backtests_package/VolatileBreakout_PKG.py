Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
import pandas as pd
import pandas_ta as ta
import talib
import numpy as np
from backtesting import Backtest, Strategy

class VolatileBreakout(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using proper self.I() wrapper
        self.chop = self.I(ta.chop, 
                          self.data.High, self.data.Low, self.data.Close, 
                          length=14, name='CHOP')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume_SMA')
        self.cons_high = self.I(talib.MAX, self.data.High, timeperiod=14, name='Cons_High')
        self.cons_low = self.I(talib.MIN, self.data.Low, timeperiod=14, name='Cons_Low')
        
        print("🌙 Initializing Moon Dev's VolatileBreakout Strategy 🌙")
        print("✨ Indicators loaded:")
        print(f"   - Choppiness Index (14)")
        print(f"   - 20-period Volume SMA")
        print(f"   - 14-period Consolidation High/Low")

    def next(self):
        # Skip if already in position
        if self.position:
            return
            
        # Get current values
        current_chop = self.chop[-1]
        prev_chop = self.chop[-2] if len(self.chop) > 1 else None
        current_volume = self.data.Volume[-1]
        volume_sma = self.volume_sma[-1]
        cons_high = self.cons_high[-1]
        cons_low = self.cons_low[-1]
        
        # Check core conditions
        chop_condition = (current_chop < 38.2) and (prev_chop and prev_chop >= 38.2)
        volume_condition = current_volume > 1.5 * volume_sma
        
        # Long entry logic
        if chop_condition and volume_condition and self.data.High[-1] > cons_high:
            entry = self.data.Close[-1]
            sl = self.data.Low[-1]
            risk = entry - sl
            position_size = int(round((self.risk_percent * self.equity) / risk))
            
            if position_size > 0:
                tp = entry + 2 * risk
                self.buy(size=position_size, sl=sl, tp=tp)
                print(f"\n🚀🌕 MOON SHOT LONG! 🚀")
                print(f"Entry: {entry:.2f}")
                print(f"Size: {position_size} units")
                print(f"SL: {sl:.2f} | TP: {tp:.2f}")
                print(f"Risk/Reward: 1:2")
        
        # Short entry logic
        elif chop_condition and volume_condition and self.data.Low[-1] < cons_low:
            entry = self.data.Close[-1]
            sl = self.data.High[-1]
            risk = sl - entry
            position_size = int(round((self.risk_percent * self.equity) / risk))
            
            if position_size > 0:
                tp = entry - 2 * risk
                self.sell(size=position_size, sl=sl, tp=tp)
                print(f"\n🌑🌪 BLACK HOLE SHORT! 🌑")
                print(f"Entry: {entry:.2f}")
                print(f"Size: {position_size} units")
                print(f"SL: {sl:.2f} | TP: {tp:.2f}")
                print(f"Risk/Reward: 1:2")

# Data preprocessing
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Execute backtest
print("\n🌙✨ MOON DEV BACKTEST INITIATED ✨🌙")
bt = Backtest(data, VolatileBreakout, cash=1_000_000,