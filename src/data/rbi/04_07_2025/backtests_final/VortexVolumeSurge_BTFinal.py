I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Clean and prepare data
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and remove unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexVolumeSurge(Strategy):
    def init(self):
        # Calculate Vortex Indicator components
        tr = self.I(talib.TRANGE, self.data.High, self.data.Low, self.data.Close)
        sum_tr = self.I(talib.SUM, tr, 14)
        
        # Calculate VM+ and VM-
        prev_low = self.I(lambda x: x.shift(1), self.data.Low)
        prev_high = self.I(lambda x: x.shift(1), self.data.High)
        vm_plus = self.I(lambda h, l: np.abs(h - l), self.data.High, prev_low)
        vm_minus = self.I(lambda h, l: np.abs(h - l), prev_high, self.data.Low)
        
        # Calculate sums
        sum_vm_plus = self.I(talib.SUM, vm_plus, 14)
        sum_vm_minus = self.I(talib.SUM, vm_minus, 14)
        
        # Final Vortex calculations
        self.vi_plus = self.I(lambda x, y: x/y, sum_vm_plus, sum_tr, name='VI+')
        self.vi_minus = self.I(lambda x, y: x/y, sum_vm_minus, sum_tr, name='VI-')
        
        # Volume indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 480)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        
        self.trailing_stop = None

    def next(self):
        # Wait for sufficient data
        if len(self.data) < 480 + 14:
            return
            
        # Entry conditions
        if ((self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]) and
            self.data.Volume[-1] > self.volume_sma[-1] * 1.2 and
            not self.position):
            
            # Calculate position size with 1% risk
            risk_amount = self.equity * 0.01
            atr_value = self.atr[-1]
            if atr_value == 0:
                return  # Safety check
            
            position_size = int(round(risk_amount / (3 * atr_value)))
            if position_size > 0:
                self.buy(size=position_size)
                self.trailing_stop = self.data.Close[-1] - 3 * atr_value
                print(f"🌙 MOON DEV ALERT! 🚀 Long entry {position_size} units at {self.data.Close[-1]} ✨")
                print(f"   Initial stop: {self.trailing_stop:.2f} | ATR: {atr_value:.2f}")
                print(f"   VI+ {self.vi_plus[-1]:.4f} crossed above VI- {self.vi_minus[-1]:.4f} 🌙")
                print(f"   Volume {self.data.Volume[-1]:.0f} vs SMA {self.volume_sma[-1]:.0f} 🌊")

        # Manage open positions
        if self.position:
            # Update trailing stop
            current_atr = self.atr[-1]
            new_stop = max(self.trailing_stop, 
                          self.data.High[-1] - 3 * current_atr)
            self.trailing_stop = new_stop
            
            # Check exit condition
            if self.data.Close[-1] < new_stop:
                self.position.close()
                print