Here's the fixed and completed backtest code with Moon Dev themed debug prints and proper position sizing:

```python
# 🌙 Moon Dev's BandwidthBreakout Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

# 🚀 DATA PREPARATION 
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])

# 📊 Ensure proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'basis': 'Basis'  # Fixed typo from 'basis' to 'Basis'
}, inplace=True)

class BandwidthBreakout(Strategy):
    risk_percent = 0.01  # 🌕 1% risk per trade
    
    def init(self):
        # 📈 INDICATORS
        self.middle_band = self.I(talib.SMA, self.data.Close, timeperiod=20, name='MiddleBB')
        self.std_dev = self.I(talib.STDDEV, self.data.Close, timeperiod=20, name='StdDev')
        self.upper_band = self.I(lambda m,s: m + 2*s, self.middle_band, self.std_dev, name='UpperBB')
        self.lower_band = self.I(lambda m,s: m - 2*s, self.middle_band, self.std_dev, name='LowerBB')
        self.bb_width = self.I(lambda u,l,m: (u-l)/m, self.upper_band, self.lower_band, self.middle_band, name='BBWidth')
        self.bb_width_pct = self.I(ta.percentile, self.bb_width, length=100, q=20, name='BBWidthPct')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolumeMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')

    def calculate_position_size(self, direction):
        """🌙 Calculate proper position size based on risk and ATR"""
        risk_amount = self.equity * self.risk_percent
        atr_value = self.atr[-1]
        
        if atr_value == 0:  # Prevent division by zero
            print("🌑 Warning: ATR is zero, using minimum position size")
            atr_value = 0.01 * self.data.Close[-1]  # 1% of current price as fallback
            
        # Calculate units based on ATR stop distance
        units = risk_amount / atr_value
        units = int(round(units))  # 🌕 Round to whole units
        
        if direction == 'long':
            self.buy(size=units)
            print(f"🌖 LONG ENTRY @ {self.data.Close[-1]:.2f} | Size: {units} units ✨")
        else:
            self.sell(size=units)
            print(f"🌘 SHORT ENTRY @ {self.data.Close[-1]:.2f} | Size: {units} units ✨")

    def next(self):
        # 🌑🌒 ENTRY LOGIC
        if not self.position:
            # LONG Conditions 🌖
            if (self.bb_width[-1] < self.bb_width_pct[-1] and
                self.data.Close[-1] > self.upper_band[-1] and
                abs(self.data.Basis[-1]) > 0.005 and
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                self.calculate_position_size('long')
            
            # SHORT Conditions 🌘
            elif (self.bb_width[-1] < self.bb_width_pct[-1] and
                  self.data.Close[-1] < self.lower_band[-1] and
                  abs(self.data.Basis[-1]) > 0.005 and
                  self.data.Volume[-1] > self.volume_ma[-1]):
                
                self.calculate_position_size('short')
        
        # 🛑 EXIT LOGIC
        else:
            if self.position.is_long:
                if self.data.Close[-1] <= self.middle