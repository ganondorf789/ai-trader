Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed implementations:

```python
# 🌙 Moon Dev's VolSqueezeAccel Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 🚀 Data Preparation Rocket Launch 🚀
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Data Cleansing Ritual
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌌 Cosmic Column Alignment
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSqueezeAccel(Strategy):
    def init(self):
        # 🌗 Phase 1: Orbital Indicator Calculations
        # 📡 Deploying Bollinger Bands Array
        self.bb_upper = self.I(self._calculate_bb_upper, self.data.Close)
        self.bb_middle = self.I(self._calculate_bb_middle, self.data.Close)
        self.bb_lower = self.I(self._calculate_bb_lower, self.data.Close)
        
        # 🌌 Band Width Nebula Analysis
        self.bb_width = self.I(lambda u, l, m: (u - l)/m, 
                              self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=20)
        
        # 🚀 Volume Thrust Accelerator
        self.volume_roc = self.I(talib.ROC, self.data.Volume, timeperiod=5)
        self.volume_roc_mean = self.I(talib.SMA, self.volume_roc, timeperiod=20)
        self.volume_roc_std = self.I(talib.STDDEV, self.volume_roc, timeperiod=20)
        self.volume_accel = self.I(lambda r, m, s: (r - m)/s,
                                  self.volume_roc, self.volume_roc_mean, self.volume_roc_std)
        
        # 🛡️ ATR Force Field Generator
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
    def _calculate_bb_upper(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper
    
    def _calculate_bb_middle(self, close):
        _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return middle
    
    def _calculate_bb_lower(self, close):
        _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return lower

    def next(self):
        # 🌙 Lunar Debug Console
        print(f"\n🌕 Moon Phase Update | Close: {self.data.Close[-1]:.2f} | Volume Z: {self.volume_accel[-1]:.2f}σ | BB Width: {self.bb_width[-1]:.4f}")
        
        if self.position:
            # 🚨 Active Position Management
            price = self.data.Close[-1]
            if self.position.is_long:
                if self.data.Low[-1] <= self.position.sl:
                    print(f"🌘 LONG STOP LOSS ACTIVATED! Collision at {self.position.sl:.2f}")
                    self.position.close()
                elif self.data.High[-1] >= self.position.tp:
                    print(f"🌖 LONG TAKE PROFIT REACHED! Docking at {self.position.tp:.2f}")
                    self.position.close()
            else:
                if self.data.High[-1] >= self.position.sl:
                    print(f"🌒 SHORT STOP LOSS TRIGGERED! Breach at {self.position.sl:.2f}")
                    self.position.close()
                elif self.data.Low[-1] <= self.position.tp:
                    print(f"🌔 SHORT TAKE PROFIT ACHIEVED! Landing