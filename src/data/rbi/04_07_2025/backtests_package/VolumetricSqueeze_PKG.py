Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's VolumetricSqueeze Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
import numpy as np

# 🚀 Data Preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# 🧹 Cleanse data columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 🌌 Strategy Core
class VolumetricSqueeze(Strategy):
    def init(self):
        # 🌀 Custom BBANDS wrapper
        def get_bbands(close, timeperiod=20, nbdev=2, which=0):
            upper, middle, lower = talib.BBANDS(close, timeperiod, nbdevup=nbdev, nbdevdn=nbdev)
            return [upper, middle, lower][which]
        
        # 📈 Indicators
        self.bb_upper = self.I(get_bbands, self.data.Close, 20, 2, 0, name='BB_Upper')
        self.bb_middle = self.I(get_bbands, self.data.Close, 20, 2, 1, name='BB_Middle') 
        self.bb_lower = self.I(get_bbands, self.data.Close, 20, 2, 2, name='BB_Lower')
        
        # 🌗 Bollinger Bandwidth
        self.bbw = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle, name='BBW')
        self.bbw_pct = self.I(ta.percentile, self.bbw, 50, q=10, name='BBW_PCT')
        
        # 🌐 Keltner Channel
        typical_price = (self.data.High + self.data.Low + self.data.Close)/3
        self.kc_mid = self.I(talib.EMA, typical_price, 20, name='KC_Mid')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 20)
        self.kc_upper = self.I(lambda: self.kc_mid + 2*self.atr, name='KC_Upper')
        self.kc_lower = self.I(lambda: self.kc_mid - 2*self.atr, name='KC_Lower')
        
        # 📊 Volume Dynamics
        self.vol_pct80 = self.I(ta.percentile, self.data.Volume, 50, q=80, name='VOL_PCT80')
        self.vol_pct90 = self.I(ta.percentile, self.data.Volume, 50, q=90, name='VOL_PCT90')
        
        print("🌙 Lunar Indicators Activated! 🚀")
        print("✨ All systems go - No backtesting.lib dependencies detected! 🌌")

    def next(self):
        # 🌑 Skip early bars
        if len(self.data) < 50:
            return
            
        # 🛑 Exit Conditions
        if self.position:
            if self.position.is_long and (self.data.Volume[-1] > self.vol_pct90[-1] or self.data.Close[-1] < self.trailing_sl):
                print(f"🌕 Closing LONG at {self.data.Close[-1]:.2f} | Moon Profit Protection! 🌙")
                self.position.close()
            elif self.position.is_short and (self.data.Volume[-1] > self.vol_pct90[-1] or self.data.Close[-1] > self.trailing_sl):
                print(f"🌑 Closing SHORT at {self.data.Close[-1]:.2f} | Stellar Exit Protocol! 🌠")
                self.position.close()
        
        # 🚀 Entry Logic
        else:
            bb_squeeze = self.bbw[-1] < self.bbw_pct[-1]
            vol_boost = self.data.Volume[-1] > self.vol_pct80[-1]
            
            if bb_squeeze and vol_boost:
                # 🔭 Long Setup