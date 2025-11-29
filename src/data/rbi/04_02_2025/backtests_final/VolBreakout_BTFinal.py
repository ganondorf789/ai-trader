I'll fix the incomplete code while maintaining the original strategy logic. Here's the complete, debugged version with Moon Dev themed prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np
import datetime

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and prepare data
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

class VolBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate technical indicators
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=1920, name='VOL_SMA_20D')
        self.realized_vol = self.I(talib.STDDEV, self.data.Close, timeperiod=4, nbdev=1, name='REAL_VOL_1H')
        self.vol_avg = self.I(talib.SMA, self.realized_vol, timeperiod=1920, name='VOL_AVG_20D')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR_14')
        
        # Overnight session tracking
        self.current_oh = -np.inf
        self.current_ol = np.inf
        self.overnight_high = None
        self.overnight_low = None
        
    def next(self):
        dt = self.data.index[-1]
        current_time = dt.time()
        
        # Update overnight levels
        if (current_time >= datetime.time(16, 0)) or (current_time < datetime.time(9, 30)):
            self.current_oh = max(self.current_oh, self.data.High[-1])
            self.current_ol = min(self.current_ol, self.data.Low[-1])
            print(f"🌒 Moon Phase Update: Tracking overnight levels | High: {self.current_oh:.2f} | Low: {self.current_ol:.2f}")
            return  # Skip other checks during overnight
        
        # Set overnight levels at regular session start
        if current_time >= datetime.time(9, 30) and current_time < datetime.time(11, 30):
            if not self.overnight_high:
                self.overnight_high = self.current_oh
                self.overnight_low = self.current_ol
                print(f"🌕 Full Moon Alert! Overnight High: {self.overnight_high:.2f} | Overnight Low: {self.overnight_low:.2f}")
                
            # Check entry conditions
            vol_condition = self.data.Volume[-1] > 1.5 * self.volume_sma[-1]
            vola_condition = self.realized_vol[-1] > self.vol_avg[-1]
            
            if vol_condition and vola_condition:
                # Long entry
                if self.data.High[-1] > self.overnight_high and not self.position:
                    print(f"🚀 Moon Rocket Launch! Long signal detected at {self.data.Close[-1]:.2f}")
                    self.buy(
                        sl=self.overnight_low + (self.overnight_high - self.overnight_low)/2,  # Midpoint stop
                        tp=self.data.Close[-1] + 2*(self.data.Close[-1] - (self.overnight_low + (self.overnight_high - self.overnight_low)/2)  # 2:1 RR
                    )
                    
                # Short entry
                elif self.data.Low[-1] < self.overnight_low and not self.position:
                    print(f"🌘 Moon Shadow Alert! Short signal detected at {self.data.Close[-1]:.2f}")
                    self.sell(
                        sl=self.overnight_high - (self.overnight_high - self.overnight_low)/2,  # Midpoint stop
                        tp=self.data.Close[-1] - 2*((self.overnight_