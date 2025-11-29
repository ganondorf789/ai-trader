I'll fix the code while maintaining the original strategy logic and adding Moon Dev themed improvements. Here's the complete corrected version:

```python
# moon_dev_deltaband_backtest.py 🌙🚀
import numpy as np
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, crossunder

# Moon Dev Data Preparation Ritual ✨
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse the data columns 🌙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🌌
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Time alignment magic 🌗
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class DeltaBandConvergence(Strategy):
    # Moon Dev Configuration Parameters 🌙
    risk_pct = 0.01
    bb_period = 20
    vwap_period = 50
    delta_volume_window = 960  # 10 days in 15m intervals
    
    def init(self):
        # Celestial Indicators 🌠
        tp = (self.data.High + self.data.Low + self.data.Close) / 3
        vwap_num = self.I(talib.SMA, tp * self.data.Volume, timeperiod=self.vwap_period)
        vwap_den = self.I(talib.SMA, self.data.Volume, timeperiod=self.vwap_period)
        self.vwap = vwap_num / vwap_den
        
        # Bollinger Bands Constellation 🌌
        self.bb_upper, _, self.bb_lower = self.I(talib.BBANDS, self.data.Close, 
                                                timeperiod=self.bb_period, 
                                                nbdevup=2, nbdevdn=2, matype=0)
        self.bb_width = self.I(lambda u, l: (u - l)/self.I(talib.SMA, self.data.Close, 20),
                              self.bb_upper, self.bb_lower)
        
        # Volume Alchemy 🌊
        self.delta_volume = self.data.Volume
        self.median_delta = self.I(ta.median, self.delta_volume, length=self.delta_volume_window)
        
        # Lunar State Tracking 🌑
        self.consecutive_losses = 0
        self.daily_high_equity = self._broker._cash
        self.last_date = None

    def next(self):
        # Moon Phase Checks 🌓
        current_date = self.data.index[-1].date()
        if current_date != self.last_date:
            self.daily_high_equity = self._broker._equity
            self.last_date = current_date
            
        # Celestial Risk Shields 🛡️
        if self._broker._equity < 0.95 * self.daily_high_equity:
            print("🌘 DAILY LOSS LIMIT BREACHED! Trading halted.")
            return
            
        if self.consecutive_losses >= 3:
            print("💥 3 CONSECUTIVE LOSSES! Moon Dev protection activated.")
            return
            
        # Cosmic Timing Filter 🕒
        bar_time = self.data.index[-1].time()
        if (bar_time.hour == 0 and bar_time.minute < 30) or \
           (bar_time.hour == 23 and bar_time.minute >= 30):
            return
            
        # Stellar Entry Conditions 🌠
        if not self.position:
            # VWAP Convergence Check
            price = self.data.Close[-1]
            if abs(price/self.vwap[-1] - 1) > 0.005:
                return
                
            # Bandwidth Contraction
            if len(self.bb_width) < 4: 
                return
            if not (self.bb_width[-1] < self.bb_width[-2] < self.bb_width[-3] < self.bb_width[-4]):
                return
                
            # Volume Confirmation
            if self.delta_volume[-1] < 2 * self.median_delta[-1]:
                return
                
            # Long Entry
            if crossover(self.data.Close, self.bb_upper):
                size = min