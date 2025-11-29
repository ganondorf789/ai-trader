Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 Moon Dev's VolumetricBandwidth Backtest Script 🚀

import pandas as pd
import pandas_ta as ta
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Data Preparation 🌐
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and format data columns 🧹
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolumetricBandwidth(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌡️
    
    def init(self):
        # Indicator Calculation Suite 📊
        self.vwap = self.I(ta.vwap, high=self.data.High, low=self.data.Low, close=self.data.Close, volume=self.data.Volume, name='VWAP')
        
        # Bollinger Bands Calculation 🔔
        bb_upper, bb_middle, bb_lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda: bb_upper, name='UPPER_BB')
        self.lower_band = self.I(lambda: bb_lower, name='LOWER_BB')
        self.mid_band = self.I(lambda: bb_middle, name='MID_BB')
        
        # Bandwidth Metrics 📏
        bb_width = (bb_upper - bb_lower) / bb_middle
        self.bb_width = self.I(lambda: bb_width, name='BB_WIDTH')
        self.bb_width_sma = self.I(talib.SMA, bb_width, timeperiod=20, name='BB_WIDTH_SMA')
        
        # Volume Filter 🌊
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOLUME_MA')
        
    def next(self):
        # Moon Dev Debug Prints 🌙
        print(f"\n✨ New Bar: {self.data.index[-1]} ✨")
        print(f"Equity: ${self.equity:,.2f} | Price: {self.data.Close[-1]:.2f}")
        
        # Entry Logic 🚪
        if not self.position:
            # Long Entry Conditions 🌟
            long_trigger = ((self.data.Close[-2] < self.vwap[-2] and self.data.Close[-1] > self.vwap[-1]) and 
                          (self.bb_width[-1] > self.bb_width_sma[-1]))
            
            if long_trigger:
                self.calculate_position('long')
            
            # Short Entry Conditions 🌪️
            short_trigger = ((self.data.Close[-2] > self.vwap[-2] and self.data.Close[-1] < self.vwap[-1]) and 
                           (self.bb_width[-1] > self.bb_width_sma[-1]) and 
                           (self.data.Volume[-1] > self.volume_ma[-1]))
            
            if short_trigger:
                self.calculate_position('short')
        
        # Exit Logic 🚪
        if self.position.is_long and self.data.Close[-1] < self.lower_band[-1]:
            self.position.close()
            print("🌙 LONG EXIT: Price pierced Lower Band! 🛑")
            
        if self.position.is_short and self.data.Close[-1] > self.upper_band[-1]:
            self.position.close()
            print("🌙 SHORT EXIT: Price breached Upper Band! 🛑")
            
        # Time-based Exit ⏳
        if self.position and (len(self.data) - self.position.entry_bar) >= 5:
            self.position.close()
            print("🌙 TIME EXIT: 5-bar limit reached! ⌛")
    
    def calculate_position(self, direction):
        # Risk Management Engine 🛡️
        entry_price = self.data.Close[-1]
        bb_mid = self.mid_band[-1]
        bb_range = self.upper_band[-1] - self.lower_band[-1]
        
        # Stop Calculation 🛑
        risk