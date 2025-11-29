Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR BANDWIDTHSURGE STRATEGY 🚀

# ================
# IMPORTS
# ================
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ================
# DATA PREPARATION
# ================
# Load and prepare data with Moon Dev magic ✨
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names like a Moon Dev wizard 🧙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping with Moon Dev precision 🌙
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})

# Set index and convert to proper format
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# ================
# STRATEGY CLASS
# ================
class BandwidthSurge(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade 🌙
    atr_multiplier = 1.5  # Trailing stop multiplier 🛡️
    
    def init(self):
        # 🌙 BOLLINGER BAND CALCULATIONS
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return upper
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return lower
        
        self.upper_band = self.I(bb_upper, self.data.Close, name='UpperBB')
        self.lower_band = self.I(bb_lower, self.data.Close, name='LowerBB')
        self.bb_width = self.I(lambda u, l: u - l, self.upper_band, self.lower_band, name='BBWidth')
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=10, name='BBWidthLow')
        
        # 🌊 VOLUME SURGE INDICATOR
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolumeSMA')
        
        # 🛡️ ATR FOR RISK MANAGEMENT
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR14')
        
    def next(self):
        # Moon Dev debug prints for key indicators 🌙
        print(f"🌙 Current BB Width: {self.bb_width[-1]:.2f} | 10D Low: {self.bb_width_low[-1]:.2f} | Volume: {self.data.Volume[-1]:.2f} vs SMA: {self.volume_sma[-1]:.2f}")
        
        # Only trade when no position exists
        if not self.position:
            # Check volatility contraction and volume surge
            if (self.bb_width[-1] < self.bb_width_low[-1] and 
                self.data.Volume[-1] > 1.5 * self.volume_sma[-1]):
                
                # Long entry condition 🌕
                if self.data.Close[-1] > self.upper_band[-1]:
                    self.entry_width = self.bb_width[-1]
                    atr_value = self.atr[-1]
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / (self.atr_multiplier * atr_value)))
                    
                    if position_size > 0:
                        self.buy(size=position_size)
                        print(f"🚀 MOON LAUNCH: LONG {position_size} BTC @ {self.data.Close[-1]:.2f} | ATR: {atr_value:.2f} 🌕")
                
                # Short entry condition 🌑
                elif self.data.Close[-1] < self.lower_band[-1]:
                    self.entry_width = self.bb_width[-1]
                    atr_value = self