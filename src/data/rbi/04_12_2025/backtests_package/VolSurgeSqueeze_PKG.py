Here's the fixed code with all backtesting.lib imports removed and proper replacements implemented:

```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLSURGESQUEEZE STRATEGY 🚀

# 1. NECESSARY IMPORTS (CLEANED OF BACKTESTING.LIB)
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# 2. DATA PREPARATION
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean column names and drop unnamed columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper column names
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 3. STRATEGY IMPLEMENTATION
class VolSurgeSqueeze(Strategy):
    def init(self):
        # 🌙 INDICATOR CALCULATIONS USING TA-LIB (MOON DEV APPROVED)
        # Volume averages
        self.volume_20_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='Volume 20MA')
        self.volume_3_avg = self.I(talib.SMA, self.data.Volume, timeperiod=3, name='Volume 3MA')
        
        # Bollinger Bands with bandwidth calculation
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[0], 
                              self.data.Close, name='BB Upper')
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[1], 
                               self.data.Close, name='BB Middle')
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[2], 
                              self.data.Close, name='BB Lower')
        self.bb_bandwidth = self.I(lambda u,l,m: ((u-l)/m)*100, 
                                  self.bb_upper, self.bb_lower, self.bb_middle, name='BB Width')
        
        # MACD with signal line (MOON DEV OPTIMIZED)
        self.macd = self.I(lambda close: talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0], 
                          self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda close: talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1], 
                                 self.data.Close, name='MACD Signal')
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
        # 🌑 TRADE STATE TRACKING (MOON DEV SPECIAL)
        self.consecutive_losses = 0
        self.in_cool_off = False

    def next(self):
        # Skip cool-off periods
        if self.in_cool_off:
            print("🌑 COOL-OFF ACTIVE: Skipping trades")
            return
            
        if not self.position:
            # 💹 LONG ENTRY CONDITIONS (MOON DEV APPROVED)
            vol_surge = self.volume_3_avg[-1] > 1.5*self.volume_20_avg[-1]
            bb_squeeze = self.bb_bandwidth[-1] < 10
            price_break = self.data.Close[-1] > self.bb_upper[-1]
            macd_bullish = (self.macd[-2] < self.macd_signal[-2] and self.macd[-1] > self.macd_signal[-1])  # Replaced crossover
            
            if all([vol_surge, bb_squeeze, price_break, macd_bullish]):
                risk_amount = self.equity * 0.02  # 2% risk
                atr_value = self.atr[-1]
                sl_price = self.data.Close[-1] - atr_value
                position_size = int(round(risk_amount / (self