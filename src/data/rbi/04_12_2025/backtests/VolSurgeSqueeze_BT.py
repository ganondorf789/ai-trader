```python
# 🌙 MOON DEV BACKTESTING SCRIPT FOR VOLSURGESQUEEZE STRATEGY 🚀

# 1. NECESSARY IMPORTS
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
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
        # 🌙 INDICATOR CALCULATIONS USING TA-LIB
        # Volume averages
        self.volume_20_avg = self.I(talib.SMA, self.data.Volume, 20, name='Volume 20MA')
        self.volume_3_avg = self.I(talib.SMA, self.data.Volume, 3, name='Volume 3MA')
        
        # Bollinger Bands with bandwidth calculation
        self.bb_upper = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[0], self.data.Close, name='BB Upper')
        self.bb_middle = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[1], self.data.Close, name='BB Middle')
        self.bb_lower = self.I(lambda close: talib.BBANDS(close, 20, 2, 2)[2], self.data.Close, name='BB Lower')
        self.bb_bandwidth = self.I(lambda u,l,m: ((u-l)/m)*100, self.bb_upper, self.bb_lower, self.bb_middle, name='BB Width')
        
        # MACD with signal line
        self.macd = self.I(lambda close: talib.MACD(close, 12, 26, 9)[0], self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda close: talib.MACD(close, 12, 26, 9)[1], self.data.Close, name='MACD Signal')
        
        # ATR for risk management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        # 🌑 TRADE STATE TRACKING
        self.consecutive_losses = 0
        self.in_cool_off = False

    def next(self):
        # Skip cool-off periods
        if self.in_cool_off:
            print("🌑 COOL-OFF ACTIVE: Skipping trades")
            return
            
        if not self.position:
            # 💹 LONG ENTRY CONDITIONS
            vol_surge = self.volume_3_avg[-1] > 1.5*self.volume_20_avg[-1]
            bb_squeeze = self.bb_bandwidth[-1] < 10
            price_break = self.data.Close[-1] > self.bb_upper[-1]
            macd_bullish = crossover(self.macd, self.macd_signal)
            
            if all([vol_surge, bb_squeeze, price_break, macd_bullish]):
                risk_amount = self.equity * 0.02  # 2% risk
                atr_value = self.atr[-1]
                sl_price = self.data.Close[-1] - atr_value
                position_size = int(round(risk_amount / (self.data.Close[-1] - sl_price))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=sl_price, tp=self.data.Close[-1] + 2*atr_value)
                    print(f"🌙 MOON DEV LONG! 🚀 Size: {position_size} | Entry: {self.data.Close[-1]:.2f} | SL: {sl_price:.2f} | TP: {self.data.Close[-1] + 2*atr_value:.2f}")
            
            # 💹 SHORT ENTRY CONDITIONS
            price_