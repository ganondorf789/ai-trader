I'll fix the code by removing the `backtesting.lib` import and ensuring all indicators use proper packages. Here's the corrected version with Moon Dev themed debug prints:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

class LiquidationAcceleration(Strategy):
    risk_percent = 0.01
    chandelier_multiplier = 3
    exit_period = 96  # 24 hours in 15m intervals
    
    def init(self):
        # Core indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Trade tracking
        self.entry_bar = 0
        self.max_high = 0
        self.min_low = 0

    def next(self):
        if not self.position:
            # Long entry logic - using array indexing instead of crossover
            if (self.data.Close[-2] < self.swing_high[-2] and 
                self.data.Close[-1] > self.swing_high[-1] and 
                self.data.Volume[-1] > self.volume_sma[-1]):
                
                atr_value = self.atr[-1]
                if atr_value == 0:
                    print("🌙⚠️ ATR value is zero - skipping trade")
                    return
                
                stop_distance = self.chandelier_multiplier * atr_value
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_bar = len(self.data) - 1
                    self.max_high = self.data.High[-1]
                    print(f"🌙🚀 LONG ENTRY | Price: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {self.data.Close[-1]-stop_distance:.2f} | Moon Rocket Activated! 🌕")

            # Short entry logic - using array indexing instead of crossover
            elif (self.data.Close[-2] > self.swing_low[-2] and 
                  self.data.Close[-1] < self.swing_low[-1] and 
                  self.data.Volume[-1] > self.volume_sma[-1]):
                
                atr_value = self.atr[-1]
                if atr_value == 0:
                    print("🌙⚠️ ATR value is zero - skipping trade")
                    return
                
                stop_distance = self.chandelier_multiplier * atr_value
                risk_amount = self.equity * self.risk_percent
                position_size = int(round(risk_amount / stop_distance))
                
                if position_size > 0:
                    self.sell(size=position_size)
                    self.entry_bar = len(self.data) - 1
                    self.min_low = self.data.Low[-1]
                    print(f"🌙📉 SHORT ENTRY | Price: {self.data.Close[-1]:.2f} | Size: {position_size} | SL: {self.data.Close[-1]+stop_distance:.2f} | Lunar Gravity Engaged! 🌑")

        else:
            # Update trailing levels
            if self.position.is_long:
                self.max_high = max(self.max_high, self.data.High[-1])
                current_atr = self.atr[-1]
                stop_level = self.max_high - (current_atr * self.chandelier_multiplier)
                
                if self.data.Low[-1] <=