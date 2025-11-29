Here's the debugged and fixed code with Moon Dev themed improvements 🌙✨:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

# Load and preprocess data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names and handle data
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

class DivergencePulse(Strategy):
    ema_short = 5
    ema_long = 20
    atr_period = 14
    swing_period = 20
    volume_multiplier = 1.5
    risk_pct = 0.01
    
    def init(self):
        # Calculate indicators with TA-Lib using self.I()
        self.ema5 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_short)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_long)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        self.trailing_stop = None  # To track dynamic stop level

    def next(self):
        # Moon-themed debug prints
        print(f"🌙 Current Close: {self.data.Close[-1]:.2f} | EMA5: {self.ema5[-1]:.2f} | EMA20: {self.ema20[-1]:.2f} | ATR: {self.atr[-1]:.2f}")
        
        # Manage existing positions
        if self.position.is_long:
            # Update trailing stop for longs
            new_swing_low = self.swing_low[-1]
            self.trailing_stop = max(self.trailing_stop, new_swing_low) if self.trailing_stop else new_swing_low
            
            if self.data.Low[-1] <= self.trailing_stop:
                self.position.close()
                print(f"🌙✨ Trailing Stop Hit! Closing Long at {self.data.Close[-1]:.2f}")
                
        elif self.position.is_short:
            # Update trailing stop for shorts
            new_swing_high = self.swing_high[-1]
            self.trailing_stop = min(self.trailing_stop, new_swing_high) if self.trailing_stop else new_swing_high
            
            if self.data.High[-1] >= self.trailing_stop:
                self.position.close()
                print(f"🌙✨ Trailing Stop Hit! Closing Short at {self.data.Close[-1]:.2f}")
        
        # Entry logic when no position exists
        if not self.position:
            # Bullish divergence with volume spike
            if (self.ema5[-2] < self.ema20[-2] and self.ema5[-1] > self.ema20[-1]) and \
               self.data.Volume[-1] > self.volume_multiplier * self.vol_sma[-1]:
                
                risk_amount = self.equity * self.risk_pct
                atr_value = self.atr[-1]
                stop_distance = 1.5 * atr_value
                entry_price = self.data.Close[-1]
                
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))  # Ensure whole number of units
                
                if position_size > 0:
                    # Set initial stop and trailing stop
                    initial_stop = entry_price - stop_distance
                    self.trailing_stop = max(initial_stop, self.swing_low[-1])
                    self.buy(size=position_size, sl=self.trailing_stop)
                    print(f"🚀