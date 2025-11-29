Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
def load_data(path):
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class DivergentBand(Strategy):
    risk_pct = 0.01  # 1% risk per trade 🌕
    atr_period = 14
    bb_period = 14
    bb_dev = 2
    swing_window = 5
    
    def init(self):
        # Moon Dev Indicators 🌙✨
        self.macd = self.I(lambda close: talib.MACD(close, 12, 26, 9)[0], self.data.Close, name='MACD')
        self.macd_signal = self.I(lambda close: talib.MACD(close, 12, 26, 9)[1], self.data.Close, name='MACD Signal')
        self.upper_bb = self.I(lambda close: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[0], 
                             self.data.Close, name='Upper BB')
        self.lower_bb = self.I(lambda close: talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)[2], 
                             self.data.Close, name='Lower BB')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period, name='ATR')
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_window, name='Swing High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_window, name='Swing Low')
        
        self.last_swing_high = self.last_swing_low = None  # Divergence tracking 🌓

    def next(self):
        # Moon Dev Signal Detection 🌙🚀
        price = self.data.Close[-1]
        upper = self.upper_bb[-1]
        lower = self.lower_bb[-1]
        
        # Calculate %b for band positioning
        if upper != lower:
            b_percent = (price - lower) / (upper - lower)
        else:
            b_percent = 0.5
            
        # Detect swing points
        current_swing_high = self.swing_high[-1]
        current_swing_low = self.swing_low[-1]
        
        # Bullish divergence detection 🌕
        bullish_div = False
        if self.data.High[-1] == current_swing_high and self.last_swing_high:
            if (self.data.High[-1] > self.last_swing_high[0] and 
                self.macd[-1] < self.last_swing_high[1]):
                print(f"🌑 Bearish divergence! Price HH {self.data.High[-1]:.2f} vs MACD LH {self.macd[-1]:.2f}")
                self.last_swing_high = (self.data.High[-1], self.macd[-1])
                
        # Bearish divergence detection 🌑
        bearish_div = False
        if self.data.Low[-1] == current_swing_low and self.last_swing_low:
            if (self.data.Low[-1] < self.last_swing_low[0] and 
                self.macd[-1] > self.last_swing_low[1]):
                print(f"🌕 Bullish divergence! Price LL {self.data.Low[-1]:.2f} vs MACD HL {self.macd[-1]:.2f}")
                self.last_swing_low = (self.data.Low[-1], self.macd[-1])
                bullish_div = True
        
        # Entry Logic 🌙✨
        if not self.position:
            # Long entry conditions
            if (