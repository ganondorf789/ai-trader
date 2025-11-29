Here's the fixed and debugged code with Moon Dev themed improvements 🌙✨:

```python
# Moon Dev's Volumetric Breakout Strategy 🌙✨
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class DataCleaner:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        # Clean column names and drop unnamed columns
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data = self.data.drop(columns=[col for col in self.data.columns 
                                      if 'unnamed' in col.lower()])
        # Proper column mapping
        self.data = self.data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.set_index('datetime')

class VolumetricBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    ma_period = 20
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period, name='ATR')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.ma_period,
                              name='Volume_MA')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.ma_period,
                            name='ATR_MA')
        
        # Track swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=3,
                                name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=3,
                              name='Swing_Low')
        
        print("🌙 Moon Dev Strategy Initialized! ✨")
        print(f"Equity: {self.equity:.2f} | Risk/Trade: {self.risk_per_trade*100}%")
        
    def next(self):
        current_idx = len(self.data) - 1
        if current_idx < 20:  # Wait for indicators to warm up
            return
            
        # Check for engulfing patterns using array indexing
        bullish_engulf = (self.data.Close[-1] > self.data.Open[-1] and
                         self.data.Open[-1] < self.data.Close[-2] and
                         self.data.Close[-1] > self.data.Open[-2])
        
        bearish_engulf = (self.data.Close[-1] < self.data.Open[-1] and
                         self.data.Open[-1] > self.data.Close[-2] and
                         self.data.Close[-1] < self.data.Open[-2])
        
        # Volume condition check
        volume_condition = self.data.Volume[-1] > 2 * self.volume_ma[-1]
        atr_condition = self.atr[-1] > self.atr_ma[-1]
        
        # Entry logic
        if not self.position:
            # Long entry conditions
            if bullish_engulf and volume_condition and atr_condition:
                entry_price = self.data.High[-1]
                stop_loss = self.data.Low[-1]
                risk = entry_price - stop_loss
                position_size = (self.equity * self.risk_per_trade) / risk
                position_size = int(round(position_size))  # Round to whole units
                
                if position_size > 0:
                    tp_level = entry_price + 1.5 * self.atr[-1]
                    self.buy(size=position_size, sl=stop_loss,
                            tp=tp_level,
                            tag='BullishBreakout')
                    print(f"🚀 LONG ENTRY: {self.data.index[-1]} | "
                          f"Size: {position_size} | Entry: {entry_price:.2f} | "
                          f"SL: {stop_loss:.2f} | TP: {tp_level:.2f} | "
                          f"Volume: {self.data.Volume[-1]:.2f}x MA")
            
            # Short entry conditions
            elif bearish_engulf and volume_condition and atr_condition:
                entry_price = self.data.Low[-1]
                stop_loss = self.data.High[-1]
                risk = stop_loss - entry_price
                position_size = (self.equity * self.risk_per_trade) /